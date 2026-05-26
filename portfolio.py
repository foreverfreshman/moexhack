"""
Portfolio management: целевые веса → orders.

Архитектурно:
    1. Strategies (mom21, gap_fade) выдают свои target weights (ticker → weight)
       нормализованные так что sum(|weight|) <= strategy_capital_share
    2. Portfolio.combine_strategies() складывает их в единый target_weights
    3. Portfolio.weights_to_shares() конвертирует в целое число лотов по текущим ценам
    4. Portfolio.compute_orders() считает diff = target − current → список Order

Ключевые точки:
    - Учитываем lot sizes MOEX (SBER = 10, VTBR = 10000, etc.)
    - Округление лот к ближайшему целому (не floor, не ceil — round-half-to-even)
    - Защита от over-allocation (sum |target| ≤ 1.0 = no leverage)
    - Tolerance: маленькие diff игнорируются (защита от шума)

Использование:
    portfolio = Portfolio(total_capital=1_000_000)
    target = portfolio.combine_strategies([
        StrategyWeights("mom21", {"SBER": 0.05, "LKOH": -0.05, ...}, capital_share=0.3),
        StrategyWeights("gap_fade", {"GAZP": -0.07, ...}, capital_share=0.7),
    ])
    shares = portfolio.weights_to_shares(target, current_prices)
    orders = portfolio.compute_orders(shares, current_positions_from_arena)
    for order in orders:
        client.submit_order(order.ticker, order.side, order.quantity)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("portfolio")


# ============================================================
# MOEX lot sizes
# ============================================================
# ВНИМАНИЕ: эти значения — best-effort guess. Перед боевым запуском нужно
# verify через MOEX API или ArenaGo (например, попробовать submit_order
# с qty=1 и посмотреть, отклоняется ли как 'not multiple of lot').
# Источник: публичные данные MOEX по лотам blue chips на 2024-2025.

DEFAULT_LOT_SIZES: Dict[str, int] = {
    "AFLT": 10,
    "ALRS": 10,
    "CHMF": 1,
    "GAZP": 10,
    "GMKN": 1,
    "LKOH": 1,
    "MGNT": 1,
    "MOEX": 10,
    "MTSS": 10,
    "NLMK": 10,
    "NVTK": 1,
    "PIKK": 10,
    "PLZL": 1,
    "ROSN": 1,
    "SBER": 10,
    "SNGSP": 100,
    "T": 1,        # TCS Group
    "VTBR": 10000, # копеечные акции, лот огромный
    "X5": 1,
    "YDEX": 1,
}


# ============================================================
# Data classes
# ============================================================

@dataclass
class Order:
    """Целевой ордер для submit к ArenaGo."""
    ticker: str
    side: str       # 'B' или 'S'
    quantity: int   # в штуках акций (не лотах!)

    def __post_init__(self):
        if self.side not in ("B", "S"):
            raise ValueError(f"side must be 'B' or 'S', got {self.side!r}")
        if self.quantity <= 0:
            raise ValueError(f"quantity must be positive, got {self.quantity}")


@dataclass
class StrategyWeights:
    """Выход стратегии.

    weights: ticker → доля от ОБЩЕГО капитала (не от capital_share стратегии).
             Например, если mom21 на 30% капитала делает long 0.05 = 5% от 1M.
             Сумма abs(weights) для одной стратегии должна быть ≤ capital_share.
    """
    strategy_name: str
    weights: Dict[str, float]
    capital_share: float = 1.0   # для self-validation, не для масштабирования

    def validate(self) -> None:
        gross = sum(abs(w) for w in self.weights.values())
        if gross > self.capital_share + 1e-6:
            raise ValueError(
                f"[{self.strategy_name}] gross weight {gross:.4f} exceeds "
                f"capital_share {self.capital_share:.4f}"
            )


# ============================================================
# Portfolio class
# ============================================================

class Portfolio:
    """Менеджер целевого портфеля для одного бота / одного ArenaGo-аккаунта."""

    def __init__(
        self,
        total_capital: float,
        lot_sizes: Optional[Dict[str, int]] = None,
        min_trade_value: float = 1000.0,   # минимальная стоимость трейда в ₽
        max_gross_exposure: float = 1.0,   # 1.0 = no leverage
    ):
        self.total_capital = total_capital
        self.lot_sizes = lot_sizes or DEFAULT_LOT_SIZES
        self.min_trade_value = min_trade_value
        self.max_gross_exposure = max_gross_exposure

    # --------------------------------------------------------
    # Combining strategies
    # --------------------------------------------------------

    def combine_strategies(
        self, strategies: List[StrategyWeights],
    ) -> Dict[str, float]:
        """Складывает target weights из разных стратегий.

        Если несколько стратегий хотят один и тот же тикер — веса суммируются
        (long+long = больше long, long+short = netting).

        Returns: единый dict ticker → weight (доля общего капитала).
        """
        combined: Dict[str, float] = {}
        for s in strategies:
            s.validate()
            for ticker, w in s.weights.items():
                if abs(w) < 1e-9:
                    continue
                combined[ticker] = combined.get(ticker, 0.0) + w

        # Проверка общей gross-экспозиции
        gross = sum(abs(w) for w in combined.values())
        if gross > self.max_gross_exposure + 1e-6:
            log.warning(
                f"Combined gross {gross:.4f} > max {self.max_gross_exposure}; "
                f"scaling all weights by {self.max_gross_exposure / gross:.4f}"
            )
            scale = self.max_gross_exposure / gross
            combined = {t: w * scale for t, w in combined.items()}

        return combined

    # --------------------------------------------------------
    # Weights → shares
    # --------------------------------------------------------

    def weights_to_shares(
        self,
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
    ) -> Dict[str, int]:
        """Конвертирует target_weights → integer количество акций.

        Округляет до ближайшего целого числа лотов (banker's rounding).
        Пропускает тикеры:
            - без цены
            - с ценой ≤ 0
            - где результат |shares| × price < min_trade_value
        """
        out: Dict[str, int] = {}
        for ticker, w in target_weights.items():
            if abs(w) < 1e-9:
                continue
            price = current_prices.get(ticker)
            if price is None or price <= 0:
                log.warning(f"No price for {ticker} — skipping")
                continue
            target_rub = w * self.total_capital   # подписанное значение
            raw_shares = target_rub / price
            lot = self.lot_sizes.get(ticker, 1)
            if lot <= 0:
                lot = 1
            # banker's rounding: round-half-to-even
            lots = round(raw_shares / lot)
            shares = int(lots * lot)
            if shares == 0:
                continue
            if abs(shares * price) < self.min_trade_value:
                log.debug(f"{ticker} target value {shares * price:.0f} < min "
                          f"{self.min_trade_value}, skipping")
                continue
            out[ticker] = shares
        return out

    # --------------------------------------------------------
    # Shares → orders (diff)
    # --------------------------------------------------------

    def compute_orders(
        self,
        target_shares: Dict[str, int],
        current_shares: Dict[str, int],
        current_prices: Optional[Dict[str, float]] = None,
        min_delta_value: Optional[float] = None,
    ) -> List[Order]:
        """Diff между target и current → список Order.

        Если current_prices задан, маленькие изменения (|delta| × price <
        min_delta_value, по умолчанию = self.min_trade_value) фильтруются
        чтобы не торговать копейки.
        """
        if min_delta_value is None:
            min_delta_value = self.min_trade_value

        all_tickers = set(target_shares.keys()) | set(current_shares.keys())
        orders: List[Order] = []

        for ticker in sorted(all_tickers):
            target = target_shares.get(ticker, 0)
            current = current_shares.get(ticker, 0)
            delta = target - current
            if delta == 0:
                continue

            # Фильтр шума: маленькие изменения не торгуем
            if current_prices and ticker in current_prices:
                price = current_prices[ticker]
                if abs(delta) * price < min_delta_value:
                    log.debug(f"{ticker}: delta {delta} value {abs(delta)*price:.0f}"
                              f" < min {min_delta_value}, skipping")
                    continue

            # Округлим delta до лота (на всякий случай, должно уже быть кратно)
            lot = self.lot_sizes.get(ticker, 1)
            if delta % lot != 0:
                # Подтянуть к ближайшему лоту в сторону target
                rounded = (delta // lot) * lot
                if rounded == 0:
                    rounded = lot if delta > 0 else -lot
                log.warning(f"{ticker}: delta {delta} not multiple of lot {lot}, "
                            f"rounding to {rounded}")
                delta = rounded

            side = "B" if delta > 0 else "S"
            qty = abs(delta)
            orders.append(Order(ticker=ticker, side=side, quantity=qty))

        return orders

    # --------------------------------------------------------
    # Convert from ArenaGo Position list
    # --------------------------------------------------------

    @staticmethod
    def positions_to_shares(positions) -> Dict[str, int]:
        """Преобразует List[Position] от ArenaGoClient в dict ticker → shares."""
        out = {}
        for p in positions:
            if p.quantity != 0:
                out[p.ticker] = int(p.quantity)
        return out


# ============================================================
# Convenience: render single rebalance plan
# ============================================================

def plan_rebalance(
    portfolio: Portfolio,
    strategies: List[StrategyWeights],
    current_prices: Dict[str, float],
    current_positions,   # List[Position]
) -> Tuple[Dict[str, int], Dict[str, int], List[Order]]:
    """Целиком собирает ребаланс: weights → shares → orders.

    Returns: (target_shares, current_shares, orders)
    """
    target_weights = portfolio.combine_strategies(strategies)
    target_shares = portfolio.weights_to_shares(target_weights, current_prices)
    current_shares = portfolio.positions_to_shares(current_positions)
    orders = portfolio.compute_orders(target_shares, current_shares, current_prices)
    return target_shares, current_shares, orders


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from arenago_client import MockArenaGoClient, Position

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # --------- Сценарий ---------
    # Капитал 1M ₽. Две стратегии:
    #  mom21 (30% cap): long SBER, LKOH, GAZP по 5% каждый; short PIKK, MTSS, ALRS по 5%
    #  gap_fade (70% cap): short NVTK на 14% (один сильный гэп вверх), long VTBR на 14%

    portfolio = Portfolio(total_capital=1_000_000.0)

    prices = {
        "SBER": 280.0, "LKOH": 6500.0, "GAZP": 142.0,
        "PIKK": 644.0, "MTSS": 243.0, "ALRS": 57.5,
        "NVTK": 1227.0, "VTBR": 0.097,
    }

    mom21_weights = StrategyWeights(
        strategy_name="mom21",
        weights={
            "SBER": +0.05, "LKOH": +0.05, "GAZP": +0.05,
            "PIKK": -0.05, "MTSS": -0.05, "ALRS": -0.05,
        },
        capital_share=0.30,
    )
    gap_fade_weights = StrategyWeights(
        strategy_name="gap_fade",
        weights={
            "NVTK": -0.14,
            "VTBR": +0.14,
        },
        capital_share=0.70,   # выделено 70% но фактически используется 28%
    )

    target_weights = portfolio.combine_strategies([mom21_weights, gap_fade_weights])
    log.info(f"Combined target weights ({len(target_weights)} positions):")
    for t, w in sorted(target_weights.items(), key=lambda x: -abs(x[1])):
        log.info(f"  {t:8s}: {w:+.4f}  ({w * portfolio.total_capital:>+12,.0f} ₽)")
    log.info(f"  Total gross: {sum(abs(w) for w in target_weights.values()):.4f}")

    # --------- weights → shares ---------
    target_shares = portfolio.weights_to_shares(target_weights, prices)
    log.info(f"\nTarget shares:")
    for t in sorted(target_shares):
        s = target_shares[t]
        log.info(f"  {t:8s}: {s:>+8,} shares  (value {s * prices[t]:>+12,.0f} ₽, "
                 f"lot={portfolio.lot_sizes.get(t, 1)})")

    # --------- Симулируем текущие позиции через Mock client ---------
    mock = MockArenaGoClient(portfolio_name="test", price_provider=prices.get)
    # Создадим стартовое состояние: уже есть SBER long 500, NVTK short 50, ROSN long 100
    mock.submit_order("SBER", "B", 500)
    mock.submit_order("NVTK", "S", 50)
    mock.submit_order("ROSN", "B", 100)
    log.info(f"\nStarting positions:")
    starting = mock.get_positions()
    for p in starting:
        log.info(f"  {p.ticker:8s}: {p.quantity:>+8,}")

    # --------- Compute orders ---------
    current_shares = portfolio.positions_to_shares(starting)
    orders = portfolio.compute_orders(target_shares, current_shares, prices)
    log.info(f"\nOrders to execute ({len(orders)}):")
    for o in orders:
        rub = o.quantity * prices.get(o.ticker, 0)
        log.info(f"  {o.side} {o.ticker:8s} qty={o.quantity:>6,}  (~{rub:>10,.0f} ₽)")

    # --------- Submit ---------
    log.info(f"\nSubmitting orders...")
    for o in orders:
        resp = mock.submit_order(o.ticker, o.side, o.quantity)
        if not resp.success:
            log.error(f"  Failed: {resp.error}")

    # --------- Verify final positions match target ---------
    final = mock.get_positions()
    final_shares = portfolio.positions_to_shares(final)
    log.info(f"\nFinal positions vs target:")
    all_t = set(target_shares.keys()) | set(final_shares.keys())
    ok = True
    for t in sorted(all_t):
        tgt = target_shares.get(t, 0)
        fin = final_shares.get(t, 0)
        match = "✓" if tgt == fin else "✗"
        if tgt != fin:
            ok = False
        log.info(f"  {match} {t:8s}: target={tgt:>+8,}, actual={fin:>+8,}")

    if ok:
        log.info("\n✓ Portfolio reconciliation: PASS")
    else:
        log.error("\n✗ Portfolio reconciliation: FAIL")

    # --------- Test plan_rebalance helper ---------
    log.info("\n=== Test plan_rebalance helper ===")
    fresh_mock = MockArenaGoClient(portfolio_name="fresh", price_provider=prices.get)
    target_shares2, current_shares2, orders2 = plan_rebalance(
        portfolio,
        [mom21_weights, gap_fade_weights],
        prices,
        fresh_mock.get_positions(),
    )
    log.info(f"  Empty start: {len(current_shares2)} current, {len(target_shares2)} target, "
             f"{len(orders2)} orders to bootstrap")
