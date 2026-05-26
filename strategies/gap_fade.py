"""
gap_fade.py — Intraday Gap Fade strategy (LIVE state machine).

Логика (подтверждено бэктестом: GapFade_0.5 Sharpe +1.64...+2.0 standalone):
    1. На открытии сессии: для каждого тикера считаем gap = (open - prev_close) / prev_close
    2. Если |gap| >= gap_threshold (0.5%) и |gap| <= max_gap (5%):
        - side = противоположный gap'у (fade): gap вверх → SHORT, gap вниз → LONG
        - entry = текущая цена открытия
        - TP = частичное закрытие гэпа (target_close_fraction × gap)
        - SL = расширение гэпа (stop_extension × gap в сторону против нас)
    3. В течение сессии: мониторим каждую минуту, закрываем по TP / SL
    4. На EOD (18:45 MSK): принудительно закрываем все оставшиеся позиции

ОТЛИЧИЕ ОТ БЭКТЕСТА: backtest-функция compute_gap_fade_trades() видит весь день
вперёд (high/low всех минут). Live-версия видит только текущую цену и реагирует
по мере поступления тиков. Поэтому здесь — state machine с явными переходами.

Использование (в main loop):
    strategy = GapFadeStrategy(capital_share=0.75)

    # Утром, когда есть prev_close и первая цена сессии:
    entries = strategy.on_session_open(prev_closes, current_prices)
    # entries = [GapSignal(ticker, side, ...), ...] → отправляем как market orders

    # Каждую минуту в течение дня:
    exits = strategy.on_tick(current_prices)
    # exits = [GapSignal(ticker, side=close, ...), ...] → закрываем

    # В конце дня:
    final_exits = strategy.on_session_close(current_prices)

    # Для интеграции с portfolio (текущее состояние как веса):
    weights = strategy.get_target_weights()
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from portfolio import StrategyWeights

log = logging.getLogger("gap_fade")


# ============================================================
# Data classes
# ============================================================

@dataclass
class GapSignal:
    """Сигнал на открытие или закрытие intraday-позиции."""
    ticker: str
    action: str          # 'open' | 'close'
    side: str            # 'B' | 'S' — направление ОРДЕРА (не позиции)
    reason: str          # 'gap_entry' | 'take_profit' | 'stop_loss' | 'eod'
    price_hint: float = 0.0   # ожидаемая цена (для логирования; исполняем по market)


@dataclass
class GapPosition:
    """Состояние активной gap-fade позиции."""
    ticker: str
    side: int            # +1 long, -1 short
    entry_price: float
    tp_price: float
    sl_price: float
    weight: float        # доля общего капитала (подписанная)
    opened_at: Optional[str] = None


# ============================================================
# Strategy
# ============================================================

class GapFadeStrategy:
    """Live Gap Fade с внутренним состоянием активных позиций."""

    def __init__(
        self,
        capital_share: float = 0.75,
        gap_threshold: float = 0.005,
        max_gap: float = 0.05,
        max_concurrent: int = 5,
        target_close_fraction: float = 0.5,
        stop_extension: float = 0.5,
    ):
        if not 0 < capital_share <= 1:
            raise ValueError("capital_share must be in (0, 1]")
        self.capital_share = capital_share
        self.gap_threshold = gap_threshold
        self.max_gap = max_gap
        self.max_concurrent = max_concurrent
        self.target_close_fraction = target_close_fraction
        self.stop_extension = stop_extension

        # Состояние
        self.active: Dict[str, GapPosition] = {}
        self._opened_today = False

    # --------------------------------------------------------
    # Session open: detect gaps, generate entries
    # --------------------------------------------------------

    def on_session_open(
        self,
        prev_closes: Dict[str, float],
        open_prices: Dict[str, float],
        current_prices: Optional[Dict[str, float]] = None,
        timestamp: Optional[str] = None,
        news_filter=None,
    ) -> List[GapSignal]:
        """Определить гэпы и сгенерировать сигналы на открытие позиций.

        Args:
            prev_closes: ticker → close предыдущего торгового дня
            open_prices: ticker → цена ОТКРЫТИЯ сессии (для детекции гэпа)
            current_prices: ticker → ТЕКУЩАЯ цена (для входа и проверки, что гэп
                ещё не закрылся). Если None — используется open_prices.
            timestamp: для логирования
            news_filter: опциональный LLM-фильтр (NewsFilter). Если передан —
                перед открытием каждой позиции проверяет экстремальные новости;
                при вето пропускает гэп и берёт следующего кандидата (слот не теряется).

        Returns:
            Список GapSignal с action='open' (до max_concurrent сильнейших гэпов).
        """
        if current_prices is None:
            current_prices = open_prices
        if self._opened_today:
            log.warning("on_session_open called twice in one day — ignoring")
            return []

        # Считаем гэпы для всех тикеров
        candidates = []
        for ticker, prev_close in prev_closes.items():
            open_px = open_prices.get(ticker)
            if open_px is None or open_px <= 0 or prev_close <= 0:
                continue
            gap = (open_px - prev_close) / prev_close
            if abs(gap) < self.gap_threshold or abs(gap) > self.max_gap:
                continue
            candidates.append((ticker, gap, prev_close, open_px))

        # Сортируем по силе гэпа (по убыванию |gap|)
        candidates.sort(key=lambda x: -abs(x[1]))

        if not candidates:
            log.info(f"Session open: no qualifying gaps (threshold {self.gap_threshold:.1%})")
            self._opened_today = True
            return []

        # Размер позиции: capital_share делим на max_concurrent
        weight_per_pos = self.capital_share / self.max_concurrent

        signals = []
        for ticker, gap, prev_close, open_px in candidates:
            if len(signals) >= self.max_concurrent:
                break   # слоты заполнены
            side = -1 if gap > 0 else 1   # fade
            cur = current_prices.get(ticker, open_px)
            if cur is None or cur <= 0:
                cur = open_px

            # TP/SL — абсолютные уровни от gap (привязаны к prev_close/open, не к entry)
            tp_price = prev_close + (1 - self.target_close_fraction) * (open_px - prev_close)
            sl_price = open_px + self.stop_extension * (open_px - prev_close)

            # Проверка: гэп ещё НЕ отыгран (есть куда фейдить от текущей цены)?
            # Иначе мгновенное срабатывание TP по цене входа = нулевая/убыточная сделка.
            if side == -1:   # short (gap up): TP ниже, SL выше
                if cur <= tp_price:
                    log.info(f"Gap {ticker}: уже отыгран (cur {cur:.2f} ≤ TP {tp_price:.2f}), skip")
                    continue
                if cur >= sl_price:
                    log.info(f"Gap {ticker}: уже за SL (cur {cur:.2f} ≥ SL {sl_price:.2f}), skip")
                    continue
            else:            # long (gap down): TP выше, SL ниже
                if cur >= tp_price:
                    log.info(f"Gap {ticker}: уже отыгран (cur {cur:.2f} ≥ TP {tp_price:.2f}), skip")
                    continue
                if cur <= sl_price:
                    log.info(f"Gap {ticker}: уже за SL (cur {cur:.2f} ≤ SL {sl_price:.2f}), skip")
                    continue

            # LLM-вето по экстремальным новостям (если фильтр включён).
            # При вето — пропускаем гэп и берём следующего кандидата (слот НЕ теряется).
            if news_filter is not None:
                verdict = news_filter.check_ticker_risk(
                    ticker,
                    gap_direction="up" if gap > 0 else "down",
                    gap_pct=gap * 100,
                )
                if verdict.blocked:
                    log.warning(f"Gap {ticker}: ВЕТО по новостям ({verdict.reason}) — "
                                f"беру следующего кандидата, слот сохранён")
                    continue

            entry = cur   # РЕАЛЬНАЯ цена входа (текущая), не теоретический open

            pos = GapPosition(
                ticker=ticker,
                side=side,
                entry_price=entry,
                tp_price=tp_price,
                sl_price=sl_price,
                weight=side * weight_per_pos,
                opened_at=timestamp,
            )
            self.active[ticker] = pos
            order_side = "B" if side == 1 else "S"
            signals.append(GapSignal(
                ticker=ticker, action="open", side=order_side,
                reason="gap_entry", price_hint=entry,
            ))
            log.info(f"Gap entry: {ticker} gap={gap:+.2%} → "
                     f"{'LONG' if side==1 else 'SHORT'} @ {entry:.2f}, "
                     f"TP={tp_price:.2f}, SL={sl_price:.2f}")

        self._opened_today = True
        return signals

    # --------------------------------------------------------
    # Tick: check TP / SL
    # --------------------------------------------------------

    def on_tick(
        self,
        current_prices: Dict[str, float],
        timestamp: Optional[str] = None,
    ) -> List[GapSignal]:
        """Проверить TP/SL для всех активных позиций. Вернуть сигналы на закрытие."""
        exits = []
        for ticker in list(self.active.keys()):
            pos = self.active[ticker]
            px = current_prices.get(ticker)
            if px is None or px <= 0:
                continue

            hit_tp = False
            hit_sl = False
            if pos.side == 1:   # long: TP выше entry, SL ниже
                if px >= pos.tp_price:
                    hit_tp = True
                elif px <= pos.sl_price:
                    hit_sl = True
            else:               # short: TP ниже entry, SL выше
                if px <= pos.tp_price:
                    hit_tp = True
                elif px >= pos.sl_price:
                    hit_sl = True

            if hit_tp or hit_sl:
                reason = "take_profit" if hit_tp else "stop_loss"
                order_side = "S" if pos.side == 1 else "B"   # закрытие = противоположно
                exits.append(GapSignal(
                    ticker=ticker, action="close", side=order_side,
                    reason=reason, price_hint=px,
                ))
                log.info(f"Gap exit ({reason}): {ticker} @ {px:.2f} "
                         f"(entry {pos.entry_price:.2f}, "
                         f"pnl {pos.side * (px - pos.entry_price) / pos.entry_price:+.2%})")
                del self.active[ticker]

        return exits

    # --------------------------------------------------------
    # Session close: force-flatten
    # --------------------------------------------------------

    def on_session_close(
        self,
        current_prices: Dict[str, float],
        timestamp: Optional[str] = None,
    ) -> List[GapSignal]:
        """Принудительно закрыть все оставшиеся позиции (EOD)."""
        exits = []
        for ticker in list(self.active.keys()):
            pos = self.active[ticker]
            px = current_prices.get(ticker, pos.entry_price)
            order_side = "S" if pos.side == 1 else "B"
            exits.append(GapSignal(
                ticker=ticker, action="close", side=order_side,
                reason="eod", price_hint=px,
            ))
            log.info(f"Gap EOD close: {ticker} @ {px:.2f} "
                     f"(pnl {pos.side * (px - pos.entry_price) / pos.entry_price:+.2%})")
            del self.active[ticker]
        return exits

    # --------------------------------------------------------
    # Integration with portfolio
    # --------------------------------------------------------

    def get_target_weights(self) -> StrategyWeights:
        """Текущее состояние активных позиций как StrategyWeights.

        Используется чтобы portfolio мог объединить gap_fade с mom_21 при
        вычислении целевых ордеров. Когда позиций нет — пустые веса.
        """
        weights = {pos.ticker: pos.weight for pos in self.active.values()}
        return StrategyWeights(
            strategy_name="gap_fade",
            weights=weights,
            capital_share=self.capital_share,
        )

    def reset_day(self) -> None:
        """Сброс флага на новый торговый день. Вызывать перед on_session_open."""
        if self.active:
            log.warning(f"reset_day called with {len(self.active)} positions still active!")
        self._opened_today = False

    @property
    def n_active(self) -> int:
        return len(self.active)


# ============================================================
# Smoke test: симулируем один торговый день минута-за-минутой
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    strategy = GapFadeStrategy(capital_share=0.75, gap_threshold=0.005, max_concurrent=5)

    # --- Сценарий: 6 тикеров, разные гэпы ---
    prev_closes = {
        "SBER": 280.0, "LKOH": 6500.0, "GAZP": 142.0,
        "NVTK": 1200.0, "PLZL": 14000.0, "ROSN": 550.0,
    }
    # Гэпы на открытии:
    #   SBER +1.5% (gap up → fade SHORT)
    #   LKOH -2.0% (gap down → fade LONG)
    #   GAZP +0.3% (ниже threshold → не торгуем)
    #   NVTK +3.0% (gap up → fade SHORT)
    #   PLZL -0.8% (gap down → fade LONG)
    #   ROSN +6.0% (выше max_gap → не торгуем, вероятно новости)
    open_prices = {
        "SBER": 280.0 * 1.015,
        "LKOH": 6500.0 * 0.980,
        "GAZP": 142.0 * 1.003,
        "NVTK": 1200.0 * 1.030,
        "PLZL": 14000.0 * 0.992,
        "ROSN": 550.0 * 1.060,
    }

    log.info("=== Session open ===")
    entries = strategy.on_session_open(prev_closes, open_prices, timestamp="10:00")
    log.info(f"Entry signals: {len(entries)} (expected 4: SBER, LKOH, NVTK, PLZL)")
    for e in entries:
        log.info(f"  {e.side} {e.ticker} ({e.reason})")
    assert len(entries) == 4, f"Expected 4 entries, got {len(entries)}"
    assert "GAZP" not in [e.ticker for e in entries], "GAZP gap too small, should skip"
    assert "ROSN" not in [e.ticker for e in entries], "ROSN gap too big, should skip"

    log.info(f"\nActive positions: {strategy.n_active}")
    w = strategy.get_target_weights()
    log.info(f"Target weights: {w.weights}")
    log.info(f"Sum |w| = {sum(abs(x) for x in w.weights.values()):.4f} "
             f"(4 of 5 slots used = {4 * 0.75/5:.3f})")

    # --- Симулируем движение цен в течение дня ---
    log.info("\n=== Intraday tick simulation ===")

    # Тик 1: SBER падает к TP (short profit), NVTK растёт к SL (short loss)
    sber_tp = strategy.active["SBER"].tp_price
    nvtk_sl = strategy.active["NVTK"].sl_price
    tick1 = {
        "SBER": sber_tp - 0.1,    # достигли TP для short
        "LKOH": 6500.0 * 0.985,   # ещё движется
        "NVTK": nvtk_sl + 1.0,    # достигли SL для short
        "PLZL": 14000.0 * 0.995,
    }
    log.info("Tick 1:")
    exits1 = strategy.on_tick(tick1, timestamp="10:30")
    log.info(f"  Exits: {[(e.ticker, e.reason) for e in exits1]}")
    assert len(exits1) == 2, f"Expected 2 exits (SBER TP, NVTK SL), got {len(exits1)}"

    log.info(f"  Active after tick 1: {strategy.n_active} (expected 2: LKOH, PLZL)")

    # Тик 2: LKOH достигает TP (long profit), PLZL движется но НЕ достигает TP
    lkoh_tp = strategy.active["LKOH"].tp_price
    tick2 = {
        "LKOH": lkoh_tp + 1.0,    # TP для long
        "PLZL": 14000.0 * 0.994,  # движется вверх, но ещё далеко от TP (13944)
    }
    log.info("Tick 2:")
    exits2 = strategy.on_tick(tick2, timestamp="12:00")
    log.info(f"  Exits: {[(e.ticker, e.reason) for e in exits2]}")
    assert len(exits2) == 1, f"Expected 1 exit (LKOH TP), got {len(exits2)}"
    log.info(f"  Active after tick 2: {strategy.n_active} (expected 1: PLZL)")
    assert strategy.n_active == 1, "PLZL should still be active"

    # --- EOD: закрываем остаток ---
    log.info("\n=== Session close (EOD) ===")
    eod_prices = {"PLZL": 14000.0 * 0.998}
    eod_exits = strategy.on_session_close(eod_prices, timestamp="18:45")
    log.info(f"  EOD exits: {[(e.ticker, e.reason) for e in eod_exits]}")
    assert len(eod_exits) == 1, "Expected 1 EOD exit (PLZL)"
    assert strategy.n_active == 0, "All positions should be closed"

    log.info("\n=== New day reset ===")
    strategy.reset_day()
    # No gaps today
    no_gap_opens = {t: prev_closes[t] * 1.001 for t in prev_closes}
    entries_day2 = strategy.on_session_open(prev_closes, no_gap_opens)
    log.info(f"  Day 2 entries (no gaps): {len(entries_day2)} (expected 0)")
    assert len(entries_day2) == 0

    log.info("\n✓ All gap_fade tests passed")