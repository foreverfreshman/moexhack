"""
mom21.py — Daily cross-sectional momentum strategy.

Логика (подтверждено бэктестом sweep_rebal_freq.py + production_search.py):
    1. На каждом trading day берём close цены последних 22+ дней
    2. Считаем mom_21 = close[today] / close[today - 21d] - 1 для каждого тикера
    3. Cross-sectional rank по mom_21 в текущий день
    4. top_k тикеров с самым высоким mom_21 → long
    5. bot_k тикеров с самым низким mom_21 → short
    6. Equal weights в каждой стороне, gross exposure = capital_share

Параметры по умолчанию (production config):
    capital_share = 0.30   # 30% общего капитала
    top_k = bot_k = 3
    mom_window = 21

Backtest на 3y MOEX данных (production_search.py): Sharpe 1.23, P(>0)=63% standalone.
В комбинации с GapFade_0.5 (на 70% cap): Sharpe 1.99, P(>0)=64%.

Использование:
    strategy = Mom21Strategy(capital_share=0.30)
    weights = strategy.compute_signals(daily_ohlcv_df)
    # weights.weights = {SBER: +0.05, LKOH: +0.05, ..., PIKK: -0.05, ...}
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Импорт из соседнего модуля
sys.path.insert(0, str(Path(__file__).parent.parent))
from portfolio import StrategyWeights

log = logging.getLogger("mom21")


class Mom21Strategy:
    """Daily cross-sectional momentum: top-K long, bot-K short по mom_window-day return."""

    def __init__(
        self,
        capital_share: float = 0.30,
        top_k: int = 3,
        bot_k: int = 3,
        mom_window: int = 21,
    ):
        if top_k < 1 or bot_k < 1:
            raise ValueError("top_k and bot_k must be ≥ 1")
        if not 0 < capital_share <= 1:
            raise ValueError("capital_share must be in (0, 1]")
        self.capital_share = capital_share
        self.top_k = top_k
        self.bot_k = bot_k
        self.mom_window = mom_window
        self.required_history = mom_window + 1   # +1 потому что нужны обе точки

    def compute_signals(
        self,
        daily_ohlcv: pd.DataFrame,
        as_of_date: Optional[pd.Timestamp] = None,
        universe: Optional[list] = None,
    ) -> StrategyWeights:
        """Generate target weights based on daily history.

        Args:
            daily_ohlcv: long-format DataFrame с колонками ts, ticker, close
            as_of_date: использовать данные ≤ этой даты. None = последняя дата в данных.
            universe: ограничить выбор только этими тикерами. None = все из data.

        Returns:
            StrategyWeights с top-K long и bot-K short, или пустой объект при
            нехватке данных.
        """
        df = daily_ohlcv[["ts", "ticker", "close"]].copy()
        df["ts"] = pd.to_datetime(df["ts"])

        if as_of_date is None:
            as_of_date = df["ts"].max()
        else:
            as_of_date = pd.to_datetime(as_of_date)
            df = df[df["ts"] <= as_of_date]

        if universe is not None:
            df = df[df["ticker"].isin(universe)]

        # Вычисляем mom_window-return для каждого тикера на as_of_date
        moms = {}
        for ticker, group in df.groupby("ticker", sort=False):
            group = group.sort_values("ts")
            if len(group) < self.required_history:
                log.warning(
                    f"{ticker}: only {len(group)} days history, "
                    f"need {self.required_history}, skipping"
                )
                continue
            current = float(group.iloc[-1]["close"])
            past = float(group.iloc[-1 - self.mom_window]["close"])
            if current <= 0 or past <= 0 or not np.isfinite(current) or not np.isfinite(past):
                log.warning(f"{ticker}: invalid prices ({past}, {current}), skipping")
                continue
            moms[ticker] = current / past - 1

        if len(moms) < self.top_k + self.bot_k:
            log.warning(
                f"Only {len(moms)} tickers с валидным mom, "
                f"need ≥ {self.top_k + self.bot_k}. Returning empty weights."
            )
            return StrategyWeights(
                strategy_name="mom21", weights={},
                capital_share=self.capital_share,
            )

        # Сортируем по mom_21 убывающе. Top — long, bot — short.
        mom_series = pd.Series(moms).sort_values(ascending=False)
        longs = mom_series.head(self.top_k).index.tolist()
        shorts = mom_series.tail(self.bot_k).index.tolist()

        # Equal weights. Long side и short side каждая получает capital_share / 2.
        long_w_each = (self.capital_share / 2) / self.top_k
        short_w_each = (self.capital_share / 2) / self.bot_k

        weights: dict = {}
        for t in longs:
            weights[t] = long_w_each
        for t in shorts:
            weights[t] = -short_w_each

        # Log decision
        log.info(
            f"mom_21 @ {as_of_date.date()}: "
            f"LONG {[f'{t}({moms[t]:+.3f})' for t in longs]}, "
            f"SHORT {[f'{t}({moms[t]:+.3f})' for t in shorts]}"
        )

        return StrategyWeights(
            strategy_name="mom21",
            weights=weights,
            capital_share=self.capital_share,
        )


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # Synthetic data: 20 тикеров × 30 дней.
    # Каждый тикер имеет известный mom_21 drift, чтобы можно было verify
    # что стратегия правильно выбирает top/bot.
    tickers = [
        "SBER", "LKOH", "GAZP", "VTBR", "ROSN", "NVTK", "GMKN", "MGNT",
        "ALRS", "AFLT", "CHMF", "NLMK", "MOEX", "MTSS", "PLZL", "X5",
        "YDEX", "PIKK", "SNGSP", "T",
    ]
    dates = pd.bdate_range("2026-04-01", periods=30)
    rng = np.random.default_rng(42)

    # drift монотонно растёт от SBER (j=0, -1%/day) до T (j=19, +0.9%/day)
    rows = []
    for j, t in enumerate(tickers):
        drift = (j - 10) * 0.001    # j=0 → -0.01, j=19 → +0.009
        log_returns = rng.normal(drift, 0.008, 30)
        prices = 100.0 * np.exp(np.cumsum(log_returns))
        for i, d in enumerate(dates):
            rows.append({"ts": d, "ticker": t, "close": prices[i]})
    daily = pd.DataFrame(rows)
    log.info(f"Synthetic: {len(daily)} rows, {daily.ts.nunique()} days, {daily.ticker.nunique()} tickers")
    log.info(f"Expected top (high drift): T(19), SNGSP(18), PIKK(17)")
    log.info(f"Expected bot (low drift): SBER(0), LKOH(1), GAZP(2)")

    # --- Test 1: дефолтный production config ---
    log.info("\n=== Test 1: Production config (30% cap, top-3 / bot-3) ===")
    strategy = Mom21Strategy(capital_share=0.30, top_k=3, bot_k=3)
    weights = strategy.compute_signals(daily)
    log.info(f"Output weights ({len(weights.weights)} positions):")
    for t, w in sorted(weights.weights.items(), key=lambda x: -x[1]):
        log.info(f"  {t:8s}: {w:+.4f}")
    gross = sum(abs(w) for w in weights.weights.values())
    log.info(f"Sum |weights| = {gross:.4f} (expected {strategy.capital_share})")
    assert abs(gross - strategy.capital_share) < 1e-9, "Gross mismatch!"

    # --- Test 2: as_of_date в прошлом ---
    log.info("\n=== Test 2: as_of в прошлом ===")
    past_date = dates[25]
    weights2 = strategy.compute_signals(daily, as_of_date=past_date)
    log.info(f"As of {past_date.date()}: longs={[t for t, w in weights2.weights.items() if w > 0]}, "
             f"shorts={[t for t, w in weights2.weights.items() if w < 0]}")

    # --- Test 3: недостаточно истории ---
    log.info("\n=== Test 3: insufficient history (<22 days) ===")
    short_daily = daily[daily.ts < daily.ts.min() + pd.Timedelta(days=15)]
    weights3 = strategy.compute_signals(short_daily)
    log.info(f"Result: {len(weights3.weights)} positions (expected 0)")
    assert len(weights3.weights) == 0, "Should return empty for short history"

    # --- Test 4: top-5 / bot-5 ---
    log.info("\n=== Test 4: top-5 / bot-5 wider strategy ===")
    strategy5 = Mom21Strategy(capital_share=0.30, top_k=5, bot_k=5)
    weights5 = strategy5.compute_signals(daily)
    longs5 = [t for t, w in weights5.weights.items() if w > 0]
    shorts5 = [t for t, w in weights5.weights.items() if w < 0]
    log.info(f"  Longs (5): {longs5}")
    log.info(f"  Shorts (5): {shorts5}")
    log.info(f"  Sum |w| = {sum(abs(w) for w in weights5.weights.values()):.4f}")

    # --- Test 5: ограничение universe ---
    log.info("\n=== Test 5: ограниченный universe ===")
    limited = ["SBER", "LKOH", "GAZP", "NVTK", "MOEX", "PLZL", "ALRS", "PIKK"]
    weights_limited = strategy.compute_signals(daily, universe=limited)
    log.info(f"  Из 8 тикеров: {dict(sorted(weights_limited.weights.items()))}")

    # --- Test 6: integrate with portfolio module ---
    log.info("\n=== Test 6: integrate с portfolio ===")
    from portfolio import Portfolio

    # Берём только подтверждённые тикеры hackathon
    portfolio = Portfolio(total_capital=1_000_000)
    target_weights = portfolio.combine_strategies([weights])
    log.info(f"  Combined weights: {len(target_weights)} positions")
    log.info(f"  Sum |w| = {sum(abs(w) for w in target_weights.values()):.4f}")

    # Используем последние цены из synthetic
    latest_prices = (
        daily.sort_values("ts").groupby("ticker").tail(1)
        .set_index("ticker")["close"].to_dict()
    )
    target_shares = portfolio.weights_to_shares(target_weights, latest_prices)
    log.info(f"  Target shares: {target_shares}")

    log.info("\n✓ All mom21 tests passed")
