"""
tinvest_stream.py — Live market data feed через T-Invest API (t_tech.invest).

Предоставляет данные для стратегий:
    - resolve_figis()        — разрешить FIGI по тикерам (один раз при старте)
    - get_daily_closes(n)    — daily OHLCV для mom_21 (последние n торговых дней)
    - get_prev_close()       — закрытие предыдущего торгового дня (для gap detection)
    - get_session_open()     — цена открытия текущей сессии (для gap detection)
    - get_current_prices()   — последние цены (для TP/SL мониторинга gap_fade)

Реализация: polling через get_candles (проверенный паттерн из tinkoffcandles.py),
НЕ streaming — надёжнее и проще в отладке. Запрос раз в минуту для 20 тикеров
укладывается в rate limits с запасом.

Токен берётся из env var TINKOFF_TOKEN (НЕ хардкодить в коде!).

Использование:
    feed = TInvestData(token=os.environ["TINKOFF_TOKEN"], tickers=TICKERS)
    feed.resolve_figis()

    # Утром для gap detection:
    prev = feed.get_prev_close()
    opens = feed.get_session_open()
    # → передать в gap_fade.on_session_open(prev, opens)

    # Каждую минуту:
    prices = feed.get_current_prices()
    # → gap_fade.on_tick(prices)

    # Раз в день для mom_21:
    daily = feed.get_daily_closes(n_days=25)
    # → mom21.compute_signals(daily)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone, date
from typing import Dict, List, Optional

import pandas as pd

log = logging.getLogger("tinvest")

# Optional импорт — mock работает без библиотеки
try:
    from t_tech.invest import Client, CandleInterval
    HAS_TINVEST = True
except ImportError:
    HAS_TINVEST = False
    Client = None
    CandleInterval = None

# MSK timezone
MSK = timezone(timedelta(hours=3))
UTC = timezone.utc

# Сессия MOEX в UTC
MAIN_OPEN_UTC_HOUR = 7    # 10:00 MSK
MAIN_OPEN_UTC_MIN = 0

# Алиасы тикеров: наш тикер → возможные тикеры в T-Invest
TICKER_ALIASES: Dict[str, List[str]] = {
    "X5": ["X5", "FIVE", "FIVEP"],   # X5 Retail после редомициляции
}

# Канонический universe (20 тикеров хакатона)
DEFAULT_UNIVERSE = [
    "LKOH", "SBER", "ROSN", "GAZP", "VTBR", "YDEX", "PLZL", "T", "NVTK", "X5",
    "GMKN", "MGNT", "ALRS", "AFLT", "CHMF", "NLMK", "MOEX", "SNGSP", "MTSS", "PIKK",
]


def quotation_to_float(q) -> float:
    """T-Invest Quotation (units + nano) → float."""
    return q.units + q.nano / 1e9


# ============================================================
# Real T-Invest data feed
# ============================================================

class TInvestData:
    """Live feed через T-Invest. Polling-based."""

    def __init__(
        self,
        token: str,
        tickers: Optional[List[str]] = None,
        request_delay: float = 0.05,
    ):
        if not HAS_TINVEST:
            raise RuntimeError(
                "t_tech.invest не установлен. pip install t-tech "
                "или используйте MockTInvestData для тестов."
            )
        if not token:
            raise ValueError("Empty token — set TINKOFF_TOKEN env var")
        self.token = token
        self.tickers = tickers or DEFAULT_UNIVERSE
        self.request_delay = request_delay
        self.figi_map: Dict[str, str] = {}
        self.figi_to_ticker: Dict[str, str] = {}
        self.lot_map: Dict[str, int] = {}      # ticker → размер лота (из API)

    # --------------------------------------------------------

    def resolve_figis(self) -> Dict[str, str]:
        """Разрешить FIGI и LOT для всех тикеров. Вызывать один раз при старте."""
        with Client(self.token) as client:
            shares = client.instruments.shares().instruments
            # ticker → (figi, lot)
            ticker_to_info = {s.ticker: (s.figi, s.lot) for s in shares}

        resolved = {}
        lots = {}
        for t in self.tickers:
            candidates = [t] + TICKER_ALIASES.get(t, [])
            found = None
            for cand in candidates:
                if cand in ticker_to_info:
                    figi, lot = ticker_to_info[cand]
                    found = figi
                    lots[t] = int(lot)
                    if cand != t:
                        log.info(f"{t}: resolved via alias '{cand}' (lot={lot})")
                    break
            if found:
                resolved[t] = found
            else:
                log.warning(f"FIGI not found for {t} (tried {candidates})")

        self.figi_map = resolved
        self.figi_to_ticker = {v: k for k, v in resolved.items()}
        self.lot_map = lots
        log.info(f"Resolved {len(resolved)}/{len(self.tickers)} FIGIs")
        log.info(f"Lot sizes from API: {lots}")
        if len(resolved) < len(self.tickers):
            missing = set(self.tickers) - set(resolved.keys())
            log.warning(f"MISSING FIGIs: {missing} — эти тикеры не будут торговаться!")
        return resolved

    def get_lot_sizes(self) -> Dict[str, int]:
        """Реальные размеры лотов из T-Invest API (после resolve_figis)."""
        return dict(self.lot_map)

    # --------------------------------------------------------

    def _get_candles(self, figi: str, from_: datetime, to: datetime, interval):
        """Один запрос свечей с retry на rate limit."""
        retries = 0
        while True:
            try:
                with Client(self.token) as client:
                    resp = client.market_data.get_candles(
                        figi=figi, from_=from_, to=to, interval=interval,
                    )
                return resp.candles
            except Exception as e:
                err = str(e).lower()
                if ("resource_exhausted" in err or "rate limit" in err) and retries < 3:
                    wait = 5 + retries * 5
                    log.warning(f"Rate limit, waiting {wait}s")
                    time.sleep(wait)
                    retries += 1
                else:
                    log.error(f"get_candles failed for {figi}: {e}")
                    return []

    # --------------------------------------------------------

    def get_daily_closes(self, n_days: int = 25) -> pd.DataFrame:
        """Daily close цены последних n_days торговых дней для всех тикеров.

        Returns: long-format DataFrame (ts, ticker, close) — для mom21.
        """
        to = datetime.now(UTC)
        # Берём с запасом по календарю (выходные, праздники)
        frm = to - timedelta(days=int(n_days * 1.8) + 10)
        rows = []
        for t, figi in self.figi_map.items():
            candles = self._get_candles(
                figi, frm, to, CandleInterval.CANDLE_INTERVAL_DAY
            )
            for c in candles:
                rows.append({
                    "ts": pd.Timestamp(c.time).tz_convert(None).normalize(),
                    "ticker": t,
                    "close": quotation_to_float(c.close),
                })
            time.sleep(self.request_delay)
        df = pd.DataFrame(rows)
        if df.empty:
            log.error("get_daily_closes returned no data!")
            return df
        # Оставляем последние n_days дат
        last_dates = sorted(df["ts"].unique())[-n_days:]
        df = df[df["ts"].isin(last_dates)].reset_index(drop=True)
        log.info(f"Daily closes: {df['ts'].nunique()} days, {df['ticker'].nunique()} tickers")
        return df

    # --------------------------------------------------------

    def get_prev_close(self, as_of: Optional[date] = None) -> Dict[str, float]:
        """Закрытие предыдущего торгового дня.

        Используем ДНЕВНЫЕ свечи: запрос минутных за несколько дней превышает
        лимит T-Invest для 1-min интервала (ошибка 30014). Берём close последней
        завершённой дневной свечи до сегодня.
        """
        now = datetime.now(UTC)
        today = (as_of or now.date())
        start_today = datetime.combine(today, datetime.min.time(), tzinfo=UTC)
        # 10 календарных дней назад — с запасом на выходные/праздники
        frm = start_today - timedelta(days=10)
        to = start_today  # строго до начала сегодня

        out = {}
        for t, figi in self.figi_map.items():
            candles = self._get_candles(
                figi, frm, to, CandleInterval.CANDLE_INTERVAL_DAY
            )
            if candles:
                out[t] = quotation_to_float(candles[-1].close)
            time.sleep(self.request_delay)
        log.info(f"prev_close (daily candle): {len(out)} tickers")
        return out

    # --------------------------------------------------------

    def get_session_open(self) -> Dict[str, float]:
        """Цена открытия текущей сессии (open первой минуты после 10:00 MSK)."""
        now = datetime.now(UTC)
        today = now.date()
        session_start = datetime.combine(today, datetime.min.time(), tzinfo=UTC).replace(
            hour=MAIN_OPEN_UTC_HOUR, minute=MAIN_OPEN_UTC_MIN
        )
        to = now
        out = {}
        for t, figi in self.figi_map.items():
            candles = self._get_candles(
                figi, session_start, to, CandleInterval.CANDLE_INTERVAL_1_MIN
            )
            if candles:
                out[t] = quotation_to_float(candles[0].open)
            time.sleep(self.request_delay)
        log.info(f"session_open: {len(out)} tickers")
        return out

    # --------------------------------------------------------

    def get_current_prices(self) -> Dict[str, float]:
        """Последняя доступная цена (close последней минутной свечи)."""
        now = datetime.now(UTC)
        frm = now - timedelta(minutes=15)   # запас чтобы точно поймать свечу
        out = {}
        for t, figi in self.figi_map.items():
            candles = self._get_candles(
                figi, frm, now, CandleInterval.CANDLE_INTERVAL_1_MIN
            )
            if candles:
                out[t] = quotation_to_float(candles[-1].close)
            time.sleep(self.request_delay)
        return out


# ============================================================
# Mock feed для тестов (без API)
# ============================================================

class MockTInvestData:
    """Эмулятор T-Invest для unit-тестов. Возвращает заданные данные."""

    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        prev_closes: Optional[Dict[str, float]] = None,
        session_opens: Optional[Dict[str, float]] = None,
        price_series: Optional[Dict[str, List[float]]] = None,
        daily_closes: Optional[pd.DataFrame] = None,
    ):
        self.tickers = tickers or DEFAULT_UNIVERSE
        self.figi_map = {t: f"FIGI_{t}" for t in self.tickers}
        self.lot_map = {t: 1 for t in self.tickers}   # mock: все лоты = 1
        self._prev = prev_closes or {}
        self._opens = session_opens or {}
        self._series = price_series or {}   # ticker -> list цен (на каждый on_tick)
        self._daily = daily_closes
        self._tick_idx = 0

    def resolve_figis(self):
        return self.figi_map

    def get_lot_sizes(self) -> Dict[str, int]:
        return dict(self.lot_map)

    def get_daily_closes(self, n_days: int = 25) -> pd.DataFrame:
        if self._daily is not None:
            return self._daily
        return pd.DataFrame(columns=["ts", "ticker", "close"])

    def get_prev_close(self, as_of=None) -> Dict[str, float]:
        return dict(self._prev)

    def get_session_open(self) -> Dict[str, float]:
        return dict(self._opens)

    def get_current_prices(self) -> Dict[str, float]:
        """Возвращает следующий «тик» из price_series. Если нет — последние opens."""
        out = {}
        for t in self.tickers:
            series = self._series.get(t)
            if series and self._tick_idx < len(series):
                out[t] = series[self._tick_idx]
            elif t in self._opens:
                out[t] = self._opens[t]
        self._tick_idx += 1
        return out


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # ---- Mock test: проверяем интеграцию с gap_fade ----
    log.info("=== Mock feed test: integration with gap_fade ===")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "strategies"))
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from strategies.gap_fade import GapFadeStrategy

    prev_closes = {"SBER": 280.0, "LKOH": 6500.0, "NVTK": 1200.0, "GAZP": 142.0}
    session_opens = {
        "SBER": 284.2,    # +1.5% gap up → fade short
        "LKOH": 6370.0,   # -2.0% gap down → fade long
        "NVTK": 1236.0,   # +3.0% gap up → fade short
        "GAZP": 142.4,    # +0.3% → ниже threshold
    }
    # Симулируем минутные тики: SBER падает (TP для short), NVTK растёт (SL для short)
    price_series = {
        "SBER": [283.0, 282.0, 281.5],
        "LKOH": [6400.0, 6435.0, 6440.0],
        "NVTK": [1245.0, 1254.0, 1255.0],
        "GAZP": [142.4, 142.4, 142.4],
    }

    feed = MockTInvestData(
        tickers=["SBER", "LKOH", "NVTK", "GAZP"],
        prev_closes=prev_closes,
        session_opens=session_opens,
        price_series=price_series,
    )
    feed.resolve_figis()

    strategy = GapFadeStrategy(capital_share=0.75, gap_threshold=0.005, max_concurrent=5)

    # Утром
    log.info("\n--- Session open ---")
    prev = feed.get_prev_close()
    opens = feed.get_session_open()
    entries = strategy.on_session_open(prev, opens)
    log.info(f"Entries: {[(e.ticker, e.side) for e in entries]} (expected 3: SBER, LKOH, NVTK)")
    assert len(entries) == 3, f"Expected 3, got {len(entries)}"

    # Тики
    log.info("\n--- Ticks ---")
    for tick_n in range(3):
        prices = feed.get_current_prices()
        exits = strategy.on_tick(prices)
        if exits:
            log.info(f"Tick {tick_n}: exits {[(e.ticker, e.reason) for e in exits]}")

    log.info(f"\nActive after ticks: {strategy.n_active}")

    # EOD
    final_prices = feed.get_current_prices()
    eod = strategy.on_session_close(final_prices)
    log.info(f"EOD exits: {[(e.ticker, e.reason) for e in eod]}")
    assert strategy.n_active == 0

    # ---- Daily closes mock test для mom21 ----
    log.info("\n=== Mock daily closes for mom21 ===")
    import numpy as np
    tickers = DEFAULT_UNIVERSE
    dates = pd.bdate_range("2026-04-01", periods=25)
    rng = np.random.default_rng(1)
    rows = []
    for j, t in enumerate(tickers):
        drift = (j - 10) * 0.001
        prices = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.008, 25)))
        for i, d in enumerate(dates):
            rows.append({"ts": d, "ticker": t, "close": prices[i]})
    daily_df = pd.DataFrame(rows)

    feed2 = MockTInvestData(tickers=tickers, daily_closes=daily_df)
    daily_out = feed2.get_daily_closes(n_days=25)
    log.info(f"Daily closes: {daily_out['ts'].nunique()} days, {daily_out['ticker'].nunique()} tickers")

    from strategies.mom21 import Mom21Strategy
    mom = Mom21Strategy(capital_share=0.25)
    w = mom.compute_signals(daily_out)
    log.info(f"mom21 signals: {len(w.weights)} positions, sum|w|={sum(abs(x) for x in w.weights.values()):.3f}")

    log.info("\n✓ All tinvest_stream mock tests passed")

    # ---- Real API test (если токен задан) ----
    log.info("\n=== Real T-Invest test (если TINKOFF_TOKEN задан и t_tech установлен) ===")
    token = os.environ.get("TINKOFF_TOKEN")
    if token and HAS_TINVEST:
        try:
            real = TInvestData(token=token, tickers=DEFAULT_UNIVERSE)
            figis = real.resolve_figis()
            log.info(f"Resolved FIGIs: {len(figis)}/{len(DEFAULT_UNIVERSE)}")
            prices = real.get_current_prices()
            log.info(f"Current prices sample: {dict(list(prices.items())[:5])}")
        except Exception as e:
            log.error(f"Real API test failed: {e}")
    else:
        log.info("  Skipped (TINKOFF_TOKEN not set or t_tech not installed)")