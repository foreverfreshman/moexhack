"""
Загрузка минутных данных MOEX из CSV в дневные OHLCV.

Формат входных файлов в директории data_dir:
    {TICKER}_3y_1m.csv

Содержимое каждого файла:
    timestamp,open,high,low,close,volume
    2023-05-18T06:59:00+00:00,39.12,39.12,39.12,39.12,4738
    ...

Использование:
    from data_loader import load_moex_daily
    daily = load_moex_daily("/path/to/market_data_final",
                            cache_path="daily_cache.parquet")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

log = logging.getLogger("data")

TICKERS_DEFAULT = [
    "LKOH", "SBER", "ROSN", "GAZP", "VTBR", "YDEX", "PLZL", "T",
    "NVTK", "X5", "GMKN", "MGNT", "ALRS", "AFLT", "CHMF", "NLMK",
    "MOEX", "SNGSP", "MTSS", "PIKK",
]


def _load_one_ticker_minute(
    filepath: Path,
    ticker: str,
    session_start_utc_hour: int = 7,   # 10:00 MSK
    session_end_utc_hour: int = 20,    # 23:59 MSK (включает evening)
) -> pd.DataFrame:
    """Минутный CSV → DataFrame с MSK trading_date.

    Фильтр сессии MOEX в UTC: 07:00-20:50 UTC = 10:00-23:50 MSK.
    Pre-market (до 07:00 UTC) отбрасывается, иначе open искажается.
    """
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    hr = df["timestamp"].dt.hour
    df = df.loc[(hr >= session_start_utc_hour) & (hr <= session_end_utc_hour)].copy()
    df["ts_msk"] = df["timestamp"].dt.tz_convert("Europe/Moscow")
    df["trading_date"] = df["ts_msk"].dt.date
    df["ticker"] = ticker
    return df


def _aggregate_to_daily(df_minute: pd.DataFrame) -> pd.DataFrame:
    """Минутные OHLCV → дневные OHLCV.

    open  = первая open в дне (хронологически)
    high  = max(high)
    low   = min(low)
    close = последняя close в дне
    volume = sum(volume)
    """
    df_minute = df_minute.sort_values("timestamp")
    daily = df_minute.groupby(["ticker", "trading_date"]).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        n_minutes=("close", "count"),
    ).reset_index()
    daily["ts"] = pd.to_datetime(daily["trading_date"])
    return daily[["ts", "ticker", "open", "high", "low", "close", "volume", "n_minutes"]]


def load_moex_daily(
    data_dir: str,
    tickers: Optional[List[str]] = None,
    file_pattern: str = "{ticker}_3y_1m.csv",
    cache_path: Optional[str] = None,
    min_minutes_per_day: int = 60,
    rebuild_cache: bool = False,
) -> pd.DataFrame:
    """Главная функция: загрузка и агрегация всех тикеров.

    Args:
        data_dir: путь к папке с CSV файлами (market_data_final)
        tickers: список тикеров (None -> дефолт хакатона)
        file_pattern: шаблон имени файла, {ticker} заменяется
        cache_path: parquet-кэш, ускоряет повторные запуски
        min_minutes_per_day: отбрасываем дни с < N минут (неполные сессии,
                             праздники с укороченной торговлей)
        rebuild_cache: True -> игнорировать кэш и пересчитать

    Returns:
        DataFrame со столбцами: ts, ticker, open, high, low, close, volume
        Long-format, отсортирован по (ts, ticker).
    """
    if tickers is None:
        tickers = TICKERS_DEFAULT

    if cache_path and Path(cache_path).exists() and not rebuild_cache:
        log.info(f"Loading cached daily data from {cache_path}")
        return pd.read_parquet(cache_path)

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data dir not found: {data_path}")

    all_daily = []
    missing = []
    for ticker in tickers:
        fp = data_path / file_pattern.format(ticker=ticker)
        if not fp.exists():
            log.warning(f"  [SKIP] {fp.name} not found")
            missing.append(ticker)
            continue
        try:
            df_min = _load_one_ticker_minute(fp, ticker)
            df_daily = _aggregate_to_daily(df_min)
            log.info(f"  {ticker:6s}: {len(df_min):>8,} min -> {len(df_daily):>4} days "
                     f"[{df_daily.ts.min().date()} -> {df_daily.ts.max().date()}]")
            all_daily.append(df_daily)
        except Exception as e:
            log.error(f"  [FAIL] {ticker}: {e}")

    if not all_daily:
        raise RuntimeError("No data loaded! Check data_dir and file_pattern.")

    daily = pd.concat(all_daily, ignore_index=True)

    # Фильтр неполных сессий
    before = len(daily)
    daily = daily[daily["n_minutes"] >= min_minutes_per_day].copy()
    log.info(f"Filtered {before - len(daily)} incomplete days (< {min_minutes_per_day} mins)")

    daily = daily.drop(columns=["n_minutes"]).sort_values(["ts", "ticker"]).reset_index(drop=True)

    if missing:
        log.warning(f"Missing tickers: {missing}")

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(cache_path)
        log.info(f"Cached daily data to {cache_path}")

    log.info(f"TOTAL: {len(daily):,} rows | {daily.ticker.nunique()} tickers | "
             f"{daily.ts.nunique()} days [{daily.ts.min().date()} -> {daily.ts.max().date()}]")
    return daily


# ============================================================
# Diagnostic / sanity checks
# ============================================================

def diagnose_daily(daily: pd.DataFrame) -> None:
    """Печатает diagnostic report по дневным данным.
    Полезно прогнать сразу после load_moex_daily для catch проблем.
    """
    log.info("=== Data diagnostics ===")
    by_ticker = daily.groupby("ticker").agg(
        days=("ts", "count"),
        first_day=("ts", "min"),
        last_day=("ts", "max"),
        avg_volume=("volume", "mean"),
        avg_close=("close", "mean"),
        max_pct_jump=("close", lambda x: x.pct_change().abs().max()),
    )
    # Round только numeric columns, чтобы избежать warning'а на datetime
    num_cols = by_ticker.select_dtypes(include="number").columns
    by_ticker[num_cols] = by_ticker[num_cols].round(4)
    log.info(f"\n{by_ticker.to_string()}")

    # Поиск аномалий
    daily_copy = daily.copy()
    daily_copy["ret"] = daily_copy.groupby("ticker")["close"].pct_change()
    extreme = daily_copy[daily_copy["ret"].abs() > 0.20]   # > 20% за день
    if len(extreme):
        log.warning(f"\nExtreme daily moves (>20%):\n"
                    f"{extreme[['ts','ticker','close','ret']].head(20).to_string()}")

    # Проверка покрытия по дням
    coverage = daily.groupby("ts")["ticker"].nunique()
    incomplete = coverage[coverage < daily.ticker.nunique()]
    if len(incomplete):
        log.warning(f"Days with incomplete coverage ({len(incomplete)} of {len(coverage)}):")
        log.warning(f"  First 5: {incomplete.head().to_dict()}")
        log.warning(f"  Last 5: {incomplete.tail().to_dict()}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python data_loader.py /path/to/market_data_final [cache.parquet]")
        sys.exit(1)
    data_dir = sys.argv[1]
    cache = sys.argv[2] if len(sys.argv) > 2 else None
    daily = load_moex_daily(data_dir, cache_path=cache)
    diagnose_daily(daily)
