"""
Opening Range Breakout (ORB) + Rolling 14-day window evaluation.

Strategy logic:
    1. Первые ORB_MINUTES после открытия = opening range (OR_high, OR_low)
    2. После OR-периода: монитор пробоя
        - high >= OR_high → buy at OR_high (entry on level)
        - low <= OR_low → sell at OR_low
    3. Стоп: возврат к середине OR (OR_mid)
    4. Тейк: конец основной сессии (15:50 UTC = 18:50 MSK)

Position sizing:
    - Капитал на стратегию: STRATEGY_CAPITAL (например 400k = 40% от 1M total)
    - Каждая позиция: STRATEGY_CAPITAL / MAX_CONCURRENT
    - До MAX_CONCURRENT одновременных позиций — берём top по OR_range (сильные сигналы)

Метрики через rolling 14-day windows:
    - 54 non-overlapping windows за 3 года
    - Дополнительно rolling step=7 для статистической силы
    - Distribution: mean, P10, P50, P90 для return, turnover, Sharpe

Запуск:
    python intraday_eval.py --data-dir market_data_final
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import _load_one_ticker_minute, TICKERS_DEFAULT

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("orb")

# ============================================================
# Config
# ============================================================
FEE_BPS = 5.0
CAPITAL = 1_000_000.0
INTRADAY_CAPITAL = 400_000.0   # 40% от total — для intraday strategies
MAX_CONCURRENT = 5             # max одновременных positions на стратегию
MIN_OR_PCT = 0.001             # минимальная ширина OR (0.1%), фильтр шума

# MOEX main session times in UTC
SESSION_OPEN_UTC = 7    # 10:00 MSK
SESSION_CLOSE_UTC = 15  # 18:00 UTC — основная закрывается 15:50, берём цельный час до 18:00 чтобы захватить все минуты


# ============================================================
# 1. Load minute data
# ============================================================

def load_all_minute(data_dir: str, tickers: Optional[List[str]] = None,
                   file_pattern: str = "{ticker}_3y_1m.csv") -> pd.DataFrame:
    if tickers is None:
        tickers = TICKERS_DEFAULT
    frames = []
    for t in tickers:
        fp = Path(data_dir) / file_pattern.format(ticker=t)
        if not fp.exists():
            log.warning(f"[skip] {fp.name} missing")
            continue
        try:
            df = _load_one_ticker_minute(fp, t)
            frames.append(df)
            log.info(f"  {t:6s}: {len(df):>8,} min loaded")
        except Exception as e:
            log.error(f"  {t}: {e}")
    if not frames:
        raise RuntimeError("No data loaded")
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    log.info(f"Total minute rows: {len(out):,}, tickers: {out.ticker.nunique()}, "
             f"days: {out.trading_date.nunique()}")
    return out


# ============================================================
# 2. ORB signal generation
# ============================================================

def compute_orb_trades(
    minute_df: pd.DataFrame,
    or_minutes: int = 15,
    stop_mode: str = "mid",      # 'mid' | 'tight' | 'none'
    min_or_pct: float = MIN_OR_PCT,
) -> pd.DataFrame:
    """Для каждого (ticker, trading_date) считаем ORB-сделку.

    stop_mode:
        'mid'   — стоп на OR_mid (классика)
        'tight' — стоп ближе к entry: для long = OR_high - 0.3*OR_range
        'none'  — нет стопа, только end-of-day exit
    """
    # Фильтруем главную сессию (10:00-18:00 MSK = 07:00-15:00 UTC)
    msk_hours = minute_df["ts_msk"].dt.hour
    minute_df = minute_df[(msk_hours >= 10) & (msk_hours < 19)].copy()

    trades = []
    for (ticker, date), day in minute_df.groupby(["ticker", "trading_date"], sort=False):
        day = day.sort_values("timestamp").reset_index(drop=True)
        if len(day) < or_minutes + 30:   # минимум полчаса торгов после OR
            continue

        or_data = day.iloc[:or_minutes]
        post_or = day.iloc[or_minutes:]

        or_high = float(or_data["high"].max())
        or_low = float(or_data["low"].min())
        or_mid = (or_high + or_low) / 2.0
        or_range = or_high - or_low
        or_range_pct = or_range / or_high
        if or_range_pct < min_or_pct:
            continue  # OR слишком узкий — нет сигнала

        # Определяем стоп-уровни
        if stop_mode == "mid":
            stop_long = or_mid
            stop_short = or_mid
        elif stop_mode == "tight":
            stop_long = or_high - 0.3 * or_range
            stop_short = or_low + 0.3 * or_range
        else:   # 'none'
            stop_long = -np.inf
            stop_short = np.inf

        # Поиск первого breakout
        side = 0
        entry_price = None
        entry_idx = None
        for idx in post_or.index:
            row = post_or.loc[idx]
            if row["high"] >= or_high:
                side = 1
                entry_price = or_high
                entry_idx = idx
                break
            if row["low"] <= or_low:
                side = -1
                entry_price = or_low
                entry_idx = idx
                break
        if side == 0:
            continue   # breakout не произошёл

        # Поиск exit: stop или EOD
        exit_price = None
        exit_idx = None
        exit_reason = "eod"
        stop_lvl = stop_long if side == 1 else stop_short
        # Двигаемся со следующей минуты после entry
        for idx in day.index[day.index > entry_idx]:
            row = day.loc[idx]
            if side == 1 and row["low"] <= stop_lvl:
                exit_price = stop_lvl
                exit_idx = idx
                exit_reason = "stop"
                break
            if side == -1 and row["high"] >= stop_lvl:
                exit_price = stop_lvl
                exit_idx = idx
                exit_reason = "stop"
                break
        if exit_price is None:
            exit_price = float(day.iloc[-1]["close"])
            exit_idx = day.index[-1]

        pnl_pct = side * (exit_price - entry_price) / entry_price

        trades.append({
            "trading_date": date,
            "ticker": ticker,
            "side": side,
            "entry_time": day.loc[entry_idx, "timestamp"],
            "entry_price": entry_price,
            "exit_time": day.loc[exit_idx, "timestamp"],
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
            "exit_reason": exit_reason,
            "or_range_pct": or_range_pct,
        })

    df = pd.DataFrame(trades)
    log.info(f"  ORB trades found: {len(df)}, "
             f"long: {(df['side']==1).sum() if len(df) else 0}, "
             f"short: {(df['side']==-1).sum() if len(df) else 0}")
    return df


def compute_reverse_orb_trades(orb_trades: pd.DataFrame) -> pd.DataFrame:
    """Reverse ORB: при breakout идём ПРОТИВ направления (fade).

    Работает точно для no-stop варианта (одинаковый exit = EOD close).
    Side и pnl flip'ятся; turnover остаётся идентичен.
    """
    if orb_trades.empty:
        return orb_trades
    df = orb_trades.copy()
    df["side"] = -df["side"]
    df["pnl_pct"] = -df["pnl_pct"]
    return df


def compute_gap_fade_trades(
    minute_df: pd.DataFrame,
    gap_threshold: float = 0.008,    # 0.8% minimum gap to trade
    max_gap: float = 0.05,           # 5% — не торгуем экстремальные гэпы (новости)
    target_close_fraction: float = 0.5,   # TP = closure 50% gap
    stop_extension: float = 0.5,     # SL = expansion gap на 50%
) -> pd.DataFrame:
    """Gap Fade: при открытии торгуем ПРОТИВ overnight гэпа.

    Логика на одну торговую сессию для тикера:
        1. prev_close = close последней минуты предыдущего trading_date (включая evening)
        2. today_open = open первой минуты текущего trading_date после 10:00 MSK
        3. gap = (today_open - prev_close) / prev_close
        4. Если |gap| < gap_threshold или |gap| > max_gap → не торгуем
        5. side = -sign(gap)  (fade direction)
        6. entry = today_open
        7. TP = prev_close + (1 - target_close_fraction) × (today_open - prev_close)
           SL = today_open + stop_extension × (today_open - prev_close)
           (т.е. PT — частичное закрытие гэпа, SL — расширение гэпа)
        8. Exit: TP / SL / EOD close (что наступит раньше)
    """
    msk = minute_df["ts_msk"]
    main_session = minute_df[(msk.dt.hour >= 10) & (msk.dt.hour < 19)].copy()

    trades = []
    for ticker, t_df in main_session.groupby("ticker", sort=False):
        t_df = t_df.sort_values("timestamp").reset_index(drop=True)
        # Группируем по trading_date; нам нужен предыдущий день close
        by_day = t_df.groupby("trading_date", sort=False)
        days = sorted(t_df["trading_date"].unique())
        prev_close_by_day = {}
        for d in days:
            day = by_day.get_group(d)
            prev_close_by_day[d] = float(day.iloc[-1]["close"])

        for i in range(1, len(days)):
            today = days[i]
            yesterday = days[i - 1]
            prev_close = prev_close_by_day[yesterday]
            today_df = by_day.get_group(today)
            if len(today_df) < 30:
                continue
            today_open = float(today_df.iloc[0]["open"])
            gap = (today_open - prev_close) / prev_close
            if abs(gap) < gap_threshold or abs(gap) > max_gap:
                continue

            side = -1 if gap > 0 else 1   # fade
            entry_price = today_open
            tp_price = prev_close + (1 - target_close_fraction) * (today_open - prev_close)
            sl_price = today_open + stop_extension * (today_open - prev_close)

            # Monitor for exit
            exit_price = None
            exit_idx = None
            exit_reason = "eod"
            for idx in today_df.index[1:]:
                row = today_df.loc[idx]
                if side == 1:   # long: TP > entry, SL < entry
                    if row["high"] >= tp_price:
                        exit_price = tp_price; exit_idx = idx; exit_reason = "tp"; break
                    if row["low"] <= sl_price:
                        exit_price = sl_price; exit_idx = idx; exit_reason = "sl"; break
                else:           # short: TP < entry, SL > entry
                    if row["low"] <= tp_price:
                        exit_price = tp_price; exit_idx = idx; exit_reason = "tp"; break
                    if row["high"] >= sl_price:
                        exit_price = sl_price; exit_idx = idx; exit_reason = "sl"; break
            if exit_price is None:
                exit_price = float(today_df.iloc[-1]["close"])
                exit_idx = today_df.index[-1]

            pnl_pct = side * (exit_price - entry_price) / entry_price
            trades.append({
                "trading_date": today,
                "ticker": ticker,
                "side": side,
                "entry_time": today_df.iloc[0]["timestamp"],
                "entry_price": entry_price,
                "exit_time": today_df.loc[exit_idx, "timestamp"],
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
                "exit_reason": exit_reason,
                "or_range_pct": abs(gap),   # для совместимости с aggregate_orb_daily
            })
    df = pd.DataFrame(trades)
    if not df.empty:
        log.info(f"  Gap Fade trades found: {len(df)}, "
                 f"long: {(df['side']==1).sum()}, short: {(df['side']==-1).sum()}")
    return df


# ============================================================
# 3. Daily aggregation with position sizing
# ============================================================

def aggregate_orb_daily(
    trades: pd.DataFrame,
    strategy_capital: float = INTRADAY_CAPITAL,
    max_concurrent: int = MAX_CONCURRENT,
    fee_bps: float = FEE_BPS,
) -> pd.DataFrame:
    """Аггрегируем сделки по торговому дню: берём top-N по OR_range, считаем PnL."""
    if trades.empty:
        return pd.DataFrame()
    fee = fee_bps / 10000.0
    rows = []
    for date, day_trades in trades.groupby("trading_date"):
        # Выбираем top по силе сигнала (широкий OR = больше momentum)
        selected = day_trades.sort_values("or_range_pct", ascending=False).head(max_concurrent)
        capital_per_pos = strategy_capital / max_concurrent
        selected = selected.copy()
        selected["pnl_rub"] = selected["pnl_pct"] * capital_per_pos
        # Turnover = open + close = 2 × capital per position
        selected["turnover_rub"] = 2 * capital_per_pos
        selected["fee_rub"] = selected["turnover_rub"] * fee
        rows.append({
            "trading_date": date,
            "n_trades": len(selected),
            "gross_pnl_rub": selected["pnl_rub"].sum(),
            "fee_rub": selected["fee_rub"].sum(),
            "net_pnl_rub": (selected["pnl_rub"] - selected["fee_rub"]).sum(),
            "turnover_rub": selected["turnover_rub"].sum(),
            "n_long": int((selected["side"] == 1).sum()),
            "n_short": int((selected["side"] == -1).sum()),
        })
    return pd.DataFrame(rows).sort_values("trading_date").reset_index(drop=True)


# ============================================================
# 4. mom_21 daily strategy (for combining)
# ============================================================

def daily_mom21_pnl(
    daily_df: pd.DataFrame,
    strategy_capital: float = CAPITAL - INTRADAY_CAPITAL,   # остаток после intraday
    top_k: int = 3,
    fee_bps: float = FEE_BPS,
) -> pd.DataFrame:
    """Daily mom_21 backtest на части капитала."""
    df = daily_df.copy().sort_values(["ts", "ticker"]).reset_index(drop=True)
    grp = df.groupby("ticker", group_keys=False)
    df["mom_21"] = grp["close"].transform(lambda x: x.pct_change(21))
    df["cs_mom_21"] = df.groupby("ts")["mom_21"].rank(pct=True)
    df["cs_mom_21"] = df.groupby("ticker")["cs_mom_21"].shift(1)
    df["fwd_1d"] = grp["close"].transform(lambda x: np.log(x.shift(-1) / x))
    df = df.dropna(subset=["cs_mom_21", "fwd_1d"]).reset_index(drop=True)

    universe = sorted(df["ticker"].unique())
    fee = fee_bps / 10000.0
    prev_w = pd.Series(0.0, index=universe)

    rows = []
    for d in sorted(df["ts"].unique()):
        snap = df[df["ts"] == d].set_index("ticker")
        ranks = snap["cs_mom_21"].rank(ascending=False)
        longs = list(ranks.nsmallest(top_k).index)
        shorts = list(ranks.nlargest(top_k).index)
        target = pd.Series(0.0, index=universe)
        for t in longs: target.loc[t] = 0.5 / top_k
        for t in shorts: target.loc[t] = -0.5 / top_k

        turnover_ratio = (target - prev_w).abs().sum()
        turnover_rub = turnover_ratio * strategy_capital
        fee_rub = turnover_rub * fee
        gross_ret_pct = (target * snap["fwd_1d"].reindex(universe).fillna(0)).sum()
        gross_pnl_rub = gross_ret_pct * strategy_capital
        net_pnl_rub = gross_pnl_rub - fee_rub

        rows.append({
            "trading_date": d.date(),
            "gross_pnl_rub": gross_pnl_rub,
            "fee_rub": fee_rub,
            "net_pnl_rub": net_pnl_rub,
            "turnover_rub": turnover_rub,
        })
        prev_w = target

    return pd.DataFrame(rows)


# ============================================================
# 5. Rolling 14-day window evaluation
# ============================================================

def rolling_14d_eval(
    daily_pnl: pd.DataFrame,
    strategy_capital: float,
    window_bdays: int = 10,   # реальное окно этапа 2 = 10 business days
    step_bdays: int = 5,      # шаг = неделя
    label: str = "strategy",
) -> dict:
    """Считает метрики по rolling 14/10-bday окнам.

    Возвращает dict с distribution: mean/median/P10/P90 для return, turnover, sharpe.
    """
    if daily_pnl.empty or "trading_date" not in daily_pnl.columns:
        log.warning(f"[{label}] нет данных")
        return {}
    daily_pnl = daily_pnl.sort_values("trading_date").reset_index(drop=True)
    n = len(daily_pnl)
    if n < window_bdays:
        log.warning(f"[{label}] недостаточно данных")
        return {}

    windows = []
    for start_idx in range(0, n - window_bdays + 1, step_bdays):
        w = daily_pnl.iloc[start_idx:start_idx + window_bdays]
        if len(w) < window_bdays:
            continue
        total_pnl = w["net_pnl_rub"].sum()
        total_turnover_M = w["turnover_rub"].sum() / 1e6
        ret_pct = total_pnl / strategy_capital
        # Sharpe в окне: mean daily PnL / std × sqrt(252)
        daily_rets = w["net_pnl_rub"] / strategy_capital
        sharpe = daily_rets.mean() / (daily_rets.std() + 1e-9) * np.sqrt(252) if daily_rets.std() > 0 else 0
        windows.append({
            "start": w["trading_date"].iloc[0],
            "end": w["trading_date"].iloc[-1],
            "return_pct": ret_pct,
            "turnover_M": total_turnover_M,
            "sharpe": sharpe,
        })
    bt = pd.DataFrame(windows)
    if bt.empty:
        return {}

    summary = {
        "label": label,
        "n_windows": len(bt),
        "ret_mean": bt["return_pct"].mean(),
        "ret_median": bt["return_pct"].median(),
        "ret_p10": bt["return_pct"].quantile(0.10),
        "ret_p90": bt["return_pct"].quantile(0.90),
        "ret_pct_positive": (bt["return_pct"] > 0).mean(),
        "turnover_mean_M": bt["turnover_M"].mean(),
        "turnover_median_M": bt["turnover_M"].median(),
        "turnover_p10_M": bt["turnover_M"].quantile(0.10),
        "turnover_p90_M": bt["turnover_M"].quantile(0.90),
        "turnover_pct_pass10M": (bt["turnover_M"] >= 10).mean(),
        "sharpe_mean": bt["sharpe"].mean(),
        "sharpe_median": bt["sharpe"].median(),
    }
    return summary


# ============================================================
# 6. Combined daily PnL (mom_21 + intraday)
# ============================================================

def combine_pnl(mom_daily: pd.DataFrame, intraday_daily: pd.DataFrame) -> pd.DataFrame:
    """Объединяем daily PnL от двух стратегий по trading_date."""
    if mom_daily.empty or "trading_date" not in mom_daily.columns:
        return intraday_daily.copy() if not intraday_daily.empty else pd.DataFrame()
    if intraday_daily.empty:
        return mom_daily.copy()
    m = mom_daily.set_index("trading_date")
    i = intraday_daily.set_index("trading_date")
    common = m.index.union(i.index)
    out = pd.DataFrame(index=common)
    out["gross_pnl_rub"] = m["gross_pnl_rub"].reindex(common, fill_value=0) + \
                            i["gross_pnl_rub"].reindex(common, fill_value=0)
    out["fee_rub"] = m["fee_rub"].reindex(common, fill_value=0) + \
                      i["fee_rub"].reindex(common, fill_value=0)
    out["net_pnl_rub"] = m["net_pnl_rub"].reindex(common, fill_value=0) + \
                          i["net_pnl_rub"].reindex(common, fill_value=0)
    out["turnover_rub"] = m["turnover_rub"].reindex(common, fill_value=0) + \
                          i["turnover_rub"].reindex(common, fill_value=0)
    return out.reset_index().rename(columns={"index": "trading_date"})


# ============================================================
# 7. Print helper
# ============================================================

def print_summary(s: dict) -> None:
    if not s:
        log.info("  (пусто)")
        return
    log.info(f"  [{s['label']}] across {s['n_windows']} rolling 10-bday windows (step=5):")
    log.info(f"    Return: mean={s['ret_mean']:+.2%}, median={s['ret_median']:+.2%}, "
             f"P10/P90: {s['ret_p10']:+.2%}/{s['ret_p90']:+.2%}, "
             f"P(>0)={s['ret_pct_positive']:.0%}")
    log.info(f"    Turnover: mean={s['turnover_mean_M']:.1f}M, "
             f"P10/P90: {s['turnover_p10_M']:.1f}M/{s['turnover_p90_M']:.1f}M, "
             f"P(≥10M)={s['turnover_pct_pass10M']:.0%}")
    log.info(f"    Sharpe: mean={s['sharpe_mean']:+.2f}, median={s['sharpe_median']:+.2f}")


# ============================================================
# 8. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--daily-cache", default=None,
                        help="parquet кэш daily OHLCV (для mom_21 модуля)")
    parser.add_argument("--limit-tickers", type=int, default=None,
                        help="Ограничить число тикеров (debug)")
    args = parser.parse_args()

    tickers = TICKERS_DEFAULT[:args.limit_tickers] if args.limit_tickers else TICKERS_DEFAULT

    log.info("=== Step 1: Loading minute data (это медленно — 3 года × 20 тикеров) ===")
    minute_df = load_all_minute(args.data_dir, tickers=tickers)

    log.info("\n=== Step 2: ORB-15min trades (stop=mid) ===")
    orb15_mid = compute_orb_trades(minute_df, or_minutes=15, stop_mode="mid")

    log.info("\n=== Step 3: ORB-15min trades (no stop, для reverse) ===")
    orb15_nostop = compute_orb_trades(minute_df, or_minutes=15, stop_mode="none")

    log.info("\n=== Step 4: Reverse ORB (fade breakout, no stop) ===")
    reverse_orb = compute_reverse_orb_trades(orb15_nostop)
    log.info(f"  Reverse ORB trades: {len(reverse_orb)} (тот же набор, side flip)")

    log.info("\n=== Step 5: Gap Fade (0.8% threshold) ===")
    gap_fade = compute_gap_fade_trades(minute_df, gap_threshold=0.008)

    log.info("\n=== Step 6: Gap Fade (1.5% threshold — только большие гэпы) ===")
    gap_fade_big = compute_gap_fade_trades(minute_df, gap_threshold=0.015)

    # Daily aggregation для каждой стратегии
    intraday_variants = {
        "ORB15_mid": aggregate_orb_daily(orb15_mid),
        "ORB15_nostop": aggregate_orb_daily(orb15_nostop),
        "ReverseORB": aggregate_orb_daily(reverse_orb),
        "GapFade_0.8": aggregate_orb_daily(gap_fade),
        "GapFade_1.5": aggregate_orb_daily(gap_fade_big),
    }

    log.info("\n=== Step 6: mom_21 daily backtest на 60% капитала ===")
    # Загружаем daily из cache или агрегируем минутные
    if args.daily_cache and Path(args.daily_cache).exists():
        daily_df = pd.read_parquet(args.daily_cache)
    else:
        log.info("  Aggregating minute→daily (no cache provided)")
        from data_loader import _aggregate_to_daily
        daily_df = _aggregate_to_daily(minute_df)
    mom21_daily = daily_mom21_pnl(daily_df, strategy_capital=CAPITAL - INTRADAY_CAPITAL)
    log.info(f"  mom_21 daily PnL: {len(mom21_daily)} business days")

    # ===== Rolling evaluation =====
    log.info("\n" + "=" * 80)
    log.info("ROLLING 10-BDAY WINDOW EVALUATION (реальное окно этапа 2)")
    log.info("=" * 80)

    log.info("\n>>> Baseline: только mom_21 на 60% капитала (без intraday):")
    s_mom_alone = rolling_14d_eval(mom21_daily, strategy_capital=CAPITAL,
                                    label="mom_21 alone 60% cap")
    print_summary(s_mom_alone)

    log.info("\n>>> Только intraday (40% капитала):")
    for name, intra_df in intraday_variants.items():
        if intra_df.empty:
            log.info(f"  [{name}] нет сделок"); continue
        s = rolling_14d_eval(intra_df, strategy_capital=INTRADAY_CAPITAL,
                             label=f"{name} alone")
        print_summary(s)

    log.info("\n>>> Combined: mom_21 (60%) + intraday (40%):")
    for name, intra_df in intraday_variants.items():
        if intra_df.empty:
            log.info(f"  [{name}] нет сделок"); continue
        combined = combine_pnl(mom21_daily, intra_df)
        s = rolling_14d_eval(combined, strategy_capital=CAPITAL,
                             label=f"mom_21 + {name}")
        print_summary(s)


if __name__ == "__main__":
    main()