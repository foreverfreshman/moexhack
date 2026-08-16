"""
capital_split_search.py — поиск оптимальной доли капитала mom/gap.

Прогоняет текущую стратегию (mom_21 2d-rebalance + gap_fade 0.5%) на всей сетке
распределений капитала от 5/95 до 95/5 с шагом 5%, на 14-ДНЕВНЫХ скользящих окнах.
Для каждой доли считает: медианную доходность, Sharpe, оборот (P10, медиана,
вероятность ≥10M). Помогает решить, оптимальна ли текущая 25/75 при 14 торговых днях.

Переиспользует проверенные функции из твоих модулей (как rebal_test.py), поэтому
результаты сопоставимы с прежними бэктестами.

Запуск:
    python capital_split_search.py --data-dir market_data_final --daily-cache daily.parquet

    --data-dir     папка с {TICKER}_3y_1m.csv (минутные данные)
    --daily-cache  (опц.) parquet с дневными свечами для ускорения; если нет — посчитается
    --window       размер окна в торговых днях (по умолчанию 14)
    --step         шаг сетки долей в % (по умолчанию 5)
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import _aggregate_to_daily, TICKERS_DEFAULT
from intraday_eval import (
    load_all_minute, compute_gap_fade_trades, aggregate_orb_daily,
    combine_pnl, CAPITAL
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("split_search")

TARGET_TURNOVER = 10_000_000.0   # порог штрафа
TRADING_DAYS_PER_YEAR = 252      # для аннуализации Sharpe


# ---- копии из rebal_test.py (чтобы скрипт был самодостаточным) ----

def scale_pnl(daily_pnl: pd.DataFrame, ref_capital: float, target_capital: float) -> pd.DataFrame:
    if daily_pnl.empty or target_capital <= 0:
        return pd.DataFrame()
    df = daily_pnl.copy()
    factor = target_capital / ref_capital
    for col in ["gross_pnl_rub", "fee_rub", "net_pnl_rub", "turnover_rub"]:
        if col in df.columns:
            df[col] = df[col] * factor
    return df


def daily_mom21_pnl_rebal(daily_df, strategy_capital=600_000.0, top_k=3,
                          fee_bps=5.0, rebal_freq=2):
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
    days = sorted(df["ts"].unique())
    for i, d in enumerate(days):
        snap = df[df["ts"] == d].set_index("ticker")
        if i % rebal_freq == 0:
            ranks = snap["cs_mom_21"].rank(ascending=False)
            longs = list(ranks.nsmallest(top_k).index)
            shorts = list(ranks.nlargest(top_k).index)
            target = pd.Series(0.0, index=universe)
            for t in longs: target.loc[t] = 0.5 / top_k
            for t in shorts: target.loc[t] = -0.5 / top_k
        else:
            target = prev_w.copy()
        turnover_ratio = (target - prev_w).abs().sum()
        turnover_rub = turnover_ratio * strategy_capital
        fee_rub = turnover_rub * fee
        gross_ret_pct = (target * snap["fwd_1d"].reindex(universe).fillna(0)).sum()
        gross_pnl_rub = gross_ret_pct * strategy_capital
        rows.append({
            "trading_date": d.date() if hasattr(d, "date") else d,
            "gross_pnl_rub": gross_pnl_rub,
            "fee_rub": fee_rub,
            "net_pnl_rub": gross_pnl_rub - fee_rub,
            "turnover_rub": turnover_rub,
        })
        prev_w = target
    return pd.DataFrame(rows)


# ---- собственная rolling-метрика с гарантированным окном ----

def _find_col(df, *candidates):
    """Найти колонку по списку возможных имён или по подстроке."""
    for c in candidates:
        if c in df.columns:
            return c
    for col in df.columns:
        for c in candidates:
            if c in col:
                return col
    raise KeyError(f"Не нашёл колонку из {candidates} в {list(df.columns)}")


def rolling_metrics(combined: pd.DataFrame, window: int = 14) -> dict:
    """Скользящие окна по `window` последовательных торговых дней.

    Возвращает агрегаты по всем окнам: медианная доходность, Sharpe,
    обороты (P10, медиана, mean), вероятность достичь порога 10M.
    """
    if combined.empty or len(combined) < window:
        return {}
    df = combined.copy()
    date_col = _find_col(df, "trading_date", "trade_date", "date")
    pnl_col = _find_col(df, "net_pnl_rub", "net_pnl")
    to_col = _find_col(df, "turnover_rub", "turnover")
    df = df.sort_values(date_col).reset_index(drop=True)

    daily_ret = df[pnl_col].values / CAPITAL          # дневная доходность доли капитала
    daily_to = df[to_col].values

    win_returns, win_sharpes, win_turnovers = [], [], []
    for start in range(len(df) - window + 1):
        r = daily_ret[start:start + window]
        t = daily_to[start:start + window]
        win_returns.append(r.sum() * 100.0)            # суммарная доходность окна, %
        win_turnovers.append(t.sum() / 1e6)            # оборот окна, M₽
        sd = r.std(ddof=1)
        win_sharpes.append((r.mean() / sd * np.sqrt(TRADING_DAYS_PER_YEAR)) if sd > 1e-12 else 0.0)

    win_returns = np.array(win_returns)
    win_sharpes = np.array(win_sharpes)
    win_turnovers = np.array(win_turnovers)
    return {
        "n_windows": len(win_returns),
        "ret_median": float(np.median(win_returns)),
        "ret_mean": float(win_returns.mean()),
        "sharpe_mean": float(win_sharpes.mean()),
        "sharpe_median": float(np.median(win_sharpes)),
        "turnover_p10": float(np.percentile(win_turnovers, 10)),
        "turnover_median": float(np.median(win_turnovers)),
        "turnover_mean": float(win_turnovers.mean()),
        "prob_ge_10m": float((win_turnovers >= TARGET_TURNOVER / 1e6).mean() * 100.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--daily-cache", default=None)
    parser.add_argument("--limit-tickers", type=int, default=None)
    parser.add_argument("--window", type=int, default=14)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--since", default=None,
                        help="YYYY-MM-DD: считать только данные С этой даты (напр. 2025-01-01)")
    args = parser.parse_args()

    tickers = TICKERS_DEFAULT[:args.limit_tickers] if args.limit_tickers else TICKERS_DEFAULT

    log.info("=== Loading minute data ===")
    minute_df = load_all_minute(args.data_dir, tickers=tickers)

    # Фильтр по дате: только свежий период (gap-альфа деградировала с 2025)
    if args.since:
        tcol = None
        for c in ["ts", "begin", "datetime", "timestamp", "time", "dt"]:
            if c in minute_df.columns:
                tcol = c
                break
        if tcol:
            before = len(minute_df)
            ts = pd.to_datetime(minute_df[tcol])
            since = pd.to_datetime(args.since)
            # данные могут быть tz-aware — приводим границу к той же tz
            if getattr(ts.dt, "tz", None) is not None:
                since = since.tz_localize(ts.dt.tz) if since.tzinfo is None else since.tz_convert(ts.dt.tz)
            minute_df = minute_df[ts >= since].copy()
            log.info(f"Фильтр since={args.since}: {before:,} → {len(minute_df):,} строк")
        else:
            log.warning(f"Не нашёл колонку времени для фильтра since. Колонки: {list(minute_df.columns)}")

    log.info("=== Computing Gap Fade trades (0.5% threshold) ===")
    gf_trades = compute_gap_fade_trades(minute_df, gap_threshold=0.005)
    GF_REF_CAP = 400_000
    gf_daily = aggregate_orb_daily(gf_trades, strategy_capital=GF_REF_CAP)

    log.info("=== Loading mom_21 daily data (2-day rebalance) ===")
    if args.daily_cache and Path(args.daily_cache).exists():
        daily_df = pd.read_parquet(args.daily_cache)
    else:
        daily_df = _aggregate_to_daily(minute_df)
    MOM_REF_CAP = 600_000
    mom_daily = daily_mom21_pnl_rebal(daily_df, strategy_capital=MOM_REF_CAP, rebal_freq=2)

    # ---- сетка долей ----
    log.info(f"\n{'='*92}")
    log.info(f"ПОИСК ДОЛИ КАПИТАЛА mom/gap | окно {args.window} дней | шаг {args.step}% | "
             f"текущая стратегия (mom 2d + gap 0.5%)")
    log.info(f"{'='*92}")
    header = (f"{'mom/gap':>8} | {'ret med%':>9} | {'ret mean%':>9} | {'Sharpe':>7} | "
              f"{'TO P10':>8} | {'TO med':>8} | {'P(>=10M)':>9}")
    log.info(header)
    log.info("-" * 92)

    results = []
    for mom_pct in range(args.step, 100, args.step):   # 5,10,...,95
        gap_pct = 100 - mom_pct
        m_cap = CAPITAL * mom_pct / 100
        g_cap = CAPITAL * gap_pct / 100
        m_scaled = scale_pnl(mom_daily, MOM_REF_CAP, m_cap)
        g_scaled = scale_pnl(gf_daily, GF_REF_CAP, g_cap)
        combined = combine_pnl(m_scaled, g_scaled)
        s = rolling_metrics(combined, window=args.window)
        if not s:
            log.warning(f"{mom_pct}/{gap_pct}: недостаточно данных для окна {args.window}")
            continue
        s["mom_pct"] = mom_pct
        s["gap_pct"] = gap_pct
        results.append(s)
        flag = " <-- порог 10M под угрозой" if s["prob_ge_10m"] < 90 else ""
        log.info(f"{mom_pct:>3}/{gap_pct:<3} | {s['ret_median']:>9.3f} | {s['ret_mean']:>9.3f} | "
                 f"{s['sharpe_mean']:>7.2f} | {s['turnover_p10']:>8.1f} | "
                 f"{s['turnover_median']:>8.1f} | {s['prob_ge_10m']:>8.0f}%{flag}")

    log.info("-" * 92)

    # ---- оптимумы по разным критериям ----
    if results:
        df = pd.DataFrame(results)
        # кандидаты, проходящие порог оборота с запасом (P>=10M >= 90%)
        safe = df[df["prob_ge_10m"] >= 90]
        log.info("\n=== ОПТИМУМЫ ===")
        best_ret = df.loc[df["ret_median"].idxmax()]
        best_sharpe = df.loc[df["sharpe_mean"].idxmax()]
        log.info(f"Макс. медианная доходность: {int(best_ret['mom_pct'])}/{int(best_ret['gap_pct'])} "
                 f"(ret {best_ret['ret_median']:.3f}%, Sharpe {best_ret['sharpe_mean']:.2f}, "
                 f"P>=10M {best_ret['prob_ge_10m']:.0f}%)")
        log.info(f"Макс. Sharpe: {int(best_sharpe['mom_pct'])}/{int(best_sharpe['gap_pct'])} "
                 f"(Sharpe {best_sharpe['sharpe_mean']:.2f}, ret {best_sharpe['ret_median']:.3f}%, "
                 f"P>=10M {best_sharpe['prob_ge_10m']:.0f}%)")
        if not safe.empty:
            best_safe = safe.loc[safe["ret_median"].idxmax()]
            log.info(f"Лучшая доходность СРЕДИ безопасных по обороту (P>=10M>=90%): "
                     f"{int(best_safe['mom_pct'])}/{int(best_safe['gap_pct'])} "
                     f"(ret {best_safe['ret_median']:.3f}%, Sharpe {best_safe['sharpe_mean']:.2f})")
        else:
            log.info("Ни одна доля не даёт P>=10M>=90% — оборот под вопросом на всей сетке!")
        # текущая 25/75 для сравнения
        cur = df[df["mom_pct"] == 25]
        if not cur.empty:
            c = cur.iloc[0]
            log.info(f"\nТекущая 25/75: ret {c['ret_median']:.3f}%, Sharpe {c['sharpe_mean']:.2f}, "
                     f"TO P10 {c['turnover_p10']:.1f}M, P>=10M {c['prob_ge_10m']:.0f}%")

        # сохраним CSV для детального разбора
        out = Path("capital_split_results.csv")
        df.to_csv(out, index=False)
        log.info(f"\nДетальные результаты сохранены: {out}")


if __name__ == "__main__":
    main()