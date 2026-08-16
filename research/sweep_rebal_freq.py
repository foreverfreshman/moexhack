"""
Sweep по rebalance frequency × momentum window для naive cross-sectional momentum.

Ключевой эксперимент: можно ли поднять оборот до 10M+ за 14 дней этапа 2,
сохранив положительный Sharpe.

Для каждой комбинации (mom_window, rebal_freq) делаем backtest:
    - signal = cs_mom_{window}
    - top-3 long, bot-3 short, equal weights, gross=1
    - rebalance каждые rebal_freq дней
    - реализованный return за rebal_freq дней (no double-counting)

Запуск:
    python sweep_rebal_freq.py --data-dir market_data_final --cache daily.parquet
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_moex_daily

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("sweep")

FEE_BPS = 5.0
TOP_K = 3
BOTTOM_K = 3
CAPITAL = 1_000_000.0


def prep_features(daily: pd.DataFrame, mom_windows: List[int]) -> pd.DataFrame:
    """Считает mom фичи и cs_ ранги для нескольких окон."""
    df = daily.copy().sort_values(["ts", "ticker"]).reset_index(drop=True)
    grp = df.groupby("ticker", group_keys=False)
    for w in mom_windows:
        df[f"mom_{w}"] = grp["close"].transform(lambda x, w=w: x.pct_change(w))
        df[f"cs_mom_{w}"] = df.groupby("ts")[f"mom_{w}"].rank(pct=True)
        # shift на 1 день, anti-lookahead
        df[f"cs_mom_{w}"] = df.groupby("ticker")[f"cs_mom_{w}"].shift(1)
    return df


def backtest_at_freq(
    feat: pd.DataFrame,
    signal_col: str,
    rebal_freq: int,
    return_horizon: int,
) -> dict:
    """Бэктест с rebalance каждые rebal_freq дней и реализацией return за return_horizon.

    rebal_freq == return_horizon — non-overlapping returns, no double-count.
    """
    fee = FEE_BPS / 10000.0
    universe = sorted(feat["ticker"].unique())
    df = feat.copy()
    # forward return на нужный горизонт
    df["fwd"] = df.groupby("ticker")["close"].transform(
        lambda x: np.log(x.shift(-return_horizon) / x)
    )
    df = df.dropna(subset=["fwd", signal_col]).reset_index(drop=True)

    days = sorted(df["ts"].unique())
    rebal_days = days[::rebal_freq]

    prev_w = pd.Series(0.0, index=universe)
    rows = []

    for d in rebal_days:
        snap = df[df["ts"] == d].set_index("ticker")
        if snap[signal_col].isna().all():
            continue
        ranks = snap[signal_col].rank(ascending=False)
        longs = list(ranks.nsmallest(TOP_K).index)
        shorts = list(ranks.nlargest(BOTTOM_K).index)
        if not longs or not shorts:
            continue

        target = pd.Series(0.0, index=universe)
        for t in longs: target.loc[t] = 0.5 / len(longs)
        for t in shorts: target.loc[t] = -0.5 / len(shorts)

        turnover = (target - prev_w).abs().sum()
        fee_cost = turnover * fee
        gross = (target * snap["fwd"].reindex(universe).fillna(0)).sum()
        net = gross - fee_cost
        rows.append({"ts": d, "gross": gross, "fee": fee_cost, "net": net,
                     "turnover": turnover})
        prev_w = target

    bt = pd.DataFrame(rows)
    if bt.empty:
        return {}
    bt["equity"] = (1 + bt["net"]).cumprod()
    total = bt["equity"].iloc[-1] - 1
    sharpe = bt["net"].mean() / (bt["net"].std() + 1e-9) * np.sqrt(252 / rebal_freq)
    dd = (bt["equity"] / bt["equity"].cummax() - 1).min()
    # Оборот в рублях для 14 дней (этап 2)
    days_in_data = (bt["ts"].max() - bt["ts"].min()).days
    if days_in_data <= 0:
        return {}
    turnover_M = bt["turnover"].sum() * CAPITAL / 1e6
    proj_14d_M = turnover_M * 14 / days_in_data * (252 / 365)   # business-day approx
    return {
        "return": total, "sharpe": sharpe, "maxdd": dd,
        "n_rebal": len(bt),
        "turnover_M": turnover_M,
        "proj_turnover_14d_M": proj_14d_M,
        "avg_turnover_per_rebal": bt["turnover"].mean(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache", default=None)
    args = parser.parse_args()

    daily = load_moex_daily(args.data_dir, cache_path=args.cache)
    log.info(f"Loaded: {daily.ts.nunique()} days, {daily.ticker.nunique()} tickers")

    # Окна momentum: вокруг 21 (победитель) + пара вариантов
    mom_windows = [10, 15, 21, 30, 42, 63]
    # Rebalance frequencies (и horizon = тот же для non-overlap)
    rebal_freqs = [1, 2, 3, 5, 7, 10]

    feat = prep_features(daily, mom_windows)
    log.info(f"Computed features for {len(mom_windows)} momentum windows")

    log.info("\nSweep mom_window × rebal_freq:")
    log.info(f"{'mom_w':>6} {'rebal':>6} {'return':>10} {'Sharpe':>8} {'MaxDD':>8} "
             f"{'rebals':>7} {'turn_tot':>10} {'proj_14d':>11}")

    results = []
    for w in mom_windows:
        for r in rebal_freqs:
            res = backtest_at_freq(feat, f"cs_mom_{w}", r, r)
            if not res:
                continue
            res["mom_w"] = w
            res["rebal"] = r
            results.append(res)
            log.info(f"{w:>6} {r:>4}d  {res['return']:>+9.2%} {res['sharpe']:>+8.2f} "
                     f"{res['maxdd']:>+8.2%} {res['n_rebal']:>7} "
                     f"{res['turnover_M']:>8.1f}M ₽ {res['proj_turnover_14d_M']:>9.1f}M ₽")

    log.info("\n" + "=" * 80)
    log.info("ТОП-10 по Sharpe (с ограничением proj_14d ≥ 10M ₽ — pass штрафа):")
    log.info("=" * 80)
    df = pd.DataFrame(results)
    eligible = df[df["proj_turnover_14d_M"] >= 10.0]
    if eligible.empty:
        log.warning("Нет конфигов с прогнозом ≥10M за 14d. Печатаю топ-10 без фильтра по обороту.")
        eligible = df
    top = eligible.sort_values("sharpe", ascending=False).head(10)
    log.info(f"{'mom_w':>6} {'rebal':>6} {'return':>10} {'Sharpe':>8} {'proj_14d':>11}")
    for _, r in top.iterrows():
        log.info(f"{int(r['mom_w']):>6} {int(r['rebal']):>4}d  {r['return']:>+9.2%} "
                 f"{r['sharpe']:>+8.2f} {r['proj_turnover_14d_M']:>9.1f}M ₽")

    log.info("\nТОП-10 по Sharpe (любой оборот, для справки):")
    top_all = df.sort_values("sharpe", ascending=False).head(10)
    log.info(f"{'mom_w':>6} {'rebal':>6} {'return':>10} {'Sharpe':>8} {'proj_14d':>11}")
    for _, r in top_all.iterrows():
        log.info(f"{int(r['mom_w']):>6} {int(r['rebal']):>4}d  {r['return']:>+9.2%} "
                 f"{r['sharpe']:>+8.2f} {r['proj_turnover_14d_M']:>9.1f}M ₽")


if __name__ == "__main__":
    main()
