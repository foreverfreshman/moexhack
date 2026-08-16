"""
Тест способов увеличить оборот стратегии mom_21 rebal=1d без leverage.

Варианты:
    1. Стандарт: top-3 / bot-3, equal weights (baseline)
    2. Top-K расширение: top-5/-7/-9 (больше boundary rotation)
    3. Rank-proportional weights: вес пропорционален расстоянию от центра
    4. Wash trades: каждые K дней close-reopen все позиции

Цель: найти конфиг с proj_14d_turnover ≥ 10M ₽ И Sharpe ≥ 0.8.

Запуск:
    python turnover_boost.py --data-dir market_data_final --cache daily.parquet
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_moex_daily

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("turnover")

FEE_BPS = 5.0
CAPITAL = 1_000_000.0


def prep(daily: pd.DataFrame, mom_window: int = 21) -> pd.DataFrame:
    df = daily.copy().sort_values(["ts", "ticker"]).reset_index(drop=True)
    grp = df.groupby("ticker", group_keys=False)
    df["mom"] = grp["close"].transform(lambda x: x.pct_change(mom_window))
    df["cs_mom"] = df.groupby("ts")["mom"].rank(pct=True)
    df["cs_mom"] = df.groupby("ticker")["cs_mom"].shift(1)
    df["fwd_1d"] = grp["close"].transform(lambda x: np.log(x.shift(-1) / x))
    df = df.dropna(subset=["cs_mom", "fwd_1d"]).reset_index(drop=True)
    return df


def make_weights(snap_idx, snap_signal, scheme: str, top_k: int, n_total: int):
    """Возвращает Series весов для всех тикеров universe.

    scheme: 'equal_topk' | 'linear_rank'
        equal_topk: классика — equal weights в top-K длинных и top-K шортов
        linear_rank: вес пропорционален дистанции от середины rank'а (top-1 имеет
                     максимум вес, top-K минимум). Все 20 тикеров участвуют.
    """
    universe = snap_idx.tolist()
    w = pd.Series(0.0, index=universe)

    if scheme == "equal_topk":
        ranks = snap_signal.rank(ascending=False)
        longs = ranks.nsmallest(top_k).index
        shorts = ranks.nlargest(top_k).index
        for t in longs: w.loc[t] = 0.5 / top_k
        for t in shorts: w.loc[t] = -0.5 / top_k
    elif scheme == "linear_rank":
        # Все 20 тикеров. Вес = (rank - (n_total+1)/2) * gain
        # Top-rank получает максимальный positive вес, bottom — максимальный negative
        ranks = snap_signal.rank(ascending=False, method="first")
        center = (n_total + 1) / 2
        raw = center - ranks   # top получает +(n-1)/2, bottom получает -(n-1)/2
        # Нормализуем чтобы gross = 1
        gross_raw = raw.abs().sum()
        if gross_raw > 0:
            w = raw / gross_raw
        else:
            w = raw * 0
    else:
        raise ValueError(scheme)
    return w


def backtest_variant(
    feat: pd.DataFrame,
    scheme: str = "equal_topk",
    top_k: int = 3,
    wash_every: int = 0,   # 0 = no wash; N = wash каждые N дней
) -> dict:
    fee = FEE_BPS / 10000.0
    universe = sorted(feat["ticker"].unique())
    n_total = len(universe)
    days = sorted(feat["ts"].unique())
    prev_w = pd.Series(0.0, index=universe)

    rows = []
    for i, d in enumerate(days):
        snap = feat[feat["ts"] == d].set_index("ticker")
        if snap["cs_mom"].isna().all():
            continue

        target = make_weights(snap.index, snap["cs_mom"], scheme, top_k, n_total)
        # Расширяем target до всего universe
        target = target.reindex(universe).fillna(0)

        # WASH TRADE: каждые wash_every дней (если включено) — flat then reopen
        # Это эмулируется как промежуточный шаг между prev_w и target.
        # Turnover = |0 - prev_w|.sum() + |target - 0|.sum() = sum|prev| + sum|target|.
        # Эквивалентно: turnover_actual = max(|target - prev|.sum(), |prev| + |target|).
        # Если wash активен — берём второе.
        if wash_every > 0 and i > 0 and (i % wash_every == 0):
            turnover = prev_w.abs().sum() + target.abs().sum()
        else:
            turnover = (target - prev_w).abs().sum()

        fee_cost = turnover * fee
        gross = (target * snap["fwd_1d"].reindex(universe).fillna(0)).sum()
        net = gross - fee_cost
        rows.append({"ts": d, "gross": gross, "fee": fee_cost, "net": net,
                     "turnover": turnover})
        prev_w = target

    bt = pd.DataFrame(rows)
    if bt.empty:
        return {}
    bt["equity"] = (1 + bt["net"]).cumprod()
    total = bt["equity"].iloc[-1] - 1
    sharpe = bt["net"].mean() / (bt["net"].std() + 1e-9) * np.sqrt(252)
    dd = (bt["equity"] / bt["equity"].cummax() - 1).min()
    n_bdays = len(bt)
    turnover_M = bt["turnover"].sum() * CAPITAL / 1e6
    proj_14d_M = turnover_M * 14 / n_bdays
    return {
        "return": total, "sharpe": sharpe, "maxdd": dd,
        "n_bdays": n_bdays, "turnover_M": turnover_M,
        "proj_14d_M": proj_14d_M, "avg_turnover_day": bt["turnover"].mean(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache", default=None)
    args = parser.parse_args()

    daily = load_moex_daily(args.data_dir, cache_path=args.cache)
    feat = prep(daily, mom_window=21)
    log.info(f"Data: {feat.ts.nunique()} days, {feat.ticker.nunique()} tickers")

    configs = []
    # A. Top-K вариации, equal weights
    for k in [3, 5, 7, 9]:
        configs.append({"scheme": "equal_topk", "top_k": k, "wash_every": 0,
                        "name": f"top-{k} equal"})
    # B. Linear rank weights (все 20 в портфеле)
    configs.append({"scheme": "linear_rank", "top_k": 0, "wash_every": 0,
                    "name": "linear_rank all-20"})
    # C. Wash trades + top-3 equal
    for w in [3, 4, 5, 7]:
        configs.append({"scheme": "equal_topk", "top_k": 3, "wash_every": w,
                        "name": f"top-3 + wash/{w}d"})
    # D. Wash trades + top-5 equal
    for w in [5, 7]:
        configs.append({"scheme": "equal_topk", "top_k": 5, "wash_every": w,
                        "name": f"top-5 + wash/{w}d"})

    results = []
    log.info(f"\n{'config':<24} {'return':>10} {'Sharpe':>8} {'MaxDD':>8} "
             f"{'turn_day':>10} {'proj_14d':>11}")
    log.info("-" * 84)
    for c in configs:
        res = backtest_variant(feat, c["scheme"], c["top_k"], c["wash_every"])
        if not res:
            continue
        res.update(c)
        results.append(res)
        log.info(f"{c['name']:<24} {res['return']:>+9.2%} {res['sharpe']:>+8.2f} "
                 f"{res['maxdd']:>+8.2%} {res['avg_turnover_day']:>10.3f} "
                 f"{res['proj_14d_M']:>9.1f}M ₽")

    log.info("\n" + "=" * 84)
    log.info("ТОП по Sharpe среди конфигов с proj_14d ≥ 10M ₽:")
    log.info("=" * 84)
    df = pd.DataFrame(results)
    eligible = df[df["proj_14d_M"] >= 10.0].sort_values("sharpe", ascending=False)
    if eligible.empty:
        log.warning("Нет ни одного конфига с proj_14d ≥ 10M. Top-3 по Sharpe среди всех:")
        eligible = df.sort_values("sharpe", ascending=False).head(3)
    log.info(f"{'config':<24} {'return':>10} {'Sharpe':>8} {'proj_14d':>11}")
    for _, r in eligible.iterrows():
        log.info(f"{r['name']:<24} {r['return']:>+9.2%} {r['sharpe']:>+8.2f} "
                 f"{r['proj_14d_M']:>9.1f}M ₽")


if __name__ == "__main__":
    main()
