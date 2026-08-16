"""

Диагностика edge'a: проверка momentum vs reversal на разных горизонтах,
feature-by-feature Rank IC, per-ticker breakdown.

НЕ требует training модели, работает напрямую на дневном OHLCV.
Запуск:
    python diagnose.py --data-dir market_data_final --cache daily.parquet
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_moex_daily, TICKERS_DEFAULT

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("diag")

FEE_BPS = 5.0


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Минимум фич для diagnostic: momentum на разных горизонтах + vol/rsi."""
    df = df.copy().sort_values(["ts", "ticker"]).reset_index(drop=True)
    grp = df.groupby("ticker", group_keys=False)
    df["log_ret_1d"] = grp["close"].transform(lambda x: np.log(x / x.shift(1)))

    for w in [1, 2, 5, 10, 21]:
        df[f"mom_{w}"] = grp["close"].transform(lambda x, w=w: x.pct_change(w))

    df["vol_21"] = grp["log_ret_1d"].transform(lambda x: x.rolling(21).std())

    def rsi(x, n=14):
        d = x.diff()
        up = d.clip(lower=0).rolling(n).mean()
        dn = -d.clip(upper=0).rolling(n).mean()
        return 100 - 100 / (1 + up / (dn + 1e-9))
    df["rsi_14"] = grp["close"].transform(rsi)

    # Shift на 1 день — anti-lookahead
    feat_cols = [f"mom_{w}" for w in [1, 2, 5, 10, 21]] + ["vol_21", "rsi_14"]
    for c in feat_cols:
        df[c] = df.groupby("ticker")[c].shift(1)
    return df


def add_forward_returns(df: pd.DataFrame, horizons=[1, 2, 5, 10]) -> pd.DataFrame:
    df = df.copy()
    for h in horizons:
        df[f"fwd_{h}d"] = df.groupby("ticker")["close"].transform(
            lambda x, h=h: np.log(x.shift(-h) / x)
        )
    return df


def rank_ic(df: pd.DataFrame, feature: str, fwd_col: str) -> float:
    """Среднее Spearman Rank IC по timestamp."""
    ics = []
    for _, g in df.groupby("ts"):
        s_f = g[feature].dropna()
        s_y = g[fwd_col].dropna()
        common = s_f.index.intersection(s_y.index)
        if len(common) < 5:
            continue
        rho, _ = spearmanr(s_f.loc[common], s_y.loc[common])
        if np.isfinite(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


def naive_backtest(df: pd.DataFrame, rank_col: str, fwd_col: str,
                   ascending: bool = False, k: int = 3,
                   rebalance_every: int = 1) -> dict:
    """Простой long-short бэктест по одному фактору.

    ascending=False: top-K ranked самые ВЫСОКИЕ значения rank_col (классический momentum)
    ascending=True: top-K ranked самые НИЗКИЕ значения rank_col (reversal)
    """
    fee = FEE_BPS / 10000.0
    days = sorted(df["ts"].unique())
    rebal_days = days[::rebalance_every]
    universe = sorted(df["ticker"].unique())
    prev_w = pd.Series(0.0, index=universe)

    rets = []
    for d in rebal_days:
        snap = df[df["ts"] == d].set_index("ticker")
        if rank_col not in snap.columns or snap[rank_col].isna().all():
            continue
        ranks = snap[rank_col].rank(ascending=ascending)
        longs = list(ranks.nsmallest(k).index)
        shorts = list(ranks.nlargest(k).index)
        if not longs or not shorts:
            continue
        target = pd.Series(0.0, index=universe)
        for t in longs: target.loc[t] = 0.5 / k
        for t in shorts: target.loc[t] = -0.5 / k
        turnover = (target - prev_w).abs().sum()
        ret = (target * snap[fwd_col].reindex(universe).fillna(0)).sum()
        net = ret - turnover * fee
        rets.append(net)
        prev_w = target

    if not rets:
        return {"return": float("nan"), "sharpe": float("nan")}
    rets = np.array(rets)
    total_ret = (1 + rets).prod() - 1
    sharpe = rets.mean() / (rets.std() + 1e-9) * np.sqrt(252 / rebalance_every)
    return {"return": total_ret, "sharpe": sharpe, "n_periods": len(rets),
            "mean_pnl": float(rets.mean()), "vol": float(rets.std())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache", default=None)
    args = parser.parse_args()

    daily = load_moex_daily(args.data_dir, cache_path=args.cache)
    feat = add_basic_features(daily)
    feat = add_forward_returns(feat)
    feat = feat.dropna(subset=["fwd_1d"]).reset_index(drop=True)

    # ===== 1. Feature-by-feature Rank IC =====
    log.info("\n=== 1. Rank IC по фичам (без модели, чистый сигнал) ===")
    log.info(f"{'feature':<12} {'IC vs fwd_1d':>14} {'IC vs fwd_5d':>14} {'IC vs fwd_10d':>14}")
    for f in ["mom_1", "mom_2", "mom_5", "mom_10", "mom_21", "vol_21", "rsi_14"]:
        if f not in feat.columns:
            continue
        ic1 = rank_ic(feat, f, "fwd_1d")
        ic5 = rank_ic(feat, f, "fwd_5d")
        ic10 = rank_ic(feat, f, "fwd_10d")
        log.info(f"{f:<12} {ic1:>+14.4f} {ic5:>+14.4f} {ic10:>+14.4f}")
    log.info("\nИнтерпретация:")
    log.info("  IC > +0.02 — фактор предсказывает форвард-доходность (momentum-side)")
    log.info("  IC < -0.02 — фактор обратно коррелирует с форвард-доходностью (reversal-side)")
    log.info("  |IC| < 0.01 — статистический шум")

    # ===== 2. Naive momentum vs reversal baselines =====
    log.info("\n=== 2. Naive baselines: long top-3 / short bottom-3 ===")
    log.info(f"{'strategy':<32} {'horizon':>8} {'rebal':>6} {'return':>10} {'Sharpe':>8} {'n':>6}")
    for h in [1, 5]:
        for f in ["mom_1", "mom_5", "mom_21"]:
            for ascending, label in [(False, "MOM (long high, short low)"),
                                      (True, "REV (long low, short high)")]:
                res = naive_backtest(feat, f, f"fwd_{h}d", ascending=ascending,
                                     k=3, rebalance_every=h)
                log.info(f"{label} {f:<6} {h:>4}d {h:>4} "
                         f"{res['return']:>+9.2%} {res['sharpe']:>+8.2f} {res.get('n_periods', 0):>6}")
        log.info("  ---")

    # ===== 3. Per-ticker autocorrelation =====
    log.info("\n=== 3. Per-ticker 1-day autocorrelation log_ret ===")
    log.info("Положительная — momentum-склонность; отрицательная — reversal.")
    log.info(f"{'ticker':<8} {'auto-corr ret(t) vs ret(t+1)':>30} {'n_obs':>7}")
    for ticker, g in feat.groupby("ticker"):
        rets = g["log_ret_1d"].dropna()
        if len(rets) < 30:
            continue
        ac = rets.autocorr(lag=1)
        log.info(f"{ticker:<8} {ac:>+30.4f} {len(rets):>7}")

    # ===== 4. Cross-sectional dispersion =====
    log.info("\n=== 4. Cross-sectional дисперсия по дням (наличие сигнала) ===")
    disp = feat.groupby("ts")["fwd_1d"].std()
    log.info(f"  Median std fwd_1d по дням: {disp.median():.4f}")
    log.info(f"  Mean std fwd_1d по дням: {disp.mean():.4f}")
    log.info(f"  Days with disp < 0.5%: {(disp < 0.005).sum()} / {len(disp)}")
    log.info(f"  Days with disp > 2%: {(disp > 0.02).sum()} / {len(disp)}")
    log.info("Если медиана < 0.5%, разлёт мал — сложно отделить top от bottom;")
    log.info("если >2%, разлёт огромный, скорее аномальный режим.")


if __name__ == "__main__":
    main()
