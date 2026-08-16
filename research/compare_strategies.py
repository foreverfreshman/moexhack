"""
Compare strategies: naive vs LightGBM, разные feature sets и rebalance frequencies.

Все стратегии работают на одних и тех же дневных данных, одни и те же WF splits,
одинаковый backtest (fee, gross exposure, top-K). Цель — выбрать стратегию с
лучшим Sharpe-after-cost, прежде чем добавлять meta-layer и position sizing.

Запуск:
    python compare_strategies.py --data-dir market_data_final --cache daily.parquet
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_moex_daily
from pipeline import (
    add_features, add_target, build_splits, train_primary,
    backtest, HORIZON_DAYS, TOP_K, BOTTOM_K, FEE_BPS_PER_SIDE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("compare")


# ============================================================
# Strategy implementations
# ============================================================

def strategy_naive_single(feat: pd.DataFrame, factor: str = "mom_21") -> pd.DataFrame:
    """Один фактор. Использует cs_ версию (rank нормализован) как pred."""
    cs = f"cs_{factor}"
    if cs not in feat.columns:
        raise ValueError(f"{cs} not in features")
    out = feat.copy()
    out["pred"] = out[cs]
    return out


def strategy_naive_multifactor(feat: pd.DataFrame) -> pd.DataFrame:
    """Multi-factor ensemble. Веса грубые, не optimized:
        +1 * rank(mom_21)   # momentum long
        +1 * rank(rsi_14)   # momentum-RSI long
        -1 * rank(vol_21)   # low-vol premium
    """
    out = feat.copy()
    out["pred"] = (
        out["cs_mom_21"].fillna(0.5)
        + out["cs_rsi_14"].fillna(0.5)
        - out["cs_vol_21"].fillna(0.5)
    )
    return out


def strategy_lightgbm(feat: pd.DataFrame, feature_cols: List[str],
                       splits, label: str = "lgb") -> pd.DataFrame:
    """LightGBM LambdaRank на заданном наборе фичей. Walk-forward OOS."""
    test_preds = []
    for i, sp in enumerate(splits):
        m_tr = (feat.ts >= sp.train_start) & (feat.ts <= sp.train_end)
        m_va = (feat.ts >= sp.val_start) & (feat.ts <= sp.val_end)
        m_te = (feat.ts >= sp.test_start) & (feat.ts <= sp.test_end)
        df_tr = feat.loc[m_tr].dropna(subset=feature_cols).reset_index(drop=True)
        df_va = feat.loc[m_va].dropna(subset=feature_cols).reset_index(drop=True)
        df_te = feat.loc[m_te].dropna(subset=feature_cols).reset_index(drop=True)
        if df_tr.empty or df_va.empty or df_te.empty:
            continue
        booster = train_primary(df_tr, df_va, feature_cols)
        df_te = df_te.copy()
        df_te["pred"] = booster.predict(df_te[feature_cols])
        test_preds.append(df_te)
        if (i + 1) % 5 == 0:
            log.info(f"  [{label}] fold {i+1}/{len(splits)} done")
    return pd.concat(test_preds, ignore_index=True)


# ============================================================
# Standardised backtest wrapper
# ============================================================

def evaluate(df: pd.DataFrame, name: str) -> Dict:
    """Прогоняет backtest и возвращает ключевые числа в dict."""
    bt = backtest(df[["ts", "ticker", "pred", "fwd_ret"]], name=name)
    if bt.empty:
        return {"name": name, "return": float("nan"), "sharpe": float("nan"),
                "maxdd": float("nan"), "turnover_M": 0.0}
    total = bt["equity"].iloc[-1] - 1
    sharpe = bt["net"].mean() / (bt["net"].std() + 1e-9) * np.sqrt(252 / HORIZON_DAYS)
    dd = (bt["equity"] / bt["equity"].cummax() - 1).min()
    turnover_M = bt["turnover"].sum() * 1.0   # 1M capital → turnover * 1M = M ₽
    return {"name": name, "return": total, "sharpe": sharpe, "maxdd": dd,
            "turnover_M": turnover_M, "n_rebal": len(bt)}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache", default=None)
    args = parser.parse_args()

    daily = load_moex_daily(args.data_dir, cache_path=args.cache)
    log.info(f"Data: {daily.ts.nunique()} days, {daily.ticker.nunique()} tickers")

    feat = add_features(daily)
    feat = add_target(feat)
    feat = feat.dropna(subset=["fwd_ret"]).reset_index(drop=True)
    feat["target"] = feat["target"].fillna(-1).astype(int)   # для LGB

    unique_days = pd.DatetimeIndex(sorted(feat["ts"].unique()))
    splits = build_splits(unique_days)
    log.info(f"WF splits: {len(splits)} (each with {len(splits[0].__dict__)//2} periods)")

    # Restrict feature subsets
    reduced_cols = ["cs_mom_21", "cs_vol_21", "cs_rsi_14",
                    "mom_21", "vol_21", "rsi_14", "dow"]
    reduced_cols = [c for c in reduced_cols if c in feat.columns]

    full_cols = [c for c in feat.columns
                 if c.startswith(("cs_", "mom_", "vol_", "atr_", "rsi_", "dist_", "volume_z", "range_pct"))]
    full_cols += ["dow"]
    full_cols = [c for c in full_cols if c in feat.columns]

    log.info(f"Reduced feature set: {len(reduced_cols)} cols ({reduced_cols})")
    log.info(f"Full feature set: {len(full_cols)} cols")

    # Для honesta сравнения тестируем все стратегии на одном OOS window —
    # объединение всех WF tests. Naive стратегии работают на всём диапазоне feat,
    # но мы их обрезаем до того же временного диапазона.
    test_window_start = splits[0].test_start
    test_window_end = splits[-1].test_end
    log.info(f"OOS window: {test_window_start.date()} → {test_window_end.date()}")

    # OOS mask
    oos_mask = (feat.ts >= test_window_start) & (feat.ts <= test_window_end)
    feat_oos = feat.loc[oos_mask].reset_index(drop=True)

    results = []

    # 1. Naive single-factor mom_21
    log.info("\n=== Strategy 1: Naive single-factor mom_21 ===")
    df1 = strategy_naive_single(feat_oos, "mom_21")
    results.append(evaluate(df1, "naive_mom21"))

    # 2. Naive single-factor rsi_14
    log.info("\n=== Strategy 2: Naive single-factor rsi_14 ===")
    df2 = strategy_naive_single(feat_oos, "rsi_14")
    results.append(evaluate(df2, "naive_rsi14"))

    # 3. Naive multi-factor ensemble
    log.info("\n=== Strategy 3: Naive multi-factor (mom_21 + rsi_14 - vol_21) ===")
    df3 = strategy_naive_multifactor(feat_oos)
    results.append(evaluate(df3, "naive_multifactor"))

    # 4. LightGBM на reduced features
    log.info("\n=== Strategy 4: LightGBM reduced (3 top-IC features) ===")
    df4 = strategy_lightgbm(feat, reduced_cols, splits, "lgb_reduced")
    results.append(evaluate(df4, "lgb_reduced"))

    # 5. LightGBM на full features
    log.info("\n=== Strategy 5: LightGBM full (25 features) ===")
    df5 = strategy_lightgbm(feat, full_cols, splits, "lgb_full")
    results.append(evaluate(df5, "lgb_full"))

    # ===== Summary =====
    log.info("\n" + "=" * 80)
    log.info("SUMMARY: all strategies on the same OOS window")
    log.info("=" * 80)
    log.info(f"{'strategy':<22} {'return':>10} {'Sharpe':>8} {'MaxDD':>8} {'turnover':>11} {'rebals':>7}")
    for r in sorted(results, key=lambda x: -x.get("sharpe", -999) if pd.notna(x.get("sharpe")) else -999):
        log.info(f"{r['name']:<22} {r['return']:>+9.2%} {r['sharpe']:>+8.2f} "
                 f"{r['maxdd']:>+8.2%} {r['turnover_M']:>9.1f}M ₽ {r.get('n_rebal',0):>7}")

    log.info("\nIntepretация:")
    log.info("  - Если naive_mom21 >> lgb_full → модель портит сигнал, нужен другой подход")
    log.info("  - Если lgb_reduced > lgb_full → проблема в количестве фич (overfit)")
    log.info("  - Если naive_multifactor > naive_mom21 → ensemble лучше single (как и ожидали)")


if __name__ == "__main__":
    main()
