"""
LambdaRank + Trailing-Barrier Meta-Labeling pipeline для MOEX.

Принимает daily OHLCV (например из data_loader.load_moex_daily) и прогоняет:
    1. Feature engineering (per-ticker + cross-sectional ranks)
    2. Walk-forward primary LambdaRank (rank-IC оптимизация)
    3. Trailing-barrier labeling сигналов primary model:
        - PT_MULT = 3.0σ (широкий profit-take)
        - SL_MULT = 1.5σ (initial stop)
        - BREAKEVEN_AT = 1.0σ (двигаем SL в entry)
        - TRAIL_AT = 1.5σ (активируем trailing stop)
        - TRAIL_DIST = 1.0σ (trail на этой дистанции от max favorable)
        - MAX_HOLDING = 7 дней
    4. Walk-forward meta-model (binary classifier поверх primary)
    5. Auto-tune threshold на out-of-sample (sweep + best Sharpe)
    6. Comparative backtest: primary-only vs primary+meta@best_threshold

Использование:
    from data_loader import load_moex_daily
    from pipeline import run_pipeline
    daily = load_moex_daily("/path/to/market_data_final", cache_path="cache.parquet")
    results = run_pipeline(daily)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score, roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
log = logging.getLogger("pipeline")

# ============================================================
# Конфиг
# ============================================================

# Учитывая что у нас 3 года данных:
#   Train 252 days + Val 42 + Embargo 7 + Test 21 = 322 минимум
#   Step 21 → ~ 20 folds на 3y истории
TRAIN_DAYS = 252
VAL_DAYS = 42
TEST_DAYS = 21
STEP_DAYS = 21

# Trailing barrier params (D+1 horizon, max holding до 7 дней)
PT_MULT = 3.0
SL_MULT = 1.5
BREAKEVEN_AT = 1.0
TRAIL_AT = 1.5
TRAIL_DIST = 1.0
MAX_HOLDING_DAYS = 10
VOL_WINDOW = 20

EMBARGO_DAYS = MAX_HOLDING_DAYS + 1   # покрываем path-overlap

# Прогноз и портфель
HORIZON_DAYS = 5
N_BINS = 5
TOP_K = 3
BOTTOM_K = 3

# Auto-threshold sweep
META_THRESHOLD_GRID = [0.45, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65]

# Комиссия
FEE_BPS_PER_SIDE = 5.0

# Стартовый капитал (для отчётности по обороту)
INITIAL_CAPITAL = 1_000_000.0


# ============================================================
# 1. Features
# ============================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["ts", "ticker"]).reset_index(drop=True)
    grp = df.groupby("ticker", group_keys=False)
    df["ret_1d"] = grp["close"].pct_change(1)
    df["log_ret_1d"] = grp["close"].transform(lambda x: np.log(x / x.shift(1)))

    for w in [5, 10, 21, 63]:
        df[f"mom_{w}"] = grp["close"].transform(lambda x, w=w: x.pct_change(w))
    for w in [5, 21]:
        df[f"vol_{w}"] = grp["log_ret_1d"].transform(lambda x, w=w: x.rolling(w).std())

    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["atr_14"] = grp["range_pct"].transform(lambda x: x.rolling(14).mean())

    df["vol_ma_21"] = grp["volume"].transform(lambda x: x.rolling(21).mean())
    df["volume_z"] = (df["volume"] - df["vol_ma_21"]) / (df["vol_ma_21"] + 1e-9)

    def rsi(x, n=14):
        d = x.diff()
        up = d.clip(lower=0).rolling(n).mean()
        dn = -d.clip(upper=0).rolling(n).mean()
        return 100 - 100 / (1 + up / (dn + 1e-9))
    df["rsi_14"] = grp["close"].transform(rsi)

    df["dist_hi20"] = (df["close"] - grp["close"].transform(lambda x: x.rolling(20).max())) / df["close"]
    df["dist_lo20"] = (df["close"] - grp["close"].transform(lambda x: x.rolling(20).min())) / df["close"]

    cs_features = ["mom_5", "mom_10", "mom_21", "mom_63", "vol_5", "vol_21",
                   "atr_14", "volume_z", "rsi_14", "dist_hi20", "dist_lo20"]
    for f in cs_features:
        df[f"cs_{f}"] = df.groupby("ts")[f].rank(pct=True)

    df["dow"] = df["ts"].dt.dayofweek.astype("category")

    # Shift на 1 день — anti-lookahead
    feature_like = [c for c in df.columns
                    if c.startswith(("mom_", "vol_", "atr_", "rsi_", "cs_", "volume_z", "range_pct", "dist_"))]
    for c in feature_like:
        df[c] = df.groupby("ticker")[c].shift(1)

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fwd_ret"] = (
        df.groupby("ticker")["close"]
          .transform(lambda x: np.log(x.shift(-HORIZON_DAYS) / x))
    )
    df["fwd_ret"] = df["fwd_ret"].replace([np.inf, -np.inf], np.nan)

    def rank_to_bin(s):
        if s.notna().sum() < N_BINS:
            return pd.Series(np.nan, index=s.index)
        r = s.rank(method="first", na_option="keep")
        n = s.notna().sum()
        return np.floor((r - 1) / n * N_BINS).clip(0, N_BINS - 1)

    df["target"] = df.groupby("ts")["fwd_ret"].transform(rank_to_bin)
    return df


# ============================================================
# 2. Walk-forward
# ============================================================

@dataclass
class Split:
    train_start: pd.Timestamp; train_end: pd.Timestamp
    val_start: pd.Timestamp; val_end: pd.Timestamp
    test_start: pd.Timestamp; test_end: pd.Timestamp


def build_splits(unique_days: pd.DatetimeIndex) -> List[Split]:
    total = TRAIN_DAYS + VAL_DAYS + EMBARGO_DAYS + TEST_DAYS
    if len(unique_days) < total:
        raise ValueError(f"Need ≥{total} days, got {len(unique_days)}. "
                         f"Уменьши TRAIN_DAYS или загрузи больше данных.")
    out, i = [], 0
    while i + total <= len(unique_days):
        out.append(Split(
            unique_days[i], unique_days[i + TRAIN_DAYS - 1],
            unique_days[i + TRAIN_DAYS], unique_days[i + TRAIN_DAYS + VAL_DAYS - 1],
            unique_days[i + TRAIN_DAYS + VAL_DAYS + EMBARGO_DAYS],
            unique_days[i + TRAIN_DAYS + VAL_DAYS + EMBARGO_DAYS + TEST_DAYS - 1],
        ))
        i += STEP_DAYS
    log.info(f"Built {len(out)} walk-forward splits")
    return out


# ============================================================
# 3. Primary LambdaRank
# ============================================================

LGB_RANK = {
    "objective": "lambdarank", "metric": "ndcg",
    "ndcg_eval_at": [3, 5], "lambdarank_truncation_level": 10,
    "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 30,
    "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 5,
    "lambda_l2": 1.0, "max_depth": 6, "verbose": -1, "num_threads": -1,
}


def _rank_dataset(df: pd.DataFrame, feat_cols: List[str]) -> lgb.Dataset:
    df = df.sort_values(["ts", "ticker"]).reset_index(drop=True)
    groups = df.groupby("ts", sort=False).size().values
    cat = [c for c in ["dow"] if c in feat_cols]
    return lgb.Dataset(df[feat_cols], label=df["target"].astype(int).values,
                       group=groups, categorical_feature=cat, free_raw_data=False)


def train_primary(df_tr, df_va, feat_cols) -> lgb.Booster:
    return lgb.train(
        LGB_RANK, _rank_dataset(df_tr, feat_cols), num_boost_round=500,
        valid_sets=[_rank_dataset(df_va, feat_cols)], valid_names=["val"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )


# ============================================================
# 4. Trailing Barrier Labeler
# ============================================================

def _simulate_one(
    close_path: np.ndarray, high_path: np.ndarray, low_path: np.ndarray,
    sigma: float, direction: int,   # +1 long, -1 short
) -> Tuple[int, float, str]:
    """Симулирует одну сделку с trailing stop.

    Returns: (label, realized_log_pnl, exit_reason)
        label: 1 если pnl > 0 else 0
        exit_reason: "pt" | "sl" | "time"

    direction=+1 (long):
        entry = close_path[0]
        PT level = entry * exp(+PT_MULT * σ)
        SL level (initial) = entry * exp(-SL_MULT * σ)
        Когда max_high >= entry * exp(+BREAKEVEN_AT * σ): SL → max(SL, entry)
        Когда max_high >= entry * exp(+TRAIL_AT * σ): trailing активен,
            SL → max(SL, max_high * exp(-TRAIL_DIST * σ))
        Exit: high касается PT → label=1; low касается SL → label по PnL знаку;
              time barrier → label по PnL знаку

    direction=-1 (short): зеркально, отслеживаем min_low.
    """
    entry = close_path[0]
    if direction == 1:
        pt = entry * np.exp(+PT_MULT * sigma)
        sl = entry * np.exp(-SL_MULT * sigma)
        be_trigger = entry * np.exp(+BREAKEVEN_AT * sigma)
        tr_trigger = entry * np.exp(+TRAIL_AT * sigma)
        extreme_favorable = entry
        for k in range(1, len(close_path)):
            extreme_favorable = max(extreme_favorable, high_path[k])
            # Обновляем SL
            if extreme_favorable >= tr_trigger:
                trail_sl = extreme_favorable * np.exp(-TRAIL_DIST * sigma)
                sl = max(sl, trail_sl)
            elif extreme_favorable >= be_trigger:
                sl = max(sl, entry)
            # Проверяем exits в порядке: PT first (optimistic), SL second
            if high_path[k] >= pt:
                pnl = np.log(pt / entry)
                return (1, pnl, "pt")
            if low_path[k] <= sl:
                pnl = np.log(sl / entry)
                return (int(pnl > 0), pnl, "sl")
        # time barrier
        exit_p = close_path[-1]
        pnl = np.log(exit_p / entry)
        return (int(pnl > 0), pnl, "time")
    else:  # short
        pt = entry * np.exp(-PT_MULT * sigma)
        sl = entry * np.exp(+SL_MULT * sigma)
        be_trigger = entry * np.exp(-BREAKEVEN_AT * sigma)
        tr_trigger = entry * np.exp(-TRAIL_AT * sigma)
        extreme_favorable = entry
        for k in range(1, len(close_path)):
            extreme_favorable = min(extreme_favorable, low_path[k])
            if extreme_favorable <= tr_trigger:
                trail_sl = extreme_favorable * np.exp(+TRAIL_DIST * sigma)
                sl = min(sl, trail_sl)
            elif extreme_favorable <= be_trigger:
                sl = min(sl, entry)
            if low_path[k] <= pt:
                pnl = -np.log(pt / entry)
                return (1, pnl, "pt")
            if high_path[k] >= sl:
                pnl = -np.log(sl / entry)
                return (int(pnl > 0), pnl, "sl")
        exit_p = close_path[-1]
        pnl = -np.log(exit_p / entry)
        return (int(pnl > 0), pnl, "time")


def trailing_barrier_labels(
    closes: pd.DataFrame, highs: pd.DataFrame, lows: pd.DataFrame,
    signals: pd.DataFrame, vol: pd.DataFrame,
    max_holding: int = MAX_HOLDING_DAYS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Возвращает 3 DataFrame (same shape as signals): labels, pnls, exit_reasons.

    labels: 1 если sign(PnL) > 0, 0 иначе, NaN если signal=0 или не хватает данных
    pnls: realized log PnL
    exit_reasons: 'pt'/'sl'/'time'
    """
    labels = pd.DataFrame(np.nan, index=signals.index, columns=signals.columns)
    pnls = pd.DataFrame(np.nan, index=signals.index, columns=signals.columns)
    reasons = pd.DataFrame("", index=signals.index, columns=signals.columns, dtype=object)

    closes_arr, highs_arr, lows_arr = closes.values, highs.values, lows.values
    sig_arr, vol_arr = signals.values, vol.values
    n_dates, n_assets = closes_arr.shape

    for t in range(n_dates - 1):
        for j in range(n_assets):
            s = sig_arr[t, j]
            sigma = vol_arr[t, j]
            if s == 0 or not np.isfinite(s) or not np.isfinite(sigma) or sigma <= 0:
                continue
            entry = closes_arr[t, j]
            if not np.isfinite(entry) or entry <= 0:
                continue
            end_idx = min(t + max_holding, n_dates - 1)
            cp = closes_arr[t:end_idx + 1, j]
            hp = highs_arr[t:end_idx + 1, j]
            lp = lows_arr[t:end_idx + 1, j]
            if np.isnan(cp).any() or np.isnan(hp).any() or np.isnan(lp).any():
                continue
            label, pnl, reason = _simulate_one(cp, hp, lp, sigma, int(np.sign(s)))
            labels.iat[t, j] = label
            pnls.iat[t, j] = pnl
            reasons.iat[t, j] = reason
    return labels, pnls, reasons


# ============================================================
# 5. Meta-model
# ============================================================

LGB_META = {
    "objective": "binary", "metric": ["binary_logloss", "auc"],
    "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 30,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    "lambda_l2": 1.0, "max_depth": 6, "verbose": -1, "num_threads": -1,
}


def train_meta(df_tr, df_va, feat_cols) -> lgb.Booster:
    cat = [c for c in ["dow"] if c in feat_cols]
    train = lgb.Dataset(df_tr[feat_cols], label=df_tr["tb_label"].values,
                        categorical_feature=cat, free_raw_data=False)
    val = lgb.Dataset(df_va[feat_cols], label=df_va["tb_label"].values,
                      categorical_feature=cat, free_raw_data=False)
    return lgb.train(
        LGB_META, train, num_boost_round=500,
        valid_sets=[val], valid_names=["val"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )


# ============================================================
# 6. Backtest
# ============================================================

def backtest(
    df: pd.DataFrame, name: str,
    meta_threshold: float | None = None,
    capital: float = INITIAL_CAPITAL,
) -> pd.DataFrame:
    """Long-short equal-weight backtest с реалистичными комиссиями.

    Если meta_threshold задан и в df есть meta_pred — фильтруем сигналы.
    """
    fee = FEE_BPS_PER_SIDE / 10000.0
    rows = []
    days = sorted(df["ts"].unique())
    rebal_days = days[::HORIZON_DAYS]
    universe = sorted(df["ticker"].unique())
    prev_w = pd.Series(0.0, index=universe)

    for d in rebal_days:
        snap = df[df["ts"] == d].set_index("ticker")
        if "pred" not in snap.columns or snap["pred"].isna().all():
            continue
        ranks = snap["pred"].rank(ascending=False)
        longs = list(ranks.nsmallest(TOP_K).index)
        shorts = list(ranks.nlargest(BOTTOM_K).index)
        if meta_threshold is not None and "meta_pred" in snap.columns:
            longs = [t for t in longs if pd.notna(snap.loc[t, "meta_pred"])
                     and snap.loc[t, "meta_pred"] >= meta_threshold]
            shorts = [t for t in shorts if pd.notna(snap.loc[t, "meta_pred"])
                      and snap.loc[t, "meta_pred"] >= meta_threshold]

        # Equal weights, gross exposure = 1 (no leverage). Если longs=shorts={},
        # позиция плоская — комиссия только за закрытие предыдущих.
        target = pd.Series(0.0, index=universe)
        long_w = 0.5 / len(longs) if longs else 0
        short_w = 0.5 / len(shorts) if shorts else 0
        for t in longs: target.loc[t] = long_w
        for t in shorts: target.loc[t] = -short_w

        turnover = (target - prev_w).abs().sum()
        fee_cost = turnover * fee
        ret_vec = snap["fwd_ret"].reindex(universe).fillna(0)
        gross = (target * ret_vec).sum()
        net = gross - fee_cost

        rows.append({"ts": d, "gross": gross, "fee": fee_cost, "net": net,
                     "turnover": turnover, "n_long": len(longs), "n_short": len(shorts)})
        prev_w = target

    bt = pd.DataFrame(rows)
    if bt.empty:
        log.warning(f"  [{name}] empty backtest")
        return bt
    bt["equity"] = (1 + bt["net"]).cumprod()
    total = bt["equity"].iloc[-1] - 1
    sharpe = bt["net"].mean() / (bt["net"].std() + 1e-9) * np.sqrt(252 / HORIZON_DAYS)
    dd = (bt["equity"] / bt["equity"].cummax() - 1).min()
    n_active = int(((bt["n_long"] + bt["n_short"]) > 0).sum())
    turnover_rub = bt["turnover"].sum() * capital   # L1 turnover уже включает обе стороны
    log.info(f"  [{name:24s}] ret {total:+.2%} | Sharpe {sharpe:5.2f} | "
             f"MaxDD {dd:+.2%} | active {n_active}/{len(bt)} | "
             f"avg turnover/day {bt['turnover'].mean():.2f} | "
             f"total turnover ≈ {turnover_rub/1e6:.1f}M ₽")
    return bt


def find_best_threshold(df: pd.DataFrame, grid: List[float] = META_THRESHOLD_GRID) -> Dict:
    """Sweep по threshold, возвращает best по Sharpe-after-cost."""
    results = []
    for thr in grid:
        bt = backtest(df, name=f"sweep@{thr:.2f}", meta_threshold=thr)
        if bt.empty:
            continue
        sharpe = bt["net"].mean() / (bt["net"].std() + 1e-9) * np.sqrt(252 / HORIZON_DAYS)
        ret = bt["equity"].iloc[-1] - 1
        results.append({"threshold": thr, "sharpe": sharpe, "return": ret, "trades": len(bt)})
    if not results:
        return {"best_threshold": grid[len(grid) // 2], "all": []}
    best = max(results, key=lambda r: r["sharpe"])
    log.info(f"  Best threshold: {best['threshold']:.2f} (Sharpe {best['sharpe']:.2f}, "
             f"return {best['return']:+.2%})")
    return {"best_threshold": best["threshold"], "all": results}


# ============================================================
# 7. Метрики
# ============================================================

def rank_ic(df: pd.DataFrame) -> Tuple[float, float, int]:
    ics = []
    for _, g in df.groupby("ts"):
        if g["fwd_ret"].notna().sum() < 5 or g["pred"].notna().sum() < 5:
            continue
        rho, _ = spearmanr(g["pred"], g["fwd_ret"])
        if np.isfinite(rho):
            ics.append(rho)
    arr = np.array(ics)
    if len(arr) == 0:
        return float("nan"), float("nan"), 0
    return arr.mean(), arr.mean() / (arr.std() + 1e-9), len(arr)


def ndcg_at_k(df: pd.DataFrame, k: int = 5) -> float:
    scores = []
    for _, g in df.groupby("ts"):
        if len(g) < k or g["target"].notna().sum() < k:
            continue
        scores.append(ndcg_score(g["target"].values.reshape(1, -1),
                                 g["pred"].values.reshape(1, -1), k=k))
    return float(np.mean(scores)) if scores else float("nan")


# ============================================================
# 8. Orchestration
# ============================================================

def run_pipeline(daily: pd.DataFrame) -> Dict:
    """Главная функция. Принимает daily OHLCV, возвращает все результаты в dict."""
    log.info(f"=== Pipeline start: {len(daily):,} rows, {daily.ticker.nunique()} tickers, "
             f"{daily.ts.nunique()} days ===")

    log.info("[1/7] Features")
    feat = add_features(daily)
    log.info("[2/7] Target")
    feat = add_target(feat)
    feat = feat.dropna(subset=["target", "fwd_ret"]).reset_index(drop=True)
    feat["target"] = feat["target"].astype(int)

    primary_cols = [c for c in feat.columns
                    if c.startswith(("cs_", "mom_", "vol_", "atr_", "rsi_", "dist_", "volume_z", "range_pct"))]
    primary_cols += ["dow"]
    log.info(f"      feature count: {len(primary_cols)}")

    unique_days = pd.DatetimeIndex(sorted(feat["ts"].unique()))
    splits = build_splits(unique_days)

    # ---- Primary walk-forward ----
    log.info(f"[3/7] Primary LambdaRank ({len(splits)} folds)")
    primary_test = []
    for i, sp in enumerate(splits):
        m_tr = (feat.ts >= sp.train_start) & (feat.ts <= sp.train_end)
        m_va = (feat.ts >= sp.val_start) & (feat.ts <= sp.val_end)
        m_te = (feat.ts >= sp.test_start) & (feat.ts <= sp.test_end)
        df_tr = feat.loc[m_tr].dropna(subset=primary_cols).reset_index(drop=True)
        df_va = feat.loc[m_va].dropna(subset=primary_cols).reset_index(drop=True)
        df_te = feat.loc[m_te].dropna(subset=primary_cols).reset_index(drop=True)
        if df_tr.empty or df_va.empty or df_te.empty:
            continue
        booster = train_primary(df_tr, df_va, primary_cols)
        df_te = df_te.copy()
        df_te["pred"] = booster.predict(df_te[primary_cols])
        primary_test.append(df_te)
        if (i + 1) % 5 == 0:
            log.info(f"      fold {i+1}/{len(splits)} done")

    primary_oos = pd.concat(primary_test, ignore_index=True)
    ic_m, ic_ir, n_obs = rank_ic(primary_oos[["ts", "pred", "fwd_ret"]])
    n5 = ndcg_at_k(primary_oos[["ts", "pred", "target"]])
    log.info(f"      Primary: Rank IC {ic_m:+.4f} | IC_IR {ic_ir:+.4f} (n={n_obs}) | NDCG@5 {n5:.4f}")
    log.info(f"      OOS samples: {len(primary_oos):,}")

    # ---- Signals + Trailing Barrier labels ----
    log.info("[4/7] Trailing Barrier labeling")
    def build_signals(df, k_long=TOP_K, k_short=BOTTOM_K):
        out = df.copy()
        out["signal"] = 0
        for _, g in out.groupby("ts"):
            r = g["pred"].rank(ascending=False)
            out.loc[r.nsmallest(k_long).index, "signal"] = 1
            out.loc[r.nlargest(k_short).index, "signal"] = -1
        return out
    primary_oos = build_signals(primary_oos)

    # Pivot для path-based labelling — нам нужны полные паттерны цен,
    # включая дни вне OOS. Используем full daily для closes/highs/lows/vol.
    full = daily.copy().sort_values(["ts", "ticker"]).reset_index(drop=True)
    closes_w = full.pivot(index="ts", columns="ticker", values="close")
    highs_w = full.pivot(index="ts", columns="ticker", values="high")
    lows_w = full.pivot(index="ts", columns="ticker", values="low")
    log_rets = np.log(closes_w / closes_w.shift(1))
    vol_w = log_rets.rolling(VOL_WINDOW).std()

    # Сигналы тоже на полной сетке (OOS период)
    sig_w = (primary_oos.pivot(index="ts", columns="ticker", values="signal")
                        .reindex_like(closes_w).fillna(0))

    log.info(f"      Simulating trailing barriers (PT={PT_MULT}σ, SL={SL_MULT}σ, "
             f"BE={BREAKEVEN_AT}σ, trail={TRAIL_AT}σ/{TRAIL_DIST}σ, max_hold={MAX_HOLDING_DAYS}d)")
    tb_labels, tb_pnls, tb_reasons = trailing_barrier_labels(
        closes_w, highs_w, lows_w, sig_w, vol_w, max_holding=MAX_HOLDING_DAYS,
    )

    n_signals = int((sig_w != 0).sum().sum())
    n_labeled = int(tb_labels.notna().sum().sum())
    n_pos = int((tb_labels == 1).sum().sum())
    reason_counts = pd.Series(tb_reasons.values.ravel()).value_counts()
    log.info(f"      Signals: {n_signals}, labeled: {n_labeled}, hit-rate: {n_pos/n_labeled:.3f}")
    log.info(f"      Exit reasons: {dict(reason_counts)}")

    # Stack labels обратно в long-формат
    tb_long = tb_labels.stack().rename("tb_label").reset_index()
    primary_oos = primary_oos.merge(tb_long, on=["ts", "ticker"], how="left")

    # ---- Meta walk-forward ----
    log.info(f"[5/7] Meta-model walk-forward")
    meta_base = primary_oos[primary_oos["signal"] != 0].dropna(subset=["tb_label"]).copy()
    meta_base["tb_label"] = meta_base["tb_label"].astype(int)
    # Регim features
    cross_disp = primary_oos.groupby("ts")["ret_1d"].std().rename("market_dispersion")
    cross_trend = primary_oos.groupby("ts")["ret_1d"].mean().rename("market_trend")
    meta_base = meta_base.merge(cross_disp, on="ts", how="left")
    meta_base = meta_base.merge(cross_trend, on="ts", how="left")

    meta_cols = primary_cols + ["pred", "signal", "market_dispersion", "market_trend"]
    meta_cols = [c for c in meta_cols if c in meta_base.columns]

    meta_test = []
    for i, sp in enumerate(splits):
        m_tr = (meta_base.ts >= sp.train_start) & (meta_base.ts <= sp.train_end)
        m_va = (meta_base.ts >= sp.val_start) & (meta_base.ts <= sp.val_end)
        m_te = (meta_base.ts >= sp.test_start) & (meta_base.ts <= sp.test_end)
        df_tr = meta_base.loc[m_tr].dropna(subset=meta_cols).reset_index(drop=True)
        df_va = meta_base.loc[m_va].dropna(subset=meta_cols).reset_index(drop=True)
        df_te = meta_base.loc[m_te].dropna(subset=meta_cols).reset_index(drop=True)
        if df_tr.empty or df_va.empty or df_te.empty or df_tr["tb_label"].nunique() < 2:
            continue
        booster = train_meta(df_tr, df_va, meta_cols)
        df_te = df_te.copy()
        df_te["meta_pred"] = booster.predict(df_te[meta_cols])
        meta_test.append(df_te[["ts", "ticker", "meta_pred", "tb_label"]])
        if (i + 1) % 5 == 0:
            log.info(f"      meta fold {i+1}/{len(splits)} done")

    meta_oos = pd.concat(meta_test, ignore_index=True)
    auc = roc_auc_score(meta_oos["tb_label"], meta_oos["meta_pred"])
    log.info(f"      Meta AUC: {auc:.4f} on {len(meta_oos):,} samples")

    # ---- Backtests ----
    log.info("[6/7] Backtests")
    bt_primary = backtest(primary_oos[["ts", "ticker", "pred", "fwd_ret"]],
                          name="primary_only")

    primary_meta = primary_oos.merge(meta_oos[["ts", "ticker", "meta_pred"]],
                                     on=["ts", "ticker"], how="left")
    bt_input = primary_meta[["ts", "ticker", "pred", "fwd_ret", "meta_pred"]]

    log.info("[7/7] Threshold sweep + best")
    sweep = find_best_threshold(bt_input)
    bt_best = backtest(bt_input, name=f"meta@best={sweep['best_threshold']:.2f}",
                       meta_threshold=sweep["best_threshold"])

    return {
        "primary_oos": primary_oos,
        "meta_oos": meta_oos,
        "tb_labels": tb_labels, "tb_pnls": tb_pnls, "tb_reasons": tb_reasons,
        "bt_primary": bt_primary, "bt_best": bt_best,
        "primary_metrics": {"rank_ic": ic_m, "ic_ir": ic_ir, "ndcg5": n5},
        "meta_metrics": {"auc": auc, "n_signals": n_signals, "hit_rate": n_pos / max(n_labeled, 1)},
        "best_threshold": sweep["best_threshold"], "threshold_sweep": sweep["all"],
    }
