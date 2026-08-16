"""
Final Rebalance & Split Search: бэктест 1d vs 2d ребалансировки
с новыми распределениями капитала (25/75 и 20/80).

Запуск:
    python final_rebal_test.py --data-dir market_data_final --daily-cache daily.parquet
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Импортируем готовые функции из твоих предыдущих скриптов
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import _aggregate_to_daily, TICKERS_DEFAULT
from intraday_eval import (
    load_all_minute, compute_gap_fade_trades, aggregate_orb_daily,
    combine_pnl, rolling_14d_eval, print_summary, CAPITAL
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("final_test")


def scale_pnl(daily_pnl: pd.DataFrame, ref_capital: float, target_capital: float) -> pd.DataFrame:
    """Масштабирует PnL для другого размера капитала."""
    if daily_pnl.empty or target_capital <= 0:
        return pd.DataFrame()
    df = daily_pnl.copy()
    factor = target_capital / ref_capital
    for col in ["gross_pnl_rub", "fee_rub", "net_pnl_rub", "turnover_rub"]:
        if col in df.columns:
            df[col] = df[col] * factor
    return df


def daily_mom21_pnl_rebal(
    daily_df: pd.DataFrame,
    strategy_capital: float = 600_000.0,
    top_k: int = 3,
    fee_bps: float = 5.0,
    rebal_freq: int = 1  # Добавлен параметр частоты ребалансировки
) -> pd.DataFrame:
    """Daily mom_21 backtest с поддержкой ребалансировки раз в N дней."""
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

        # Пересчитываем позиции только если это день ребалансировки
        if i % rebal_freq == 0:
            ranks = snap["cs_mom_21"].rank(ascending=False)
            longs = list(ranks.nsmallest(top_k).index)
            shorts = list(ranks.nlargest(top_k).index)
            target = pd.Series(0.0, index=universe)
            for t in longs: target.loc[t] = 0.5 / top_k
            for t in shorts: target.loc[t] = -0.5 / top_k
        else:
            # Иначе держим вчерашние позиции
            target = prev_w.copy()

        turnover_ratio = (target - prev_w).abs().sum()
        turnover_rub = turnover_ratio * strategy_capital
        fee_rub = turnover_rub * fee
        
        # PnL считается каждый день по текущим позициям
        gross_ret_pct = (target * snap["fwd_1d"].reindex(universe).fillna(0)).sum()
        gross_pnl_rub = gross_ret_pct * strategy_capital
        net_pnl_rub = gross_pnl_rub - fee_rub

        rows.append({
            "trading_date": d.date() if hasattr(d, 'date') else d,
            "gross_pnl_rub": gross_pnl_rub,
            "fee_rub": fee_rub,
            "net_pnl_rub": net_pnl_rub,
            "turnover_rub": turnover_rub,
        })
        prev_w = target

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--daily-cache", default=None)
    parser.add_argument("--limit-tickers", type=int, default=None)
    args = parser.parse_args()

    tickers = TICKERS_DEFAULT[:args.limit_tickers] if args.limit_tickers else TICKERS_DEFAULT

    log.info("=== Loading minute data ===")
    minute_df = load_all_minute(args.data_dir, tickers=tickers)

    log.info("\n=== Computing Gap Fade trades (0.5% threshold) ===")
    gf05_trades = compute_gap_fade_trades(minute_df, gap_threshold=0.005)
    
    GF_REF_CAP = 400_000
    gf05_daily = aggregate_orb_daily(gf05_trades, strategy_capital=GF_REF_CAP)

    log.info("\n=== Loading mom_21 daily data ===")
    if args.daily_cache and Path(args.daily_cache).exists():
        daily_df = pd.read_parquet(args.daily_cache)
    else:
        daily_df = _aggregate_to_daily(minute_df)
    
    MOM_REF_CAP = 600_000
    
    # Считаем базовый mom_21 (ребаланс каждый день)
    log.info("  Calculating mom_21 (1-day rebalance)...")
    mom_daily_1d = daily_mom21_pnl_rebal(daily_df, strategy_capital=MOM_REF_CAP, rebal_freq=1)
    
    # Считаем улучшенный mom_21 (ребаланс раз в 2 дня)
    log.info("  Calculating mom_21 (2-day rebalance)...")
    mom_daily_2d = daily_mom21_pnl_rebal(daily_df, strategy_capital=MOM_REF_CAP, rebal_freq=2)


    # =========================================================
    # БЛОК ТЕСТОВ
    # =========================================================
    configs = [
        # Название, данные моментума, доля мом%, доля гэп%
        ("1d rebalance | 30% mom + 70% GF_0.5 (Текущий прод)", mom_daily_1d, 30, 70),
        ("2d rebalance | 25% mom + 75% GF_0.5", mom_daily_2d, 25, 75),
        ("2d rebalance | 20% mom + 80% GF_0.5", mom_daily_2d, 20, 80),
    ]

    log.info("\n" + "=" * 80)
    log.info("СРАВНЕНИЕ ПРОДАКШН КОНФИГОВ (10-BDAY ROLLING WINDOWS)")
    log.info("=" * 80)

    for label, mom_data, mom_pct, gf_pct in configs:
        m_cap = CAPITAL * mom_pct / 100
        g_cap = CAPITAL * gf_pct / 100
        
        m_scaled = scale_pnl(mom_data, MOM_REF_CAP, m_cap)
        g_scaled = scale_pnl(gf05_daily, GF_REF_CAP, g_cap)
        
        combined = combine_pnl(m_scaled, g_scaled)
        s = rolling_14d_eval(combined, strategy_capital=CAPITAL, label=label)
        
        print_summary(s)
        log.info("-" * 80)


if __name__ == "__main__":
    main()