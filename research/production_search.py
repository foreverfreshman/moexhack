"""
Production config search: оптимальный capital split между mom_21 и Gap Fade.

Использует кэшированные PnL daily от mom_21 и Gap Fade, масштабирует их линейно
для разных capital splits и оценивает на rolling 10-bday окнах.

Запуск:
    python production_search.py --data-dir market_data_final --daily-cache daily.parquet

Время выполнения: ~5-10 минут (Gap Fade compute + mom_21).
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_moex_daily, _aggregate_to_daily, TICKERS_DEFAULT
from intraday_eval import (
    load_all_minute, compute_gap_fade_trades, aggregate_orb_daily,
    daily_mom21_pnl, combine_pnl, rolling_14d_eval, print_summary,
    CAPITAL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("prod_search")


def scale_pnl(daily_pnl: pd.DataFrame, ref_capital: float, target_capital: float) -> pd.DataFrame:
    """Масштабирует PnL для другого размера капитала. PnL и turnover линейны."""
    if daily_pnl.empty or target_capital <= 0:
        return pd.DataFrame()
    df = daily_pnl.copy()
    factor = target_capital / ref_capital
    for col in ["gross_pnl_rub", "fee_rub", "net_pnl_rub", "turnover_rub"]:
        if col in df.columns:
            df[col] = df[col] * factor
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--daily-cache", default=None)
    parser.add_argument("--limit-tickers", type=int, default=None)
    args = parser.parse_args()

    tickers = TICKERS_DEFAULT[:args.limit_tickers] if args.limit_tickers else TICKERS_DEFAULT

    # ---- Load data ----
    log.info("=== Loading minute data ===")
    minute_df = load_all_minute(args.data_dir, tickers=tickers)

    log.info("\n=== Computing Gap Fade trades (0.8% threshold) ===")
    gf08_trades = compute_gap_fade_trades(minute_df, gap_threshold=0.008)
    log.info("\n=== Computing Gap Fade trades (1.5% threshold) ===")
    gf15_trades = compute_gap_fade_trades(minute_df, gap_threshold=0.015)
    log.info("\n=== Computing Gap Fade trades (0.5% threshold) — больше сделок ===")
    gf05_trades = compute_gap_fade_trades(minute_df, gap_threshold=0.005)

    # Daily aggregation для каждой версии Gap Fade (reference cap = 400k)
    GF_REF_CAP = 400_000
    gf08_daily = aggregate_orb_daily(gf08_trades, strategy_capital=GF_REF_CAP)
    gf15_daily = aggregate_orb_daily(gf15_trades, strategy_capital=GF_REF_CAP)
    gf05_daily = aggregate_orb_daily(gf05_trades, strategy_capital=GF_REF_CAP)

    # ---- mom_21 ----
    log.info("\n=== mom_21 daily PnL ===")
    if args.daily_cache and Path(args.daily_cache).exists():
        daily_df = pd.read_parquet(args.daily_cache)
    else:
        daily_df = _aggregate_to_daily(minute_df)
    MOM_REF_CAP = 600_000
    mom_daily = daily_mom21_pnl(daily_df, strategy_capital=MOM_REF_CAP)
    log.info(f"  mom_21: {len(mom_daily)} business days")

    # ---- Grid of capital splits ----
    log.info("\n" + "=" * 100)
    log.info("CAPITAL SPLIT SEARCH (mom_21 + GapFade_0.8)")
    log.info("=" * 100)
    splits = [
        (0, 100),
        (10, 90),
        (20, 80),
        (30, 70),
        (40, 60),
        (50, 50),
        (60, 40),
        (70, 30),
        (100, 0),
    ]
    for mom_pct, gf_pct in splits:
        m_cap = CAPITAL * mom_pct / 100
        g_cap = CAPITAL * gf_pct / 100
        m_scaled = scale_pnl(mom_daily, MOM_REF_CAP, m_cap)
        g_scaled = scale_pnl(gf08_daily, GF_REF_CAP, g_cap)
        combined = combine_pnl(m_scaled, g_scaled)
        s = rolling_14d_eval(combined, strategy_capital=CAPITAL,
                             label=f"{mom_pct:3d}% mom + {gf_pct:3d}% GF_0.8")
        print_summary(s)

    log.info("\n" + "=" * 100)
    log.info("CAPITAL SPLIT SEARCH (mom_21 + GapFade_0.5 — низкий threshold, больше сделок)")
    log.info("=" * 100)
    for mom_pct, gf_pct in splits:
        m_cap = CAPITAL * mom_pct / 100
        g_cap = CAPITAL * gf_pct / 100
        m_scaled = scale_pnl(mom_daily, MOM_REF_CAP, m_cap)
        g_scaled = scale_pnl(gf05_daily, GF_REF_CAP, g_cap)
        combined = combine_pnl(m_scaled, g_scaled)
        s = rolling_14d_eval(combined, strategy_capital=CAPITAL,
                             label=f"{mom_pct:3d}% mom + {gf_pct:3d}% GF_0.5")
        print_summary(s)

    log.info("\n" + "=" * 100)
    log.info("BONUS: Multi-threshold Gap Fade (распределяем capital между разными threshold)")
    log.info("=" * 100)
    # 20% mom + 80% split between GF strategies
    for split_name, weights in [
        ("20%m+40%GF08+40%GF15", (200_000, 400_000, 400_000, 0)),
        ("20%m+50%GF08+30%GF15", (200_000, 500_000, 300_000, 0)),
        ("30%m+50%GF08+20%GF15", (300_000, 500_000, 200_000, 0)),
        ("20%m+30%GF05+30%GF08+20%GF15", (200_000, 300_000, 300_000, 200_000)),
    ]:
        m_cap, g08_cap, g15_cap, g05_cap = weights
        m_scaled = scale_pnl(mom_daily, MOM_REF_CAP, m_cap)
        g08_scaled = scale_pnl(gf08_daily, GF_REF_CAP, g08_cap)
        g15_scaled = scale_pnl(gf15_daily, GF_REF_CAP, g15_cap)
        g05_scaled = scale_pnl(gf05_daily, GF_REF_CAP, g05_cap) if g05_cap > 0 else pd.DataFrame()

        # Combine all
        combined = m_scaled
        if not g08_scaled.empty:
            combined = combine_pnl(combined, g08_scaled)
        if not g15_scaled.empty:
            combined = combine_pnl(combined, g15_scaled)
        if not g05_scaled.empty:
            combined = combine_pnl(combined, g05_scaled)

        s = rolling_14d_eval(combined, strategy_capital=CAPITAL, label=split_name)
        print_summary(s)


if __name__ == "__main__":
    main()
