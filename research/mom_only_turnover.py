"""Оборот ТОЛЬКО mom (gap сломан в проде) — наберём ли 10M за 14 дней.
Запуск: python mom_only_turnover.py --data-dir market_data_final --since 2025-01-01"""
import argparse, logging, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import _aggregate_to_daily, TICKERS_DEFAULT
from intraday_eval import load_all_minute, CAPITAL
from capital_split_search import daily_mom21_pnl_rebal, scale_pnl
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("mom_to")

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True)
ap.add_argument("--since", default="2025-01-01")
ap.add_argument("--window", type=int, default=14)
ap.add_argument("--mom-share", type=float, default=0.55)  # реальная доля mom в проде
a = ap.parse_args()

log.info("Загрузка минуток...")
mdf = load_all_minute(a.data_dir, tickers=TICKERS_DEFAULT)
tcol = next((c for c in ["ts","begin","datetime","timestamp","time","dt"] if c in mdf.columns), None)
ts = pd.to_datetime(mdf[tcol]); since = pd.to_datetime(a.since)
if getattr(ts.dt,"tz",None) is not None:
    since = since.tz_localize(ts.dt.tz) if since.tzinfo is None else since.tz_convert(ts.dt.tz)
mdf = mdf[ts >= since].copy()
log.info(f"С {a.since}: {len(mdf):,} строк")

daily = _aggregate_to_daily(mdf)
MOM_REF = 600_000
mom = daily_mom21_pnl_rebal(daily, strategy_capital=MOM_REF, rebal_freq=2)
mom = scale_pnl(mom, MOM_REF, CAPITAL * a.mom_share)

dcol = next((c for c in ["trading_date","trade_date","date"] if c in mom.columns), None)
to_col = next((c for c in mom.columns if "turnover" in c), None)
mom = mom.sort_values(dcol).reset_index(drop=True)
to = mom[to_col].values

# скользящие 14-дневные окна — оборот только mom
W = a.window
wins = [to[i:i+W].sum()/1e6 for i in range(len(to)-W+1)]
wins = np.array(wins)
log.info(f"\n=== ОБОРОТ ТОЛЬКО MOM ({a.mom_share:.0%} капитала), окно {W} дней ===")
log.info(f"Окон: {len(wins)}")
log.info(f"Медиана оборота: {np.median(wins):.2f}M")
log.info(f"P10 (10й перцентиль): {np.percentile(wins,10):.2f}M")
log.info(f"P25: {np.percentile(wins,25):.2f}M")
log.info(f"Минимум: {wins.min():.2f}M")
log.info(f"P(>=10M): {(wins>=10).mean()*100:.0f}%")
log.info(f"\nПорог 10M: {'ПРОХОДИМ с запасом' if np.percentile(wins,10)>=10 else 'РИСК штрафа -70' if np.median(wins)<12 else 'на грани'}")
