"""
gap_alpha_by_year.py — деградирует ли альфа Gap Fade со временем?

Прогоняет gap fade отдельно по каждому календарному году и сравнивает метрики,
чтобы увидеть, жив ли edge сейчас или «выеден» рынком. Короткие окна (14 дней)
для этого не годятся — шум доминирует; разбивка по годам даёт честный тренд.

Метрики по годам:
  • число гэпов (не пропадает ли сам сигнал);
  • win rate (доля прибыльных фейдов);
  • средний PnL на сделку в б.п. ДО и ПОСЛЕ комиссии (ключевое: edge на гэп);
  • суммарная доходность и Sharpe (при доле 45% капитала, как в боте);
  • для контекста — то же для mom_21, чтобы видеть, на что опираться, если
    gap-альфа просела.

Запуск:
    python gap_alpha_by_year.py --data-dir market_data_final
    python gap_alpha_by_year.py --data-dir market_data_final --gap-threshold 0.005
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
    CAPITAL
)
from capital_split_search import daily_mom21_pnl_rebal, scale_pnl

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("gap_alpha")

GF_REF_CAP = 400_000
MOM_REF_CAP = 600_000
TRADING_DAYS = 252
FEE_BPS = 5.0   # 0.05% за сторону


def _find(df, *cands):
    for c in cands:
        if c in df.columns:
            return c
    for cc in df.columns:
        for c in cands:
            if c in cc:
                return cc
    return None


def analyze_gap_trades_by_year(gf_trades: pd.DataFrame):
    """Метрики gap-сделок по годам напрямую из сделок (edge на гэп)."""
    if gf_trades.empty:
        log.warning("Нет gap-сделок вообще.")
        return

    df = gf_trades.copy()
    tcol = _find(df, "entry_time", "entry_ts", "ts", "time", "begin", "datetime")
    pcol = _find(df, "pnl_pct", "ret", "return", "pnl_ratio")          # доходность сделки (доля)
    if tcol is None:
        log.error(f"Не нашёл колонку времени в сделках. Колонки: {list(df.columns)}")
        return
    df["year"] = pd.to_datetime(df[tcol]).dt.year

    # если нет готовой колонки доходности — попробуем посчитать из entry/exit цен
    if pcol is None:
        ep = _find(df, "entry_price", "entry", "price_in")
        xp = _find(df, "exit_price", "exit", "price_out")
        sd = _find(df, "side", "direction", "dir")
        if ep and xp:
            side = df[sd] if sd else 1
            # для шорта доходность инвертируется; если side кодируется -1/1 или S/B
            if sd and df[sd].dtype == object:
                sign = df[sd].map(lambda s: -1 if str(s).upper().startswith("S") else 1)
            elif sd:
                sign = np.sign(df[sd]).replace(0, 1)
            else:
                sign = 1
            df["pnl_pct"] = sign * (df[xp] - df[ep]) / df[ep]
            pcol = "pnl_pct"
        else:
            log.warning(f"Не нашёл ни доходность, ни entry/exit цены. Колонки: {list(df.columns)}")
            log.warning("Покажу только число сделок по годам.")

    log.info(f"\n{'='*78}")
    log.info("GAP FADE — edge на сделку по годам (доходность ДО вычета комиссии)")
    log.info(f"{'='*78}")
    log.info(f"{'год':>6} | {'сделок':>7} | {'win rate':>9} | {'ср.PnL бп':>10} | "
             f"{'ср.PnL-комис бп':>15} | {'медиана бп':>11}")
    log.info("-" * 78)

    fee_round_bps = FEE_BPS * 2   # вход+выход
    for year in sorted(df["year"].unique()):
        sub = df[df["year"] == year]
        n = len(sub)
        if pcol:
            rets_bps = sub[pcol].values * 10000.0
            win = (rets_bps > 0).mean() * 100
            mean_bps = rets_bps.mean()
            net_bps = mean_bps - fee_round_bps
            med_bps = np.median(rets_bps)
            log.info(f"{year:>6} | {n:>7} | {win:>8.1f}% | {mean_bps:>10.2f} | "
                     f"{net_bps:>15.2f} | {med_bps:>11.2f}")
        else:
            log.info(f"{year:>6} | {n:>7} | {'—':>9} | {'—':>10} | {'—':>15} | {'—':>11}")

    if pcol:
        log.info("-" * 78)
        log.info("Читать так: 'ср.PnL-комис бп' — средний возврат на гэп ПОСЛЕ комиссии (0.1%=10бп).")
        log.info("Если этот столбец падает к нулю/минусу в свежих годах — альфа деградирует.")
        log.info("Если держится положительным — edge жив.")


def yearly_strategy_metrics(minute_df, gf_trades, gap_pct=0.45):
    """Доходность/Sharpe gap (и mom для контекста) по годам при долях бота."""
    # gap daily PnL
    gf_daily = aggregate_orb_daily(gf_trades, strategy_capital=GF_REF_CAP)
    g_scaled = scale_pnl(gf_daily, GF_REF_CAP, CAPITAL * gap_pct)

    # mom daily PnL (для контекста; доля mom 0.55)
    daily_df = _aggregate_to_daily(minute_df)
    mom_daily = daily_mom21_pnl_rebal(daily_df, strategy_capital=MOM_REF_CAP, rebal_freq=2)
    m_scaled = scale_pnl(mom_daily, MOM_REF_CAP, CAPITAL * 0.55)

    def by_year(daily, label):
        if daily.empty:
            return
        d = daily.copy()
        dcol = _find(d, "trading_date", "trade_date", "date")
        pcol = _find(d, "net_pnl_rub", "net_pnl")
        d["year"] = pd.to_datetime(d[dcol]).dt.year
        log.info(f"\n{label} — по годам (доходность от капитала, Sharpe):")
        log.info(f"{'год':>6} | {'дней':>5} | {'доход %':>8} | {'Sharpe':>7}")
        log.info("-" * 38)
        for year in sorted(d["year"].unique()):
            sub = d[d["year"] == year]
            ret = sub[pcol].sum() / CAPITAL * 100
            daily_ret = sub[pcol].values / CAPITAL
            sd = daily_ret.std(ddof=1)
            sharpe = (daily_ret.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 1e-12 else 0.0
            log.info(f"{year:>6} | {len(sub):>5} | {ret:>+8.2f} | {sharpe:>7.2f}")

    log.info(f"\n{'='*78}")
    log.info("ДОХОДНОСТЬ ПО ГОДАМ при долях бота (gap=45%, mom=55%)")
    log.info(f"{'='*78}")
    by_year(g_scaled, "GAP FADE (45% капитала)")
    by_year(m_scaled, "MOM_21 (55% капитала, для контекста)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--gap-threshold", type=float, default=0.005)
    args = ap.parse_args()

    log.info("=== Загрузка минутных данных ===")
    minute_df = load_all_minute(args.data_dir, tickers=TICKERS_DEFAULT)

    log.info(f"=== Расчёт gap-сделок (порог {args.gap_threshold*100:.1f}%) ===")
    gf_trades = compute_gap_fade_trades(minute_df, gap_threshold=args.gap_threshold)

    # 1) edge на сделку по годам (главный ответ про деградацию)
    analyze_gap_trades_by_year(gf_trades)

    # 2) доходность/Sharpe стратегии по годам
    yearly_strategy_metrics(minute_df, gf_trades, gap_pct=0.45)

    log.info(f"\n{'='*78}")
    log.info("ВЫВОД: смотри тренд 'ср.PnL-комис бп' и Sharpe gap по годам.")
    log.info("Падение к нулю в 2025-2026 = деградация альфы. Стабильность = edge жив.")
    log.info("mom_21 показан для контекста: на него можно опереться, если gap просел.")


if __name__ == "__main__":
    main()
