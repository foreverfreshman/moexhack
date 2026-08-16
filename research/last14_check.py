"""
last14_check.py — диагностика последних N торговых дней (по умолчанию 14).

Отвечает на конкретный вопрос: «нормально ли, что гэпов нет пару дней подряд,
и наберётся ли оборот?» Считает по твоим минутным данным:
  • частоту гэпов: в скольких днях из N был хотя бы один гэп >порога, сколько
    гэпов в среднем за день, серии пустых дней подряд;
  • оборот: суммарный turnover стратегии (mom 2d + gap) за окно при доле 55/45;
  • PnL: суммарная доходность за окно.

ВАЖНО: это ОДНО конкретное окно (последние N дней), а не прогноз на этап 2.
Основная оценка — на 3 годах (rolling-окна). Этот скрипт — диагностика «что было
недавно», чтобы понять, аномалия ли текущая тишина по гэпам.

Запуск:
    python last14_check.py --data-dir market_data_final
    python last14_check.py --data-dir market_data_final --days 14 --gap-threshold 0.005
    python last14_check.py --data-dir market_data_final --as-of 2026-05-27
        --as-of: считать «последними» дни до этой даты включительно (если данные
                 заканчиваются раньше — просто берёт последние доступные).
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
# переиспользуем mom-функцию из соседнего скрипта поиска долей
from capital_split_search import daily_mom21_pnl_rebal, scale_pnl

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("last14")

MOM_PCT, GAP_PCT = 0.55, 0.45      # текущая боевая доля
MOM_REF_CAP, GF_REF_CAP = 600_000, 400_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--gap-threshold", type=float, default=0.005)
    ap.add_argument("--as-of", default=None, help="YYYY-MM-DD; дни до этой даты включительно")
    args = ap.parse_args()

    log.info("=== Загрузка минутных данных ===")
    minute_df = load_all_minute(args.data_dir, tickers=TICKERS_DEFAULT)

    # --- определить окно последних N торговых дней ---
    minute_df = minute_df.copy()
    # имя колонки времени в минутных данных может отличаться (ts/begin/datetime/...)
    TCOL = None
    for c in ["ts", "begin", "datetime", "timestamp", "time", "dt", "date", "tradedate"]:
        if c in minute_df.columns:
            TCOL = c
            break
    if TCOL is None:
        log.error(f"Не нашёл колонку времени. Доступные колонки: {list(minute_df.columns)}")
        log.error("Укажи нужную колонку вручную в скрипте (переменная TCOL).")
        return
    log.info(f"Колонка времени в минутках: '{TCOL}'")
    minute_df["date"] = pd.to_datetime(minute_df[TCOL]).dt.date
    all_dates = sorted(minute_df["date"].unique())
    if args.as_of:
        cutoff = pd.to_datetime(args.as_of).date()
        all_dates = [d for d in all_dates if d <= cutoff]
    window_dates = all_dates[-args.days:]
    if len(window_dates) < args.days:
        log.warning(f"В данных только {len(window_dates)} дней (запрошено {args.days})")
    start, end = window_dates[0], window_dates[-1]
    log.info(f"Окно: {start} … {end} ({len(window_dates)} торговых дней)")

    win = minute_df[minute_df["date"].isin(window_dates)].copy()

    # ===== 1. ЧАСТОТА ГЭПОВ =====
    gf_trades = compute_gap_fade_trades(win, gap_threshold=args.gap_threshold)
    log.info(f"\n{'='*70}\nЧАСТОТА ГЭПОВ (порог {args.gap_threshold*100:.1f}%)\n{'='*70}")

    if gf_trades.empty:
        log.warning("За окно НЕ найдено ни одного гэпа выше порога!")
        days_with_gaps, gap_counts = set(), {}
    else:
        gf_trades = gf_trades.copy()
        # дата сделки — из времени входа
        tcol = "entry_time" if "entry_time" in gf_trades.columns else gf_trades.columns[
            [("time" in c or "ts" in c) for c in gf_trades.columns].index(True)]
        gf_trades["date"] = pd.to_datetime(gf_trades[tcol]).dt.date
        gap_counts = gf_trades.groupby("date").size().to_dict()
        days_with_gaps = set(gap_counts.keys())

    # пройдём по всем дням окна, отметим есть/нет гэп, посчитаем серии пустых
    empty_streak, max_empty_streak = 0, 0
    per_day = []
    for d in window_dates:
        n = gap_counts.get(d, 0)
        per_day.append((d, n))
        if n == 0:
            empty_streak += 1
            max_empty_streak = max(max_empty_streak, empty_streak)
        else:
            empty_streak = 0

    n_with = len(days_with_gaps)
    n_total = len(window_dates)
    total_gaps = sum(gap_counts.values())
    log.info(f"Дней с гэпами: {n_with} из {n_total} ({100*n_with/n_total:.0f}%)")
    log.info(f"Всего гэпов за окно: {total_gaps} (в среднем {total_gaps/n_total:.1f}/день)")
    log.info(f"Макс. серия дней БЕЗ гэпов подряд: {max_empty_streak}")
    log.info("Гэпы по дням:")
    for d, n in per_day:
        bar = "█" * n if n else "·  (нет гэпов)"
        log.info(f"  {d} ({d.strftime('%a')}): {bar}")

    # ===== 2. ОБОРОТ И PnL =====
    log.info(f"\n{'='*70}\nОБОРОТ И PnL (доля 55/45, mom 2d-rebal)\n{'='*70}")
    gf_daily = aggregate_orb_daily(gf_trades, strategy_capital=GF_REF_CAP)
    daily_df = _aggregate_to_daily(win)
    mom_daily = daily_mom21_pnl_rebal(daily_df, strategy_capital=MOM_REF_CAP, rebal_freq=2)

    m_scaled = scale_pnl(mom_daily, MOM_REF_CAP, CAPITAL * MOM_PCT)
    g_scaled = scale_pnl(gf_daily, GF_REF_CAP, CAPITAL * GAP_PCT)
    combined = combine_pnl(m_scaled, g_scaled)

    if combined.empty:
        log.warning("Нет данных для оборота/PnL за окно.")
        return

    # гибкий поиск колонок
    def col(df, *cands):
        for c in cands:
            if c in df.columns: return c
        for cc in df.columns:
            for c in cands:
                if c in cc: return cc
        return None
    to_col = col(combined, "turnover_rub", "turnover")
    pnl_col = col(combined, "net_pnl_rub", "net_pnl")

    total_turnover = combined[to_col].sum()
    total_pnl = combined[pnl_col].sum()
    log.info(f"Суммарный оборот за {n_total} дней: {total_turnover/1e6:.2f}M ₽")
    log.info(f"Порог штрафа: 10.00M ₽ → {'ПРОЙДЕН ✓' if total_turnover>=10e6 else 'НЕ ПРОЙДЕН ✗'}")
    log.info(f"Суммарный PnL за окно: {total_pnl:,.0f} ₽ ({100*total_pnl/CAPITAL:+.2f}%)")
    log.info(f"Средний оборот в день: {total_turnover/1e6/n_total:.2f}M ₽")

    # вклад mom vs gap в оборот
    mom_to = m_scaled[col(m_scaled,'turnover_rub','turnover')].sum() if not m_scaled.empty else 0
    gap_to = g_scaled[col(g_scaled,'turnover_rub','turnover')].sum() if not g_scaled.empty else 0
    log.info(f"  из них mom-ребаланс: {mom_to/1e6:.2f}M, gap-фейды: {gap_to/1e6:.2f}M")

    log.info(f"\n{'='*70}")
    log.info("ВЫВОД: это одно конкретное окно последних дней, не прогноз этапа 2.")
    log.info("Для прогноза смотри rolling-оценку на 3 годах (capital_split_search.py).")


if __name__ == "__main__":
    main()