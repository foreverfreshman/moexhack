"""
Вывод каждого 10-дневного окна для финального конфига:
25% mom_21 (2d rebal) + 75% Gap Fade 0.5
"""

import argparse
import pandas as pd
from pathlib import Path

# Импортируем готовые функции из твоих предыдущих скриптов
from data_loader import _aggregate_to_daily, TICKERS_DEFAULT
from intraday_eval import (
    load_all_minute, compute_gap_fade_trades, aggregate_orb_daily,
    combine_pnl, CAPITAL
)
from final_rebal_test import daily_mom21_pnl_rebal, scale_pnl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--daily-cache", default=None)
    args = parser.parse_args()

    print("Загрузка данных и расчет стратегий (займет пару минут)...")
    
    # 1. Загрузка минуток и расчет Gap Fade
    minute_df = load_all_minute(args.data_dir, tickers=TICKERS_DEFAULT)
    gf05_trades = compute_gap_fade_trades(minute_df, gap_threshold=0.005)
    GF_REF_CAP = 400_000
    gf05_daily = aggregate_orb_daily(gf05_trades, strategy_capital=GF_REF_CAP)

    # 2. Загрузка дневок и расчет mom21 (rebal 2d)
    if args.daily_cache and Path(args.daily_cache).exists():
        daily_df = pd.read_parquet(args.daily_cache)
    else:
        daily_df = _aggregate_to_daily(minute_df)
        
    MOM_REF_CAP = 600_000
    mom_daily_2d = daily_mom21_pnl_rebal(daily_df, strategy_capital=MOM_REF_CAP, rebal_freq=2)

    # 3. Масштабируем под финальный капитал (25% mom / 75% gap)
    m_cap = CAPITAL * 0.25
    g_cap = CAPITAL * 0.75
    
    m_scaled = scale_pnl(mom_daily_2d, MOM_REF_CAP, m_cap)
    g_scaled = scale_pnl(gf05_daily, GF_REF_CAP, g_cap)
    combined = combine_pnl(m_scaled, g_scaled)
    
    # Сортируем по дате
    combined = combined.sort_values("trading_date").reset_index(drop=True)

    print("\n" + "="*70)
    print(" ДЕТАЛЬНЫЙ ОТЧЕТ ПО 10-ДНЕВНЫМ ТОРГОВЫМ ОКНАМ (Шаг сдвига = 5 дней)")
    print("="*70)
    print(f"{'Дата Старта':<12} | {'Дата Конца':<12} | {'PnL (%)':<10} | {'Оборот (Млн ₽)':<15} | {'Статус Оборота'}")
    print("-" * 70)

    window_bdays = 10
    step_bdays = 5
    n = len(combined)
    
    total_windows = 0
    passed_windows = 0
    
    # Проходим скользящим окном
    for start_idx in range(0, n - window_bdays + 1, step_bdays):
        w = combined.iloc[start_idx:start_idx + window_bdays]
        if len(w) < window_bdays:
            continue
            
        start_date = w["trading_date"].iloc[0]
        end_date = w["trading_date"].iloc[-1]
        
        total_pnl = w["net_pnl_rub"].sum()
        ret_pct = (total_pnl / CAPITAL) * 100
        
        total_turnover_M = w["turnover_rub"].sum() / 1_000_000
        
        turnover_status = "✅ PASS" if total_turnover_M >= 10.0 else "❌ ШТРАФ"
        
        if total_turnover_M >= 10.0:
            passed_windows += 1
        total_windows += 1
            
        print(f"{str(start_date):<12} | {str(end_date):<12} | {ret_pct:>+8.2f}% | {total_turnover_M:>10.2f} M | {turnover_status}")

    print("-" * 70)
    print(f"Всего окон проанализировано: {total_windows}")
    print(f"Окон, прошедших порог 10 млн оборта: {passed_windows} ({passed_windows/total_windows*100:.1f}%)")
    print("="*70)

if __name__ == "__main__":
    main()