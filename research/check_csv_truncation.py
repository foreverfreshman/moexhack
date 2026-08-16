"""Проверка: обрезаны ли дни в CSV (докачаны ли до конца сессии).
Если CSV обрывается не на ~18:55/23:50, а раньше — bэктест считал prev_close
на неполных данных, и оценку надо пересматривать.
Запуск из D:\moex:  python check_csv_truncation.py"""
import pandas as pd

df = pd.read_csv('market_data_final/ALRS_3y_1m.csv')
ts = pd.to_datetime(df['timestamp'])
# привести к МСК (данные в UTC)
if ts.dt.tz is not None:
    ts_msk = ts.dt.tz_convert('Europe/Moscow')
else:
    ts_msk = ts.dt.tz_localize('UTC').dt.tz_convert('Europe/Moscow')
df = df.assign(t_msk=ts_msk, d=ts_msk.dt.date)

print("ALRS — последний бар каждого дня (МСК), close:")
for day in ['2026-05-25', '2026-05-26', '2026-05-27', '2026-05-28', '2026-05-29']:
    d = df[df['d'] == pd.Timestamp(day).date()]
    if len(d):
        first = d.iloc[0]
        last = d.iloc[-1]
        print(f"  {day}: баров={len(d):4d} | "
              f"первый={first['t_msk'].strftime('%H:%M')} (close={first['close']}) | "
              f"последний={last['t_msk'].strftime('%H:%M')} (close={last['close']})")
    else:
        print(f"  {day}: нет данных")

print("\nЕсли 'последний' для прошлых дней = ~18:5x или ~23:4x — день полный.")
print("Если обрывается раньше (напр. 14:xx) — CSV неполный, bэктест на битых prev_close.")
