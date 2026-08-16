import os
import csv
import time
import asyncio
from datetime import datetime, timedelta, timezone

# Импортируем AsyncClient вместо обычного Client
from t_tech.invest import AsyncClient, CandleInterval

# ================= НАСТРОЙКИ =================
TOKEN = "t.zCgCQwYt29Oep4Uf9y4oAkj4RJ08Z_h_3X_yMAdbnHM0JkwmY1r727Nnf_YrQsQ8bSN1HO5P5qM8e0qzdiMQZg"

TICKERS = [
    "LKOH", "SBER", "ROSN", "GAZP", "VTBR", 
    "YDEX", "PLZL", "T", "NVTK", "FIVE", "X5", 
    "GMKN", "ALRS", "AFLT", "CHMF", 
    "NLMK", "MOEX", "SNGSP", "MTSS", "PIKK"
]

DATA_DIR = "market_data_raw"
# Семафор: сколько запросов делаем одновременно (5 параллельных потоков)
SEMAPHORE = asyncio.Semaphore(5)
# =============================================

async def download_day(client, ticker, figi, target_date):
    """Асинхронная функция для скачивания одного дня"""
    ticker_dir = os.path.join(DATA_DIR, ticker)
    os.makedirs(ticker_dir, exist_ok=True)
    
    date_str = target_date.strftime("%Y-%m-%d")
    file_path = os.path.join(ticker_dir, f"{date_str}.csv")
    
    # Если файл уже есть - мгновенно пропускаем
    if os.path.exists(file_path):
        return

    from_dt = datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc)
    to_dt = datetime.combine(target_date, datetime.max.time(), tzinfo=timezone.utc)
    
    retries = 0
    while True:
        # Захватываем 1 слот из 5 доступных
        async with SEMAPHORE:
            try:
                # Асинхронный запрос к API
                response = await client.market_data.get_candles(
                    figi=figi,
                    from_=from_dt,
                    to=to_dt,
                    interval=CandleInterval.CANDLE_INTERVAL_1_MIN
                )
                
                # Быстрое синхронное сохранение в файл
                with open(file_path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
                    for candle in response.candles:
                        o = candle.open.units + candle.open.nano / 1e9
                        h = candle.high.units + candle.high.nano / 1e9
                        l = candle.low.units + candle.low.nano / 1e9
                        c = candle.close.units + candle.close.nano / 1e9
                        writer.writerow([candle.time.isoformat(), o, h, l, c, candle.volume])
                
                if len(response.candles) > 0:
                    print(f"[{ticker}] {date_str} -> Скачано: {len(response.candles)}")
                
                # ИСКУССТВЕННАЯ ЗАДЕРЖКА: держим слот 1 секунду. 
                # 5 слотов * 1 сек задержки = ровно 5 запросов в секунду (300 в минуту).
                # Это гарантирует, что мы никогда не словим бан за Rate Limit.
                await asyncio.sleep(1.0)
                break # Выходим из цикла при успехе
                
            except Exception as e:
                err_msg = str(e).lower()
                if "resource_exhausted" in err_msg or "rate limit" in err_msg:
                    wait_time = 15 + retries * 10
                    print(f"  ![Лимит] Пауза {wait_time}с для {ticker} {date_str}")
                    await asyncio.sleep(wait_time)
                    retries += 1
                else:
                    print(f"  ! Ошибка {ticker} {date_str}: {e}")
                    await asyncio.sleep(2)
                    retries += 1
                    if retries > 5:
                        print(f"  !!! Пропуск {ticker} {date_str}")
                        break

async def main_async():
    async with AsyncClient(TOKEN) as client:
        print("Получаем FIGI...")
        shares_resp = await client.instruments.shares()
        all_shares = shares_resp.instruments
        
        ticker_to_figi = {s.ticker: s.figi for s in all_shares if s.ticker in TICKERS}
        
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=3 * 365)
        
        # 1. Собираем все задачи (комбинации тикер + день) в один огромный список
        tasks = []
        for ticker, figi in ticker_to_figi.items():
            current_date = start_date
            while current_date <= end_date:
                # Добавляем задачу в список, но еще не запускаем
                tasks.append(download_day(client, ticker, figi, current_date))
                current_date += timedelta(days=1)
                
        print(f"Сформировано задач на скачивание: {len(tasks)}")
        print("Начинаем ковровое скачивание...\n")
        
        # 2. Запускаем все задачи параллельно (семафор внутри не даст им сломать сервер)
        await asyncio.gather(*tasks)

def merge_data_into_final_csv():
    # Эта функция остается без изменений, она просто склеивает файлы
    print("\nСклейка файлов...")
    final_dir = "market_data_final"
    os.makedirs(final_dir, exist_ok=True)
    
    for ticker in os.listdir(DATA_DIR):
        ticker_path = os.path.join(DATA_DIR, ticker)
        if not os.path.isdir(ticker_path): continue
            
        final_file_path = os.path.join(final_dir, f"{ticker}_3y_1m.csv")
        with open(final_file_path, mode="w", newline="", encoding="utf-8") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            csv_files = sorted([f for f in os.listdir(ticker_path) if f.endswith(".csv")])
            for file in csv_files:
                with open(os.path.join(ticker_path, file), mode="r", encoding="utf-8") as infile:
                    reader = csv.reader(infile)
                    next(reader) 
                    for row in reader:
                        writer.writerow(row)
    print(f"Готово! Данные в {os.path.abspath(final_dir)}")

if __name__ == "__main__":
    start_time = time.time()
    
    # Запускаем асинхронную часть
    asyncio.run(main_async())
    
    # Склеиваем файлы
    merge_data_into_final_csv()
    
    print(f"\nВремя выполнения: {round((time.time() - start_time) / 60, 1)} мин.")