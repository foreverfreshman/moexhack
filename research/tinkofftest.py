import os
from datetime import datetime, timedelta, timezone
# Обратите внимание: теперь мы импортируем из t_tech
from t_tech.invest import Client, CandleInterval

# Укажите ваш токен здесь или задайте через переменную окружения TINKOFF_TOKEN
TOKEN = os.getenv("TINKOFF_TOKEN", "t.zCgCQwYt29Oep4Uf9y4oAkj4RJ08Z_h_3X_yMAdbnHM0JkwmY1r727Nnf_YrQsQ8bSN1HO5P5qM8e0qzdiMQZg")

# Уникальный идентификатор (FIGI) для обыкновенных акций Сбербанка
SBER_FIGI = "BBG004730N88"

def main():
    # Инициализируем клиента
    with Client(TOKEN) as client:
        # Настраиваем временной интервал (последний 1 час в UTC)
        to_time = datetime.now(timezone.utc)
        from_time = to_time - timedelta(hours=1)

        print(f"Запрос свечей с {from_time.strftime('%H:%M:%S')} по {to_time.strftime('%H:%M:%S')} (UTC)...")

        # Получаем свечи
        response = client.market_data.get_candles(
            figi=SBER_FIGI,
            from_=from_time,
            to=to_time,
            interval=CandleInterval.CANDLE_INTERVAL_1_MIN
        )

        print(f"Успешно! Получено свечей: {len(response.candles)}\n")

        # Выводим первые 5 свечей для проверки структуры данных
        for candle in response.candles[:5]:
            open_p = candle.open.units + candle.open.nano / 1e9
            close_p = candle.close.units + candle.close.nano / 1e9
            
            print(
                f"Время: {candle.time.strftime('%H:%M:%S')} | "
                f"Open: {open_p:<7.2f} | "
                f"Close: {close_p:<7.2f} | "
                f"Vol: {candle.volume}"
            )

if __name__ == "__main__":
    if TOKEN == "ВАШ_ТОКЕН_ЗДЕСЬ":
        print("Ошибка: Замените 'ВАШ_ТОКЕН_ЗДЕСЬ' на реальный токен API.")
    else:
        main()