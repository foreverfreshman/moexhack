#!/usr/bin/env python3
"""
Тест возможности открытия шорта через ArenaGo API.

Использование:
    $env:SANDBOX_API_KEY="c497d13a30f1df8e18eef3b5b4d59becb5bdc39314195a1fa4a66202111173aa"
    python test_short.py

Опционально, если у вас несколько портфелей или имя портфеля не совпадает с именем бота:
    export ARENAGO_PORTFOLIO="MyPortfolioName"
    export ARENAGO_BOT="MyBotName"

Что делает:
1. Получает список ботов и текущие позиции
2. Выбирает тикер, которого нет в позициях
3. Пытается продать 1 акцию этого тикера (это либо шорт, либо ошибка)
4. Проверяет, появилась ли отрицательная позиция (= шорт открылся)
5. Если шорт открылся, сразу его закрывает (покупает 1 акцию обратно)
6. Логирует все ответы API

Минимально-инвазивно: торгуется 1 лот, который сразу же закрывается.
"""

import os
import sys
import json
import time
import requests

BASE_URL = "https://arenago.ru/api"
TOKEN = os.environ.get("SANDBOX_API_KEY")
PORTFOLIO_OVERRIDE = os.environ.get("ARENAGO_PORTFOLIO")
BOT_OVERRIDE = os.environ.get("ARENAGO_BOT")

if not TOKEN:
    sys.exit("ERROR: переменная окружения SANDBOX_API_KEY не установлена")

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": TOKEN,
}

ALL_TICKERS = [
    "LKOH", "SBER", "ROSN", "GAZP", "VTBR", "YDEX", "PLZL", "T", "NVTK", "X5",
    "GMKN", "MGNT", "ALRS", "AFLT", "CHMF", "NLMK", "MOEX", "SNGSP", "MTSS", "PIKK",
]


def pretty(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def http(method: str, path: str, **kwargs):
    """HTTP-запрос с логированием URL и статуса."""
    url = f"{BASE_URL}{path}"
    print(f"  -> {method} {url}")
    r = requests.request(method, url, headers=HEADERS, timeout=10, **kwargs)
    print(f"  <- HTTP {r.status_code}")
    try:
        body = r.json()
    except ValueError:
        body = {"raw_text": r.text}
    return r.status_code, body


def get_bots():
    status, body = http("GET", "/bots")
    return status, body


def get_positions(portfolio: str):
    status, body = http("GET", f"/positions/{portfolio}")
    return status, body


def submit_order(direction: str, secid: str, quantity: int, bot: str):
    payload = {
        "direction": direction,
        "secid": secid,
        "quantity": quantity,
        "bot": bot,
    }
    print(f"  payload: {payload}")
    status, body = http("POST", "/submit_order", json=payload)
    return status, body


def find_target_ticker(positions_list) -> str:
    """Выбрать тикер которого нет в позициях."""
    held = set()
    if isinstance(positions_list, list):
        for p in positions_list:
            secid = p.get("secid") if isinstance(p, dict) else None
            if secid:
                held.add(secid)
    available = [t for t in ALL_TICKERS if t not in held]
    if not available:
        sys.exit("ERROR: все 20 тикеров уже в позициях")
    return available[0]


def find_position(positions_list, secid: str):
    """Найти позицию по тикеру."""
    if not isinstance(positions_list, list):
        return None
    for p in positions_list:
        if isinstance(p, dict) and p.get("secid") == secid:
            return p
    return None


def main():
    print("=" * 70)
    print("ТЕСТ ШОРТА через ArenaGo API")
    print("=" * 70)

    # ---- Шаг 1: список ботов ----
    print("\n[1] GET /api/bots")
    status, bots = get_bots()
    print(pretty(bots))

    if status != 200 or not bots:
        sys.exit("ERROR: не удалось получить ботов или список пустой. "
                 "Создайте бота в ArenaGo UI или проверьте токен.")

    # Имя бота и портфеля
    if BOT_OVERRIDE:
        bot_name = BOT_OVERRIDE
    else:
        bot_name = bots[0].get("name") if isinstance(bots, list) else None
    if not bot_name:
        sys.exit("ERROR: не нашли поле 'name' в ответе /api/bots. "
                 "Задайте ARENAGO_BOT вручную.")

    portfolio = PORTFOLIO_OVERRIDE or bot_name
    print(f"\nИспользуем bot='{bot_name}', portfolio='{portfolio}'")
    print(f"(если портфель называется иначе, задайте ARENAGO_PORTFOLIO)")

    # ---- Шаг 2: текущие позиции ----
    print(f"\n[2] GET /api/positions/{portfolio}")
    status, positions_before = get_positions(portfolio)
    print(pretty(positions_before))
    if status != 200:
        sys.exit("ERROR: не удалось получить позиции. "
                 "Проверьте ARENAGO_PORTFOLIO.")

    # ---- Шаг 3: выбор тикера для теста ----
    target = find_target_ticker(positions_before)
    print(f"\nДля теста выбран тикер: {target} (нет в позициях)")

    # ---- Шаг 4: попытка открыть шорт ----
    print(f"\n[3] POST /api/submit_order  (продаём 1 акцию {target})")
    status, sell_response = submit_order("S", target, 1, bot_name)
    print(pretty(sell_response))

    success = isinstance(sell_response, dict) and sell_response.get("success")
    if not success:
        err = sell_response.get("error") if isinstance(sell_response, dict) else sell_response
        print("\n" + "=" * 70)
        print("РЕЗУЛЬТАТ: ШОРТ НЕДОСТУПЕН ИЛИ ОШИБКА")
        print(f"Ответ API: {err}")
        print("=" * 70)
        print("\nВозможные причины:")
        print("  - market closed (время вне торгов)")
        print("  - шорт реально недоступен через API")
        print("  - какая-то другая ошибка - см. сообщение")
        return

    # ---- Шаг 5: проверим позицию ----
    print(f"\n[4] Ждём 1 сек, затем проверяем позицию по {target}...")
    time.sleep(1)
    status, positions_after_sell = get_positions(portfolio)
    print(pretty(positions_after_sell))

    target_pos = find_position(positions_after_sell, target)

    print("\n" + "=" * 70)
    if target_pos is None:
        print("РЕЗУЛЬТАТ: позиция не создана (странно — сделка прошла как success)")
        print(f"Возможно тикер {target} в API отображается иначе. Проверьте /api/trades")
        print("=" * 70)
        return

    pos_qty = target_pos.get("position")
    avg_price = target_pos.get("average_price")

    if pos_qty is not None and pos_qty < 0:
        print(f"✓ ШОРТ ПОДТВЕРЖДЁН")
        print(f"  Тикер: {target}")
        print(f"  Позиция: {pos_qty} (отрицательная)")
        print(f"  Средняя цена: {avg_price}")
        print(f"  remaining_cash после открытия: {sell_response.get('remaining_cash')}")
    else:
        print(f"⚠ Странно: позиция = {pos_qty} (не отрицательная)")
        print(f"  Может быть в системе уже была позиция, которую не показывал /api/positions")
    print("=" * 70)

    # ---- Шаг 6: закрываем шорт ----
    if pos_qty is not None and pos_qty < 0:
        print(f"\n[5] Закрываем шорт: покупаем |{pos_qty}| = {abs(pos_qty)} акций {target}")
        status, buy_response = submit_order("B", target, abs(pos_qty), bot_name)
        print(pretty(buy_response))

        time.sleep(1)
        print(f"\n[6] Финальная проверка позиций:")
        status, positions_final = get_positions(portfolio)
        print(pretty(positions_final))

        final_pos = find_position(positions_final, target)
        final_qty = final_pos.get("position") if final_pos else 0
        print(f"\nИтоговая позиция по {target}: {final_qty} (ожидаем 0)")

    print("\n" + "=" * 70)
    print("ТЕСТ ЗАВЕРШЁН")
    print("=" * 70)
    print("\nЧто можно сделать дальше:")
    print("  - Если шорт открылся: проверить размер плеча, попробовать большой шорт")
    print("  - Если шорт не открылся: уточнить в чате хакатона условия по шортам")


if __name__ == "__main__":
    main()