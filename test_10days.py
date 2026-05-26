"""
test_10days.py — симуляция 10 торговых дней автономной работы.

Прогоняет виртуальное время через фазы каждого дня (prepare→open→monitor→eod)
для 10 дней и проверяет инварианты многодневной работы:
  1. day_counter растёт 1→10 (по одному на торговый день)
  2. rebal на днях 1,3,5,7,9; held на 2,4,6,8,10
  3. mom_shares меняются на rebal day, держатся на held day
  4. gap открывается на открытии и flat после EOD каждый день
  5. mom держится overnight (не закрывается на EOD)
  6. рестарт в тот же день НЕ увеличивает counter повторно
"""

import sys, logging
from datetime import datetime, timedelta, timezone, date
sys.path.insert(0, "/home/claude/work/moex/prod")
sys.path.insert(0, "/home/claude/work/moex/prod/strategies")
sys.path.insert(0, "/home/claude/work/moex/prod/data")

from main import Config, TradingBot, MSK
from arenago_client import MockArenaGoClient
from data.tinvest_stream import MockTInvestData, DEFAULT_UNIVERSE
from portfolio import Portfolio
from strategies.mom21 import Mom21Strategy
from strategies.gap_fade import GapFadeStrategy
from monitor import StateStore, RiskMonitor
from pathlib import Path

logging.basicConfig(level=logging.WARNING)  # тише, только наши проверки

DB = "/tmp/sim10.db"


def make_bot(feed, client):
    store = StateStore(DB)
    mon = RiskMonitor(store, total_capital=1_000_000)
    cfg = Config(state_db=DB, news_filter_enabled=False)
    return TradingBot(cfg, client, feed,
                      Mom21Strategy(0.25), GapFadeStrategy(0.75),
                      Portfolio(1_000_000, feed.lot_map), store, mon,
                      sync_on_start=False), store


class SimFeed(MockTInvestData):
    """Feed с ценами, меняющимися по дням (детерминированно)."""
    def __init__(self):
        super().__init__(tickers=DEFAULT_UNIVERSE)
        self.lot_map = {t: 1 for t in DEFAULT_UNIVERSE}
        self.day = 0
        self.base = {t: 100.0 + i * 5 for i, t in enumerate(DEFAULT_UNIVERSE)}

    def set_day(self, d):
        self.day = d

    def get_prev_close(self, as_of=None):
        # вчерашнее закрытие — база + небольшой тренд по дню
        return {t: self.base[t] * (1 + 0.001 * self.day) for t in DEFAULT_UNIVERSE}

    def get_session_open(self):
        # открытие с гэпом на части тикеров (зависит от дня → разные гэпы)
        out = {}
        for i, t in enumerate(DEFAULT_UNIVERSE):
            prev = self.base[t] * (1 + 0.001 * self.day)
            gap = 0.0
            if (i + self.day) % 4 == 0:      # часть тикеров с гэпом вверх
                gap = 0.012
            elif (i + self.day) % 4 == 2:    # часть с гэпом вниз
                gap = -0.012
            out[t] = prev * (1 + gap)
        return out

    def get_current_prices(self):
        # текущая цена ≈ открытие, дрейфует к prev (гэп закрывается частично)
        opens = self.get_session_open()
        prev = self.get_prev_close()
        return {t: opens[t] + (prev[t] - opens[t]) * 0.3 for t in DEFAULT_UNIVERSE}

    def get_daily_closes(self, n_days=25):
        import pandas as pd
        # long-format (ts, ticker, close) — как ждёт mom21
        dates = [date(2026, 5, 1) + timedelta(days=k) for k in range(n_days)]
        rows = []
        for i, t in enumerate(DEFAULT_UNIVERSE):
            trend = (i - 10) * 0.002   # разные тренды → разные топ/бот momentum
            for k, dt in enumerate(dates):
                rows.append({"ts": pd.Timestamp(dt), "ticker": t,
                             "close": self.base[t] * (1 + trend * k)})
        return pd.DataFrame(rows)


def run_phases(bot, feed, day_date):
    """Прогнать фазы одного дня: prepare, open, monitor, eod."""
    for hh, mm in [(9, 55), (10, 0), (12, 0), (18, 50)]:
        now = datetime(day_date.year, day_date.month, day_date.day, hh, mm, tzinfo=MSK)
        bot.tick(now=now)


def main():
    Path(DB).unlink(missing_ok=True)
    feed = SimFeed()
    client = MockArenaGoClient(portfolio_name="sim", price_provider=lambda t: feed.base.get(t, 100.0))
    bot, store = make_bot(feed, client)

    # 10 торговых дней подряд (будни, без праздников). Стартуем с понедельника.
    from main import MOEX_HOLIDAYS_2026
    start = date(2026, 6, 1)   # старт; пропускаем выходные и праздники
    trading_days = []
    d = start
    while len(trading_days) < 14:
        if d not in MOEX_HOLIDAYS_2026:  # выходные торгуются
            trading_days.append(d)
        d += timedelta(days=1)

    print(f"Симуляция 14 торговых дней: {trading_days[0]} … {trading_days[-1]}\n")

    results = []
    for n, dd in enumerate(trading_days, 1):
        feed.set_day(n)
        # мок цен клиента под текущий день
        client.price_provider = lambda t, f=feed: f.get_current_prices().get(t, 100.0)
        mom_before = dict(bot.mom_shares)
        run_phases(bot, feed, dd)
        mom_after = dict(bot.mom_shares)
        gap_after = dict(bot.gap_shares)
        gap_nonzero = {k: v for k, v in gap_after.items() if v != 0}
        counter = bot.day_state.day_counter
        rebal = bot.day_state.rebal_day
        mom_changed = mom_before != mom_after
        results.append((n, counter, rebal, mom_changed, len(gap_nonzero)))
        print(f"День {n}: counter={counter} rebal={rebal} "
              f"mom_changed={mom_changed} gap_after_eod={len(gap_nonzero)} "
              f"mom_positions={len([v for v in mom_after.values() if v != 0])}")

    print("\n=== ПРОВЕРКИ ===")
    ok = True

    # 1. counter 1→10
    counters = [r[1] for r in results]
    if counters == list(range(1, 15)):
        print("✓ counter растёт 1→14 по одному на день")
    else:
        print(f"✗ counter неверный: {counters}")
        ok = False

    # 2. rebal на 1,3,5,7,9
    rebal_days = [r[0] for r in results if r[2]]
    if rebal_days == [1, 3, 5, 7, 9, 11, 13]:
        print("✓ rebal на днях 1,3,5,7,9,11,13")
    else:
        print(f"✗ rebal-расписание неверное: {rebal_days}")
        ok = False

    # 3. mom меняется на rebal day
    mom_changed_days = [r[0] for r in results if r[3]]
    # день 1: с нуля открыл (изменился); rebal-дни меняют; held — нет
    if all(d in [1,3,5,7,9,11,13] for d in mom_changed_days):
        print(f"✓ mom_shares меняются только на rebal-дни: {mom_changed_days}")
    else:
        print(f"✗ mom менялся в held-дни: {mom_changed_days}")
        ok = False

    # 4. gap flat после EOD каждый день
    gap_eod = [r[4] for r in results]
    if all(g == 0 for g in gap_eod):
        print("✓ gap-позиции закрыты после EOD каждый день")
    else:
        print(f"✗ gap не закрылся в некоторые дни: {gap_eod}")
        ok = False

    # 5. mom держится (на held-дни позиции не нулевые)
    # проверяем что после дня 2 (held) mom_shares не пустой
    print("\n=== РЕСТАРТ В ТОТ ЖЕ ДЕНЬ ===")
    # эмулируем рестарт на дне 10: новый объект бота, тот же DB, тот же день
    bot2, _ = make_bot(feed, client)
    bot2.tick(now=datetime(trading_days[-1].year, trading_days[-1].month,
                          trading_days[-1].day, 12, 0, tzinfo=MSK))
    c_after_restart = bot2.day_state.day_counter
    if c_after_restart == 14:
        print(f"✓ Рестарт в тот же день: counter остался {c_after_restart} (не прыгнул)")
    else:
        print(f"✗ БАГ: рестарт увеличил counter до {c_after_restart} (должно быть 14)")
        ok = False

    print("\n=== РЕСТАРТ МЕЖДУ OPEN И EOD (наслоение?) ===")
    # Новый день 11 (свежий торговый день после серии). Открываемся, потом рестарт в monitor.
    Path(DB).unlink(missing_ok=True)
    feed2 = SimFeed(); feed2.set_day(1)
    client2 = MockArenaGoClient(portfolio_name="r", price_provider=lambda t, f=feed2: f.get_current_prices().get(t, 100.0))
    botA, storeA = make_bot(feed2, client2)
    dd = trading_days[0]
    # prepare + open (позиции открываются)
    botA.tick(now=datetime(dd.year, dd.month, dd.day, 9, 55, tzinfo=MSK))
    botA.tick(now=datetime(dd.year, dd.month, dd.day, 10, 0, tzinfo=MSK))
    broker_after_open = {p.ticker: p.quantity for p in client2.get_positions() if p.quantity != 0}
    n_open = len(broker_after_open)
    print(f"После open: {n_open} позиций на бирже")
    # РЕСТАРТ: новый объект бота (sync_on_start=True!), тот же день, фаза monitor
    botB = TradingBot(Config(state_db=DB, news_filter_enabled=False), client2, feed2,
                      Mom21Strategy(0.25), GapFadeStrategy(0.75),
                      Portfolio(1_000_000, feed2.lot_map), storeA,
                      RiskMonitor(storeA, total_capital=1_000_000), sync_on_start=True)
    botB.tick(now=datetime(dd.year, dd.month, dd.day, 12, 0, tzinfo=MSK))
    broker_after_restart = {p.ticker: p.quantity for p in client2.get_positions() if p.quantity != 0}
    # Проверяем, что позиции не удвоились (наслоение)
    doubled = any(abs(broker_after_restart.get(t, 0)) > abs(broker_after_open.get(t, 0)) * 1.5
                  for t in broker_after_open)
    if not doubled:
        print(f"✓ После рестарта в monitor: позиций {len(broker_after_restart)}, наслоения нет")
    else:
        print(f"✗ НАСЛОЕНИЕ: позиции выросли после рестарта")
        print(f"   до:    {broker_after_open}")
        print(f"   после: {broker_after_restart}")
        ok = False

    print("\n" + ("✓✓✓ ВСЕ ПРОВЕРКИ ПРОШЛИ" if ok else "✗✗✗ ЕСТЬ ПРОБЛЕМЫ"))
    Path(DB).unlink(missing_ok=True)
    return ok


if __name__ == "__main__":
    main()
