"""
main.py — Event loop торгового бота MOEX ArenaGo.

Архитектура:
    Раздельные реестры позиций mom_shares и gap_shares (разный lifecycle):
        - mom_21: ребаланс раз в 2 торговых дня, держится overnight
        - gap_fade: вход на открытии, выход intraday по TP/SL, закрытие на EOD

    Два пути исполнения:
        ПУТЬ 1 (rebalance, на открытии): mom (если rebal day) + gap entries
        ПУТЬ 2 (gap exits, каждые 30 сек): немедленное закрытие по TP/SL

Расписание (MSK):
    09:55  подготовка: prev_close + daily_closes
    10:00  открытие: gap entries + mom rebalance (на rebal day)
    10:00-18:45  мониторинг каждые 30 сек: gap TP/SL + kill-switch
    18:45  EOD: закрытие всех gap позиций (mom держится)
    18:50  конец сессии

Запуск:
    # Mock (тест без реального API):
    python main.py --mock

    # Реальный (нужны env vars):
    export SANDBOX_API_KEY=...
    export TINKOFF_TOKEN=...
    export ARENAGO_BASE_URL=...
    python main.py --portfolio my_bot
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, date, time as dtime
from pathlib import Path
from typing import Dict, List, Optional

# Локальные модули
import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "strategies"))
sys.path.insert(0, str(Path(__file__).parent / "data"))

from arenago_client import (
    ArenaGoClient, MockArenaGoClient, ArenaGoError, NetworkError,
)
from portfolio import Portfolio, Order
from strategies.mom21 import Mom21Strategy
from strategies.gap_fade import GapFadeStrategy
from data.tinvest_stream import TInvestData, MockTInvestData, DEFAULT_UNIVERSE
from monitor import StateStore, RiskMonitor, recover_bot_state
from news_filter import NewsFilter

log = logging.getLogger("main")

MSK = timezone(timedelta(hours=3))

# T-Invest токен по умолчанию (личный, read-only данные). Можно переопределить
# через env TINKOFF_TOKEN. ВНИМАНИЕ: при коммите в репозиторий токен станет
# виден всем с доступом к репо — при необходимости вынеси в env и здесь оставь "".
DEFAULT_TINKOFF_TOKEN = "t.zCgCQwYt29Oep4Uf9y4oAkj4RJ08Z_h_3X_yMAdbnHM0JkwmY1r727Nnf_YrQsQ8bSN1HO5P5qM8e0qzdiMQZg"


def _parse_available_cash(error_msg: str):
    """Извлечь «Доступно: X» из ошибки ArenaGo о недостатке средств.
    Пример: 'Недостаточно средств. Требуется: 149606.00, Доступно: 46838.36' → 46838.36
    """
    import re
    m = re.search(r"[Дд]оступно:?\s*([\d]+(?:\.[\d]+)?)", error_msg)
    return float(m.group(1)) if m else None

# Праздники MOEX 2026 (нерабочие дни биржи). TODO: уточнить полный список.
MOEX_HOLIDAYS_2026 = {
    date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 7),
    date(2026, 2, 23), date(2026, 3, 9), date(2026, 5, 1),
    date(2026, 5, 11), date(2026, 6, 12), date(2026, 11, 4),
}


# ============================================================
# Config
# ============================================================

def _resolve_data_dir() -> str:
    """Определить директорию для persistent-состояния.

    Приоритет:
      1. env BOT_DATA_DIR (если задан)
      2. /data — постоянный диск на серверах организаторов (ТЗ: данные туда)
      3. текущая директория (локальная разработка / Windows)
    """
    env_dir = os.environ.get("BOT_DATA_DIR")
    if env_dir:
        Path(env_dir).mkdir(parents=True, exist_ok=True)
        return env_dir
    data_dir = Path("/data")
    if data_dir.exists() and os.access("/data", os.W_OK):
        return "/data"
    return "."


DATA_DIR = _resolve_data_dir()


def setup_logging(data_dir: str = None) -> str:
    """Настроить логирование: stdout (для Dashboard) + файл на persistent-диске.

    Файл logs/bot_YYYY-MM-DD.log с ротацией в полночь, хранится 30 дней.
    Возвращает путь к лог-файлу.
    """
    from logging.handlers import TimedRotatingFileHandler

    data_dir = data_dir or DATA_DIR
    log_dir = Path(data_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "bot.log"

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Убираем старые handler'ы (если basicConfig уже вызывался)
    for h in list(root.handlers):
        root.removeHandler(h)

    # stdout — для docker logs / Dashboard организаторов
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # Файл — с ротацией в полночь, суффикс с датой, хранить 30 дней
    fh = TimedRotatingFileHandler(
        log_path, when="midnight", backupCount=30, encoding="utf-8"
    )
    fh.suffix = "%Y-%m-%d"
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Заглушаем шумные библиотеки: T-Invest логирует каждый GetCandles (десятки
    # тысяч строк/день), urllib3 — каждый retry. Оставляем только WARNING+.
    for noisy in ("t_tech", "t_tech.invest.logging", "urllib3", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return str(log_path)


@dataclass
class Config:
    total_capital: float = 1_000_000.0
    mom_capital_share: float = 0.55
    gap_capital_share: float = 0.45
    mom_rebal_freq: int = 2          # ребаланс mom каждые N торговых дней
    mom_top_k: int = 3
    gap_threshold: float = 0.005
    gap_max_concurrent: int = 5
    tick_interval_sec: int = 30
    kill_switch_dd: float = -0.05    # -5% дневная просадка → flatten all (backstop против бага/data error/fat-tail, не против нормальной воли ~0.29%/день)
    portfolio_name: str = "my_bot"
    state_file: str = field(default_factory=lambda: str(Path(DATA_DIR) / "bot_state.json"))
    state_db: str = field(default_factory=lambda: str(Path(DATA_DIR) / "bot_state.db"))
    stage_trading_days: int = 14   # 28 мая–10 июня 2026, выходные торгуются, без праздников
    turnover_target: float = 10_000_000.0
    # LLM news filter
    news_filter_enabled: bool = True
    news_min_turnover_pace: float = 1_000_000.0   # < 1M/день к началу дня → фильтр off (оборот важнее)
    news_model: str = "qwen/qwen-2.5-72b-instruct"   # открытая модель (Apache 2.0)

    # Расписание MSK
    prepare_time: dtime = dtime(9, 55)
    session_open: dtime = dtime(10, 0)
    gap_window_end: dtime = dtime(10, 15)
    eod_flatten: dtime = dtime(18, 48)    # закрываем gap ДО вечернего аукциона 18:55
    session_close: dtime = dtime(18, 55)  # 18:55–19:00 аукцион (сделок нет); вечёрку не торгуем


@dataclass
class DayState:
    trade_date: date
    day_counter: int
    rebal_day: bool
    prepared: bool = False
    opened: bool = False
    eod_done: bool = False
    kill_switched: bool = False
    day_realized_pnl: float = 0.0


# ============================================================
# Trading Bot
# ============================================================

class TradingBot:
    def __init__(self, config: Config, client, feed, mom, gap, portfolio,
                 store: StateStore, monitor: RiskMonitor, sync_on_start: bool = True):
        self.config = config
        self.client = client
        self.feed = feed
        self.mom = mom
        self.gap = gap
        self.portfolio = portfolio
        self.store = store
        self.monitor = monitor

        # Раздельные реестры позиций — восстанавливаем из persistence при старте
        recovered = recover_bot_state(store)
        self.mom_shares: Dict[str, int] = recovered["mom_shares"]
        self.gap_shares: Dict[str, int] = recovered["gap_shares"]
        self.entry_prices: Dict[str, float] = recovered["entry_prices"]
        if self.mom_shares or self.gap_shares:
            log.info(f"Restored {len(self.mom_shares)} mom + {len(self.gap_shares)} "
                     f"gap positions from persistence")

        self.day_state: Optional[DayState] = None
        self.prev_closes: Dict[str, float] = {}
        self.daily_closes = None
        self.news_filter = None   # устанавливается в build_bot, если включён

        # Сверка с фактическими позициями биржи (предотвращает наслоение при рестарте)
        if sync_on_start:
            self._sync_with_broker()

    def _sync_with_broker(self) -> None:
        """Сверка реестров с фактическими позициями ArenaGo при старте.

        Биржа — источник истины. Если БД совпадает с биржей — доверяем
        разделению mom/gap из БД. Если расходится (БД стёрта/устарела) —
        приводим к факту биржи, расхождение относим на mom (overnight-стратегия;
        gap внутридневной и при старте вне сессии его быть не должно).
        Это предотвращает наслоение позиций после перезапуска.
        """
        try:
            positions = self.client.get_positions()
        except Exception as e:
            log.warning(f"Старт: не удалось свериться с биржей ({e}). Доверяю БД.")
            return

        broker = {p.ticker: p.quantity * self._lot(p.ticker) for p in positions}  # лоты→штуки
        broker_avg = {p.ticker: p.avg_price for p in positions}

        all_t = set(broker) | set(self.mom_shares) | set(self.gap_shares)
        corrections = 0
        for t in all_t:
            b = broker.get(t, 0)
            m = self.mom_shares.get(t, 0)
            g = self.gap_shares.get(t, 0)
            if b == (m + g):
                continue
            corrections += 1
            if b == 0:
                # биржа пуста — обнуляем оба реестра
                log.warning(f"Sync {t}: биржа 0, обнуляю реестр (было mom={m} gap={g})")
                self.mom_shares.pop(t, None)
                self.gap_shares.pop(t, None)
            else:
                # доверяем gap из БД, разницу относим на mom
                new_m = b - g
                log.warning(f"Sync {t}: биржа {b}шт ≠ БД {m+g} (mom={m} gap={g}) → mom={new_m}")
                if new_m == 0:
                    self.mom_shares.pop(t, None)
                else:
                    self.mom_shares[t] = new_m
                # entry неизвестен для новых позиций — берём avg_price биржи
                if t not in self.entry_prices and broker_avg.get(t):
                    self.entry_prices[t] = broker_avg[t]

        if corrections:
            log.warning(f"Старт-sync с биржей: {corrections} коррекций "
                        f"(биржа — источник истины, наслоение предотвращено)")
            self._persist()
        else:
            log.info(f"Старт-sync: расхождений с биржей нет "
                     f"({len(broker)} позиций на бирже)")

    def _persist(self) -> None:
        """Сохранить текущие реестры позиций в SQLite."""
        try:
            self.store.save_positions(self.mom_shares, self.gap_shares, self.entry_prices)
        except Exception as e:
            log.warning(f"Persist positions failed: {e}")

    # --------------------------------------------------------
    # Persistence через StateStore (SQLite, см. monitor.py)
    # --------------------------------------------------------

    def _load_counter(self) -> int:
        return self.store.load_counter()

    def _save_counter(self, counter: int) -> None:
        self.store.save_counter(counter)
        self.store.save_last_trade_date(datetime.now(MSK).date())

    # --------------------------------------------------------
    # Day / phase helpers
    # --------------------------------------------------------

    def _is_trading_day(self, d: date) -> bool:
        # MOEX торгует и в выходные (сб/вс) — исключаем только праздники.
        return d not in MOEX_HOLIDAYS_2026

    def _phase(self, now: datetime) -> str:
        t = now.timetz().replace(tzinfo=None)
        c = self.config
        if t < c.prepare_time:
            return "pre"
        if c.prepare_time <= t < c.session_open:
            return "prepare"
        if c.session_open <= t < c.gap_window_end:
            return "open"
        if c.gap_window_end <= t < c.eod_flatten:
            return "monitor"
        if c.eod_flatten <= t < c.session_close:
            return "eod"
        return "closed"

    def _init_new_day(self, now: datetime) -> None:
        today = now.date()
        last = self.store.load_last_trade_date()
        if last == today:
            # Рестарт в ТОТ ЖЕ торговый день — не инкрементируем counter повторно,
            # восстанавливаем уже присвоенный номер дня.
            counter = self._load_counter()
            log.info(f"Рестарт в тот же день {today} (#{counter}) — counter сохранён")
        else:
            counter = self._load_counter() + 1
            self._save_counter(counter)
            self.store.save_last_trade_date(today)
            log.info(f"=== New trading day {today} (#{counter}, "
                     f"mom_rebal={((counter - 1) % self.config.mom_rebal_freq == 0)}) ===")
        rebal_day = ((counter - 1) % self.config.mom_rebal_freq == 0)
        self.day_state = DayState(
            trade_date=today,
            day_counter=counter,
            rebal_day=rebal_day,
        )
        self.gap.reset_day()
        # Reconcile наши реестры с реальными позициями брокера
        self._reconcile_positions()

    def _reconcile_positions(self) -> None:
        """Сверить mom_shares + gap_shares с реальными позициями брокера."""
        try:
            positions = self.client.get_positions()
        except ArenaGoError as e:
            log.warning(f"Reconcile failed (get_positions): {e}")
            return
        # ArenaGo возвращает позиции в ЛОТАХ → переводим в штуки для сравнения
        broker = {}
        for p in positions:
            lot = self._lot(p.ticker)
            broker[p.ticker] = p.quantity * lot
        ours = {}
        for t, q in self.mom_shares.items():
            ours[t] = ours.get(t, 0) + q
        for t, q in self.gap_shares.items():
            ours[t] = ours.get(t, 0) + q
        all_t = set(broker) | set(ours)
        for t in all_t:
            b, o = broker.get(t, 0), ours.get(t, 0)
            if b != o:
                log.warning(f"Position mismatch {t}: broker={b}, tracked={o}")

    # --------------------------------------------------------
    # Order execution
    # --------------------------------------------------------

    def _lot(self, ticker: str) -> int:
        """Размер лота тикера (из T-Invest API). По умолчанию 1."""
        lot = getattr(self.feed, "lot_map", {}).get(ticker, 1)
        return lot if lot and lot > 0 else 1

    def _submit(self, ticker: str, side: str, shares: int, price: float = None):
        """Отправить ордер. Внутри бот считает в ШТУКАХ, ArenaGo принимает ЛОТЫ —
        конвертируем здесь. При ответе «недостаточно средств» урезаем ордер
        до максимума, который влезает в доступный кэш (вместо отклонения).
        """
        lot = self._lot(ticker)
        lots = abs(int(shares)) // lot
        if lots == 0:
            log.warning(f"{ticker}: {shares} шт < 1 лота ({lot}) — пропуск")
            return None

        resp = self.client.submit_order(ticker, side, lots)

        # Урезание при нехватке средств
        if (not resp.success and resp.error
                and "недостаточно" in resp.error.lower()):
            avail = _parse_available_cash(resp.error)
            px = price or 0
            if avail and avail > 0 and px > 0:
                # сколько целых лотов влезает в доступный кэш (с запасом 0.5% на комиссию/движение)
                max_lots = int((avail * 0.995) / (px * lot))
                if 0 < max_lots < lots:
                    log.warning(f"{ticker}: недостаточно средств, урезаю "
                                f"{lots}→{max_lots} лот (доступно {avail:.0f}, цена {px})")
                    resp = self.client.submit_order(ticker, side, max_lots)
                elif max_lots == 0:
                    log.warning(f"{ticker}: на доступные {avail:.0f} не влезает даже 1 лот — пропуск")
        return resp

    def _reconcile_ticker(self, ticker: str, tag: str, prices: Dict[str, float]) -> None:
        """После таймаута submit_order состояние ордера неизвестно.
        Сверяемся с фактической позицией ArenaGo (источник истины) и
        подстраиваем реестр стратегии tag под факт.

        Логика: брокерская нетто-позиция = mom + gap по тикеру. Вклад «другой»
        стратегии этим ордером не менялся → его считаем верным, а всю разницу
        относим на стратегию tag, которая делала ордер.
        """
        positions = self.client.get_positions()   # в ЛОТАХ
        broker_lots = next((p.quantity for p in positions if p.ticker == ticker), 0)
        broker_shares = broker_lots * self._lot(ticker)   # → штуки

        other = self.gap_shares if tag == "mom" else self.mom_shares
        reg = self.mom_shares if tag == "mom" else self.gap_shares
        other_shares = other.get(ticker, 0)
        new_tag = broker_shares - other_shares
        old_tag = reg.get(ticker, 0)

        if new_tag == old_tag:
            log.info(f"Reconcile {ticker} [{tag}]: ордер НЕ исполнился "
                     f"(позиция без изменений: {broker_shares}шт)")
            return

        log.warning(f"Reconcile {ticker} [{tag}]: реестр {old_tag} → {new_tag}шт "
                    f"(broker net {broker_shares}, other-strat {other_shares})")
        if new_tag == 0:
            reg.pop(ticker, None)
        else:
            reg[ticker] = new_tag
        # Если позиция появилась/изменилась — приблизим entry текущей ценой
        # (точную fill-цену таймаут не вернул). Для mom не критично (нет per-pos exit),
        # для gap — приближение, помечаем в логе.
        cur = prices.get(ticker)
        if new_tag != 0 and cur:
            self.entry_prices[ticker] = cur
            log.info(f"  entry_price[{ticker}] ≈ {cur} (приближение после таймаута)")
        self._persist()

    def _execute(self, orders: List[Order], prices: Dict[str, float], tag: str) -> None:
        for o in orders:
            try:
                resp = self._submit(o.ticker, o.side, o.quantity, price=prices.get(o.ticker))
                if resp is None:
                    continue
                if resp.success:
                    fill_px = resp.filled_price or prices.get(o.ticker, 0)
                    self._update_entry_price(o.ticker, o.side, o.quantity, fill_px)
                    # log_trade в ШТУКАХ — turnover = штуки × цена_за_штуку (реальные ₽)
                    self.store.log_trade(o.ticker, o.side, o.quantity, fill_px,
                                         strategy=tag, reason="rebalance" if tag == "mom" else "gap_entry")
                    log.info(f"[{tag}] {o.side} {o.ticker} x{o.quantity}шт "
                             f"({o.quantity // self._lot(o.ticker)} лот) @ {fill_px}")
                else:
                    log.error(f"[{tag}] REJECTED {o.side} {o.ticker} x{o.quantity}: {resp.error}")
            except NetworkError as e:
                log.error(f"[{tag}] TIMEOUT {o.ticker} — состояние неизвестно, сверяюсь с биржей")
                try:
                    self._reconcile_ticker(o.ticker, tag, prices)
                except Exception as re:
                    log.error(f"Reconcile after timeout failed {o.ticker}: {re}")
            except ArenaGoError as e:
                log.error(f"[{tag}] FAILED {o.ticker}: {e}")

    def _update_entry_price(self, ticker, side, qty, price) -> None:
        """Обновить средневзвешенную entry price для mark-to-market."""
        if price <= 0:
            return
        signed = qty if side == "B" else -qty
        cur_qty = self._total_qty(ticker)
        cur_entry = self.entry_prices.get(ticker, price)
        new_qty = cur_qty + signed
        if cur_qty == 0 or (cur_qty > 0) != (new_qty > 0):
            # открытие новой / разворот
            self.entry_prices[ticker] = price
        elif abs(new_qty) > abs(cur_qty):
            # увеличение позиции — взвешенное среднее
            self.entry_prices[ticker] = (
                (cur_entry * abs(cur_qty) + price * abs(signed)) / abs(new_qty)
            )
        # частичное закрытие — entry не меняется

    def _total_qty(self, ticker: str) -> int:
        return self.mom_shares.get(ticker, 0) + self.gap_shares.get(ticker, 0)

    # --------------------------------------------------------
    # Phase actions
    # --------------------------------------------------------

    def _prepare_day(self) -> None:
        log.info("Prepare: загружаем prev_close" +
                 (" + daily_closes (rebal day)" if self.day_state.rebal_day else ""))
        try:
            self.prev_closes = self.feed.get_prev_close()
            if self.day_state.rebal_day:
                self.daily_closes = self.feed.get_daily_closes(n_days=25)
            self.day_state.prepared = True
        except Exception as e:
            log.error(f"Prepare failed: {e}")

    def _do_open(self) -> None:
        ds = self.day_state
        log.info("=== SESSION OPEN ===")
        prices = self.feed.get_current_prices()
        opens = self.feed.get_session_open()

        # --- mom rebalance (только rebal day) ---
        if ds.rebal_day and self.daily_closes is not None:
            mom_w = self.mom.compute_signals(self.daily_closes)
            mom_target = self.portfolio.weights_to_shares(mom_w.weights, prices)
            mom_orders = self.portfolio.compute_orders(mom_target, self.mom_shares, prices)
            log.info(f"mom rebalance: {len(mom_orders)} orders")
            self._execute(mom_orders, prices, "mom")
            self.mom_shares = mom_target
        else:
            log.info("mom: held day, не трогаем")

        # --- gap entries ---
        # LLM-вето включаем только если: фильтр есть, оборот НЕ отстаёт от темпа,
        # и это НЕ последний торговый день (тогда оборот критичнее новостной фильтрации).
        active_filter = None
        if self.news_filter is not None:
            day_n = self.day_state.day_counter
            is_last_day = day_n >= self.config.stage_trading_days
            pace = self.monitor.turnover_pace(day_n) if day_n > 0 else {"cumulative_M": 0}
            # «Отстаём» считаем по ЗАВЕРШЁННЫМ дням (day_n - 1): в первый день оборот
            # ещё 0, и фильтр не должен из-за этого выключаться.
            expected = self.config.news_min_turnover_pace * max(day_n - 1, 0)
            behind = (pace["cumulative_M"] * 1e6) < expected
            if is_last_day:
                log.info("News filter: ОТКЛЮЧЁН (последний день — оборот важнее)")
            elif behind:
                log.info(f"News filter: ОТКЛЮЧЁН (оборот отстаёт от темпа "
                         f"{self.config.news_min_turnover_pace/1e6:.1f}M/день)")
            else:
                active_filter = self.news_filter
                log.info("News filter: ВКЛЮЧЁН")

        entries = self.gap.on_session_open(self.prev_closes, opens,
                                           current_prices=prices, news_filter=active_filter)
        if entries:
            gap_w = self.gap.get_target_weights()
            gap_target = self.portfolio.weights_to_shares(gap_w.weights, prices)
            gap_orders = self.portfolio.compute_orders(gap_target, self.gap_shares, prices)
            log.info(f"gap entries: {len(gap_orders)} orders")
            self._execute(gap_orders, prices, "gap")
            self.gap_shares = {**self.gap_shares, **gap_target}
        ds.opened = True
        self._persist()

    def _monitor_tick(self) -> None:
        prices = self.feed.get_current_prices()
        if not prices:
            return

        # Kill-switch
        day_pnl = self._compute_day_pnl(prices)
        dd = day_pnl / self.config.total_capital
        if dd < self.config.kill_switch_dd:
            log.critical(f"!!! KILL SWITCH: day PnL {day_pnl:,.0f} ₽ ({dd:.2%}) "
                         f"< threshold {self.config.kill_switch_dd:.0%}")
            self._trigger_kill_switch()
            return

        # Gap exits (TP/SL)
        exits = self.gap.on_tick(prices)
        for sig in exits:
            qty = abs(self.gap_shares.get(sig.ticker, 0))
            if qty <= 0:
                continue
            try:
                resp = self._submit(sig.ticker, sig.side, qty, price=prices.get(sig.ticker))
                if resp is None:
                    self.gap_shares[sig.ticker] = 0
                    continue
                if resp.success:
                    fill = resp.filled_price or prices.get(sig.ticker, 0)
                    self._record_realized(sig.ticker, qty, fill)
                    self.store.log_trade(sig.ticker, sig.side, qty, fill,
                                         strategy="gap", reason=sig.reason)
                    log.info(f"[gap-exit:{sig.reason}] {sig.side} {sig.ticker} x{qty}шт @ {fill}")
                    self.gap_shares[sig.ticker] = 0
            except ArenaGoError as e:
                log.error(f"Gap exit failed {sig.ticker}: {e}")
        if exits:
            self._persist()

    def _do_eod(self) -> None:
        log.info("=== EOD: закрываем gap позиции (mom держим overnight) ===")
        prices = self.feed.get_current_prices()
        exits = self.gap.on_session_close(prices)
        for sig in exits:
            qty = abs(self.gap_shares.get(sig.ticker, 0))
            if qty <= 0:
                continue
            try:
                resp = self._submit(sig.ticker, sig.side, qty, price=prices.get(sig.ticker))
                if resp is None:
                    self.gap_shares[sig.ticker] = 0
                    continue
                if resp.success:
                    fill = resp.filled_price or prices.get(sig.ticker, 0)
                    self._record_realized(sig.ticker, qty, fill)
                    self.store.log_trade(sig.ticker, sig.side, qty, fill,
                                         strategy="gap", reason="eod")
                    log.info(f"[gap-eod] {sig.side} {sig.ticker} x{qty}шт @ {fill}")
                    self.gap_shares[sig.ticker] = 0
            except ArenaGoError as e:
                log.error(f"EOD close failed {sig.ticker}: {e}")
        self.gap_shares = {t: q for t, q in self.gap_shares.items() if q != 0}
        self.day_state.eod_done = True
        self._persist()

        # Запись дневного PnL + проверка темпа оборота
        day_pnl = self._compute_day_pnl(prices)
        unrealized = day_pnl - self.day_state.day_realized_pnl
        equity = self.config.total_capital + day_pnl
        try:
            self.store.record_daily_pnl(
                self.day_state.trade_date,
                realized=self.day_state.day_realized_pnl,
                unrealized=unrealized,
                equity=equity,
                turnover=self.monitor.cumulative_turnover(),
            )
        except Exception as e:
            log.warning(f"record_daily_pnl failed: {e}")

        alert = self.monitor.turnover_alert(self.day_state.day_counter)
        if alert:
            log.warning(alert)
        log.info(self.monitor.summary(self.day_state.day_counter))
        log.info(f"EOD done. Realized PnL today: {self.day_state.day_realized_pnl:,.0f} ₽")

    # --------------------------------------------------------
    # PnL / kill-switch
    # --------------------------------------------------------

    def _compute_day_pnl(self, prices: Dict[str, float]) -> float:
        """Дневной PnL = realized + unrealized (mark-to-market)."""
        unrealized = 0.0
        for ticker in set(self.mom_shares) | set(self.gap_shares):
            qty = self._total_qty(ticker)
            if qty == 0:
                continue
            entry = self.entry_prices.get(ticker)
            cur = prices.get(ticker)
            if entry and cur:
                unrealized += qty * (cur - entry)
        return self.day_state.day_realized_pnl + unrealized

    def _record_realized(self, ticker: str, qty: int, exit_price: float) -> None:
        """Зафиксировать realized PnL при закрытии gap позиции."""
        entry = self.entry_prices.get(ticker)
        if entry is None or exit_price <= 0:
            return
        # gap_shares[ticker] до закрытия — подписанное
        signed_qty = self.gap_shares.get(ticker, 0)
        pnl = signed_qty * (exit_price - entry)   # signed_qty>0 long, <0 short
        self.day_state.day_realized_pnl += pnl

    def _trigger_kill_switch(self) -> None:
        log.critical("FLATTENING ALL POSITIONS")
        try:
            results = self.client.flatten_all_positions()
            log.critical(f"Flattened {len(results)} positions")
        except ArenaGoError as e:
            log.critical(f"FLATTEN FAILED: {e} — MANUAL INTERVENTION NEEDED")
        self.mom_shares = {}
        self.gap_shares = {}
        self.day_state.kill_switched = True
        self._persist()

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    def tick(self, now: Optional[datetime] = None) -> None:
        """Один проход event loop. now можно инжектить для тестов."""
        now = now or datetime.now(MSK)

        if not self._is_trading_day(now.date()):
            return

        if self.day_state is None or self.day_state.trade_date != now.date():
            self._init_new_day(now)

        ds = self.day_state
        phase = self._phase(now)

        try:
            if phase == "prepare" and not ds.prepared:
                self._prepare_day()
            elif phase == "open" and not ds.opened:
                if not ds.prepared:
                    self._prepare_day()
                self._do_open()
            elif phase == "monitor":
                if not ds.opened:
                    # бот стартовал посреди сессии — наверстать
                    if not ds.prepared:
                        self._prepare_day()
                    self._do_open()
                if not ds.kill_switched:
                    self._monitor_tick()
            elif phase == "eod" and not ds.eod_done:
                if not ds.kill_switched:
                    self._do_eod()
        except Exception as e:
            log.exception(f"Tick error in phase {phase}: {e}")

    def run(self) -> None:
        """Бесконечный цикл (для production)."""
        # Стартовый баннер — диагностика окружения (видно в логах контейнера)
        log.info("=" * 60)
        log.info("MOEX ArenaGo bot — STARTING")
        log.info(f"  portfolio='{self.config.portfolio_name}', "
                 f"bot='{getattr(self.client, 'bot_name', '?')}'")
        log.info(f"  capital={self.config.total_capital:,.0f}, "
                 f"mom={self.config.mom_capital_share:.0%}/gap={self.config.gap_capital_share:.0%}, "
                 f"rebal every {self.config.mom_rebal_freq}d")
        log.info(f"  gap_threshold={self.config.gap_threshold:.1%}, "
                 f"max_concurrent={self.config.gap_max_concurrent}, "
                 f"kill_switch={self.config.kill_switch_dd:.0%}")
        log.info(f"  state_db={self.config.state_db}")
        log.info(f"  universe={len(self.feed.figi_map)} tickers, "
                 f"lot_sizes loaded={bool(getattr(self.feed, 'lot_map', None))}")
        log.info(f"  schedule MSK: prepare={self.config.prepare_time}, "
                 f"open={self.config.session_open}, eod={self.config.eod_flatten}")
        log.info(f"  restored: {len(self.mom_shares)} mom + {len(self.gap_shares)} gap positions")
        log.info(f"  current time MSK: {datetime.now(MSK).strftime('%Y-%m-%d %H:%M:%S')}")
        log.info("=" * 60)

        last_heartbeat = 0.0
        heartbeat_interval = 600   # лог "жив" раз в 10 минут в простое
        while True:
            try:
                self.tick()
                # Heartbeat — чтобы в логах было видно, что бот жив вне торгов
                now_ts = time.time()
                if now_ts - last_heartbeat >= heartbeat_interval:
                    now_msk = datetime.now(MSK)
                    phase = self._phase(now_msk) if self._is_trading_day(now_msk.date()) else "closed"
                    log.info(f"[heartbeat] {now_msk.strftime('%H:%M')} MSK | phase={phase} | "
                             f"mom={len(self.mom_shares)} gap={len(self.gap_shares)} | "
                             f"turnover={self.monitor.cumulative_turnover()/1e6:.1f}M")
                    last_heartbeat = now_ts
            except KeyboardInterrupt:
                log.info("Shutdown requested. mom/gap позиции оставлены как есть.")
                break
            except Exception as e:
                log.exception(f"Unhandled error in run loop: {e}")
            time.sleep(self.config.tick_interval_sec)


# ============================================================
# Factory
# ============================================================

def build_bot(config: Config, use_mock: bool) -> TradingBot:
    if use_mock:
        feed = MockTInvestData(tickers=DEFAULT_UNIVERSE)
        client = MockArenaGoClient(portfolio_name=config.portfolio_name,
                                   price_provider=lambda t: 100.0)
        lot_sizes = feed.get_lot_sizes()
    else:
        # T-Invest токен: хардкод-дефолт, env TINKOFF_TOKEN переопределяет при наличии.
        # (ArenaGo SANDBOX_API_KEY остаётся строго в env — его подменяют на этапе 2.)
        token = os.environ.get("TINKOFF_TOKEN", DEFAULT_TINKOFF_TOKEN)
        feed = TInvestData(token=token, tickers=DEFAULT_UNIVERSE)
        feed.resolve_figis()
        lot_sizes = feed.get_lot_sizes()   # реальные лоты из API!

        # Определяем имя бота/портфеля: env override → иначе из /bots (с retry).
        # Сетевой сбой на старте НЕ должен ронять процесс (иначе restart-петля).
        bot_name = os.environ.get("ARENAGO_BOT")
        portfolio = os.environ.get("ARENAGO_PORTFOLIO")
        if not bot_name or not portfolio:
            resolved = None
            probe = ArenaGoClient.from_env(portfolio_name=portfolio or "_probe")
            for attempt in range(5):
                try:
                    bots = probe.get_bots()
                    if bots:
                        resolved = bots[0].get("name")
                        log.info(f"Resolved from /bots: '{resolved}'")
                        break
                    log.warning(f"/bots пустой (попытка {attempt+1}/5)")
                except Exception as e:
                    wait = 2 ** attempt
                    log.warning(f"/bots недоступен (попытка {attempt+1}/5): {e}. "
                                f"Retry через {wait}с")
                    time.sleep(wait)
            if resolved:
                bot_name = bot_name or resolved
                portfolio = portfolio or resolved
            else:
                # Fallback: env или config default. Бот стартует, не падает.
                bot_name = bot_name or config.portfolio_name
                portfolio = portfolio or config.portfolio_name
                log.warning(f"Не удалось определить имя бота из /bots. "
                            f"Использую bot='{bot_name}', portfolio='{portfolio}'. "
                            f"Если неверно — задайте ARENAGO_BOT и ARENAGO_PORTFOLIO.")
        config.portfolio_name = portfolio
        client = ArenaGoClient.from_env(portfolio_name=portfolio, bot_name=bot_name)

    portfolio = Portfolio(total_capital=config.total_capital, lot_sizes=lot_sizes)
    mom = Mom21Strategy(capital_share=config.mom_capital_share, top_k=config.mom_top_k)
    gap = GapFadeStrategy(capital_share=config.gap_capital_share,
                          gap_threshold=config.gap_threshold,
                          max_concurrent=config.gap_max_concurrent)
    store = StateStore(config.state_db)
    monitor = RiskMonitor(store, total_capital=config.total_capital,
                          stage_trading_days=config.stage_trading_days,
                          turnover_target=config.turnover_target,
                          kill_switch_dd=abs(config.kill_switch_dd))
    bot = TradingBot(config, client, feed, mom, gap, portfolio, store, monitor)

    # LLM news filter (если включён). Ключ: env POLZA_API_KEY (от модератора) →
    # иначе хардкод-дефолт. Баланс пополняется автоматически, ключ не меняется.
    if config.news_filter_enabled:
        from news_filter import DEFAULT_POLZA_KEY
        polza_key = os.environ.get("POLZA_API_KEY", DEFAULT_POLZA_KEY)
        if polza_key:
            bot.news_filter = NewsFilter(
                api_key=polza_key,
                model=os.environ.get("POLZA_MODEL", config.news_model),
            )
            src = "env" if os.environ.get("POLZA_API_KEY") else "хардкод"
            log.info(f"News filter активирован (модель {config.news_model}, ключ из {src})")
        else:
            log.warning("News filter включён, но ключ polza.ai пуст — фильтр выключен")
    return bot


# ============================================================
# Smoke test: simulated trading day
# ============================================================

def _smoke_test():
    """Прогон полного торгового дня через инжектированное время."""
    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log.info("=== SMOKE TEST: simulated trading day ===")

    import tempfile
    tmp = tempfile.gettempdir()
    config = Config(
        state_file=str(Path(tmp) / "bot_state_test.json"),
        state_db=str(Path(tmp) / "bot_state_test.db"),
    )
    Path(config.state_file).unlink(missing_ok=True)
    Path(config.state_db).unlink(missing_ok=True)

    # Mock feed с гэпами и движением цен
    tickers = ["SBER", "LKOH", "GAZP", "NVTK", "PLZL", "ROSN", "MOEX", "T"]
    prev_closes = {t: 100.0 for t in tickers}
    session_opens = {
        "SBER": 101.5, "LKOH": 98.0, "NVTK": 103.0,  # гэпы для fade
        "GAZP": 100.2, "PLZL": 99.2,
        "ROSN": 100.0, "MOEX": 100.0, "T": 100.0,
    }
    # daily closes для mom (25 дней)
    dates = pd.bdate_range("2026-04-01", periods=25)
    rng = np.random.default_rng(7)
    rows = []
    for j, t in enumerate(tickers):
        drift = (j - 4) * 0.002
        px = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.008, 25)))
        for i, d in enumerate(dates):
            rows.append({"ts": d, "ticker": t, "close": px[i]})
    daily = pd.DataFrame(rows)
    # price series: SBER падает (short TP), NVTK растёт (short SL)
    price_series = {
        "SBER": [101.0, 100.5, 100.0],
        "LKOH": [98.5, 99.0, 99.5],
        "NVTK": [103.5, 104.0, 104.5],
        "GAZP": [100.2] * 3, "PLZL": [99.3, 99.5, 99.6],
        "ROSN": [100.0]*3, "MOEX": [100.0]*3, "T": [100.0]*3,
    }
    feed = MockTInvestData(tickers=tickers, prev_closes=prev_closes,
                           session_opens=session_opens, price_series=price_series,
                           daily_closes=daily)
    feed.resolve_figis()
    client = MockArenaGoClient(portfolio_name="test", price_provider=lambda t: session_opens.get(t, 100.0))
    portfolio = Portfolio(total_capital=1_000_000, lot_sizes=feed.get_lot_sizes())
    mom = Mom21Strategy(capital_share=0.25, top_k=3)
    gap = GapFadeStrategy(capital_share=0.75, gap_threshold=0.005)
    store = StateStore(config.state_db)
    monitor = RiskMonitor(store, total_capital=1_000_000)
    bot = TradingBot(config, client, feed, mom, gap, portfolio, store, monitor, sync_on_start=False)

    base = datetime(2026, 5, 28, tzinfo=MSK)   # четверг
    # Прогоняем фазы дня
    for label, t in [
        ("prepare", dtime(9, 56)),
        ("open", dtime(10, 1)),
        ("monitor-1", dtime(10, 30)),
        ("monitor-2", dtime(12, 0)),
        ("monitor-3", dtime(15, 0)),
        ("eod", dtime(18, 46)),
    ]:
        now = base.replace(hour=t.hour, minute=t.minute)
        log.info(f"\n----- Phase: {label} ({t}) -----")
        bot.tick(now)
        log.info(f"  mom_shares: {bot.mom_shares}")
        log.info(f"  gap_shares: { {k:v for k,v in bot.gap_shares.items() if v} }")

    log.info(f"\nFinal: realized PnL = {bot.day_state.day_realized_pnl:,.0f} ₽")
    log.info(f"mom positions held overnight: { {k:v for k,v in bot.mom_shares.items() if v} }")
    log.info(f"gap positions (should be empty after EOD): { {k:v for k,v in bot.gap_shares.items() if v} }")
    assert all(v == 0 for v in bot.gap_shares.values()), "Gap should be flat after EOD!"
    log.info("\n✓ Smoke test passed: gap flat after EOD, mom held")

    # --- Отдельный тест kill-switch ---
    log.info("\n=== KILL-SWITCH TEST ===")
    ks_db = str(Path(tmp) / "bot_ks.db")
    Path(ks_db).unlink(missing_ok=True)
    store_ks = StateStore(ks_db)
    monitor_ks = RiskMonitor(store_ks, total_capital=1_000_000)
    bot2 = TradingBot(config, MockArenaGoClient(portfolio_name="ks", price_provider=lambda t: 100.0),
                      feed, mom, gap, portfolio, store_ks, monitor_ks, sync_on_start=False)
    bot2._init_new_day(base)
    # Симулируем открытую позицию с большим убытком
    bot2.gap_shares = {"SBER": 5000}       # long 5000 @ entry 100
    bot2.entry_prices = {"SBER": 100.0}
    # current price упал до 88 → unrealized = 5000 × (88-100) = -60,000 ₽ = -6% от 1M
    loss_prices = {"SBER": 88.0}
    pnl = bot2._compute_day_pnl(loss_prices)
    dd = pnl / config.total_capital
    log.info(f"Simulated day PnL: {pnl:,.0f} ₽ ({dd:.2%}), threshold {config.kill_switch_dd:.0%}")
    assert dd < config.kill_switch_dd, "Should breach kill-switch threshold"
    bot2._trigger_kill_switch()
    assert bot2.day_state.kill_switched, "Kill-switch flag should be set"
    assert len(bot2.gap_shares) == 0, "Positions should be flattened"
    log.info("✓ Kill-switch test passed: breach detected, positions flattened")

    # --- Тест: первый день = rebal day ---
    log.info("\n=== FIRST-DAY REBAL TEST ===")
    fd_db = str(Path(tmp) / "bot_fd.db")
    Path(fd_db).unlink(missing_ok=True)
    store_fd = StateStore(fd_db)
    monitor_fd = RiskMonitor(store_fd, total_capital=1_000_000)
    bot3 = TradingBot(config, MockArenaGoClient(portfolio_name="fd", price_provider=lambda t: 100.0),
                      feed, mom, gap, portfolio, store_fd, monitor_fd, sync_on_start=False)
    bot3._init_new_day(base)
    assert bot3.day_state.rebal_day, "Day 1 must be a rebal day (bootstrap mom)!"
    log.info(f"✓ Day 1 (#{bot3.day_state.day_counter}) is rebal day: {bot3.day_state.rebal_day}")


if __name__ == "__main__":
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Run smoke test with mock")
    parser.add_argument("--portfolio", default="my_bot")
    args = parser.parse_args()

    if args.mock:
        _smoke_test()
    else:
        log_path = setup_logging()
        log.info(f"Logging to stdout + file: {log_path}")
        cfg = Config(portfolio_name=args.portfolio)
        bot = build_bot(cfg, use_mock=False)
        bot.run()