"""
monitor.py — Персистенция состояния, восстановление, риск-мониторинг.

Закрывает ограничения main.py:
    - Полное состояние (mom_shares, gap_shares, entry_prices, counter) в SQLite
    - Восстановление при перезапуске (бот знает разделение mom/gap после рестарта)
    - Аудит-лог всех сделок (прозрачность для проверки кода — никаких wash trades)
    - Точный tracking дневного PnL и накопленного оборота за этап

Два компонента:
    StateStore  — SQLite persistence (позиции, счётчик, сделки, дневной PnL)
    RiskMonitor — drawdown, накопленный оборот (проверка порога 10M в реальном времени)

Использование в main.py:
    store = StateStore("bot_state.db")
    # При старте — восстановление:
    mom_shares, gap_shares, entry_prices = store.load_positions()
    counter = store.load_counter()
    # После каждой сделки:
    store.log_trade(ticker, side, qty, price, strategy, reason)
    store.save_positions(mom_shares, gap_shares, entry_prices)
    # В конце дня:
    store.record_daily_pnl(date, realized, unrealized, equity, turnover)

    monitor = RiskMonitor(store, total_capital=1_000_000, stage_days=10)
    monitor.add_turnover(order_value)
    if monitor.turnover_behind_pace(day_n): alert(...)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("monitor")

MSK = timezone(timedelta(hours=3))


# ============================================================
# State Store (SQLite)
# ============================================================

class StateStore:
    """SQLite-персистенция состояния бота. Restart-safe."""

    def __init__(self, db_path: str = "bot_state.db"):
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS positions (
                    strategy TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL DEFAULT 0,
                    updated_at TEXT,
                    PRIMARY KEY (strategy, ticker)
                );
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    strategy TEXT NOT NULL,
                    reason TEXT
                );
                CREATE TABLE IF NOT EXISTS daily_pnl (
                    trade_date TEXT PRIMARY KEY,
                    realized REAL NOT NULL DEFAULT 0,
                    unrealized REAL NOT NULL DEFAULT 0,
                    equity REAL NOT NULL DEFAULT 0,
                    turnover REAL NOT NULL DEFAULT 0,
                    max_drawdown REAL NOT NULL DEFAULT 0
                );
            """)

    # --------------------------------------------------------
    # Generic key-value state
    # --------------------------------------------------------

    def set_value(self, key: str, value) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO state(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)),
            )

    def get_value(self, key: str, default=None):
        with self._conn() as c:
            row = c.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    # --------------------------------------------------------
    # Day counter (для 2d phase)
    # --------------------------------------------------------

    def save_counter(self, counter: int) -> None:
        self.set_value("day_counter", counter)

    def load_counter(self) -> int:
        return int(self.get_value("day_counter", 0))

    def save_last_trade_date(self, d: date) -> None:
        self.set_value("last_trade_date", d.isoformat())

    def load_last_trade_date(self) -> Optional[date]:
        v = self.get_value("last_trade_date")
        return date.fromisoformat(v) if v else None

    # --------------------------------------------------------
    # Position registries (mom / gap раздельно)
    # --------------------------------------------------------

    def save_positions(
        self,
        mom_shares: Dict[str, int],
        gap_shares: Dict[str, int],
        entry_prices: Dict[str, float],
    ) -> None:
        """Полная перезапись реестров позиций."""
        now = datetime.now(MSK).isoformat()
        with self._conn() as c:
            c.execute("DELETE FROM positions")
            rows = []
            for t, q in mom_shares.items():
                if q != 0:
                    rows.append(("mom", t, int(q), float(entry_prices.get(t, 0)), now))
            for t, q in gap_shares.items():
                if q != 0:
                    rows.append(("gap", t, int(q), float(entry_prices.get(t, 0)), now))
            c.executemany(
                "INSERT INTO positions(strategy, ticker, quantity, entry_price, updated_at) "
                "VALUES(?, ?, ?, ?, ?)",
                rows,
            )

    def load_positions(self) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, float]]:
        """Восстановить mom_shares, gap_shares, entry_prices из БД."""
        mom_shares, gap_shares, entry_prices = {}, {}, {}
        with self._conn() as c:
            rows = c.execute("SELECT strategy, ticker, quantity, entry_price FROM positions").fetchall()
        for r in rows:
            if r["strategy"] == "mom":
                mom_shares[r["ticker"]] = r["quantity"]
            else:
                gap_shares[r["ticker"]] = r["quantity"]
            entry_prices[r["ticker"]] = r["entry_price"]
        return mom_shares, gap_shares, entry_prices

    # --------------------------------------------------------
    # Trade log (аудит)
    # --------------------------------------------------------

    def log_trade(
        self,
        ticker: str,
        side: str,
        quantity: int,
        price: float,
        strategy: str,
        reason: str = "",
        timestamp: Optional[str] = None,
    ) -> None:
        ts = timestamp or datetime.now(MSK).isoformat()
        with self._conn() as c:
            c.execute(
                "INSERT INTO trades(timestamp, ticker, side, quantity, price, strategy, reason) "
                "VALUES(?, ?, ?, ?, ?, ?, ?)",
                (ts, ticker, side, int(quantity), float(price), strategy, reason),
            )

    def get_trades(self, since: Optional[str] = None) -> List[dict]:
        with self._conn() as c:
            if since:
                rows = c.execute(
                    "SELECT * FROM trades WHERE timestamp >= ? ORDER BY id", (since,)
                ).fetchall()
            else:
                rows = c.execute("SELECT * FROM trades ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def total_turnover(self) -> float:
        """Суммарный оборот по всем залогированным сделкам (в ₽)."""
        with self._conn() as c:
            row = c.execute(
                "SELECT COALESCE(SUM(ABS(quantity) * price), 0) AS t FROM trades"
            ).fetchone()
        return float(row["t"])

    # --------------------------------------------------------
    # Daily PnL
    # --------------------------------------------------------

    def record_daily_pnl(
        self, trade_date: date, realized: float, unrealized: float,
        equity: float, turnover: float, max_drawdown: float = 0.0,
    ) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO daily_pnl(trade_date, realized, unrealized, equity, turnover, max_drawdown) "
                "VALUES(?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(trade_date) DO UPDATE SET "
                "realized=excluded.realized, unrealized=excluded.unrealized, "
                "equity=excluded.equity, turnover=excluded.turnover, "
                "max_drawdown=excluded.max_drawdown",
                (trade_date.isoformat(), realized, unrealized, equity, turnover, max_drawdown),
            )

    def get_daily_history(self) -> List[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM daily_pnl ORDER BY trade_date").fetchall()
        return [dict(r) for r in rows]


# ============================================================
# Risk Monitor
# ============================================================

class RiskMonitor:
    """Отслеживание риска и оборота в реальном времени."""

    def __init__(
        self,
        store: StateStore,
        total_capital: float = 1_000_000.0,
        stage_trading_days: int = 10,
        turnover_target: float = 10_000_000.0,
        kill_switch_dd: float = 0.02,
    ):
        self.store = store
        self.total_capital = total_capital
        self.stage_trading_days = stage_trading_days
        self.turnover_target = turnover_target
        self.kill_switch_dd = kill_switch_dd

    # --------------------------------------------------------
    # Turnover tracking (проверка порога 10M)
    # --------------------------------------------------------

    def cumulative_turnover(self) -> float:
        return self.store.total_turnover()

    def turnover_pace(self, current_trading_day: int) -> dict:
        """Анализ темпа оборота относительно цели 10M.

        Returns dict с: cumulative, required_pace, projected_final, on_track.
        """
        cum = self.cumulative_turnover()
        # Сколько должно быть к этому дню для линейного достижения цели
        required_by_now = self.turnover_target * current_trading_day / self.stage_trading_days
        # Прогноз на конец этапа при текущем темпе
        if current_trading_day > 0:
            projected = cum / current_trading_day * self.stage_trading_days
        else:
            projected = 0.0
        on_track = cum >= required_by_now
        return {
            "cumulative_M": cum / 1e6,
            "required_by_now_M": required_by_now / 1e6,
            "projected_final_M": projected / 1e6,
            "target_M": self.turnover_target / 1e6,
            "on_track": on_track,
            "deficit_M": max(0, required_by_now - cum) / 1e6,
        }

    def turnover_alert(self, current_trading_day: int) -> Optional[str]:
        """Вернуть строку-предупреждение если отстаём от темпа оборота."""
        pace = self.turnover_pace(current_trading_day)
        if not pace["on_track"]:
            return (
                f"⚠ TURNOVER BEHIND PACE day {current_trading_day}/{self.stage_trading_days}: "
                f"{pace['cumulative_M']:.1f}M cum, need {pace['required_by_now_M']:.1f}M, "
                f"projected final {pace['projected_final_M']:.1f}M "
                f"(target {pace['target_M']:.0f}M). Deficit {pace['deficit_M']:.1f}M."
            )
        return None

    # --------------------------------------------------------
    # Drawdown
    # --------------------------------------------------------

    def day_drawdown(self, day_start_equity: float, current_equity: float) -> float:
        if day_start_equity <= 0:
            return 0.0
        return (current_equity - day_start_equity) / day_start_equity

    def kill_switch_breached(self, day_start_equity: float, current_equity: float) -> bool:
        return self.day_drawdown(day_start_equity, current_equity) < -self.kill_switch_dd

    # --------------------------------------------------------
    # Summary report
    # --------------------------------------------------------

    def summary(self, current_trading_day: int) -> str:
        pace = self.turnover_pace(current_trading_day)
        history = self.store.get_daily_history()
        total_realized = sum(d["realized"] for d in history)
        lines = [
            "=" * 60,
            f"RISK SUMMARY — day {current_trading_day}/{self.stage_trading_days}",
            "=" * 60,
            f"Cumulative turnover: {pace['cumulative_M']:.1f}M / {pace['target_M']:.0f}M target",
            f"  Required by now: {pace['required_by_now_M']:.1f}M — "
            f"{'ON TRACK ✓' if pace['on_track'] else 'BEHIND ✗'}",
            f"  Projected final: {pace['projected_final_M']:.1f}M",
            f"Total realized PnL: {total_realized:+,.0f} ₽ "
            f"({total_realized / self.total_capital:+.2%})",
            f"Days recorded: {len(history)}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ============================================================
# Recovery helper
# ============================================================

def recover_bot_state(store: StateStore) -> dict:
    """Восстановить состояние для TradingBot при перезапуске.

    Returns dict с mom_shares, gap_shares, entry_prices, counter, last_date.
    """
    mom_shares, gap_shares, entry_prices = store.load_positions()
    counter = store.load_counter()
    last_date = store.load_last_trade_date()
    log.info(f"Recovered state: counter={counter}, last_date={last_date}, "
             f"mom={len(mom_shares)} pos, gap={len(gap_shares)} pos")
    return {
        "mom_shares": mom_shares,
        "gap_shares": gap_shares,
        "entry_prices": entry_prices,
        "counter": counter,
        "last_date": last_date,
    }


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    import tempfile
    import os

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    db = tempfile.mktemp(suffix=".db")
    log.info(f"=== StateStore test (db: {db}) ===")
    store = StateStore(db)

    # --- Save/load positions ---
    log.info("\n--- Test 1: positions persistence ---")
    mom = {"SBER": -413, "LKOH": -423, "T": 417, "GAZP": -416, "ROSN": 417, "MOEX": 417}
    gap = {"NVTK": -1449, "PLZL": 1511}
    entry = {"SBER": 280.0, "LKOH": 6500.0, "T": 3500.0, "NVTK": 1200.0, "PLZL": 14000.0,
             "GAZP": 142.0, "ROSN": 550.0, "MOEX": 200.0}
    store.save_positions(mom, gap, entry)
    mom2, gap2, entry2 = store.load_positions()
    assert mom2 == mom, f"mom mismatch: {mom2} != {mom}"
    assert gap2 == gap, f"gap mismatch: {gap2} != {gap}"
    log.info(f"  Saved & loaded: {len(mom2)} mom, {len(gap2)} gap positions ✓")

    # --- Counter ---
    log.info("\n--- Test 2: counter & date persistence ---")
    store.save_counter(5)
    store.save_last_trade_date(date(2026, 6, 3))
    assert store.load_counter() == 5
    assert store.load_last_trade_date() == date(2026, 6, 3)
    log.info(f"  counter=5, last_date=2026-06-03 ✓")

    # --- Trade log + turnover ---
    log.info("\n--- Test 3: trade log & turnover ---")
    store.log_trade("SBER", "B", 100, 280.0, "mom", "rebalance")
    store.log_trade("LKOH", "S", 10, 6500.0, "mom", "rebalance")
    store.log_trade("NVTK", "S", 1449, 1200.0, "gap", "gap_entry")
    store.log_trade("NVTK", "B", 1449, 1210.0, "gap", "stop_loss")
    trades = store.get_trades()
    expected_turnover = 100*280 + 10*6500 + 1449*1200 + 1449*1210
    actual_turnover = store.total_turnover()
    log.info(f"  Logged {len(trades)} trades, turnover={actual_turnover:,.0f} ₽ "
             f"(expected {expected_turnover:,.0f})")
    assert abs(actual_turnover - expected_turnover) < 1, "Turnover mismatch"
    log.info("  ✓ Trade log & turnover correct")

    # --- Recovery ---
    log.info("\n--- Test 4: recovery after 'restart' ---")
    store2 = StateStore(db)   # новое подключение к той же БД
    recovered = recover_bot_state(store2)
    assert recovered["counter"] == 5
    assert recovered["mom_shares"] == mom
    assert recovered["gap_shares"] == gap
    log.info(f"  Recovered: counter={recovered['counter']}, "
             f"mom={len(recovered['mom_shares'])}, gap={len(recovered['gap_shares'])} ✓")

    # --- RiskMonitor turnover pace ---
    log.info("\n--- Test 5: turnover pace tracking ---")
    monitor = RiskMonitor(store, total_capital=1_000_000, stage_trading_days=10,
                          turnover_target=10_000_000)
    # Текущий оборот ~3.5M от 4 сделок выше
    cum = monitor.cumulative_turnover()
    log.info(f"  Cumulative turnover: {cum/1e6:.2f}M")
    # День 1: need 1M, have 3.5M → on track
    pace_d1 = monitor.turnover_pace(current_trading_day=1)
    log.info(f"  Day 1 pace: {pace_d1['cumulative_M']:.1f}M cum, "
             f"need {pace_d1['required_by_now_M']:.1f}M, on_track={pace_d1['on_track']}")
    assert pace_d1["on_track"], "Day 1 should be on track"
    # День 8: need 8M, have 3.5M → behind
    pace_d8 = monitor.turnover_pace(current_trading_day=8)
    log.info(f"  Day 8 pace: need {pace_d8['required_by_now_M']:.1f}M, "
             f"on_track={pace_d8['on_track']}, projected {pace_d8['projected_final_M']:.1f}M")
    assert not pace_d8["on_track"], "Day 8 should be behind"
    alert = monitor.turnover_alert(current_trading_day=8)
    log.info(f"  Alert: {alert}")
    assert alert is not None

    # --- Daily PnL ---
    log.info("\n--- Test 6: daily PnL recording ---")
    store.record_daily_pnl(date(2026, 5, 28), realized=3500, unrealized=-1200,
                           equity=1002300, turnover=2_500_000, max_drawdown=-0.008)
    store.record_daily_pnl(date(2026, 5, 29), realized=-800, unrealized=2100,
                           equity=1003400, turnover=2_800_000, max_drawdown=-0.015)
    history = store.get_daily_history()
    log.info(f"  Recorded {len(history)} days")
    for d in history:
        log.info(f"    {d['trade_date']}: realized={d['realized']:+.0f}, "
                 f"equity={d['equity']:,.0f}, turnover={d['turnover']/1e6:.1f}M")

    # --- Drawdown / kill-switch ---
    log.info("\n--- Test 7: drawdown & kill-switch ---")
    dd = monitor.day_drawdown(day_start_equity=1_000_000, current_equity=975_000)
    log.info(f"  Drawdown: {dd:.2%}")
    assert monitor.kill_switch_breached(1_000_000, 975_000), "Should breach at -2.5%"
    assert not monitor.kill_switch_breached(1_000_000, 990_000), "Should not breach at -1%"
    log.info("  ✓ Kill-switch logic correct")

    # --- Summary ---
    log.info("\n--- Test 8: summary report ---")
    print(monitor.summary(current_trading_day=2))

    os.unlink(db)
    log.info("\n✓ All monitor tests passed")
