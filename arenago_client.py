"""
ArenaGo REST API client для хакатона MOEX.

Base URL: https://arenago.ru/api  (подтверждён рабочим скриптом test_short.py)

API endpoints:
    POST /submit_order        — отправить market-ордер
    GET  /trades/<portfolio>  — история сделок портфеля
    GET  /positions/<portfolio> — текущие позиции
    GET  /bots                — список ботов команды (каждый с полем 'name')

Auth: header `Authorization: <token>` (без префикса Bearer)

submit_order body (ПОДТВЕРЖДЁН):
    {"direction": "B"|"S", "secid": <ticker>, "quantity": <int>, "bot": <bot_name>}

submit_order response:
    {"success": bool, "error"?: str, "remaining_cash": float, "average_price"?: float}
    Успех определяется полем "success" в body, НЕ только HTTP-статусом.

positions response: list of {"secid": str, "position": int (отриц.=short), "average_price": float}

Short selling РАБОТАЕТ (подтверждено: продажа отсутствующего тикера открывает шорт).

Использование:
    client = ArenaGoClient.from_env(portfolio_name="my_bot")  # bot_name = portfolio по умолч.
    resp = client.submit_order("SBER", "B", 100)
    if resp.success:
        print(f"Filled at {resp.filled_price}, cash left {client.last_remaining_cash}")

    positions = client.get_positions()   # list of Position(ticker, quantity, avg_price)
    trades = client.get_trades()
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict, field
from datetime import date, datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger("arenago")


# ============================================================
# Configuration
# ============================================================

DEFAULT_BASE_URL = "https://arenago.ru/api"   # подтверждено рабочим скриптом
DEFAULT_TIMEOUT = 10.0
DEFAULT_RETRY_TIMES = 3
DEFAULT_BACKOFF = 0.5

DAILY_TRADE_LIMIT = 1000        # на одного бота
DAILY_TRADE_WARNING_PCT = 0.8   # log warning if used > 80%

VALID_SIDES = ("B", "S")


# ============================================================
# Exceptions
# ============================================================

class ArenaGoError(Exception):
    """Базовый класс для всех ArenaGo-специфичных ошибок."""

class OrderRejectedError(ArenaGoError):
    """HTTP 400 от submit_order с человекочитаемым reason."""
    def __init__(self, message: str, raw: dict):
        super().__init__(message)
        self.raw = raw

class TradeLimitExceededError(ArenaGoError):
    """Подошли к лимиту 1000 сделок/день."""

class AuthenticationError(ArenaGoError):
    """HTTP 401/403 — проблема с токеном."""

class NetworkError(ArenaGoError):
    """Сетевые проблемы, таймауты, retry exhausted."""


# ============================================================
# Data classes
# ============================================================

@dataclass
class OrderResponse:
    """Ответ от submit_order, нормализованный из возможных формат API."""
    success: bool
    order_id: Optional[str] = None
    ticker: Optional[str] = None
    side: Optional[str] = None
    quantity: Optional[int] = None
    filled_price: Optional[float] = None
    filled_quantity: Optional[int] = None
    status: Optional[str] = None
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    http_status: Optional[int] = None


@dataclass
class Position:
    """Открытая позиция в портфеле. Quantity положительная = long, отрицательная = short."""
    ticker: str
    quantity: int
    avg_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trade:
    """История сделки."""
    ticker: str
    side: str
    quantity: int
    price: float
    timestamp: Optional[str] = None
    order_id: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Helper: parse ArenaGo error
# ============================================================

def _parse_error_message(response_json: Any) -> str:
    """Извлекает человекочитаемое сообщение из {'error': 'ERROR: ...'}."""
    if not isinstance(response_json, dict):
        return str(response_json)[:200]
    err = response_json.get("error") or response_json.get("message") or ""
    if isinstance(err, str):
        # Формат ТЗ "ERROR: <message>" — отрезаем prefix
        return err.removeprefix("ERROR:").strip() or err
    return str(err)[:200]


# ============================================================
# Daily trade counter
# ============================================================

class TradeCounter:
    """In-memory счётчик сделок с reset на midnight MSK.

    В проде лучше хранить в SQLite чтобы пережить перезапуск.
    """
    def __init__(self):
        self._count = 0
        self._reset_date = self._current_msk_date()
        self._warned_at = -1   # последний счёт на котором было предупреждение

    @staticmethod
    def _current_msk_date() -> date:
        msk = timezone(timedelta(hours=3))
        return datetime.now(msk).date()

    def _maybe_reset(self) -> None:
        today = self._current_msk_date()
        if today != self._reset_date:
            log.info(f"Trade counter reset: {self._count} -> 0 (new day {today})")
            self._count = 0
            self._warned_at = -1
            self._reset_date = today

    def increment(self) -> int:
        self._maybe_reset()
        self._count += 1
        return self._count

    def check_limit(self, limit: int = DAILY_TRADE_LIMIT) -> None:
        """Бросает TradeLimitExceededError если близко к лимиту.

        Warning логируется один раз на каждый 50-trade milestone после 80%.
        """
        self._maybe_reset()
        if self._count >= limit:
            raise TradeLimitExceededError(
                f"Daily trade limit reached: {self._count}/{limit}"
            )
        threshold = int(limit * DAILY_TRADE_WARNING_PCT)
        if self._count >= threshold and self._count // 50 > self._warned_at // 50:
            log.warning(f"Approaching daily trade limit: {self._count}/{limit}")
            self._warned_at = self._count

    @property
    def count(self) -> int:
        self._maybe_reset()
        return self._count


# ============================================================
# Main client
# ============================================================

class ArenaGoClient:
    """REST-клиент к ArenaGo.

    Стиль использования:
        client = ArenaGoClient(base_url, api_key, portfolio_name)
        client.submit_order("SBER", "B", 100)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        portfolio_name: str,
        bot_name: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        retry_times: int = DEFAULT_RETRY_TIMES,
        backoff: float = DEFAULT_BACKOFF,
    ):
        if not api_key:
            raise AuthenticationError("Empty API key — set SANDBOX_API_KEY env var")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.portfolio_name = portfolio_name
        # bot_name используется в submit_order (поле "bot"); по умолчанию = portfolio_name
        self.bot_name = bot_name or portfolio_name
        self.timeout = timeout
        self.trade_counter = TradeCounter()
        self.last_remaining_cash: Optional[float] = None   # из ответов submit_order

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

        # urllib3 Retry — только для idempotent методов (GET).
        # submit_order — POST, retry делаем вручную с проверкой что ордер реально не прошёл.
        retry_strategy = Retry(
            total=retry_times,
            backoff_factor=backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],   # POST не retry автоматически
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    @classmethod
    def from_env(
        cls,
        portfolio_name: str,
        bot_name: Optional[str] = None,
        base_url: Optional[str] = None,
        env_var: str = "SANDBOX_API_KEY",
    ) -> "ArenaGoClient":
        api_key = os.environ.get(env_var)
        if not api_key:
            raise AuthenticationError(f"Env var {env_var} not set")
        return cls(
            base_url=base_url or os.environ.get("ARENAGO_BASE_URL", DEFAULT_BASE_URL),
            api_key=api_key,
            portfolio_name=portfolio_name,
            bot_name=bot_name or os.environ.get("ARENAGO_BOT", portfolio_name),
        )

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _do_get(self, path: str) -> Tuple[int, Any]:
        """GET с автоматическими retry для transient errors."""
        try:
            r = self.session.get(self._url(path), timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"GET {path} failed: {e}") from e

        try:
            data = r.json()
        except json.JSONDecodeError:
            data = {"raw_text": r.text[:500]}

        if r.status_code == 401 or r.status_code == 403:
            raise AuthenticationError(f"GET {path}: {_parse_error_message(data)}")
        if r.status_code >= 500:
            raise NetworkError(f"GET {path} server error {r.status_code}: {_parse_error_message(data)}")
        return r.status_code, data

    def _do_post(self, path: str, body: dict) -> Tuple[int, Any]:
        """POST БЕЗ автоматического retry — слишком рискованно для ордеров."""
        log.debug(f"POST {path} body={body}")
        try:
            r = self.session.post(self._url(path), json=body, timeout=self.timeout)
        except requests.exceptions.Timeout as e:
            # Critical: timeout на submit_order — мы не знаем прошёл ордер или нет.
            # Стратегия должна это handle через сверку с positions.
            raise NetworkError(f"POST {path} TIMEOUT (order state unknown!): {e}") from e
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"POST {path} failed: {e}") from e

        try:
            data = r.json()
        except json.JSONDecodeError:
            data = {"raw_text": r.text[:500]}

        if r.status_code == 401 or r.status_code == 403:
            raise AuthenticationError(f"POST {path}: {_parse_error_message(data)}")
        return r.status_code, data

    # --------------------------------------------------------
    # Public API methods
    # --------------------------------------------------------

    def submit_order(
        self,
        ticker: str,
        side: str,
        quantity: int,
        skip_limit_check: bool = False,
    ) -> OrderResponse:
        """Отправить market-ордер.

        Args:
            ticker: тикер (например 'SBER')
            side: 'B' (buy / open long / close short) или 'S' (sell / open short / close long)
            quantity: количество лотов (положительное число)
            skip_limit_check: пропустить проверку daily trade limit (для emergency close)

        Returns:
            OrderResponse с success=True/False и параметрами.

        Raises:
            ValueError — невалидный input
            TradeLimitExceededError — превышен дневной лимит
            AuthenticationError — проблема с токеном
            OrderRejectedError — биржа отклонила ордер (insufficient balance, etc.)
            NetworkError — таймаут или сетевая проблема (СОСТОЯНИЕ ОРДЕРА НЕИЗВЕСТНО)
        """
        if side not in VALID_SIDES:
            raise ValueError(f"side must be one of {VALID_SIDES}, got {side!r}")
        if not isinstance(quantity, int) or quantity <= 0:
            raise ValueError(f"quantity must be positive int, got {quantity!r}")

        if not skip_limit_check:
            self.trade_counter.check_limit()

        body = self._build_order_body(ticker, side, quantity)
        status_code, data = self._do_post("/submit_order", body)

        # ArenaGo: ответ содержит поле "success" (true/false) + "error" при отказе.
        body_success = isinstance(data, dict) and data.get("success") is True

        if status_code == 400 or (isinstance(data, dict) and data.get("success") is False):
            err = _parse_error_message(data)
            log.warning(f"Order rejected: {ticker} {side} {quantity} → {err}")
            return OrderResponse(
                success=False, ticker=ticker, side=side, quantity=quantity,
                error=err, raw=data if isinstance(data, dict) else {},
                http_status=status_code,
            )
        if status_code not in (200, 201) and not body_success:
            log.error(f"Unexpected status {status_code} from submit_order: {data}")
            return OrderResponse(
                success=False, ticker=ticker, side=side, quantity=quantity,
                error=f"HTTP {status_code}",
                raw=data if isinstance(data, dict) else {},
                http_status=status_code,
            )

        self.trade_counter.increment()
        resp = self._parse_order_response(data, ticker, side, quantity)
        resp.http_status = status_code
        # Сохраняем remaining_cash для equity-tracking
        if isinstance(data, dict) and data.get("remaining_cash") is not None:
            try:
                self.last_remaining_cash = float(data["remaining_cash"])
            except (ValueError, TypeError):
                pass
        log.info(f"Order OK: {ticker} {side} {quantity} → "
                 f"filled {resp.filled_quantity}@{resp.filled_price}, "
                 f"cash={resp.raw.get('remaining_cash')}")
        return resp

    def _build_order_body(self, ticker: str, side: str, quantity: int) -> dict:
        """Body для submit_order ArenaGo (формат подтверждён рабочим скриптом).

        {"direction": "B"|"S", "secid": <ticker>, "quantity": <int>, "bot": <bot_name>}
        """
        return {
            "direction": side,
            "secid": ticker,
            "quantity": quantity,
            "bot": self.bot_name,
        }

    def _parse_order_response(
        self, data: Any, ticker: str, side: str, quantity: int,
    ) -> OrderResponse:
        """Парсит ответ submit_order ArenaGo."""
        if not isinstance(data, dict):
            return OrderResponse(
                success=True, ticker=ticker, side=side, quantity=quantity,
                raw={"raw": data},
            )
        # ArenaGo market orders исполняются сразу; цена может прийти в average_price
        price = (data.get("average_price") or data.get("price")
                 or data.get("filled_price"))
        filled_qty = (data.get("filled_quantity") or data.get("quantity") or quantity)
        order_id = (data.get("order_id") or data.get("id"))
        status = data.get("status") or ("filled" if data.get("success") else "unknown")

        return OrderResponse(
            success=True,
            order_id=str(order_id) if order_id is not None else None,
            ticker=ticker,
            side=side,
            quantity=quantity,
            filled_price=float(price) if price is not None else None,
            filled_quantity=int(filled_qty) if filled_qty is not None else None,
            status=status,
            raw=data,
        )

    def get_positions(self) -> List[Position]:
        """Получить текущие позиции портфеля."""
        status_code, data = self._do_get(f"/positions/{self.portfolio_name}")
        if status_code != 200:
            raise ArenaGoError(f"get_positions failed: {status_code} {data}")
        return self._parse_positions(data)

    def _parse_positions(self, data: Any) -> List[Position]:
        """Парсинг позиций ArenaGo: list of {secid, position, average_price}."""
        positions = []
        if isinstance(data, dict) and "positions" in data:
            data = data["positions"]
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                ticker = item.get("secid") or item.get("ticker") or item.get("symbol")
                if not ticker:
                    continue
                positions.append(self._make_position(ticker, item))
        elif isinstance(data, dict):
            for ticker, details in data.items():
                if isinstance(details, dict):
                    positions.append(self._make_position(ticker, details))
        return positions

    @staticmethod
    def _make_position(ticker: str, details: dict) -> Position:
        # ArenaGo: position (отриц.=short), average_price
        qty = (details.get("position") if details.get("position") is not None
               else details.get("quantity") or details.get("qty") or 0)
        avg = (details.get("average_price") or details.get("avg_price")
               or details.get("entry_price") or 0)
        mv = (details.get("market_value") or details.get("value") or 0)
        upnl = (details.get("unrealized_pnl") or details.get("pnl") or 0)
        return Position(
            ticker=ticker,
            quantity=int(qty),
            avg_price=float(avg),
            market_value=float(mv),
            unrealized_pnl=float(upnl),
            raw=details,
        )

    def get_trades(self) -> List[Trade]:
        """История сделок портфеля."""
        status_code, data = self._do_get(f"/trades/{self.portfolio_name}")
        if status_code != 200:
            raise ArenaGoError(f"get_trades failed: {status_code} {data}")
        if isinstance(data, dict) and "trades" in data:
            data = data["trades"]
        if not isinstance(data, list):
            log.warning(f"Unexpected trades format: {type(data)}")
            return []
        out = []
        for item in data:
            if not isinstance(item, dict):
                continue
            ticker = item.get("secid") or item.get("ticker") or item.get("symbol")
            if not ticker:
                continue
            out.append(Trade(
                ticker=ticker,
                side=item.get("direction") or item.get("side", "?"),
                quantity=int(item.get("quantity") or item.get("qty") or 0),
                price=float(item.get("average_price") or item.get("price") or 0),
                timestamp=item.get("timestamp") or item.get("time") or item.get("created_at"),
                order_id=str(item.get("order_id") or item.get("id") or ""),
                raw=item,
            ))
        return out

    def get_bots(self) -> List[Dict[str, Any]]:
        """Список ботов команды. Каждый бот имеет поле 'name'."""
        status_code, data = self._do_get("/bots")
        if status_code != 200:
            raise ArenaGoError(f"get_bots failed: {status_code} {data}")
        if isinstance(data, dict) and "bots" in data:
            return data["bots"]
        if isinstance(data, list):
            return data
        return []

    # --------------------------------------------------------
    # Convenience methods
    # --------------------------------------------------------

    def flatten_all_positions(self) -> List[OrderResponse]:
        """Закрыть все открытые позиции market-ордерами.

        Использовать для emergency stop, kill-switch, конца этапа.
        """
        positions = self.get_positions()
        results = []
        for pos in positions:
            if pos.quantity == 0:
                continue
            side = "S" if pos.quantity > 0 else "B"
            qty = abs(pos.quantity)
            try:
                resp = self.submit_order(pos.ticker, side, qty, skip_limit_check=True)
                results.append(resp)
            except ArenaGoError as e:
                log.error(f"Failed to flatten {pos.ticker}: {e}")
                results.append(OrderResponse(
                    success=False, ticker=pos.ticker, side=side, quantity=qty,
                    error=str(e),
                ))
        return results

    def trades_today(self) -> int:
        """Сколько сделок отправлено сегодня."""
        return self.trade_counter.count

    def ping(self) -> bool:
        """Простая проверка живости API через get_bots."""
        try:
            self.get_bots()
            return True
        except (NetworkError, AuthenticationError) as e:
            log.error(f"Ping failed: {e}")
            return False


# ============================================================
# Mock client (для разработки и unit-тестов без живого API)
# ============================================================

class MockArenaGoClient:
    """In-memory эмулятор ArenaGo. Используется в тестах и dev-окружении.

    Поведение: каждый submit_order сразу 'fills' по фиксированной цене
    (или цене из price_provider если передан).
    """

    def __init__(self, portfolio_name: str = "mock_bot", price_provider=None,
                 fail_rate: float = 0.0):
        self.portfolio_name = portfolio_name
        self.price_provider = price_provider  # callable: ticker -> price
        self.fail_rate = fail_rate
        self.trade_counter = TradeCounter()
        self._positions: Dict[str, Position] = {}
        self._trades: List[Trade] = []
        self._order_id_counter = 0
        # для воспроизводимости фейлов
        import random
        self._rng = random.Random(42)

    def submit_order(
        self, ticker: str, side: str, quantity: int,
        skip_limit_check: bool = False,
    ) -> OrderResponse:
        if side not in VALID_SIDES:
            raise ValueError(f"bad side {side}")
        if quantity <= 0:
            raise ValueError(f"bad quantity {quantity}")
        if not skip_limit_check:
            self.trade_counter.check_limit()

        if self.fail_rate > 0 and self._rng.random() < self.fail_rate:
            return OrderResponse(
                success=False, ticker=ticker, side=side, quantity=quantity,
                error="MOCK: random failure",
            )

        price = self.price_provider(ticker) if self.price_provider else 100.0
        self._order_id_counter += 1
        order_id = f"mock-{self._order_id_counter}"

        # Обновляем позицию
        pos = self._positions.get(ticker, Position(ticker=ticker, quantity=0, avg_price=0.0))
        delta = quantity if side == "B" else -quantity
        new_qty = pos.quantity + delta
        # Простая логика avg_price: пересчёт если direction увеличивает позицию
        if pos.quantity == 0:
            new_avg = price
        elif (pos.quantity > 0 and delta > 0) or (pos.quantity < 0 and delta < 0):
            # увеличение позиции — взвешенное среднее
            new_avg = (pos.avg_price * abs(pos.quantity) + price * abs(delta)) / abs(new_qty)
        else:
            new_avg = pos.avg_price   # частичное/полное закрытие — avg не меняется
        if new_qty == 0:
            self._positions.pop(ticker, None)
        else:
            self._positions[ticker] = Position(
                ticker=ticker, quantity=new_qty, avg_price=new_avg,
            )

        self._trades.append(Trade(
            ticker=ticker, side=side, quantity=quantity, price=price,
            timestamp=datetime.now(timezone.utc).isoformat(),
            order_id=order_id,
        ))
        self.trade_counter.increment()
        return OrderResponse(
            success=True, order_id=order_id, ticker=ticker, side=side,
            quantity=quantity, filled_price=price, filled_quantity=quantity,
            status="filled",
        )

    def get_positions(self) -> List[Position]:
        return list(self._positions.values())

    def get_trades(self) -> List[Trade]:
        return list(self._trades)

    def get_bots(self) -> List[Dict[str, Any]]:
        return [{"name": self.portfolio_name, "status": "active"}]

    def flatten_all_positions(self) -> List[OrderResponse]:
        results = []
        for pos in list(self._positions.values()):
            if pos.quantity == 0:
                continue
            side = "S" if pos.quantity > 0 else "B"
            results.append(self.submit_order(pos.ticker, side, abs(pos.quantity),
                                              skip_limit_check=True))
        return results

    def trades_today(self) -> int:
        return self.trade_counter.count

    def ping(self) -> bool:
        return True


# ============================================================
# Example usage / smoke test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # Smoke test через mock
    log.info("=== Smoke test with MockArenaGoClient ===")
    prices = {"SBER": 280.5, "LKOH": 6500.0, "GAZP": 142.0}
    client = MockArenaGoClient(portfolio_name="test_bot", price_provider=prices.get)

    # Open positions
    r1 = client.submit_order("SBER", "B", 100)
    log.info(f"  Buy SBER: success={r1.success}, price={r1.filled_price}, id={r1.order_id}")
    r2 = client.submit_order("LKOH", "S", 10)
    log.info(f"  Short LKOH: success={r2.success}, price={r2.filled_price}")

    # Check positions
    positions = client.get_positions()
    log.info(f"  Positions: {[(p.ticker, p.quantity, p.avg_price) for p in positions]}")

    # Check trades
    trades = client.get_trades()
    log.info(f"  Trades count: {len(trades)}")

    # Flatten
    flat_results = client.flatten_all_positions()
    log.info(f"  Flattened {len(flat_results)} positions, trades today: {client.trades_today()}")
    assert len(client.get_positions()) == 0
    log.info("  ✓ All flat")

    # Daily limit test
    log.info("\n=== Daily limit smoke test ===")
    client2 = MockArenaGoClient(portfolio_name="limit_test", price_provider=lambda t: 100.0)
    try:
        for i in range(1005):
            client2.submit_order("SBER", "B", 1)
    except TradeLimitExceededError as e:
        log.info(f"  ✓ Caught limit: {e}")
        log.info(f"  Trades executed: {client2.trades_today()}")

    log.info("\n=== Real ArenaGo test (если SANDBOX_API_KEY задан) ===")
    log.info("Воспроизводит логику test_short.py через ArenaGoClient")
    if os.environ.get("SANDBOX_API_KEY"):
        try:
            # bot/portfolio: из env или из первого бота
            bots_client = ArenaGoClient.from_env(portfolio_name="_tmp")
            bots = bots_client.get_bots()
            log.info(f"  [1] Bots: {bots}")
            if not bots:
                log.error("  Нет ботов — создайте бота в ArenaGo UI")
            else:
                bot_name = os.environ.get("ARENAGO_BOT") or bots[0].get("name")
                portfolio = os.environ.get("ARENAGO_PORTFOLIO") or bot_name
                log.info(f"  Using bot='{bot_name}', portfolio='{portfolio}'")

                real = ArenaGoClient.from_env(portfolio_name=portfolio, bot_name=bot_name)

                # [2] Позиции
                positions = real.get_positions()
                held = {p.ticker for p in positions}
                log.info(f"  [2] Positions: {[(p.ticker, p.quantity, p.avg_price) for p in positions]}")

                # [3] Выбрать тикер не в позициях
                universe = ["LKOH", "SBER", "ROSN", "GAZP", "VTBR", "YDEX", "PLZL",
                            "T", "NVTK", "X5", "GMKN", "MGNT", "ALRS", "AFLT",
                            "CHMF", "NLMK", "MOEX", "SNGSP", "MTSS", "PIKK"]
                target = next((t for t in universe if t not in held), None)
                log.info(f"  [3] Target ticker (not held): {target}")

                if target:
                    # [4] Открыть шорт 1 лот
                    log.info(f"  [4] Short 1 {target}")
                    resp = real.submit_order(target, "S", 1)
                    log.info(f"      success={resp.success}, price={resp.filled_price}, "
                             f"error={resp.error}, cash={real.last_remaining_cash}")
                    if resp.success:
                        time.sleep(1)
                        # [5] Проверить позицию
                        pos = next((p for p in real.get_positions() if p.ticker == target), None)
                        log.info(f"  [5] Position after short: "
                                 f"qty={pos.quantity if pos else None}, "
                                 f"avg={pos.avg_price if pos else None}")
                        if pos and pos.quantity < 0:
                            log.info(f"  ✓ SHORT CONFIRMED via ArenaGoClient")
                            # [6] Закрыть
                            log.info(f"  [6] Closing: buy {abs(pos.quantity)} {target}")
                            close = real.submit_order(target, "B", abs(pos.quantity))
                            time.sleep(1)
                            final = next((p for p in real.get_positions()
                                          if p.ticker == target), None)
                            log.info(f"      Final qty: {final.quantity if final else 0} (ожидаем 0)")
                        else:
                            log.warning(f"  ⚠ Position not negative: {pos.quantity if pos else None}")
        except ArenaGoError as e:
            log.error(f"  Real API test failed: {e}")
        except Exception as e:
            log.exception(f"  Unexpected error: {e}")
    else:
        log.info("  Skipped (SANDBOX_API_KEY not set)")