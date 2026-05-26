"""
news_filter.py — LLM-риск-оверлей через polza.ai для gap-фейдов.

Идея: gap_fade ставит на то, что гэп — это перереакция и он закроется обратно.
Но если гэп вызван РЕАЛЬНОЙ экстремальной новостью (санкции, делистинг, дефолт,
крупная авария, смена собственника), то это не перереакция, а repricing —
фейдить его опасно. LLM проверяет свежие новости по тикеру и при экстремальном
событии накладывает ВЕТО на открытие gap-позиции.

Архитектура (честная, измеримая роль LLM — не для красоты, а реально влияет):
    - Открытая модель (Qwen, Apache 2.0) через polza.ai + web-плагин (свежие новости)
    - Узкий критерий блокировки: только подтверждённое экстремальное событие
    - Дефолт ответа = НЕ блокировать (LLM не должна зря резать оборот)
    - Любая ошибка/таймаут → НЕ блокировать (LLM не single point of failure)

Управление включением (в main.py):
    - turnover отстаёт от темпа (< 1M/день) → фильтр ОТКЛЮЧАЕТСЯ (оборот важнее)
    - последний торговый день → фильтр ПРИНУДИТЕЛЬНО отключён (не рисковать оборотом)

polza.ai: OpenAI-совместимый, base https://api.polza.ai/api/v1, Bearer-авторизация,
веб-поиск через plugins=[{"id":"web"}]. Модель в формате provider/model.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

log = logging.getLogger("news_filter")

DEFAULT_BASE_URL = "https://api.polza.ai/api/v1"
# Ключ polza.ai по умолчанию (баланс организации пополняется автоматически на этапе 2,
# сам ключ не меняется). Env POLZA_API_KEY переопределяет, если задан модератором.
DEFAULT_POLZA_KEY = "pza_pI0xLZ5jk0M5Iw5QhFaztQotK4w7RgLa"
# Открытая модель (Apache 2.0). qwen-2.5-72b: стабильнее и дешевле flash на polza,
# хорошо знает русский (важно для новостей MOEX).
DEFAULT_MODEL = "qwen/qwen-2.5-72b-instruct"

# Тикер → название компании (для осмысленного поиска новостей)
TICKER_NAMES: Dict[str, str] = {
    "LKOH": "Лукойл", "SBER": "Сбербанк", "ROSN": "Роснефть", "GAZP": "Газпром",
    "VTBR": "ВТБ", "YDEX": "Яндекс", "PLZL": "Полюс", "T": "Т-Технологии (ТКС)",
    "NVTK": "Новатэк", "X5": "X5 Retail Group", "GMKN": "Норникель",
    "MGNT": "Магнит", "ALRS": "Алроса", "AFLT": "Аэрофлот", "CHMF": "Северсталь",
    "NLMK": "НЛМК", "MOEX": "Московская биржа", "SNGSP": "Сургутнефтегаз преф",
    "MTSS": "МТС", "PIKK": "ПИК",
}


@dataclass
class RiskVerdict:
    """Результат проверки тикера на экстремальные новости."""
    ticker: str
    blocked: bool
    reason: str = ""
    sources: List[str] = field(default_factory=list)
    error: Optional[str] = None   # если была ошибка (тогда blocked=False по fallback)


# Промпт: блокировка — РЕДКОЕ исключение. По умолчанию НЕ блокируем.
# НАПРАВЛЕННАЯ логика: вето только если новость объясняет движение гэпа
# (фундаментал в ту же сторону, что и гэп → fade против фундаментала → опасно).
_SYSTEM_PROMPT = (
    "Ты — риск-аналитик торгового бота на Московской бирже. Бот применяет стратегию "
    "fade: когда цена резко прыгает (гэп), бот ставит на ВОЗВРАТ цены обратно "
    "(гэп вверх → шорт, гэп вниз → лонг). Это работает, если прыжок — перереакция. "
    "Но если прыжок вызван РЕАЛЬНЫМ свежим событием, оправдывающим движение, то "
    "возврат маловероятен и fade опасен.\n\n"
    "Твоя задача — заблокировать сделку ТОЛЬКО если свежая (1-2 дня) сильная новость "
    "ОБЪЯСНЯЕТ движение гэпа В ТУ ЖЕ СТОРОНУ (для гэпа вверх — позитивная новость, "
    "оправдывающая рост; для гэпа вниз — негативная, оправдывающая падение). "
    "Новость, направленная ПРОТИВ гэпа, fade НЕ ломает — её игнорируй.\n\n"
    "КРИТИЧЕСКИ ВАЖНО: по умолчанию block=false. Блокировка — РЕДКОЕ исключение. "
    "При ЛЮБЫХ сомнениях, неподтверждённых данных, старых новостях, обычном фоне, "
    "или если новость направлена против гэпа → block=false. Лучше пропустить, чем "
    "зря заблокировать.\n\n"
    "Блокируй (block=true) ТОЛЬКО если ОДНОВРЕМЕННО:\n"
    "1) Новость свежая (последние 1-2 дня), подтверждена надёжным источником.\n"
    "2) Событие значимое и оправдывает движение цены в сторону гэпа:\n"
    "   • для падения: новые санкции, делистинг, дефолт, банкротство, арест активов, "
    "крупная авария с остановкой производства, национализация;\n"
    "   • для роста: крупная сделка/поглощение по высокой цене, оферта выкупа с премией, "
    "прорывной позитив, резко поднимающий справедливую стоимость.\n"
    "3) Из-за этого события возврат цены к прежнему уровню маловероятен.\n\n"
    "Если хотя бы одно условие НЕ выполнено → block=false.\n"
    "НЕ блокируй: отчётность, обычные дивиденды, прогнозы аналитиков, отраслевые "
    "новости, слухи, форумы, телеграм, старые события, обычную волатильность, "
    "и любые новости, направленные ПРОТИВ движения гэпа.\n\n"
    "Доверяй только надёжным источникам: interfax.ru, e-disclosure.ru, moex.com, "
    "rbc.ru, tass.ru, finam.ru, kommersant.ru.\n\n"
    "Ответ — строго один JSON-объект без markdown и текста вне него:\n"
    '{"block": true|false, "reason": "краткое обоснование на русском"}'
)


class NewsFilter:
    """LLM-фильтр экстремальных новостей через polza.ai."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 45.0,
        use_web: bool = True,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.use_web = use_web

    def check_ticker_risk(self, ticker: str, gap_direction: str = None,
                          gap_pct: float = None) -> RiskVerdict:
        """Проверить тикер на новости, ОБЪЯСНЯЮЩИЕ направление гэпа.

        Вето имеет смысл только когда свежая новость толкает цену в ТУ ЖЕ сторону,
        что и гэп (объясняет движение) — тогда fade (ставка на возврат) идёт против
        фундаментала и опасен. Если новость направлена против гэпа — не блокируем.

        Args:
            ticker: тикер
            gap_direction: "up" (гэп вверх → бот шортит) или "down" (гэп вниз → бот лонгует).
                Если None — направление не учитывается (старое поведение, любая катастрофа).
            gap_pct: величина гэпа в % (для контекста)

        При любой ошибке/таймауте → blocked=False (fallback).
        """
        company = TICKER_NAMES.get(ticker, ticker)

        if gap_direction == "up":
            move_desc = (
                f"Цена сделала гэп ВВЕРХ"
                + (f" на {abs(gap_pct):.1f}%" if gap_pct else "")
                + ". Бот собирается ШОРТИТЬ (ставит на откат вниз). "
                "Блокируй ТОЛЬКО если есть свежая сильная ПОЗИТИВНАЯ новость, которая "
                "ОБЪЯСНЯЕТ рост и делает откат вниз маловероятным (рост обоснован "
                "фундаментально). Негативные новости НЕ блокируй — они играют за шорт."
            )
        elif gap_direction == "down":
            move_desc = (
                f"Цена сделала гэп ВНИЗ"
                + (f" на {abs(gap_pct):.1f}%" if gap_pct else "")
                + ". Бот собирается ЛОНГовать (ставит на отскок вверх). "
                "Блокируй ТОЛЬКО если есть свежая сильная НЕГАТИВНАЯ новость, которая "
                "ОБЪЯСНЯЕТ падение и делает отскок вверх маловероятным (падение обосновано "
                "фундаментально: санкции/делистинг/дефолт/авария). Позитивные новости НЕ "
                "блокируй — они играют за лонг."
            )
        else:
            move_desc = (
                "Оцени, есть ли свежее катастрофическое событие, делающее возврат "
                "цены маловероятным."
            )

        user_msg = (
            f"Компания: {company} (тикер {ticker} на MOEX). {move_desc}\n"
            f"Ищи новости ТОЛЬКО за последние 1-2 дня на надёжных источниках "
            f"(Интерфакс, e-disclosure, MOEX, РБК, ТАСС, Финам, Коммерсант). "
            f"Помни: блокировка — редкое исключение, по умолчанию block=false. "
            f"Верни строго один JSON."
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.0,
            "max_tokens": 300,
        }
        if self.use_web:
            payload["plugins"] = [{"id": "web"}]   # свежие новости из интернета

        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            if r.status_code != 200:
                log.warning(f"News check {ticker}: HTTP {r.status_code} — fallback (не блокируем)")
                return RiskVerdict(ticker, blocked=False, error=f"HTTP {r.status_code}")
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            verdict = self._parse_verdict(ticker, content)
            # извлечём ссылки на источники, если есть (annotations)
            ann = data["choices"][0]["message"].get("annotations") or []
            verdict.sources = [a.get("url", "") for a in ann if isinstance(a, dict)][:5]
            # Структурированный лог для пост-анализа (легко грепать: "NEWS_CHECK")
            dir_str = gap_direction or "n/a"
            gap_str = f"{gap_pct:+.1f}%" if gap_pct is not None else "n/a"
            mark = "VETO" if verdict.blocked else "OK"
            log.info(f"NEWS_CHECK | {ticker} ({company}) | dir={dir_str} gap={gap_str} "
                     f"| verdict={mark} | sources={len(verdict.sources)} | reason={verdict.reason}")
            if verdict.sources:
                log.info(f"NEWS_SRC | {ticker} | " + " ; ".join(s for s in verdict.sources if s))
            return verdict
        except requests.Timeout:
            log.warning(f"News check {ticker}: timeout — fallback (не блокируем)")
            return RiskVerdict(ticker, blocked=False, error="timeout")
        except Exception as e:
            log.warning(f"News check {ticker}: {e} — fallback (не блокируем)")
            return RiskVerdict(ticker, blocked=False, error=str(e))

    @staticmethod
    def _parse_verdict(ticker: str, content: str) -> RiskVerdict:
        """Распарсить JSON-вердикт из ответа LLM. При сбое парсинга → не блокируем."""
        # Вырезаем JSON из возможного markdown/текста
        m = re.search(r"\{[^{}]*\"block\"[^{}]*\}", content, re.DOTALL)
        raw = m.group(0) if m else content.strip()
        try:
            obj = json.loads(raw)
            blocked = bool(obj.get("block", False))
            reason = str(obj.get("reason", ""))[:300]
            return RiskVerdict(ticker, blocked=blocked, reason=reason)
        except Exception as e:
            log.warning(f"News {ticker}: не смог распарсить вердикт '{content[:80]}' ({e}) — не блокируем")
            return RiskVerdict(ticker, blocked=False, error="parse_error")


class MockNewsFilter:
    """Эмулятор для тестов. Блокирует тикеры из blocked_set."""

    def __init__(self, blocked_set: Optional[set] = None):
        self.blocked_set = blocked_set or set()
        self.calls: List[str] = []

    def check_ticker_risk(self, ticker: str, gap_direction: str = None,
                          gap_pct: float = None) -> RiskVerdict:
        self.calls.append(ticker)
        if ticker in self.blocked_set:
            return RiskVerdict(ticker, blocked=True, reason=f"[mock] экстремальная новость по {ticker}")
        return RiskVerdict(ticker, blocked=False, reason="[mock] чисто")


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # --- Mock test ---
    log.info("=== Mock NewsFilter test ===")
    mock = MockNewsFilter(blocked_set={"GAZP"})
    v1 = mock.check_ticker_risk("SBER")
    v2 = mock.check_ticker_risk("GAZP")
    log.info(f"SBER: blocked={v1.blocked} ({v1.reason})")
    log.info(f"GAZP: blocked={v2.blocked} ({v2.reason})")
    assert not v1.blocked and v2.blocked
    log.info("✓ Mock: GAZP заблокирован, SBER пропущен")

    # --- Parse test ---
    log.info("\n=== JSON parse test ===")
    tests = [
        ('{"block": true, "reason": "санкции"}', True),
        ('```json\n{"block": false, "reason": "чисто"}\n```', False),
        ('Вот ответ: {"block": true, "reason": "делистинг"} — конец', True),
        ('мусор без json', False),   # fallback не блокирует
    ]
    for content, expected in tests:
        v = NewsFilter._parse_verdict("TEST", content)
        status = "✓" if v.blocked == expected else "✗"
        log.info(f"  {status} '{content[:40]}...' → block={v.blocked}")
        assert v.blocked == expected
    log.info("✓ Парсинг JSON-вердиктов корректен (мусор → не блокируем)")

    # --- Real polza.ai test (если ключ задан) ---
    log.info("\n=== Real polza.ai test (если POLZA_API_KEY задан) ===")
    key = os.environ.get("POLZA_API_KEY", DEFAULT_POLZA_KEY)
    if key:
        model = os.environ.get("POLZA_MODEL", DEFAULT_MODEL)
        nf = NewsFilter(api_key=key, model=model)
        log.info(f"Модель: {model}, web-поиск: вкл")
        # Демонстрируем направленную логику: один тикер в обе стороны
        for tk, direction, pct in [("SBER", "up", 3.0), ("GAZP", "down", -3.0)]:
            v = nf.check_ticker_risk(tk, gap_direction=direction, gap_pct=pct)
            log.info(f"  {tk} гэп {direction} {pct:+.0f}%: blocked={v.blocked}, "
                     f"reason='{v.reason}', sources={len(v.sources)}, error={v.error}")
    else:
        log.info("  Skipped (POLZA_API_KEY не задан). Для теста:")
        log.info("    export POLZA_API_KEY=... ; export POLZA_MODEL=qwen/...")

    log.info("\n✓ All news_filter tests passed")