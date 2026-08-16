# Paper-trade через Docker — имитация этапа 2

Цель: запустить бота локально максимально близко к тому, как его запустят
организаторы на этапе 2 (тот же Dockerfile, те же env-переменные, тот же
persistent-диск `/data`). За пару дней увидим реальную частоту gap-фейдов
и убедимся, что live-путь рабочий.

## Что воспроизводим из этапа 2

| Этап 2 (организаторы) | Локально (ты) |
|---|---|
| CI собирает Dockerfile | `docker build` |
| Контейнер на их сервере | `docker run` у тебя |
| `SANDBOX_API_KEY` в env | `-e SANDBOX_API_KEY=...` |
| Диск примонтирован в `/data` | `-v ...:/data` |
| Логи в Dashboard | `docker logs -f` |

Отличие только одно: на этапе 1 у тебя **свой** `SANDBOX_API_KEY` (с этапа 1),
на этапе 2 организаторы подменят его автоматически. Код один и тот же.

---

## Шаг 1. Собрать образ

Из папки `prod/` (где лежит Dockerfile):

```powershell
docker build -t arenago-bot .
```

Первый билд ~2-4 мин (ставит pandas/numpy/t-tech). Если упадёт на
`pip install t-tech` — значит на твоей сети нет доступа к нужному индексу;
напиши, добавим `--index-url` или внутренний источник.

## Шаг 2. Подготовить папку для persistent-состояния

Эмулируем диск `/data`, который переживёт перезапуски контейнера:

```powershell
mkdir D:\moex\bot_data
```

Тут будет жить `bot_state.db` (позиции, счётчик дней, аудит сделок) —
ровно как `/data` на этапе 2.

## Шаг 3. Запустить (имитация этапа 2)

```powershell
docker run -d `
  --name arenago-bot `
  --restart unless-stopped `
  -e SANDBOX_API_KEY="c497d13a30f1df8e18eef3b5b4d59becb5bdc39314195a1fa4a66202111173aa" `
  -e TINKOFF_TOKEN="t.zCgCQwYt29Oep4Uf9y4oAkj4RJ08Z_h_3X_yMAdbnHM0JkwmY1r727Nnf_YrQsQ8bSN1HO5P5qM8e0qzdiMQZg" `
  -v D:\moex\bot_data:/data `
  arenago-bot
```

Что тут что:
- `-d` — фоновый режим (как сервер)
- `--restart unless-stopped` — автоперезапуск при падении (проверим recovery!)
- `-e SANDBOX_API_KEY` — твой ключ ArenaGo (этап 2 подменит сам)
- `-e TINKOFF_TOKEN` — данные для feed
- `-v D:\moex\bot_data:/data` — persistent-диск

Имя бота определится автоматически из `/bots` (как в `test_short.py`).
Если у тебя несколько ботов и нужен конкретный — добавь
`-e ARENAGO_BOT="ИмяБота" -e ARENAGO_PORTFOLIO="ИмяБота"`.

## Шаг 4. Смотреть логи (= их Dashboard)

```powershell
docker logs -f arenago-bot
```

Логи пишутся **в два места одновременно**:
1. **stdout** → `docker logs` (на этапе 2 — их Dashboard)
2. **Файл на диске** → `D:\moex\bot_data\logs\bot.log` (переживает перезапуски
   и `docker rm`!). Ротация в полночь, хранится 30 дней (`bot.log.2026-05-28` и т.д.)

Поскольку файл на persistent-диске `/data`, логи **не теряются** даже если
пересоздать контейнер. Можно открыть напрямую:

```powershell
Get-Content D:\moex\bot_data\logs\bot.log -Tail 50 -Wait
```

Сразу увидишь стартовый баннер: режим, капитал, расписание, сколько FIGI
разрешилось, какие позиции восстановлены. Дальше:
- `[heartbeat] HH:MM MSK | phase=... | turnover=X.XM` — раз в 10 мин, значит бот жив
- В торговые часы (10:00-18:50 MSK): `Gap entry: ...`, `[gap-exit:...]`, `[mom] ...`
- На EOD (18:45): `RISK SUMMARY` с дневным оборотом

**Считаем gap-фейды:** каждый `Gap entry:` в логе = один фейд. За пару дней
будет видно среднее число в день — это и есть данные для решения по LLM.

## Полезные команды

```powershell
# Сколько gap-входов сегодня (из лог-файла, не теряется)
Select-String "Gap entry" D:\moex\bot_data\logs\bot.log

# Все события за день из файла
Get-Content D:\moex\bot_data\logs\bot.log -Tail 200

# Аудит всех сделок из persistent-БД
docker exec arenago-bot python -c "from monitor import StateStore; import json; print(json.dumps(StateStore('/data/bot_state.db').get_trades(), ensure_ascii=False, indent=2))"

# Накопленный оборот
docker exec arenago-bot python -c "from monitor import StateStore; print(StateStore('/data/bot_state.db').total_turnover())"

# Проверить recovery: перезапустить и убедиться, что позиции восстановились
docker restart arenago-bot
docker logs arenago-bot | Select-String "restored|Restored"

# Остановить
docker stop arenago-bot && docker rm arenago-bot
```

## Что проверяем за эти дни

1. **Live-путь рабочий** — бот реально открывает/закрывает позиции (видно в логах + на arenago.ru)
2. **Частота гэпов** — сколько `Gap entry` в день (для решения по LLM-вето)
3. **Turnover pace** — `RISK SUMMARY` на EOD: идём ли к 10M
4. **Recovery** — после `docker restart` позиции восстанавливаются из `/data`
5. **Heartbeat** — в простое (ночью) бот не «молчит», видно что жив
6. **Нет крэшей** — за пару суток ни одного `Unhandled error`

После этого — данные на руках, решаем по LLM предметно.
