# MOEX ArenaGo trading bot
# Требование ТЗ: Dockerfile в корне проекта.
# Деплой: GitLab CI Pipeline на серверах организаторов (4 vCPU, 16 GB RAM, 10 GB).

FROM python:3.11-slim

# Логи сразу в stdout без буферизации (для мониторинга организаторов через Dashboard)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Persistent-состояние на подключаемом диске /data (переживает редеплои)
    BOT_DATA_DIR=/data

WORKDIR /app

# Системные зависимости. grpcio (зависимость t-tech) обычно ставится как
# manylinux-wheel без компиляции; gcc на случай отсутствия wheel под платформу.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Зависимости отдельным слоем — кэшируются, пока requirements.txt не менялся.
# --extra-index-url: t-tech-investments лежит на внутреннем индексе Т-Банка
# (pypi.org остаётся основным для requests/pandas/numpy).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple

# Код бота
COPY . .

# Постоянное хранилище: bot_state.db, логи. Монтируется организаторами.
# Директория /data создаётся; если диск не подключён — код сделает fallback.
RUN mkdir -p /data
VOLUME ["/data"]

# Точка входа — live-режим (без --mock): build_bot(use_mock=False) → run()
# SANDBOX_API_KEY прокидывается организаторами автоматически.
# TINKOFF_TOKEN нужно задать в переменных Pipeline.
CMD ["python", "main.py"]