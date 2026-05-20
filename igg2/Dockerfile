FROM python:3.14-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIPENV_VENV_IN_PROJECT=1 \
    PYTHONPATH=/app

RUN pip install --no-cache-dir pipenv

WORKDIR /app

COPY Pipfile Pipfile.lock ./
RUN pipenv sync --system --dev

COPY src/ ./src/

# Run as an unprivileged user. /data is the conventional mount point used by
# docker-compose.yml; pre-create it so the non-root user can write there.
RUN groupadd --system --gid 1000 igg2 \
 && useradd  --system --uid 1000 --gid igg2 --home-dir /app --shell /sbin/nologin igg2 \
 && mkdir -p /data \
 && chown -R igg2:igg2 /app /data
USER igg2:igg2

ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["--help"]
