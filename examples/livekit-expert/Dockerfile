# syntax=docker/dockerfile:1
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1
COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /usr/local/bin/uv

RUN adduser --disabled-password --gecos "" --home /app --uid 10001 appuser
WORKDIR /app
USER appuser

COPY pyproject.toml ./
RUN uv sync --no-cache --no-dev
ENV PATH="/app/.venv/bin:${PATH}"

COPY . .
CMD ["python", "main.py", "start"]
