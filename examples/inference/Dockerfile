# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
USER appuser

COPY requirements.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt

# Pre-download any model weights plugins ship with a download_files
# hook (e.g. the turn detector ONNX), so the container is ready to
# take traffic without a first-request download stall.
RUN python -m livekit.agents download-files

COPY . .

CMD ["python", "agent.py", "start"]
