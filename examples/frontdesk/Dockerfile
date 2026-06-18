# syntax=docker/dockerfile:1
#
# Shared Dockerfile for every example under examples/. Byte-identical
# across the tree — each example's entry script is named `agent.py`,
# so there's no per-example variation left.
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
    git \
    git-lfs \
    gcc \
    g++ \
    python3-dev \
  && rm -rf /var/lib/apt/lists/*

# Enable git-lfs so pip's git installs smudge LFS-tracked binaries
# (e.g. silero's bundled VAD onnx) instead of leaving pointer files.
# --system so the unprivileged appuser below inherits the filters.
RUN git lfs install --system

WORKDIR /app
USER appuser

COPY requirements.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt

# Pre-download model weights plugins ship (silero VAD, turn-detector, …)
# so the container is ready to take traffic without a cold-download stall.
RUN python -m livekit.agents download-files

COPY . .

CMD ["python", "agent.py", "start"]
