# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.6
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Install gcc and other build dependencies.
RUN apt-get update && \
    apt-get install -y \
    gcc \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

FROM base

# Copy the source code into the container.
COPY . .

RUN python3 -m pip install -r requirements.txt

RUN python3 minimal_assistant.py download-files

# Switch to the non-privileged user to run the application.
USER appuser

# Run the application.
CMD python3 minimal_assistant.py start