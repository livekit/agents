# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.6
FROM python:${PYTHON_VERSION}-slim 

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

WORKDIR /home/appuser

# Give ownership of the /home/appuser directory to the appuser user.
RUN command chown -R appuser:appuser /home/appuser

# Install gcc and other build dependencies.
RUN apt-get update && \
    apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

USER appuser

COPY . .

RUN python -m pip install --no-cache-dir -r requirements.txt
RUN python -c "import torch; torch.hub.load(repo_or_dir='snakers4/silero-vad',model='silero_vad',onnx=True)"
# RUN python minimal_assistant.py download-files 

# Run the application.
CMD python minimal_assistant.py start