#!/bin/bash
set -e

until curl -sf http://toxiproxy:8474/proxies; do
  echo "Waiting for toxiproxy..."
  sleep 1
done

uv sync --all-extras --dev
uv run pytest -s --tb=short tests/test_tts.py --show-capture=all