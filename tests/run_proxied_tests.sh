#!/bin/bash
set -e

until curl -sf http://toxiproxy:8474/proxies; do
  echo "Waiting for toxiproxy..."
  sleep 1
done

uv run pytest tests/test_tts.py --tb=short