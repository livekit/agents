from __future__ import annotations

import os

API_BASE_URL = "wss://api.telnyx.com/v2"
TTS_ENDPOINT = f"{API_BASE_URL}/text-to-speech/speech"
STT_ENDPOINT = f"{API_BASE_URL}/speech-to-text/transcription"

SAMPLE_RATE = 16000
NUM_CHANNELS = 1


def get_api_key(api_key: str | None = None) -> str:
    resolved_key = api_key or os.environ.get("TELNYX_API_KEY")
    if not resolved_key:
        raise ValueError("Telnyx API key required. Set TELNYX_API_KEY or provide api_key.")
    return resolved_key
