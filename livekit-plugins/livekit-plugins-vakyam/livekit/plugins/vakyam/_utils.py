# Copyright 2025 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

from livekit.agents import APIStatusError

from .models import (
    CUSTOM_VOICE_PREFIX,
    MAX_SPEED,
    MAX_TEXT_CHARACTERS,
    MIN_SPEED,
    SUPPORTED_LANGUAGES,
    SUPPORTED_MODELS,
    SUPPORTED_SAMPLE_RATES,
    TTS_STREAM_PATH,
    TTS_WEBSOCKET_PATH,
)

_RETRYABLE_WS_CODES = {
    "rate_limit_exceeded",
    "concurrency_limit_exceeded",
    "tts_workers_busy",
    "tts_workers_unconfigured",
    "tts_workers_unavailable",
    "internal_error",
}


def normalize_voice(voice: str) -> str:
    """Normalize a preset voice name or custom ``vc_`` voice ID."""
    if not isinstance(voice, str) or not voice.strip():
        raise ValueError("voice must be a non-empty string")
    normalized = voice.strip()
    if normalized == CUSTOM_VOICE_PREFIX:
        raise ValueError("custom voice IDs must include a value after 'vc_'")
    return normalized


def normalize_base_url(base_url: str, *, allow_insecure_base_url: bool = False) -> str:
    """Normalize and validate a public API base URL."""
    normalized = base_url.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme != "http":
        return normalized

    host = parsed.hostname or ""
    if allow_insecure_base_url or host in {"localhost", "127.0.0.1", "::1"}:
        return normalized

    raise ValueError(
        "base_url must use HTTPS unless it points to localhost. "
        "Pass allow_insecure_base_url=True only for trusted development networks."
    )


def websocket_url(base_url: str) -> str:
    """Build the Vakyam TTS WebSocket URL from an HTTP(S) or WS(S) base URL."""
    root = base_url.rstrip("/")
    if root.startswith("https://"):
        root = "wss://" + root[len("https://") :]
    elif root.startswith("http://"):
        root = "ws://" + root[len("http://") :]
    elif not root.startswith(("ws://", "wss://")):
        root = "wss://" + root
    if root.endswith(TTS_WEBSOCKET_PATH):
        return root
    return root + TTS_WEBSOCKET_PATH


def http_stream_url(base_url: str) -> str:
    """Build the Vakyam HTTP streaming TTS URL."""
    return base_url.rstrip("/") + TTS_STREAM_PATH


def validate_tts_options(
    *,
    model: str,
    language: str,
    sample_rate: int,
    speed: float,
    voice: str,
) -> None:
    """Validate constructor / update_options values."""
    if model not in SUPPORTED_MODELS:
        valid = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(f"model '{model}' is not supported. Valid values are: {valid}.")
    if language not in SUPPORTED_LANGUAGES:
        valid = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise ValueError(f"language '{language}' is not supported. Valid values are: {valid}.")
    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        valid = ", ".join(str(v) for v in sorted(SUPPORTED_SAMPLE_RATES))
        raise ValueError(
            f"sample_rate '{sample_rate}' is not supported. Valid values are: {valid}."
        )
    if not MIN_SPEED <= speed <= MAX_SPEED:
        raise ValueError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}")
    normalize_voice(voice)


def validate_text(text: str) -> None:
    """Validate utterance text for a single synthesis request."""
    if not isinstance(text, str) or not text:
        raise ValueError("text is required")
    character_count = len(text)
    if character_count > MAX_TEXT_CHARACTERS:
        raise ValueError(
            f"Input text is {character_count} characters. Maximum allowed is "
            f"{MAX_TEXT_CHARACTERS} Unicode characters."
        )


def split_text(text: str, *, max_characters: int = MAX_TEXT_CHARACTERS) -> list[str]:
    """Split an oversized utterance at whitespace, falling back to a hard boundary."""
    if max_characters <= 0:
        raise ValueError("max_characters must be greater than zero")

    remaining = text.strip()
    chunks: list[str] = []
    while len(remaining) > max_characters:
        split_at = remaining.rfind(" ", 0, max_characters + 1)
        if split_at <= 0:
            split_at = max_characters
        chunk = remaining[:split_at].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def speech_payload(
    *,
    text: str,
    model: str,
    voice: str,
    language: str,
    sample_rate: int,
    speed: float,
    output_format: str = "pcm",
) -> dict[str, Any]:
    """JSON body for HTTP generate/stream requests."""
    validate_text(text)
    validate_tts_options(
        model=model, language=language, sample_rate=sample_rate, speed=speed, voice=voice
    )
    return {
        "text": text,
        "model_id": model,
        "voice": normalize_voice(voice),
        "language": language,
        "output_format": output_format,
        "sample_rate": sample_rate,
        "speed": speed,
    }


def raise_http_error(status: int, body: str) -> None:
    """Raise ``APIStatusError`` from a Vakyam HTTP error envelope."""
    parsed: object | None = None
    try:
        parsed = json.loads(body) if body else None
    except json.JSONDecodeError:
        parsed = None

    error_code: str | None = None
    if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
        error = parsed["error"]
        code = error.get("code")
        if isinstance(code, (str, int)):
            error_code = str(code)

    message = f"Vakyam TTS request failed with status {status}"
    safe_body: dict[str, object] = {"status_code": status}
    if error_code is not None:
        message += f" (error code: {error_code})"
        safe_body["error_code"] = error_code

    raise APIStatusError(message, status_code=status, body=safe_body)


def raise_ws_error(data: dict[str, Any]) -> None:
    """Raise ``APIStatusError`` from a Vakyam WebSocket ``error`` frame."""
    error = data.get("error") if isinstance(data.get("error"), dict) else {}
    code = error.get("code") if isinstance(error, dict) else None
    error_code = str(code) if isinstance(code, (str, int)) else None
    retryable = error_code in _RETRYABLE_WS_CODES
    message = "Vakyam TTS WebSocket request failed"
    safe_body: dict[str, str] = {"type": "error"}
    if error_code is not None:
        message += f" (error code: {error_code})"
        safe_body["code"] = error_code
    raise APIStatusError(
        message,
        status_code=-1,
        body=safe_body,
        retryable=retryable,
    )
