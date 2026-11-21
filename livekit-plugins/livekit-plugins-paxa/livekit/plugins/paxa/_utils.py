# Copyright 2026 LiveKit, Inc.
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
import re
import unicodedata
from email.message import Message
from typing import Any

from livekit.agents import APIConnectionError, APIError, APIStatusError

from .models import (
    MAX_SPEED,
    MIN_SPEED,
    MODEL_SAMPLE_RATES,
    NUM_CHANNELS,
    SUPPORTED_LANGUAGES,
    SUPPORTED_SAMPLE_RATES,
)
from .version import __version__

CLIENT_NAME = f"livekit-plugins-paxa/{__version__}"

# Paxa error codes are lowercase identifiers. Anything else found where a code belongs
# is dropped, so an error never carries response text into logs or telemetry.
_ERROR_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")

# The HTTP status Paxa's error catalog gives each code the live socket can send.
_LIVE_ERROR_STATUS: dict[str, int] = {
    "validation": 400,
    "live_protocol": 400,
    "unknown_model": 400,
    "unsupported_format": 400,
    "unknown_voice": 400,
    "text_too_long": 400,
    "tag_invalid": 400,
    "unauthorized": 401,
    "insufficient_credits": 402,
    "key_limit": 403,
    "unspeakable_text": 422,
    "rate_limited": 429,
    "concurrency_limited": 429,
    "internal": 500,
    "provider_error": 502,
    "provider_unavailable": 503,
}


def check_pcm_content_type(content_type: str, *, sample_rate: int) -> None:
    """Raise unless a ``POST /v1/tts`` response is labeled as the pcm the plugin asked for.

    Paxa labels pcm ``audio/pcm;rate=<Hz>;channels=1``. Audio at another rate would play
    at the wrong speed and pitch, so a mismatch is an error, not something to play.
    """
    header = Message()
    header["Content-Type"] = content_type
    found = (header.get_content_type(), header.get_param("rate"), header.get_param("channels"))
    if found != ("audio/pcm", str(sample_rate), str(NUM_CHANNELS)):
        raise APIError(
            f"Paxa TTS returned audio in an unexpected format, not mono pcm at {sample_rate} Hz",
            retryable=False,
        )


def request_headers(api_key: str) -> dict[str, str]:
    """Headers for every Paxa request, the WebSocket upgrade included.

    ``x-paxa-client`` tells Paxa which integration sent the request.
    """
    return {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": CLIENT_NAME,
        "x-paxa-client": CLIENT_NAME,
    }


def normalize_base_url(base_url: str) -> str:
    """Return the API origin, also accepting an OpenAI-style base URL ending in ``/v1``."""
    origin = base_url.strip().rstrip("/")
    if origin.endswith("/v1"):
        origin = origin[: -len("/v1")]
    if not origin.startswith(("https://", "http://")):
        raise ValueError("base_url must start with https:// or http://")
    return origin


def live_url(base_url: str) -> str:
    """Build the ``/v1/tts/live`` WebSocket URL from a normalized API origin."""
    scheme, rest = base_url.split("://", 1)
    return f"{'wss' if scheme == 'https' else 'ws'}://{rest}/v1/tts/live"


def validate_options(
    *, model: str, voice: str, language: str, speed: float, sample_rate: int
) -> None:
    """Check options locally, so a bad value fails before any request is sent."""
    if not model.strip():
        raise ValueError("model must be a non-empty string")
    if not voice.strip():
        raise ValueError("voice must be a non-empty string")
    if language not in SUPPORTED_LANGUAGES:
        valid = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise ValueError(f"language must be one of {valid}, got {language!r}")
    if not MIN_SPEED <= speed <= MAX_SPEED:
        raise ValueError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}, got {speed}")
    # A model the plugin doesn't list may serve any rate Paxa serves; Paxa answers
    # unsupported_format for the rest.
    rates = MODEL_SAMPLE_RATES.get(model, SUPPORTED_SAMPLE_RATES)
    if sample_rate not in rates:
        valid = ", ".join(str(rate) for rate in sorted(rates))
        on_model = f" on {model}" if model in MODEL_SAMPLE_RATES else ""
        raise ValueError(f"sample_rate must be one of {valid}{on_model}, got {sample_rate!r}")


def _error_code(value: object) -> str | None:
    if isinstance(value, str) and _ERROR_CODE_RE.fullmatch(value):
        return value
    return None


def http_status_error(status: int, body: bytes, *, request_id: str | None) -> APIStatusError:
    """Map a failed ``POST /v1/tts`` response to ``APIStatusError``.

    Paxa answers errors as RFC 9457 bodies whose ``title`` is a stable code, such as
    ``{"title": "unknown_voice", "status": 400}``. Only that code is kept.
    """
    code: str | None = None
    try:
        problem = json.loads(body)
    except ValueError:
        problem = None
    if isinstance(problem, dict):
        code = _error_code(problem.get("title"))

    message = f"Paxa TTS request failed with status {status}"
    if code:
        message += f" ({code})"
    return APIStatusError(
        message,
        status_code=status,
        request_id=request_id,
        body={"code": code} if code else None,
    )


def live_event_error(event: dict[str, Any], *, connection_id: str | None) -> APIStatusError:
    """Map a live ``error`` event to ``APIStatusError`` with its code's HTTP status."""
    code = _error_code(event.get("code"))
    status = _LIVE_ERROR_STATUS.get(code, -1) if code else -1
    return APIStatusError(
        f"Paxa TTS live session failed ({code or 'unknown error'})",
        status_code=status,
        request_id=connection_id,
        body={"code": code} if code else None,
    )


def live_close_error(close_code: int | None, *, connection_id: str | None) -> APIError:
    """Map a live session that closed before its ``done`` event.

    Paxa closes a failed session with 4000 plus the failure's HTTP status, such as 4402
    for insufficient credits. Any other code is a transport close.
    """
    if close_code is not None and 4000 <= close_code < 5000:
        return APIStatusError(
            f"Paxa TTS live session closed with code {close_code}",
            status_code=close_code - 4000,
            request_id=connection_id,
        )
    return APIConnectionError(
        f"Paxa TTS live session closed before delivery completed (code {close_code})"
    )


def take_frame(pending: list[str], max_chars: int) -> str:
    """Pop queued sentences into one text frame of at most ``max_chars`` characters.

    Sentences are joined with a space. A sentence longer than ``max_chars`` is cut at
    its last whitespace inside the limit, and its rest stays first in the queue.
    """
    parts: list[str] = []
    size = 0
    while pending:
        sentence = pending[0]
        added = len(sentence) + (1 if parts else 0)
        if size + added <= max_chars:
            parts.append(pending.pop(0))
            size += added
            continue
        if not parts:
            head, rest = _split_sentence(sentence, max_chars)
            parts.append(head)
            if rest:
                pending[0] = rest
            else:
                pending.pop(0)
        break
    return " ".join(parts)


def _split_sentence(text: str, max_chars: int) -> tuple[str, str]:
    cut = max_chars
    while cut > 0 and not text[cut].isspace():
        cut -= 1
    if cut == 0:
        # No whitespace to cut at: cut at the limit, but never between a character and
        # the combining marks after it, such as Thai vowel and tone marks.
        cut = max_chars
        while cut > 1 and unicodedata.category(text[cut]).startswith("M"):
            cut -= 1
    return text[:cut].rstrip(), text[cut:].lstrip()
