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
import os
import re
from collections.abc import Sequence
from typing import Any, Literal, get_args
from urllib.parse import urlencode

from livekit.agents import APIStatusError, LanguageCode, create_api_error_from_http, stt
from livekit.agents.types import NOT_GIVEN, NotGivenOr, TimedString

from .log import logger
from .version import __version__

DEFAULT_API_URL = "https://api.reson8.dev"
TURNS_PATH = "/v1/speech-to-text/turns"
PRERECORDED_PATH = "/v1/speech-to-text/prerecorded"
INTEGRATION_HEADER = "X-Reson8-Integration"
INTEGRATION_NAME = "livekit-python"

ERROR_MESSAGE_HEADER = "X-Error-Message"

_STATUS_HINTS = {
    401: "check the provided api_key or RESON8_API_KEY",
    402: "see https://docs.reson8.dev/limits/",
    429: "see https://docs.reson8.dev/limits/",
}


SupportedLanguage = Literal["de", "en", "es", "fr", "fy", "it", "nl", "pl", "pt", "sv"]
"""The languages Reson8 can recognize, as ISO 639-1 codes.

See https://docs.reson8.dev/speech-to-text/features/languages/.
"""

SUPPORTED_LANGUAGES: tuple[str, ...] = get_args(SupportedLanguage)
"""``SupportedLanguage`` as a runtime tuple, for validation and error messages."""

Encoding = Literal["pcm_s16le"]
"""
The only encoding this plugin can send.

``rtc.AudioFrame`` carries signed 16-bit PCM and this plugin forwards those
bytes as they are, so every other encoding Reson8 accepts -- the companded
ones, the container formats, and the ``auto`` mode that sniffs a container
header -- would mislabel what is on the wire. See
https://docs.reson8.dev/speech-to-text/features/audio-formats/.
"""

ENCODINGS: tuple[str, ...] = get_args(Encoding)

# https://docs.reson8.dev/api/speech-to-text/turns/
MIN_CHANNELS = 1
MAX_CHANNELS = 10
MAX_PHRASES = 250

FillerMode = Literal["clean", "natural", "verbatim"]
"""How filler words are rendered: removed, left to the model, or preserved.

See https://docs.reson8.dev/api/speech-to-text/turns/.
"""

FILLER_MODES: tuple[str, ...] = get_args(FillerMode)

_COMMA_OUTSIDE_BRACES = re.compile(r",(?![^{}]*})")

# problem+json codes are lower_snake identifiers; anything else is treated as
# free text, so nothing arbitrary can ride into an exception message on it
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,63}")


def normalize_languages(value: str | Sequence[str] | None) -> str | None:
    """
    Normalize and validate a language selection into Reson8's query form.

    ``"nl"`` -> ``"nl"``; ``"nl,de"`` -> ``"nl,de"``; ``["nl", "de"]`` ->
    ``"nl,de"``; ``None``/``""``/``[]`` -> ``None`` (auto-detect).

    Raises ``ValueError`` if any code is not a :data:`SupportedLanguage`, so
    invalid selections fail locally rather than after a request to the API.
    """

    if value is None:
        return None

    codes = value.split(",") if isinstance(value, str) else list(value)
    codes = [c.strip().lower() for c in codes if c and c.strip()]
    if not codes:
        return None

    unsupported = [c for c in codes if c not in SUPPORTED_LANGUAGES]
    if unsupported:
        supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise ValueError(
            f"unsupported language(s): {', '.join(unsupported)}. Reson8 supports: {supported}."
        )

    return ",".join(codes)


def check_comma_joined(
    name: str,
    values: Sequence[str] | None,
    *,
    limit: int | None = None,
    allow_braced_commas: bool = False,
) -> None:
    """
    Validate entries that reach Reson8 joined into one comma-separated value.

    ``allow_braced_commas`` keeps a comma inside ``{}`` — a ``{m,n}`` repeat
    range in a pattern — which the server does not treat as a separator.

    A rejected entry is never quoted back. Phrases and patterns are customer
    vocabulary — names, identifiers, domain terms — and an exception message is
    not a structured attribute, so anything put there survives redaction.
    Closed-set values such as a language code or an encoding are named, being
    configuration rather than data.
    """

    if values is None:
        return

    if isinstance(values, str):
        raise ValueError(
            f"{name} takes a sequence of strings, not a single string; "
            f"wrap a single entry in a list"
        )

    if limit is not None and len(values) > limit:
        raise ValueError(f"{name} accepts at most {limit} entries, got {len(values)}")

    for value in values:
        if not value.strip():
            raise ValueError(f"{name} cannot contain an empty entry")

        if allow_braced_commas:
            if _COMMA_OUTSIDE_BRACES.search(value):
                raise ValueError(
                    f"{name} entries cannot contain a comma outside braces, since they "
                    f"are comma-separated on the wire; a comma inside a {{m,n}} range is fine"
                )
        elif "," in value:
            raise ValueError(
                f"{name} entries cannot contain a comma, since they are comma-separated on the wire"
            )


def check_channels(num_channels: int) -> None:
    if not MIN_CHANNELS <= num_channels <= MAX_CHANNELS:
        raise ValueError(
            f"num_channels must be between {MIN_CHANNELS} and {MAX_CHANNELS}, got {num_channels}"
        )


def check_probability(name: str, value: float | None) -> None:
    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


def resolve_base_url(base_url: str | None) -> str:
    resolved = base_url or os.environ.get("RESON8_BASE_URL")

    if not resolved and (legacy := os.environ.get("RESON8_API_URL")):
        logger.warning("RESON8_API_URL is deprecated, use RESON8_BASE_URL instead")
        resolved = legacy

    return (resolved or DEFAULT_API_URL).rstrip("/")


def build_url(base_url: str, path: str, params: dict[str, str], *, websocket: bool = False) -> str:
    base = base_url.rstrip("/")
    if websocket:
        base = base.replace("https://", "wss://", 1).replace("http://", "ws://", 1)

    return f"{base}{path}?{urlencode(params)}"


def auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"ApiKey {api_key}"}


def integration_headers() -> dict[str, str]:
    return {INTEGRATION_HEADER: f"{INTEGRATION_NAME}:{__version__}"}


def problem_parts(body: str) -> tuple[str | None, str | None]:
    """
    Split a problem+json body into its machine-readable code and its free text.

    They are returned apart because only the code belongs on an exception; see
    :func:`status_error`.
    """

    try:
        parsed = json.loads(body)
    except ValueError:
        return None, None

    if not isinstance(parsed, dict):
        return None, None

    code = parsed.get("code")
    detail = parsed.get("detail")

    return (
        code if isinstance(code, str) and code else None,
        detail if isinstance(detail, str) and detail else None,
    )


def status_error(
    status_code: int, *, code: str | None = None, detail: str | None = None
) -> APIStatusError:
    """
    Map a Reson8 rejection onto an actionable error.

    ``APIStatusError`` marks non-transient 4xx as non-retryable, so an
    exhausted credit balance or a bad key fails fast instead of backing off.
    """

    if code or detail:
        logger.warning(
            "Reson8 rejected the request",
            extra={"status": status_code, "lk.pii.code": code, "lk.pii.detail": detail},
        )

    hint = _STATUS_HINTS.get(status_code)
    safe_code = code if code and _IDENTIFIER.fullmatch(code) else None
    message = ": ".join(p for p in (safe_code, hint) if p)
    return create_api_error_from_http(message, status=status_code)


def _confidence(word: dict[str, Any]) -> NotGivenOr[float]:
    """
    Reson8 reports word confidence as a probability in (0, 1].

    See https://docs.reson8.dev/glossary/.
    """

    confidence: float | None = word.get("confidence")
    if confidence is None or not confidence > 0:
        return NOT_GIVEN

    return min(confidence, 1.0)


def _word_time(word: dict[str, Any], key: str, *, offset: float) -> NotGivenOr[float]:
    if "start_ms" not in word:
        return NOT_GIVEN
    start: float = word.get("start_ms", 0)
    if key == "start":
        return offset + start / 1000.0
    duration: float = word.get("duration_ms", 0)
    return offset + (start + duration) / 1000.0


def build_speech_data(
    msg: dict[str, Any],
    *,
    language: str | None,
    start_time_offset: float = 0.0,
) -> stt.SpeechData:
    """
    Build a LiveKit ``SpeechData`` from a Reson8 transcript/turn payload.

    Handles the optional ``start_ms``/``duration_ms``/``words`` fields that are
    only present when the matching ``include_*`` options are enabled.
    """

    raw_words = msg.get("words") or []
    confidences = [_confidence(w) for w in raw_words]
    words = [
        TimedString(
            text=w.get("text", ""),
            start_time=_word_time(w, "start", offset=start_time_offset),
            end_time=_word_time(w, "end", offset=start_time_offset),
            confidence=c,
            start_time_offset=start_time_offset,
        )
        for w, c in zip(raw_words, confidences, strict=True)
    ]

    known = [c for c in confidences if isinstance(c, float)]
    confidence = sum(known) / len(known) if known else 1.0

    start_ms = msg.get("start_ms")
    duration_ms = msg.get("duration_ms") or 0
    if start_ms is not None:
        start_time = start_time_offset + start_ms / 1000.0
        end_time = start_time_offset + (start_ms + duration_ms) / 1000.0
    else:
        start_time = start_time_offset
        end_time = start_time_offset

    fallback = language if language and "," not in language else ""
    return stt.SpeechData(
        language=LanguageCode(msg.get("language") or fallback or ""),
        text=msg.get("text", ""),
        start_time=start_time,
        end_time=end_time,
        confidence=confidence,
        words=words,
    )
