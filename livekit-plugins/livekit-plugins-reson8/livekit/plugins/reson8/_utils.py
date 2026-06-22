from __future__ import annotations

import math
from typing import Any

from livekit.agents import LanguageCode, stt
from livekit.agents.types import NOT_GIVEN, NotGivenOr, TimedString

DEFAULT_API_URL = "https://api.reson8.dev"


def to_ws_base(api_url: str) -> str:
    """Convert an http(s) API base URL into its ws(s) equivalent."""
    return api_url.rstrip("/").replace("https://", "wss://", 1).replace("http://", "ws://", 1)


def auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"ApiKey {api_key}"}


def _to_probability(log_prob: float | None) -> NotGivenOr[float]:
    """Reson8 returns confidence as a natural log-probability (<= 0).

    Convert it to a probability in (0, 1] for LiveKit's confidence fields.
    """
    if log_prob is None:
        return NOT_GIVEN
    try:
        return math.exp(log_prob)
    except (OverflowError, ValueError):
        return 1.0


def _word_time(word: dict[str, Any], key: str, *, offset: float) -> NotGivenOr[float]:
    if "start_ms" not in word:
        return NOT_GIVEN
    start = float(word.get("start_ms", 0) or 0)
    if key == "start":
        return offset + start / 1000.0
    duration = float(word.get("duration_ms", 0) or 0)
    return offset + (start + duration) / 1000.0


def build_speech_data(
    msg: dict[str, Any],
    *,
    language: str | None,
    start_time_offset: float = 0.0,
) -> stt.SpeechData:
    """Build a LiveKit ``SpeechData`` from a Reson8 transcript/turn payload.

    Handles the optional ``start_ms``/``duration_ms``/``words`` fields that are
    only present when the matching ``include_*`` options are enabled.
    """
    raw_words = msg.get("words") or []
    words = [
        TimedString(
            text=w.get("text", ""),
            start_time=_word_time(w, "start", offset=start_time_offset),
            end_time=_word_time(w, "end", offset=start_time_offset),
            confidence=_to_probability(w.get("confidence")),
            start_time_offset=start_time_offset,
        )
        for w in raw_words
    ]

    probs = [
        p
        for p in (_to_probability(w.get("confidence")) for w in raw_words if "confidence" in w)
        if isinstance(p, float)
    ]
    confidence = sum(probs) / len(probs) if probs else 1.0

    start_ms = msg.get("start_ms")
    duration_ms = msg.get("duration_ms") or 0
    if start_ms is not None:
        start_time = start_time_offset + start_ms / 1000.0
        end_time = start_time_offset + (start_ms + duration_ms) / 1000.0
    else:
        start_time = start_time_offset
        end_time = start_time_offset

    return stt.SpeechData(
        language=LanguageCode(msg.get("language") or language or ""),
        text=msg.get("text", ""),
        start_time=start_time,
        end_time=end_time,
        confidence=confidence,
        words=words,
    )
