from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Generic, TypeVar
from urllib.parse import urlencode

T = TypeVar("T")


class PeriodicCollector(Generic[T]):
    """Accumulates pushed values and reports the total on a fixed interval."""

    def __init__(
        self,
        callback: Callable[[T], None],
        *,
        duration: float,
    ) -> None:
        """Create a new periodic collector.

        Args:
            callback: Function to call with the accumulated value once
                the duration expires.
            duration: Time in seconds between callback invocations.
        """
        self._duration = duration
        self._callback = callback
        self._last_flush_time = time.monotonic()
        self._total: T | None = None

    def push(self, value: T) -> None:
        """Add a value, flushing once the duration has elapsed."""
        if self._total is None:
            self._total = value
        else:
            self._total += value  # type: ignore[operator]

        if time.monotonic() - self._last_flush_time >= self._duration:
            self.flush()

    def flush(self) -> None:
        """Invoke the callback with the total, if anything is pending."""
        if self._total is not None:
            self._callback(self._total)
            self._total = None
        self._last_flush_time = time.monotonic()


# Keyterm prompting limits, enforced by Sarvam as fatal request errors.
KEYTERMS_MODELS = frozenset({"saaras:v4"})
MAX_KEYTERMS = 50
MAX_KEYTERM_CHARS = 64
# Sarvam rejects the WebSocket handshake once the URL-encoded keyterms pass about 8 KB
# (16 Devanagari terms of 64 characters pass, 17 fail; a Devanagari character is 8
# bytes encoded). The limit leaves room for the other query parameters.
MAX_KEYTERMS_ENCODED_BYTES = 8000


def keyterms_encoded_size(keyterms: list[str]) -> int:
    """Return the size in bytes of the keyterms query parameter, URL-encoded."""
    return len(urlencode({"keyterms": json.dumps(keyterms)}))


def validate_keyterms(keyterms: list[str] | None, model: str) -> list[str] | None:
    """Validate user-supplied keyterms against Sarvam's limits.

    Args:
        keyterms: Terms to bias recognition toward, or ``None``.
        model: The model the terms will be sent with.

    Returns:
        The terms with duplicates removed (first occurrence kept), or ``None`` if empty.

    Raises:
        ValueError: If a term is empty or too long, there are too many distinct terms, the
            encoded list is too large for the connection URL, or the model doesn't accept
            keyterms.
    """
    if not keyterms:
        return None
    terms = list(dict.fromkeys(keyterms))
    for term in terms:
        if not isinstance(term, str) or not term.strip():
            raise ValueError("keyterms must be non-empty strings")
        if len(term) > MAX_KEYTERM_CHARS:
            raise ValueError(
                f"each keyterm must be at most {MAX_KEYTERM_CHARS} characters, got {len(term)}"
            )
    if len(terms) > MAX_KEYTERMS:
        raise ValueError(f"at most {MAX_KEYTERMS} keyterms are allowed, got {len(terms)}")
    size = keyterms_encoded_size(terms)
    if size > MAX_KEYTERMS_ENCODED_BYTES:
        raise ValueError(
            f"keyterms are {size} bytes URL-encoded, over the {MAX_KEYTERMS_ENCODED_BYTES}-byte "
            "limit of the connection URL; use fewer or shorter terms (non-Latin scripts take "
            "about 8 bytes per character)"
        )
    if model not in KEYTERMS_MODELS:
        raise ValueError(
            f"keyterms are only supported by {', '.join(sorted(KEYTERMS_MODELS))}, got {model}"
        )
    return terms


def merge_keyterms(
    user_keyterms: list[str] | None, session_keyterms: list[str], model: str
) -> list[str] | None:
    """Merge the user's keyterms with the framework-managed session keyterms.

    User terms come first and are always kept. Session terms the model can't take, that
    break Sarvam's limits, or that don't fit are dropped, so a framework update can never make
    the connection fail.
    """
    merged = list(user_keyterms or [])
    if model in KEYTERMS_MODELS:
        for term in session_keyterms:
            if len(merged) >= MAX_KEYTERMS:
                break
            if not term.strip() or len(term) > MAX_KEYTERM_CHARS or term in merged:
                continue
            if keyterms_encoded_size([*merged, term]) > MAX_KEYTERMS_ENCODED_BYTES:
                continue
            merged.append(term)
    return merged or None
