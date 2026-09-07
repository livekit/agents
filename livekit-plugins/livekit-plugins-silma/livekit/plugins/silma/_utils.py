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
from typing import Any
from urllib.parse import urlparse

import numpy as np

from livekit.agents import APIStatusError

from .models import HTTP_STREAM_PATH, MAX_TEXT_CHARACTERS, WEBSOCKET_PATH

__all__ = [
    "Float32Decoder",
    "http_stream_url",
    "normalize_base_url",
    "raise_for_status",
    "raise_ws_error",
    "split_text",
    "websocket_url",
]

# SILMA pronunciation hints, e.g. <STAG_PN>92005455</STAG_PN>. These must survive
# text splitting intact or the server sees a broken tag and reads it literally.
_STAG_BLOCK = re.compile(r"<STAG_[A-Z]+>.*?</STAG_[A-Z]+>", re.DOTALL)
_STAG_INNER = re.compile(r"<STAG_[A-Z]+>(.*?)</STAG_[A-Z]+>", re.DOTALL)

# A provider error code such as `invalid_api_key`, as opposed to a prose message
# that may quote the text being synthesized.
_ERROR_CODE = re.compile(r"[a-z][a-z0-9_.\-]{0,48}")

# 4xx codes that will never succeed on retry.
_NON_RETRYABLE_STATUS = frozenset({400, 401, 403, 404, 405, 409, 413, 422})

# WebSocket close codes that indicate a rejected credential rather than a
# transient failure.
_AUTH_CLOSE_CODES = frozenset({4001, 4003, 1008})


def normalize_base_url(base_url: str, *, allow_insecure_base_url: bool = False) -> str:
    """Strip a trailing slash and reject plaintext HTTP to non-local hosts.

    Args:
        base_url: The API root, e.g. ``https://api.silma.ai/tts/v2``.
        allow_insecure_base_url: Permit ``http://`` to an arbitrary host. Only
            set this for a trusted development network.

    Returns:
        The normalized base URL.

    Raises:
        ValueError: If a plaintext URL points at a non-local host.
    """
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


def http_stream_url(base_url: str) -> str:
    """Build the binary waveform streaming URL from an API base URL."""
    return f"{base_url.rstrip('/')}{HTTP_STREAM_PATH}"


def websocket_url(base_url: str) -> str:
    """Build the WebSocket streaming URL from an HTTP(S) or WS(S) base URL."""
    root = base_url.rstrip("/")
    if root.startswith("https://"):
        root = "wss://" + root[len("https://") :]
    elif root.startswith("http://"):
        root = "ws://" + root[len("http://") :]
    elif not root.startswith(("ws://", "wss://")):
        root = "wss://" + root
    return f"{root}{WEBSOCKET_PATH}"


class Float32Decoder:
    """Convert a stream of little-endian float32 samples into 16-bit PCM.

    SILMA emits a raw float32 waveform, while LiveKit's ``AudioEmitter``
    consumes signed 16-bit PCM. Chunk boundaries do not necessarily land on a
    4-byte sample boundary, so the trailing partial sample is buffered until the
    rest of it arrives.
    """

    def __init__(self) -> None:
        self._tail = b""

    def decode(self, data: bytes) -> bytes:
        """Convert as many whole samples as ``data`` completes.

        Returns:
            16-bit little-endian PCM bytes, possibly empty.
        """
        if self._tail:
            data = self._tail + data

        usable = len(data) - (len(data) % 4)
        self._tail = data[usable:]
        if usable == 0:
            return b""

        samples = np.frombuffer(data[:usable], dtype="<f4")
        # Clip before scaling: a sample slightly outside [-1, 1] would otherwise
        # wrap around to the opposite polarity and click audibly.
        clipped = np.clip(samples, -1.0, 1.0)
        pcm: bytes = (clipped * 32767.0).astype("<i2").tobytes()
        return pcm

    def flush(self) -> bytes:
        """Drop any trailing partial sample and reset the decoder."""
        self._tail = b""
        return b""


def split_text(text: str, *, max_characters: int = MAX_TEXT_CHARACTERS) -> list[str]:
    """Split text into chunks the API will accept.

    Splits on whitespace so words stay whole, and treats a ``<STAG_*>`` block as
    a single indivisible token. A token longer than ``max_characters`` on its own
    (a very long URL, say) is hard-split as a last resort.

    Args:
        text: The text to split.
        max_characters: The per-request character budget.

    Returns:
        A list of chunks, each at most ``max_characters`` long. Empty if the
        input holds no non-whitespace characters.
    """
    if max_characters <= 0:
        raise ValueError("max_characters must be positive")

    tokens: list[str] = []
    cursor = 0
    for match in _STAG_BLOCK.finditer(text):
        tokens.extend(text[cursor : match.start()].split())
        block = match.group(0)
        if len(block) <= max_characters:
            tokens.append(block)
        else:
            # The tag cannot fit in one request. Splitting it would leave a
            # dangling `<STAG_...>` that the model reads out literally, so drop
            # the hint and keep the content instead.
            tokens.extend(_STAG_INNER.sub(r"\1", block).split())
        cursor = match.end()
    tokens.extend(text[cursor:].split())

    chunks: list[str] = []
    current = ""
    for token in tokens:
        while len(token) > max_characters:
            # An indivisible token that cannot fit; emit what we have and carve
            # the token down to size.
            if current:
                chunks.append(current)
                current = ""
            chunks.append(token[:max_characters])
            token = token[max_characters:]

        if not current:
            current = token
        elif len(current) + 1 + len(token) <= max_characters:
            current = f"{current} {token}"
        else:
            chunks.append(current)
            current = token

    if current:
        chunks.append(current)

    return chunks


def _error_message(status_code: int) -> str:
    return f"SILMA TTS returned status {status_code}"


def raise_for_status(status_code: int, body: str) -> None:
    """Raise an ``APIStatusError`` describing a failed HTTP response.

    The response body is parsed for an error code but is deliberately not
    included in the message: it can echo the synthesized text, which may be
    end-user content.
    """
    detail: str | None = None
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        parsed = None

    if isinstance(parsed, dict):
        raw_detail = parsed.get("detail")
        if isinstance(raw_detail, str):
            detail = raw_detail
        elif isinstance(raw_detail, dict):
            code = raw_detail.get("code")
            detail = code if isinstance(code, str) else None

    error_body: dict[str, Any] = {"status_code": status_code}
    if detail is not None and _ERROR_CODE.fullmatch(detail):
        # Only machine-readable error codes are carried through. A prose
        # `detail` can quote the text that was being synthesized.
        error_body["detail"] = detail

    raise APIStatusError(
        _error_message(status_code),
        status_code=status_code,
        request_id=None,
        body=error_body,
        retryable=status_code not in _NON_RETRYABLE_STATUS,
    )


def raise_ws_error(event: dict[str, Any]) -> None:
    """Raise an ``APIStatusError`` for a ``failed`` WebSocket event."""
    code = event.get("code")
    error_body: dict[str, Any] = {"status": "failed"}
    if isinstance(code, (str, int)):
        error_body["code"] = code

    raise APIStatusError(
        "SILMA TTS reported a failed synthesis",
        status_code=500,
        request_id=None,
        body=error_body,
        retryable=True,
    )


def is_auth_close_code(close_code: int | None) -> bool:
    """Whether a WebSocket close code means the API key was rejected."""
    return close_code in _AUTH_CLOSE_CODES
