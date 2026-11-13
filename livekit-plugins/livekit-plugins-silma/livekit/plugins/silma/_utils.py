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

import numpy as np

from livekit.agents import APIStatusError

from .models import HTTP_STREAM_PATH, MAX_TEXT_CHARACTERS, WEBSOCKET_PATH

__all__ = [
    "Float32Decoder",
    "close_code_status",
    "http_stream_url",
    "is_retryable_status",
    "raise_for_status",
    "raise_ws_error",
    "split_text",
    "websocket_url",
]

# SILMA pronunciation hints, e.g. <STAG_PN>92005455</STAG_PN>. These must survive
# text splitting intact or the server sees a broken tag and reads it literally.
_STAG = r"<STAG_[A-Z]+>.*?</STAG_[A-Z]+>"
_STAG_BLOCK = re.compile(_STAG, re.DOTALL)
_STAG_INNER = re.compile(r"<STAG_[A-Z]+>(.*?)</STAG_[A-Z]+>", re.DOTALL)

# A run of text with no whitespace to break on. Whitespace inside a tag does not
# count, so a tag and any punctuation touching it stay in one unit.
_UNIT = re.compile(rf"(?:{_STAG}|\S)+", re.DOTALL)

# A provider error code such as `invalid_api_key`, as opposed to a prose message
# that may quote the text being synthesized.
_ERROR_CODE = re.compile(r"[a-z][a-z0-9_.\-]{0,48}")

# 4xx codes that will never succeed on retry.
_NON_RETRYABLE_STATUS = frozenset({400, 401, 403, 404, 405, 409, 413, 422})

# WebSocket close codes the server uses to refuse a request outright, and the
# HTTP status each one stands for. SILMA closes with 1008 after rejecting a
# payload (unknown model, invalid parameters, no credits left) and with 1011
# after an internal error; 4001 and 4003 are the usual codes for a rejected
# credential. Anything not listed here is treated as transient.
_CLOSE_CODE_STATUS = {1008: 400, 4001: 401, 4003: 403}


def http_stream_url(base_url: str) -> str:
    """Build the binary waveform streaming URL from an HTTP(S) or WS(S) base URL."""
    root = base_url.rstrip("/")
    if root.startswith("wss://"):
        root = "https://" + root[len("wss://") :]
    elif root.startswith("ws://"):
        root = "http://" + root[len("ws://") :]
    return f"{root}{HTTP_STREAM_PATH}"


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


def _unwrap_oversized_tags(text: str, max_characters: int) -> str:
    """Drop the markup of any ``<STAG_*>`` block too long for one request.

    Cutting such a block would leave a dangling ``<STAG_...>`` that the model
    reads out literally, so the hint is dropped and its content kept.
    """

    def unwrap(match: re.Match[str]) -> str:
        block = match.group(0)
        if len(block) <= max_characters:
            return block
        return _STAG_INNER.sub(r"\1", block)

    return _STAG_BLOCK.sub(unwrap, text)


def _split_unit(text: str, start: int, end: int, max_characters: int) -> list[tuple[int, int]]:
    """Cut ``text[start:end]``, which has no whitespace to break on, into spans.

    Breaks fall between a tag and the text around it where possible, so a tag
    is never cut. Plain text longer than the budget is cut at the limit.
    """
    spans: list[tuple[int, int]] = []
    span_start = start

    def take_plain(until: int) -> None:
        nonlocal span_start
        while until - span_start > max_characters:
            spans.append((span_start, span_start + max_characters))
            span_start += max_characters

    for match in _STAG_BLOCK.finditer(text, start, end):
        take_plain(match.start())
        if match.end() - span_start > max_characters:
            # The tag does not fit after what is already in this span.
            spans.append((span_start, match.start()))
            span_start = match.start()

    take_plain(end)
    if span_start < end:
        spans.append((span_start, end))
    return spans


def split_text(text: str, *, max_characters: int = MAX_TEXT_CHARACTERS) -> list[str]:
    """Split text into chunks the API will accept.

    Text that already fits is returned exactly as given. Longer text is cut at
    whitespace, and every chunk is a verbatim slice of the input: only the
    whitespace at a cut is dropped, so spacing, line breaks and the punctuation
    around a ``<STAG_*>`` block reach the server untouched. A ``<STAG_*>`` block
    is never cut, and a run of text longer than ``max_characters`` with no
    whitespace in it (a very long URL, say) is hard-split as a last resort.

    The one case where the text itself changes is a ``<STAG_*>`` block that is
    longer than ``max_characters`` on its own: its markup is dropped and its
    content kept, since half a tag would be read out literally.

    Args:
        text: The text to split.
        max_characters: The per-request character budget.

    Returns:
        A list of chunks, each at most ``max_characters`` long. Empty if the
        input holds no non-whitespace characters.
    """
    if max_characters <= 0:
        raise ValueError("max_characters must be positive")

    if not text.strip():
        return []

    text = _unwrap_oversized_tags(text, max_characters)
    if len(text) <= max_characters:
        return [text]

    chunks: list[str] = []
    # The span of the chunk being built, as offsets into `text`.
    chunk_start = chunk_end = -1
    for match in _UNIT.finditer(text):
        unit_start, unit_end = match.span()
        if chunk_start >= 0:
            if unit_end - chunk_start <= max_characters:
                chunk_end = unit_end
                continue
            chunks.append(text[chunk_start:chunk_end])

        if unit_end - unit_start <= max_characters:
            chunk_start, chunk_end = unit_start, unit_end
            continue

        # The last piece stays open so the text after it can share its request.
        *full, (chunk_start, chunk_end) = _split_unit(text, unit_start, unit_end, max_characters)
        chunks.extend(text[a:b] for a, b in full)

    if chunk_start >= 0:
        chunks.append(text[chunk_start:chunk_end])

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
        retryable=is_retryable_status(status_code),
    )


def is_retryable_status(status_code: int) -> bool:
    """Whether a request that failed with this HTTP status is worth repeating."""
    return status_code not in _NON_RETRYABLE_STATUS


def close_code_status(close_code: int | None) -> int | None:
    """The HTTP status a WebSocket close code stands for.

    Returns:
        The equivalent status when the code means the server refused the request
        or the credential, and ``None`` when it says nothing either way.
    """
    if close_code is None:
        return None
    return _CLOSE_CODE_STATUS.get(close_code)


def _ws_failure_status(event: dict[str, Any], close_code: int | None) -> int:
    """Work out the HTTP status a ``failed`` WebSocket event amounts to."""
    code = event.get("code")
    if isinstance(code, str) and code.isdigit():
        code = int(code)
    if isinstance(code, int) and not isinstance(code, bool) and 400 <= code <= 599:
        return code

    # The event names no status, so the close code that follows it is the only
    # thing that tells a refused request apart from a server-side failure.
    return close_code_status(close_code) or 500


def raise_ws_error(event: dict[str, Any], *, close_code: int | None = None) -> None:
    """Raise an ``APIStatusError`` for a ``failed`` WebSocket event.

    A request the server refused is marked non-retryable, the same way the HTTP
    path treats a 4xx: sending it again cannot succeed, and each attempt would
    cost a new connection.

    Args:
        event: The decoded ``failed`` event.
        close_code: The close code the server sent after the event, if any.
    """
    code = event.get("code")
    error_body: dict[str, Any] = {"status": "failed"}
    if isinstance(code, (str, int)):
        error_body["code"] = code
    if close_code is not None:
        error_body["close_code"] = close_code

    status_code = _ws_failure_status(event, close_code)
    raise APIStatusError(
        "SILMA TTS reported a failed synthesis",
        status_code=status_code,
        request_id=None,
        body=error_body,
        retryable=is_retryable_status(status_code),
    )
