"""One committed PCM16 turn per socket; adapted from pipecat-oruk 0.1.0rc3.

Copyright (c) 2026 oruk labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import re
import sys
from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import aiohttp

if sys.version_info >= (3, 11):
    from asyncio import timeout
else:
    from async_timeout import timeout

SAMPLE_RATE = 16_000
BYTES_PER_SECOND = SAMPLE_RATE * 2
MODEL = "oruk-realtime"
ENDPOINT = "wss://speech-api.oruk.ai/v1/realtime?model=oruk-realtime"
TRANSCRIPT = "conversation.item.input_audio_transcription."
EMOTION = "conversation.item.input_audio_emotion."
EventHandler = Callable[[dict[str, Any]], None]


class RealtimeError(Exception):
    def __init__(self, code: str, *, retryable: bool = False, status: int | None = None):
        # Codes are safe to log; arbitrary provider bodies, URLs and credentials are not.
        self.code = (
            code
            if isinstance(code, str) and re.fullmatch(r"[a-zA-Z0-9_.-]{1,100}", code)
            else "realtime_error"
        )
        self.retryable = retryable
        self.status = status
        super().__init__(self.code)


@dataclass(frozen=True)
class RealtimeOptions:
    """Oruk session settings and finite per-turn audio/completion limits."""

    language: str = "auto"
    phrase_emotions: bool = True
    diarize: bool = False
    phrase_silence_ms: int = 600
    phrase_max_ms: int = 8_000
    finish_timeout: float = 20.0
    max_turn_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not isinstance(self.language, str) or not re.fullmatch(
            r"[a-zA-Z-]{2,20}", self.language
        ):
            raise ValueError("language must be an Oruk locale or 'auto'")
        for name in ("phrase_emotions", "diarize"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a bool")
        for name, low, high in (
            ("phrase_silence_ms", 200, 2_000),
            ("phrase_max_ms", 1_000, 15_000),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        for name, high in (("finish_timeout", 120), ("max_turn_seconds", 590)):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= high:
                raise ValueError(f"{name} must be positive and at most {high}")

    def session(self) -> dict[str, Any]:
        return {
            "model": MODEL,
            "language": self.language,
            "sample_rate": SAMPLE_RATE,
            "word_timestamps": False,
            "phrase_emotions": self.phrase_emotions,
            "phrase_silence_ms": self.phrase_silence_ms,
            "phrase_max_ms": self.phrase_max_ms,
            "diarize": self.diarize,
        }


@dataclass
class TurnResult:
    """A cleanly closed turn, including original phrase events and provider usage."""

    request_id: str
    transcript: str = ""
    phrases: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    audio_seconds: float = 0.0


def validate_endpoint(endpoint: str) -> None:
    """Require secure WebSockets, credential-free URLs and model-only query parameters."""
    parsed = urlsplit(endpoint)
    local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.username or parsed.password or parsed.fragment or not parsed.hostname:
        raise ValueError("endpoint must not contain credentials or a fragment")
    if parsed.scheme != "wss" and not (parsed.scheme == "ws" and local):
        raise ValueError("endpoint requires wss (ws is allowed only on loopback for tests)")
    # Only model belongs in the URL. Credentials always use the upgrade header.
    from urllib.parse import parse_qsl

    if any(k != "model" for k, _ in parse_qsl(parsed.query)):
        raise ValueError("endpoint query may contain only model")


def new_http_session() -> aiohttp.ClientSession:
    """Create a reusable client that refuses redirects on authenticated upgrades."""
    trace = aiohttp.TraceConfig()

    async def reject_redirect(*_: Any) -> None:
        raise RealtimeError("websocket_redirect_refused")

    trace.on_request_redirect.append(reject_redirect)
    return aiohttp.ClientSession(trace_configs=[trace], timeout=aiohttp.ClientTimeout(total=None))


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_phrase(event: dict[str, Any]) -> None:
    """Check phrase identity, timing and finite scores without relabeling the output."""
    if not isinstance(event.get("phrase_id"), str) or not event["phrase_id"]:
        raise RealtimeError("invalid_phrase_id")
    if not _number(event.get("start")) or not _number(event.get("end")):
        raise RealtimeError("invalid_phrase_time")
    if not 0 <= event["start"] <= event["end"]:
        raise RealtimeError("invalid_phrase_time")
    if event.get("speaker") is not None and not isinstance(event["speaker"], str):
        raise RealtimeError("invalid_phrase_speaker")
    if event["type"] == EMOTION + "completed":
        if not isinstance(event.get("text"), str) or not isinstance(event.get("emotions"), list):
            raise RealtimeError("invalid_phrase_payload")
        for score in event["emotions"]:
            if (
                not isinstance(score, dict)
                or not isinstance(score.get("label"), str)
                or not _number(score.get("score"))
                or not 0 <= score["score"] <= 1
            ):
                raise RealtimeError("invalid_phrase_score")


async def run_turn(
    *,
    session: aiohttp.ClientSession,
    api_key: str,
    endpoint: str,
    audio: AsyncIterable[bytes],
    request_id: str,
    options: RealtimeOptions,
    on_event: EventHandler,
    connect_timeout: float = 10.0,
    max_connect_retries: int = 3,
    retry_interval: float = 2.0,
) -> TurnResult:
    """Stream an utterance; accept completion only after final text, usage and clean close.

    The callback receives provisional transcript and independently timed phrase events.
    A final transcript may arrive before late emotion events and the usage receipt.
    """
    validate_endpoint(endpoint)
    if not api_key or any(c.isspace() for c in api_key):
        raise ValueError("Provide a nonempty ORUK_API_KEY without whitespace")
    if not math.isfinite(connect_timeout) or connect_timeout <= 0:
        raise ValueError("connect_timeout must be positive")
    if max_connect_retries < 0 or not math.isfinite(retry_interval) or retry_interval < 0:
        raise ValueError("invalid connection retry options")
    sent = 0

    async def attempt() -> TurnResult:
        nonlocal sent
        result = TurnResult(request_id=request_id)
        committed = asyncio.Event()
        seen_final = False
        seen_usage = False
        phrases: dict[str, dict[str, Any]] = {}

        async with timeout(connect_timeout):
            socket = await session.ws_connect(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}", "X-Request-ID": request_id},
                protocols=["oruk-realtime"],
                heartbeat=15,
                max_msg_size=1_048_576,
            )
            try:

                async def receive() -> dict[str, Any] | None:
                    message = await socket.receive()
                    if message.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                        return None
                    if message.type != aiohttp.WSMsgType.TEXT:
                        raise RealtimeError("invalid_websocket_message")
                    try:
                        event = json.loads(message.data)
                    except (ValueError, TypeError):
                        raise RealtimeError("invalid_event_json") from None
                    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                        raise RealtimeError("invalid_event")
                    if event.get("request_id", result.request_id) != result.request_id:
                        raise RealtimeError("request_id_mismatch")
                    if event["type"] == "error":
                        error = event.get("error")
                        code = (
                            error.get("code", "realtime_error")
                            if isinstance(error, dict)
                            else "realtime_error"
                        )
                        if not isinstance(code, str) or not re.fullmatch(
                            r"[a-zA-Z0-9_.-]{1,100}", code
                        ):
                            code = "realtime_error"
                        raise RealtimeError(
                            code,
                            retryable=sent == 0
                            and code
                            in {
                                "realtime_upstream_unavailable",
                                "rate_limit_exceeded",
                                "rate_limited",
                            },
                        )
                    return event

                created = await receive()
                if created is None or created["type"] != "session.created":
                    raise RealtimeError("missing_session_created")
                server_id = created.get("request_id", request_id)
                if not isinstance(server_id, str) or not re.fullmatch(r"[\w.-]{1,128}", server_id):
                    raise RealtimeError("invalid_request_id")
                result.request_id = server_id
                if server_id != request_id:
                    raise RealtimeError("request_id_mismatch")
                await socket.send_json({"type": "session.update", "session": options.session()})
                updated = await receive()
                if updated is None or updated["type"] != "session.updated":
                    raise RealtimeError("missing_session_updated")
            except BaseException:
                await socket.close()
                raise

        async def send_audio() -> None:
            nonlocal sent
            async for pcm in audio:
                if not isinstance(pcm, bytes) or len(pcm) % 2:
                    raise RealtimeError("invalid_pcm16")
                if sent + len(pcm) > options.max_turn_seconds * BYTES_PER_SECOND:
                    raise RealtimeError("turn_too_long")
                for offset in range(0, len(pcm), 10_240):
                    chunk = pcm[offset : offset + 10_240]
                    # A failed write has ambiguous delivery: do not replay it.
                    sent += len(chunk)
                    await socket.send_bytes(chunk)
            if not sent:
                raise RealtimeError("empty_audio")
            committed.set()
            await socket.send_json({"type": "input_audio_buffer.commit"})

        async def receive_events() -> TurnResult:
            nonlocal seen_final, seen_usage
            metadata_bytes = 0
            transcript_chars = 0
            while True:
                event = await receive()
                if event is None:
                    if socket.close_code != 1000 or not seen_final or not seen_usage:
                        raise RealtimeError("incomplete_turn")
                    result.phrases = sorted(
                        phrases.values(), key=lambda p: (p["start"], p["phrase_id"])
                    )
                    result.audio_seconds = sent / BYTES_PER_SECOND
                    return result
                kind = event["type"]
                if event.get("request_id", result.request_id) != result.request_id:
                    raise RealtimeError("request_id_mismatch")
                event["request_id"] = result.request_id
                if kind == TRANSCRIPT + "delta":
                    if seen_final or not isinstance(event.get("delta"), str):
                        raise RealtimeError("invalid_transcript_delta")
                    transcript_chars += len(event["delta"])
                    if transcript_chars > 32_768:
                        raise RealtimeError("transcript_limit")
                elif kind == TRANSCRIPT + "completed":
                    if not committed.is_set() or not isinstance(event.get("transcript"), str):
                        raise RealtimeError("invalid_final_transcript")
                    if seen_final:
                        if event["transcript"] == result.transcript:
                            continue
                        raise RealtimeError("conflicting_final_transcript")
                    seen_final = True
                    result.transcript = event["transcript"]
                    if len(result.transcript) > 32_768:
                        raise RealtimeError("transcript_limit")
                elif kind in (EMOTION + "completed", EMOTION + "failed"):
                    validate_phrase(event)
                    previous = phrases.get(event["phrase_id"])
                    if previous is not None:
                        if previous == event:
                            continue
                        raise RealtimeError("conflicting_phrase_event")
                    metadata_bytes += len(json.dumps(event).encode())
                    if len(phrases) >= 256 or metadata_bytes > 262_144:
                        raise RealtimeError("phrase_metadata_limit")
                    phrases[event["phrase_id"]] = copy.deepcopy(event)
                elif kind == "session.usage":
                    usage = event.get("usage")
                    if (
                        not committed.is_set()
                        or not seen_final
                        or not isinstance(usage, dict)
                        or not _number(usage.get("audio_seconds"))
                        or usage["audio_seconds"] < 0
                    ):
                        raise RealtimeError("invalid_usage")
                    if seen_usage:
                        if usage == result.usage:
                            continue
                        raise RealtimeError("conflicting_usage")
                    if len(json.dumps(usage).encode()) > 65_536:
                        raise RealtimeError("usage_metadata_limit")
                    seen_usage = True
                    result.usage = copy.deepcopy(usage)
                else:
                    # Speaker boundary and future auxiliary events remain separate from text.
                    if not kind.startswith("conversation.item.input_audio_speaker."):
                        continue
                on_event(copy.deepcopy(event))

        sender = asyncio.create_task(send_audio())
        receiver = asyncio.create_task(receive_events())

        async def finish_deadline() -> None:
            await committed.wait()
            await asyncio.wait_for(asyncio.shield(receiver), options.finish_timeout)

        deadline = asyncio.create_task(finish_deadline())
        tasks = [sender, receiver, deadline]
        try:
            async with timeout(options.max_turn_seconds + options.finish_timeout):
                await asyncio.gather(*tasks)
            return receiver.result()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await socket.close()

    for attempt_index in range(max_connect_retries + 1):
        try:
            return await attempt()
        except RealtimeError as exc:
            error = exc
        except aiohttp.WSServerHandshakeError as exc:
            error = RealtimeError(
                "websocket_upgrade_failed",
                status=exc.status,
                retryable=sent == 0 and (exc.status == 429 or exc.status >= 500),
            )
        except (aiohttp.ClientError, OSError, TimeoutError):
            error = RealtimeError("realtime_connection_failed", retryable=sent == 0)
        if sent or not error.retryable or attempt_index == max_connect_retries:
            # Frameworks must not retry a consumed input iterator.
            error.retryable = False
            raise error from None
        await asyncio.sleep(min(retry_interval * (2**attempt_index), 10))
    raise AssertionError("unreachable")
