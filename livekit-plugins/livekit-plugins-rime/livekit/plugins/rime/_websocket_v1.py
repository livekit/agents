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

import asyncio
import base64
import binascii
import ipaddress
import json
from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    tts,
)

SUBPROTOCOL = "rime.v1.json"
PROTOCOL_VERSION = 1
CANCEL_TIMEOUT = 1.0
NUM_CHANNELS = 1


@dataclass(frozen=True)
class SynthesisOptions:
    speaker: str
    language: str
    sampling_rate: int | None = None
    time_scale_factor: float | None = None


@dataclass(frozen=True)
class RunResult:
    reusable: bool


class _ContaminatedConnection(APIConnectionError):
    pass


class _ContextCancelled(asyncio.CancelledError):
    def __init__(self, *, reusable: bool) -> None:
        super().__init__()
        self.reusable = reusable


def validate_websocket_url(websocket_url: str) -> None:
    """Validate a caller-supplied v1 WebSocket endpoint."""
    parts = urlsplit(websocket_url)
    if parts.scheme not in ("ws", "wss") or not parts.netloc:
        raise ValueError("Rime v1 websocket_url must be an absolute ws or wss URL")
    if parts.scheme == "wss":
        return

    try:
        is_loopback = (
            parts.hostname is not None and ipaddress.ip_address(parts.hostname).is_loopback
        )
    except ValueError:
        is_loopback = False

    if not is_loopback:
        raise ValueError("Rime v1 websocket_url must use wss unless it uses a loopback IP address")


async def connect(
    session: aiohttp.ClientSession,
    *,
    websocket_url: str,
    api_key: str,
    timeout: float,
) -> aiohttp.ClientWebSocketResponse:
    """Open a v1 JSON socket and consume its one connection-level ready event."""
    validate_websocket_url(websocket_url)
    try:
        ws = await asyncio.wait_for(
            session.ws_connect(
                websocket_url,
                headers={"Authorization": f"Bearer {api_key}"},
                protocols=(SUBPROTOCOL,),
            ),
            timeout,
        )
        if ws.protocol != SUBPROTOCOL:
            await ws.close()
            raise APIConnectionError(
                f"Rime selected WebSocket subprotocol {ws.protocol!r}, expected {SUBPROTOCOL!r}",
                retryable=False,
            )

        try:
            envelope = await _receive_envelope(ws, timeout=timeout)
            payload = _payload(envelope)
            if payload == "error":
                raise _rime_error(envelope["error"], fallback_request_id=None)
            if payload != "ready" or envelope.get("contextId", ""):
                raise _ContaminatedConnection("Rime v1 did not send a connection-level ready event")
            ready = envelope["ready"]
            if not isinstance(ready, dict) or ready.get("protocol") != PROTOCOL_VERSION:
                raise _ContaminatedConnection("Rime v1 reported an unsupported protocol version")
        except BaseException:
            await ws.close()
            raise
        return ws
    except asyncio.TimeoutError:
        raise APITimeoutError("Timed out waiting for the Rime v1 ready event") from None
    except aiohttp.ClientResponseError as e:
        raise APIStatusError(
            message=e.message,
            status_code=e.status,
            request_id=None,
            body=None,
        ) from None
    except APIError:
        raise
    except Exception:
        raise APIConnectionError("Failed to connect to Rime v1") from None


async def close(ws: aiohttp.ClientWebSocketResponse) -> None:
    await ws.close()


async def run_context(
    ws: aiohttp.ClientWebSocketResponse,
    *,
    context_id: str,
    options: SynthesisOptions,
    sample_rate: int,
    input_events: AsyncIterable[str],
    output_emitter: tts.AudioEmitter,
    timeout: float,
    mark_started: Callable[[], None],
) -> RunResult:
    """Run one LiveKit output turn as one Rime synthesis context."""
    context_started = asyncio.Event()
    input_complete = asyncio.Event()
    input_ended = asyncio.Event()
    server_started = asyncio.Event()
    server_activity = asyncio.Event()
    terminal_received = asyncio.Event()
    state = _ContextState()

    async def _send() -> None:
        async for event in input_events:
            if not event:
                continue
            if not state.active:
                state.active = True
                await _send_envelope(ws, context_id, "start", _start_payload(options))
                context_started.set()
                mark_started()
            await _send_envelope(ws, context_id, "text", event)

        if state.active:
            await _send_envelope(ws, context_id, "end", {})
            input_ended.set()
        else:
            input_complete.set()
            context_started.set()
            output_emitter.initialize(
                request_id=context_id,
                sample_rate=sample_rate,
                num_channels=NUM_CHANNELS,
                mime_type="audio/pcm",
                stream=True,
            )
            state.emitter_initialized = True
            output_emitter.end_input()

    async def _receive() -> None:
        await context_started.wait()
        if input_complete.is_set():
            return
        while True:
            # A live context can remain silent during an input pause. Start and
            # terminal watchdogs enforce bounded waits where the protocol requires
            # progress.
            envelope = await _receive_envelope(ws, timeout=None)
            server_activity.set()
            payload = _payload(envelope)
            if payload == "error" and not envelope.get("contextId", ""):
                raise _rime_error(envelope["error"], fallback_request_id=state.request_id)
            _check_context(envelope, context_id)
            if payload == "started":
                if state.server_started:
                    raise _ContaminatedConnection("Rime v1 sent started more than once")
                started = envelope["started"]
                if not isinstance(started, dict) or not isinstance(
                    request_id := started.get("requestId"), str
                ):
                    raise _ContaminatedConnection("Rime v1 sent a malformed started event")
                state.server_started = True
                server_started.set()
                state.request_id = request_id
                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/pcm",
                    stream=True,
                )
                state.emitter_initialized = True
                output_emitter.start_segment(segment_id=context_id)
            elif payload == "audio":
                if not state.server_started:
                    raise _ContaminatedConnection("Rime v1 sent audio before started")
                audio = envelope["audio"]
                if not isinstance(audio, str):
                    raise _ContaminatedConnection("Rime v1 sent a non-string audio payload")
                try:
                    output_emitter.push(base64.b64decode(audio, validate=True))
                except (binascii.Error, ValueError) as e:
                    raise _ContaminatedConnection("Rime v1 sent invalid Base64 audio") from e
            elif payload == "done":
                if not state.server_started:
                    raise _ContaminatedConnection("Rime v1 sent done before started")
                state.terminal = True
                terminal_received.set()
                output_emitter.end_input()
                return
            elif payload == "cancelled":
                terminal_received.set()
                raise _ContaminatedConnection("Rime v1 cancelled an active context unexpectedly")
            elif payload == "error":
                state.terminal = True
                terminal_received.set()
                raise _rime_error(envelope["error"], fallback_request_id=state.request_id)
            else:
                raise _ContaminatedConnection(f"Unexpected Rime v1 event {payload!r}")

    async def _watch_started() -> None:
        await context_started.wait()
        if input_complete.is_set():
            return
        try:
            await asyncio.wait_for(server_started.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise APITimeoutError("Timed out waiting for the Rime v1 started event") from None

    async def _watch_terminal() -> None:
        await context_started.wait()
        if input_complete.is_set():
            return
        await input_ended.wait()

        while not terminal_received.is_set():
            server_activity.clear()
            if terminal_received.is_set():
                return
            try:
                await asyncio.wait_for(server_activity.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                raise APITimeoutError("Timed out waiting for a Rime v1 event after end") from None

    tasks = [
        asyncio.create_task(_send()),
        asyncio.create_task(_receive()),
        asyncio.create_task(_watch_started()),
        asyncio.create_task(_watch_terminal()),
    ]
    try:
        await asyncio.gather(*tasks)
        return RunResult(reusable=True)
    except asyncio.CancelledError:
        await _cancel_tasks(*tasks)
        if not state.emitter_initialized:
            output_emitter.initialize(
                request_id=context_id,
                sample_rate=sample_rate,
                num_channels=NUM_CHANNELS,
                mime_type="audio/pcm",
                stream=True,
            )
            state.emitter_initialized = True
        reusable = not state.active or state.terminal
        if not reusable:
            reusable = await _cancel_and_drain(
                ws,
                context_id=context_id,
                timeout=min(timeout, CANCEL_TIMEOUT),
            )
        raise _ContextCancelled(reusable=reusable) from None
    finally:
        context_started.set()
        await _cancel_tasks(*tasks)


@dataclass
class _ContextState:
    active: bool = False
    server_started: bool = False
    emitter_initialized: bool = False
    terminal: bool = False
    request_id: str | None = None


def _start_payload(options: SynthesisOptions) -> dict[str, Any]:
    audio_parameters: dict[str, Any] = {
        "audioFormat": "audio/pcm",
    }
    if options.sampling_rate is not None:
        audio_parameters["samplingRate"] = options.sampling_rate
    if options.time_scale_factor is not None:
        audio_parameters["timeScaleFactor"] = options.time_scale_factor

    return {
        "speaker": options.speaker,
        "language": options.language,
        "text": "",
        "audioParameters": audio_parameters,
    }


async def _send_envelope(
    ws: aiohttp.ClientWebSocketResponse,
    context_id: str,
    payload: str,
    value: object,
) -> None:
    data = json.dumps({"contextId": context_id, payload: value})
    try:
        await ws.send_str(data)
    except asyncio.CancelledError:
        raise
    except Exception:
        raise APIConnectionError("Failed to write to the Rime v1 WebSocket") from None


async def _receive_envelope(
    ws: aiohttp.ClientWebSocketResponse, *, timeout: float | None
) -> dict[str, Any]:
    try:
        if timeout is None:
            message = await ws.receive()
        else:
            message = await ws.receive(timeout=timeout)
    except asyncio.TimeoutError:
        raise APITimeoutError("Timed out waiting for a Rime v1 event") from None

    if message.type in (
        aiohttp.WSMsgType.CLOSE,
        aiohttp.WSMsgType.CLOSED,
        aiohttp.WSMsgType.CLOSING,
    ):
        raise _ContaminatedConnection("Rime v1 closed the WebSocket unexpectedly")
    if message.type == aiohttp.WSMsgType.ERROR:
        raise _ContaminatedConnection("Rime v1 WebSocket transport error")
    if message.type != aiohttp.WSMsgType.TEXT:
        raise _ContaminatedConnection(
            f"Rime v1 sent unexpected WebSocket frame type {message.type}"
        )
    try:
        envelope = json.loads(message.data)
    except (TypeError, json.JSONDecodeError):
        raise _ContaminatedConnection("Rime v1 sent invalid JSON") from None
    if not isinstance(envelope, dict):
        raise _ContaminatedConnection("Rime v1 envelope must be an object")
    return envelope


def _payload(envelope: dict[str, Any]) -> str:
    payloads = [
        name
        for name in ("ready", "started", "audio", "done", "cancelled", "error")
        if name in envelope
    ]
    if len(payloads) != 1:
        raise _ContaminatedConnection("Rime v1 envelope must contain exactly one payload")
    payload = payloads[0]
    if payload in ("done", "cancelled") and not isinstance(envelope[payload], dict):
        raise _ContaminatedConnection(f"Rime v1 sent a malformed {payload} event")
    return payload


def _check_context(envelope: dict[str, Any], expected: str) -> None:
    if envelope.get("contextId", "") != expected:
        raise _ContaminatedConnection("Rime v1 event has an unexpected contextId")


def _rime_error(error: object, *, fallback_request_id: str | None) -> APIError:
    if not isinstance(error, dict):
        return APIError("Rime v1 sent a malformed error", retryable=False)
    kind = error.get("kind")
    message = error.get("message")
    request_id = error.get("requestId") or fallback_request_id
    if not isinstance(kind, str) or not isinstance(message, str):
        return APIError("Rime v1 sent a malformed error", retryable=False)
    status_codes = {
        "invalid_input": 400,
        "unauthenticated": 401,
        "permission_denied": 403,
        "not_found": 404,
        "resource_exhausted": 429,
        "timeout": 504,
        "unavailable": 503,
        "unimplemented": 501,
        "internal": 500,
    }
    safe_kind = kind if kind in status_codes else "unknown"
    return APIStatusError(
        message="Rime v1 request failed",
        status_code=status_codes.get(kind, 500),
        request_id=request_id if isinstance(request_id, str) else None,
        body={"kind": safe_kind},
    )


async def _cancel_and_drain(
    ws: aiohttp.ClientWebSocketResponse, *, context_id: str, timeout: float
) -> bool:
    try:
        await _send_envelope(ws, context_id, "cancel", {})

        async def _drain() -> None:
            while True:
                envelope = await _receive_envelope(ws, timeout=timeout)
                payload = _payload(envelope)
                if payload == "error" and not envelope.get("contextId", ""):
                    raise _rime_error(envelope["error"], fallback_request_id=None)
                _check_context(envelope, context_id)
                if payload in ("cancelled", "done"):
                    return
                if payload == "error":
                    raise _rime_error(envelope["error"], fallback_request_id=None)
                if payload not in ("started", "audio"):
                    raise _ContaminatedConnection(
                        f"Unexpected Rime v1 event while cancelling: {payload!r}"
                    )

        await asyncio.wait_for(_drain(), timeout=timeout)
        return True
    except (Exception, asyncio.CancelledError):
        return False


async def _cancel_tasks(*tasks: asyncio.Task[None]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
