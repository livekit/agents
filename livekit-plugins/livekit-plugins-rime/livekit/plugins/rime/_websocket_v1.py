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
from typing import Any, Literal, Protocol, cast
from urllib.parse import unquote, urlsplit

import aiohttp
import numpy as np
from google.protobuf import json_format
from google.protobuf.message import DecodeError

from livekit.agents import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    tts,
)

from ._proto import websocket_v1_pb2 as proto
from .models import is_mist_model

WebSocketProtocol = Literal["binary", "json"]
RimeAudioFormat = Literal[
    "audio/wav",
    "audio/mpeg",
    "audio/ogg;codecs=opus",
    "audio/pcm",
    "audio/pcmu",
    "audio/webm;codecs=opus",
]
BINARY_SUBPROTOCOL = "rime.v1.binary"
JSON_SUBPROTOCOL = "rime.v1.json"
PROTOCOL_VERSION = 1
CANCEL_TIMEOUT = 1.0
NUM_CHANNELS = 1
DEFAULT_AUDIO_FORMAT: RimeAudioFormat = "audio/pcm"
SUPPORTED_AUDIO_FORMATS: frozenset[str] = frozenset(
    {
        "audio/wav",
        "audio/mpeg",
        "audio/ogg;codecs=opus",
        "audio/pcm",
        "audio/pcmu",
        "audio/webm;codecs=opus",
    }
)


@dataclass(frozen=True)
class SynthesisOptions:
    model: str
    speaker: str
    language: str
    audio_format: RimeAudioFormat = DEFAULT_AUDIO_FORMAT
    sampling_rate: int | None = None
    time_scale_factor: float | None = None
    pause_between_brackets: bool | None = None
    phonemize_between_brackets: bool | None = None


@dataclass(frozen=True)
class RunResult:
    reusable: bool


@dataclass(frozen=True)
class Connection:
    websocket: aiohttp.ClientWebSocketResponse
    codec: _EnvelopeCodec


def validate_audio_format(audio_format: str) -> RimeAudioFormat:
    """Return a supported canonical Rime audio MIME type."""
    if audio_format not in SUPPORTED_AUDIO_FORMATS:
        choices = ", ".join(sorted(SUPPORTED_AUDIO_FORMATS))
        raise ValueError(
            f"unsupported Rime audio_format {audio_format!r}; choose one of: {choices}"
        )
    return cast(RimeAudioFormat, audio_format)


def _build_mulaw_table() -> np.ndarray:
    """Build an ITU-T G.711 mu-law to linear 16-bit PCM lookup table."""
    table = np.zeros(256, dtype=np.int16)
    for value in range(256):
        encoded = (~value) & 0xFF
        sign = -1 if encoded & 0x80 else 1
        exponent = (encoded >> 4) & 0x07
        mantissa = encoded & 0x0F
        sample = ((mantissa << 3) + 0x84) << exponent
        table[value] = sign * (sample - 0x84)
    return table


_MULAW_TABLE = _build_mulaw_table()


def _decode_audio(audio_format: RimeAudioFormat, data: bytes) -> bytes:
    if audio_format != "audio/pcmu":
        return data
    pcm = _MULAW_TABLE[np.frombuffer(data, dtype=np.uint8)]
    return pcm.astype("<i2").tobytes()


def _emitter_mime_type(audio_format: RimeAudioFormat) -> str:
    if audio_format == "audio/pcmu":
        return "audio/pcm"
    return audio_format.partition(";")[0]


class _EnvelopeCodec(Protocol):
    """Encode and decode one semantic v1 envelope per WebSocket frame."""

    subprotocol: str

    async def send_request(
        self, websocket: aiohttp.ClientWebSocketResponse, request: proto.WebSocketRequest
    ) -> None: ...

    def decode_response(self, message: aiohttp.WSMessage) -> proto.WebSocketResponse: ...


class _BinaryEnvelopeCodec:
    subprotocol = BINARY_SUBPROTOCOL

    async def send_request(
        self, websocket: aiohttp.ClientWebSocketResponse, request: proto.WebSocketRequest
    ) -> None:
        await websocket.send_bytes(request.SerializeToString())

    def decode_response(self, message: aiohttp.WSMessage) -> proto.WebSocketResponse:
        if message.type != aiohttp.WSMsgType.BINARY:
            raise _ContaminatedConnection(
                f"Rime v1 sent unexpected WebSocket frame type {message.type}"
            )
        response = proto.WebSocketResponse()
        try:
            response.ParseFromString(message.data)
        except (DecodeError, TypeError):
            raise _ContaminatedConnection("Rime v1 sent invalid protobuf") from None
        return response


class _JsonEnvelopeCodec:
    subprotocol = JSON_SUBPROTOCOL

    async def send_request(
        self, websocket: aiohttp.ClientWebSocketResponse, request: proto.WebSocketRequest
    ) -> None:
        await websocket.send_str(
            json_format.MessageToJson(
                request,
                preserving_proto_field_name=False,
                indent=None,
            )
        )

    def decode_response(self, message: aiohttp.WSMessage) -> proto.WebSocketResponse:
        if message.type != aiohttp.WSMsgType.TEXT:
            raise _ContaminatedConnection(
                f"Rime v1 sent unexpected WebSocket frame type {message.type}"
            )
        try:
            envelope = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            raise _ContaminatedConnection("Rime v1 sent invalid JSON") from None
        _validate_json_envelope(envelope)

        response = proto.WebSocketResponse()
        try:
            json_format.ParseDict(envelope, response, ignore_unknown_fields=True)
        except (json_format.ParseError, TypeError):
            raise _ContaminatedConnection("Rime v1 sent invalid JSON") from None
        return response


_JSON_RESPONSE_PAYLOADS = ("ready", "started", "audio", "done", "cancelled", "error")


def _validate_json_envelope(envelope: Any) -> None:
    if not isinstance(envelope, dict):
        raise _ContaminatedConnection("Rime v1 envelope must be an object")

    payloads = [name for name in _JSON_RESPONSE_PAYLOADS if name in envelope]
    if len(payloads) != 1:
        raise _ContaminatedConnection("Rime v1 envelope must contain exactly one payload")

    payload = payloads[0]
    value = envelope[payload]
    if payload == "audio":
        if not isinstance(value, str):
            raise _ContaminatedConnection("Rime v1 sent a non-string audio payload")
        try:
            base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError):
            raise _ContaminatedConnection("Rime v1 sent invalid Base64 audio") from None
    elif not isinstance(value, dict):
        raise _ContaminatedConnection(f"Rime v1 sent a malformed {payload} event")


def _codec_for_protocol(protocol: WebSocketProtocol | str) -> _EnvelopeCodec:
    if protocol == "binary":
        return _BinaryEnvelopeCodec()
    if protocol == "json":
        return _JsonEnvelopeCodec()
    raise ValueError('websocket_protocol must be "binary" or "json"')


class _ContaminatedConnection(APIConnectionError):
    pass


class _ContextCancelled(asyncio.CancelledError):
    def __init__(self, *, reusable: bool) -> None:
        super().__init__()
        self.reusable = reusable


def _is_loopback_host(hostname: str | None) -> bool:
    try:
        return hostname is not None and ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _is_trusted_rime_host(hostname: str | None) -> bool:
    if hostname is None:
        return False
    normalized = hostname.rstrip(".").lower()
    return normalized == "rime.ai" or normalized.endswith(".rime.ai")


def validate_endpoint_host(endpoint_url: str, *, allow_custom_endpoint: bool = False) -> None:
    """Reject endpoints that could receive Rime credentials without caller consent."""
    parts = urlsplit(endpoint_url)
    if not parts.netloc or parts.hostname is None:
        raise ValueError("Rime endpoint must be an absolute URL")
    if (
        not allow_custom_endpoint
        and not _is_loopback_host(parts.hostname)
        and not _is_trusted_rime_host(parts.hostname)
    ):
        raise ValueError(
            "Rime endpoint must use a trusted Rime host; "
            "set allow_custom_endpoint=True to send credentials to another host"
        )


def validate_websocket_url(websocket_url: str, *, allow_custom_endpoint: bool = False) -> None:
    """Validate a caller-supplied v1 WebSocket endpoint."""
    parts = urlsplit(websocket_url)
    if parts.scheme not in ("ws", "wss") or not parts.netloc:
        raise ValueError("Rime v1 websocket_url must be an absolute ws or wss URL")
    if parts.scheme == "ws" and not _is_loopback_host(parts.hostname):
        raise ValueError("Rime v1 websocket_url must use wss unless it uses a loopback IP address")
    validate_endpoint_host(websocket_url, allow_custom_endpoint=allow_custom_endpoint)


def _model_endpoint_identity(websocket_url: str) -> tuple[str, str, int, str]:
    """Return the URL parts that identify the model behind an endpoint."""
    parts = urlsplit(websocket_url)
    assert parts.hostname is not None
    scheme = parts.scheme.lower()
    default_port = 443 if scheme == "wss" else 80
    return (
        scheme,
        parts.hostname.rstrip(".").lower(),
        parts.port or default_port,
        parts.path.rstrip("/"),
    )


def model_from_websocket_url(
    websocket_url: str, *, allow_custom_endpoint: bool = False
) -> str | None:
    """Return the model from /{model}/ws, or None for a dedicated /ws endpoint."""
    validate_websocket_url(websocket_url, allow_custom_endpoint=allow_custom_endpoint)
    path_segments = urlsplit(websocket_url).path.rstrip("/").split("/")
    if not path_segments or path_segments[-1] != "ws":
        raise ValueError("Rime v1 websocket_url path must end with /ws")
    if len(path_segments) < 2 or not path_segments[-2]:
        return None
    return unquote(path_segments[-2])


async def connect(
    session: aiohttp.ClientSession,
    *,
    websocket_url: str,
    api_key: str,
    protocol: WebSocketProtocol,
    timeout: float,
    allow_custom_endpoint: bool = False,
) -> Connection:
    """Open a v1 socket and consume its one connection-level ready event."""
    validate_websocket_url(websocket_url, allow_custom_endpoint=allow_custom_endpoint)
    codec = _codec_for_protocol(protocol)
    try:
        ws = await asyncio.wait_for(
            session.ws_connect(
                websocket_url,
                headers={"Authorization": f"Bearer {api_key}"},
                protocols=(codec.subprotocol,),
            ),
            timeout,
        )
        if ws.protocol != codec.subprotocol:
            await ws.close()
            raise APIConnectionError(
                f"Rime selected WebSocket subprotocol {ws.protocol!r}, "
                f"expected {codec.subprotocol!r}",
                retryable=False,
            )

        connection = Connection(websocket=ws, codec=codec)
        try:
            envelope = await _receive_envelope(connection, timeout=timeout)
            payload = _payload(envelope)
            if payload == "error":
                raise _rime_error(envelope.error, fallback_request_id=None)
            if payload != "ready" or envelope.context_id:
                raise _ContaminatedConnection("Rime v1 did not send a connection-level ready event")
            if envelope.ready.protocol != PROTOCOL_VERSION:
                raise _ContaminatedConnection("Rime v1 reported an unsupported protocol version")
        except BaseException:
            await ws.close()
            raise
        return connection
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


async def close(connection: Connection) -> None:
    await connection.websocket.close()


async def run_context(
    connection: Connection,
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
    emitter_mime_type = _emitter_mime_type(options.audio_format)
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
                await _send_envelope(connection, context_id, "start", _start_payload(options))
                context_started.set()
                mark_started()
            await _send_envelope(connection, context_id, "text", event)

        if state.active:
            await _send_envelope(connection, context_id, "end", None)
            input_ended.set()
        else:
            input_complete.set()
            context_started.set()
            output_emitter.initialize(
                request_id=context_id,
                sample_rate=sample_rate,
                num_channels=NUM_CHANNELS,
                mime_type=emitter_mime_type,
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
            envelope = await _receive_envelope(connection, timeout=None)
            server_activity.set()
            payload = _payload(envelope)
            if payload == "error" and not envelope.context_id:
                raise _rime_error(envelope.error, fallback_request_id=state.request_id)
            _check_context(envelope, context_id)
            if payload == "started":
                if state.server_started:
                    raise _ContaminatedConnection("Rime v1 sent started more than once")
                request_id = envelope.started.request_id
                if not request_id:
                    raise _ContaminatedConnection("Rime v1 sent a malformed started event")
                state.server_started = True
                server_started.set()
                state.request_id = request_id
                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type=emitter_mime_type,
                    stream=True,
                )
                state.emitter_initialized = True
                output_emitter.start_segment(segment_id=context_id)
            elif payload == "audio":
                if not state.server_started:
                    raise _ContaminatedConnection("Rime v1 sent audio before started")
                output_emitter.push(_decode_audio(options.audio_format, envelope.audio))
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
                raise _rime_error(envelope.error, fallback_request_id=state.request_id)
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
                mime_type=emitter_mime_type,
                stream=True,
            )
            state.emitter_initialized = True
        reusable = not state.active or state.terminal
        if not reusable:
            reusable = await _cancel_and_drain(
                connection,
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


def _start_payload(options: SynthesisOptions) -> proto.SynthesisRequest:
    audio_parameters = proto.AudioParameters(audio_format=options.audio_format)
    if options.sampling_rate is not None:
        audio_parameters.sampling_rate = options.sampling_rate
    if options.time_scale_factor is not None:
        audio_parameters.time_scale_factor = options.time_scale_factor

    request = proto.SynthesisRequest(
        speaker=options.speaker,
        language=options.language,
        text="",
        audio_parameters=audio_parameters,
    )
    if is_mist_model(options.model):
        if options.pause_between_brackets is not None:
            request.mist_parameters.pause_between_brackets = options.pause_between_brackets
        if options.phonemize_between_brackets is not None:
            request.mist_parameters.phonemize_between_brackets = options.phonemize_between_brackets
    return request


async def _send_envelope(
    connection: Connection,
    context_id: str,
    payload: str,
    value: object,
) -> None:
    request = proto.WebSocketRequest(context_id=context_id)
    if payload == "start" and isinstance(value, proto.SynthesisRequest):
        request.start.CopyFrom(value)
    elif payload == "text" and isinstance(value, str):
        request.text = value
    elif payload == "end":
        request.end.SetInParent()
    elif payload == "cancel":
        request.cancel.SetInParent()
    else:
        raise ValueError(f"unsupported Rime v1 request payload {payload!r}")
    try:
        await connection.codec.send_request(connection.websocket, request)
    except asyncio.CancelledError:
        raise
    except Exception:
        raise APIConnectionError("Failed to write to the Rime v1 WebSocket") from None


async def _receive_envelope(
    connection: Connection, *, timeout: float | None
) -> proto.WebSocketResponse:
    websocket = connection.websocket
    try:
        if timeout is None:
            message = await websocket.receive()
        else:
            message = await websocket.receive(timeout=timeout)
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
    return connection.codec.decode_response(message)


def _payload(envelope: proto.WebSocketResponse) -> str:
    payload = envelope.WhichOneof("payload")
    if payload is None:
        raise _ContaminatedConnection("Rime v1 envelope must contain exactly one payload")
    return cast(str, payload)


def _check_context(envelope: proto.WebSocketResponse, expected: str) -> None:
    if envelope.context_id != expected:
        raise _ContaminatedConnection("Rime v1 event has an unexpected contextId")


def _rime_error(error: proto.WebSocketError, *, fallback_request_id: str | None) -> APIError:
    kind = error.kind
    if not kind or not error.message:
        return APIError("Rime v1 sent a malformed error", retryable=False)
    request_id = error.request_id if error.HasField("request_id") else fallback_request_id
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
        request_id=request_id,
        body={"kind": safe_kind},
        retryable=False if kind == "unimplemented" else None,
    )


async def _cancel_and_drain(connection: Connection, *, context_id: str, timeout: float) -> bool:
    try:
        await _send_envelope(connection, context_id, "cancel", None)

        async def _drain() -> None:
            while True:
                envelope = await _receive_envelope(connection, timeout=timeout)
                payload = _payload(envelope)
                if payload == "error" and not envelope.context_id:
                    raise _rime_error(envelope.error, fallback_request_id=None)
                _check_context(envelope, context_id)
                if payload in ("cancelled", "done"):
                    return
                if payload == "error":
                    raise _rime_error(envelope.error, fallback_request_id=None)
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
