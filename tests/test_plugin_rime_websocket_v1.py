"""Tests for the Rime WebSocket v1 protocol and LiveKit adapter."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import traceback
from typing import Any, cast

import aiohttp
import av
import numpy as np
import pytest
from aiohttp import web
from google.protobuf import json_format

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
)
from livekit.plugins.rime._proto import websocket_v1_pb2 as proto

pytestmark = pytest.mark.unit


_PCM = b"\x01\x00" * 2205
_SECRET = "customer-secret-marker"


def _encode_container_audio(container_format: str, codec: str) -> bytes:
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format=container_format) as container:
        stream = container.add_stream(codec, rate=24000)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(
            np.zeros((1, 2400), dtype=np.int16), format="s16", layout="mono"
        )
        frame.sample_rate = 24000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return buffer.getvalue()


@pytest.fixture(scope="module")
def encoded_audio_by_format() -> dict[str, bytes]:
    return {
        "audio/pcm": _PCM,
        "audio/wav": _encode_container_audio("wav", "pcm_s16le"),
        "audio/mpeg": _encode_container_audio("mp3", "mp3"),
        "audio/ogg;codecs=opus": _encode_container_audio("ogg", "libopus"),
        "audio/webm;codecs=opus": _encode_container_audio("webm", "libopus"),
        "audio/pcmu": b"\xff" * 2400,
    }


def _assert_exception_is_safe(exc: BaseException) -> None:
    rendered = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    assert _SECRET not in str(exc)
    assert _SECRET not in repr(exc)
    assert _SECRET not in rendered


class _RimeV1Server:
    def __init__(
        self,
        *,
        response_mode: str = "normal",
        ready: dict[str, Any] | None = None,
        cancel_reply: bool = True,
        fail_before_audio: int = 0,
        error_kind: str = "invalid_input",
        audio: bytes = _PCM,
    ) -> None:
        self.response_mode = response_mode
        self.ready = ready or {"protocol": 1, "languages": ["eng"]}
        self.cancel_reply = cancel_reply
        self.fail_before_audio = fail_before_audio
        self.error_kind = error_kind
        self.audio = audio
        self.connections = 0
        self.ready_events = 0
        self.requests: list[dict[str, Any]] = []
        self.request_connections: list[int] = []
        self.headers: list[dict[str, str]] = []
        self.protocols: list[str | None] = []
        self.paths: list[str] = []
        self.text_messages_received = 0
        self._text_received_condition = asyncio.Condition()
        self.request_received = asyncio.Event()
        self.unexpected_requests: list[dict[str, Any]] = []
        self.connection_opened = asyncio.Event()
        self.connection_closed = asyncio.Event()
        self.closed_connections = 0

    async def __aenter__(self) -> _RimeV1Server:
        app = web.Application()
        app.router.add_get("/{path:.*}", self._handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await self._site.start()
        port = self._runner.addresses[0][1]
        self.websocket_url = f"ws://127.0.0.1:{port}/coda/ws"
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.session.close()
        await self._runner.cleanup()
        assert not self.unexpected_requests, (
            f"fake Rime v1 server received unexpected requests: {self.unexpected_requests!r}"
        )

    async def wait_for_text_messages(self, count: int) -> None:
        async with self._text_received_condition:
            await self._text_received_condition.wait_for(
                lambda: self.text_messages_received >= count
            )

    async def _handle(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(protocols=("rime.v1.binary", "rime.v1.json"))
        await ws.prepare(request)
        connection = self.connections
        self.connections += 1
        self.headers.append(dict(request.headers))
        self.protocols.append(ws.ws_protocol)
        self.paths.append(request.path)
        await self._send(ws, {"ready": self.ready})
        self.ready_events += 1
        self.connection_opened.set()

        try:
            async for message in ws:
                envelope = self._decode_request(message)
                self.requests.append(envelope)
                self.request_connections.append(connection)
                self.request_received.set()
                context_id = envelope.get("contextId", "")
                if "start" in envelope:
                    started = (
                        {}
                        if self.response_mode == "missing_started_request_id"
                        else {"requestId": f"request-{connection}"}
                    )
                    await self._send(
                        ws,
                        {
                            "contextId": context_id,
                            "started": started,
                        },
                    )
                elif "text" in envelope:
                    async with self._text_received_condition:
                        self.text_messages_received += 1
                        self._text_received_condition.notify_all()
                    if self.fail_before_audio > 0:
                        self.fail_before_audio -= 1
                        await self._send(
                            ws,
                            {
                                "contextId": context_id,
                                "error": {
                                    "kind": "unavailable",
                                    "message": "retry later",
                                    "requestId": f"request-{connection}",
                                },
                            },
                        )
                    else:
                        await self._respond_to_text(ws, context_id)
                elif "end" in envelope:
                    if self.response_mode == "malformed_done":
                        await self._send_missing_payload(ws, context_id)
                    elif self.response_mode == "invalid_done_type":
                        await ws.send_json({"contextId": context_id, "done": "bad"})
                    elif self.response_mode not in ("error", "partial_error", "no_done"):
                        await self._send(ws, {"contextId": context_id, "done": {}})
                elif "cancel" in envelope and self.cancel_reply:
                    if self.response_mode == "malformed_cancelled":
                        await self._send_missing_payload(ws, context_id)
                    else:
                        await self._send(ws, {"contextId": context_id, "cancelled": {}})
                elif "flush" in envelope:
                    self.unexpected_requests.append(envelope)
                    await self._send(
                        ws,
                        {
                            "contextId": context_id,
                            "error": {
                                "kind": "invalid_input",
                                "message": "flush is not part of the Rime v1 protocol",
                            },
                        },
                    )
        finally:
            self.closed_connections += 1
            self.connection_closed.set()
        return ws

    def _decode_request(self, message: aiohttp.WSMessage) -> dict[str, Any]:
        request = proto.WebSocketRequest()
        if message.type == aiohttp.WSMsgType.BINARY:
            request.ParseFromString(message.data)
        elif message.type == aiohttp.WSMsgType.TEXT:
            json_format.Parse(message.data, request, ignore_unknown_fields=True)
        else:
            raise AssertionError(f"unexpected request frame type: {message.type}")
        return json_format.MessageToDict(
            request,
            preserving_proto_field_name=False,
            always_print_fields_with_no_presence=True,
        )

    async def _send(self, ws: web.WebSocketResponse, payload: dict[str, Any]) -> None:
        response = proto.WebSocketResponse()
        json_format.ParseDict(payload, response, ignore_unknown_fields=True)
        if ws.ws_protocol == "rime.v1.binary":
            await ws.send_bytes(response.SerializeToString())
        else:
            await ws.send_json(payload)

    async def _send_missing_payload(self, ws: web.WebSocketResponse, context_id: str) -> None:
        await self._send(ws, {"contextId": context_id})

    async def _respond_to_text(self, ws: web.WebSocketResponse, context_id: str) -> None:
        if self.response_mode == "normal":
            await self._send(
                ws, {"contextId": context_id, "audio": base64.b64encode(self.audio).decode()}
            )
        elif self.response_mode == "wrong_context":
            await self._send(ws, {"contextId": "wrong", "done": {}})
        elif self.response_mode == "invalid_envelope":
            if ws.ws_protocol == "rime.v1.binary":
                await ws.send_bytes(b"\xff")
            else:
                await ws.send_str("{")
        elif self.response_mode == "wrong_frame":
            if ws.ws_protocol == "rime.v1.binary":
                await ws.send_str("{}")
            else:
                await ws.send_bytes(b"")
        elif self.response_mode == "invalid_base64":
            await ws.send_json({"contextId": context_id, "audio": "AQI=%%%"})
        elif self.response_mode == "error":
            await self._send(
                ws,
                {
                    "contextId": context_id,
                    "error": {"kind": self.error_kind, "message": "bad input"},
                },
            )
        elif self.response_mode == "connection_error":
            await self._send(
                ws, {"error": {"kind": self.error_kind, "message": "connection failed"}}
            )
        elif self.response_mode == "partial_error":
            await self._send(
                ws, {"contextId": context_id, "audio": base64.b64encode(_PCM).decode()}
            )
            await asyncio.sleep(0.05)
            await self._send(
                ws,
                {
                    "contextId": context_id,
                    "error": {"kind": "unavailable", "message": "failed late"},
                },
            )


def _v1_tts(server: _RimeV1Server, *, endpoint_model: str = "coda", **kwargs: Any):
    from livekit.plugins.rime import TTS

    websocket_url = server.websocket_url.replace("/coda/ws", f"/{endpoint_model}/ws")
    return TTS(
        api_key="test-key",
        websocket_url=websocket_url,
        http_session=server.session,
        **kwargs,
    )


async def _collect(stream) -> list:
    events = []
    async for event in stream:
        events.append(event)
    return events


def _payloads(server: _RimeV1Server) -> list[str]:
    assert all("flush" not in request for request in server.requests)
    return [next(key for key in request if key != "contextId") for request in server.requests]


def test_v1_binary_envelope_goldens_match_rime_field_numbers() -> None:
    request = proto.WebSocketRequest(context_id="turn-42", text="hello")
    response = proto.WebSocketResponse(context_id="turn-42", audio=b"\x01\x02")

    assert request.SerializeToString() == b"\x0a\x07turn-42\x22\x05hello"
    assert response.SerializeToString() == b"\x0a\x07turn-42\x22\x02\x01\x02"


def test_v1_decodes_pcmu_to_little_endian_pcm16() -> None:
    from livekit.plugins.rime import _websocket_v1

    decoded = _websocket_v1._decode_audio("audio/pcmu", bytes([0xFF, 0x7F, 0x80, 0x00]))

    assert np.frombuffer(decoded, dtype="<i2").tolist() == [0, 0, 32124, -32124]


@pytest.mark.parametrize("websocket_protocol", ["binary", "json"])
async def test_v1_streams_audio_before_end_and_maps_supported_start_options(
    websocket_protocol: str,
) -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(
            server,
            websocket_protocol=websocket_protocol,
            lang="eng",
            sample_rate=22050,
            time_scale_factor=1.2,
        )
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        try:
            first_event = asyncio.create_task(anext(stream))
            stream.push_text("Hello from ")
            stream.push_text("LiveKit today. Next")
            first = await asyncio.wait_for(first_event, timeout=2)
            assert first.frame.data
            assert first.frame.sample_rate == 22050

            stream.push_text(" sentence.")
            stream.end_input()
            remaining = await _collect(stream)
        finally:
            await stream.aclose()
            await tts.aclose()

    assert remaining[-1].is_final
    assert server.paths == ["/coda/ws"]
    assert server.headers[0]["Authorization"] == "Bearer test-key"
    assert server.headers[0]["Sec-WebSocket-Protocol"] == f"rime.v1.{websocket_protocol}"
    assert server.protocols == [f"rime.v1.{websocket_protocol}"]
    assert _payloads(server) == ["start", "text", "text", "end"]
    assert [request["text"] for request in server.requests if "text" in request] == [
        "Hello from LiveKit today. ",
        "Next sentence. ",
    ]
    start = server.requests[0]["start"]
    assert start == {
        "speaker": "astra",
        "language": "eng",
        "text": "",
        "audioParameters": {
            "audioFormat": "audio/pcm",
            "samplingRate": 22050,
            "timeScaleFactor": 1.2,
        },
    }


async def test_v1_defaults_to_binary_protocol() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("hello")
        stream.end_input()
        await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert server.protocols == ["rime.v1.binary"]


@pytest.mark.parametrize("websocket_protocol", ["binary", "json"])
@pytest.mark.parametrize(
    "audio_format",
    [
        "audio/pcm",
        "audio/wav",
        "audio/mpeg",
        "audio/ogg;codecs=opus",
        "audio/webm;codecs=opus",
        "audio/pcmu",
    ],
)
async def test_v1_supports_each_rime_audio_format_with_each_protocol(
    websocket_protocol: str,
    audio_format: str,
    encoded_audio_by_format: dict[str, bytes],
) -> None:
    async with _RimeV1Server(audio=encoded_audio_by_format[audio_format]) as server:
        tts = _v1_tts(
            server,
            websocket_protocol=websocket_protocol,
            audio_format=audio_format,
        )
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        try:
            stream.push_text("hello")
            stream.end_input()
            events = await _collect(stream)
        finally:
            await stream.aclose()
            await tts.aclose()

    assert any(event.frame.data for event in events)
    assert events[-1].is_final
    assert server.requests[0]["start"]["audioParameters"]["audioFormat"] == audio_format
    assert server.protocols == [f"rime.v1.{websocket_protocol}"]


async def test_v1_maps_mist_options_without_a_second_stream_implementation() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(
            server,
            endpoint_model="mist",
            pause_between_brackets=True,
            phonemize_between_brackets=False,
        )
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("hello")
        stream.end_input()
        await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    start = server.requests[0]["start"]
    assert start["mistParameters"] == {
        "pauseBetweenBrackets": True,
        "phonemizeBetweenBrackets": False,
        "inlineTimeScaleFactors": [],
    }


@pytest.mark.parametrize(
    ("endpoint_model", "expected_sample_rate"),
    [("coda", 24000), ("mist", 24000), ("mistv2", 22050), ("future-model", 22050)],
)
async def test_v1_sends_resolved_sample_rate(
    endpoint_model: str, expected_sample_rate: int
) -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server, endpoint_model=endpoint_model)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        try:
            stream.push_text("hello")
            stream.end_input()
            events = await _collect(stream)
        finally:
            await stream.aclose()
            await tts.aclose()

    assert tts.sample_rate == expected_sample_rate
    assert events[0].frame.sample_rate == expected_sample_rate
    assert server.requests[0]["start"]["audioParameters"] == {
        "audioFormat": "audio/pcm",
        "samplingRate": expected_sample_rate,
    }


async def test_v1_stream_keeps_sample_rate_after_parent_update() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server, sample_rate=22050)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        tts.update_options(sample_rate=16000)
        try:
            stream.push_text("hello")
            stream.end_input()
            events = await _collect(stream)
        finally:
            await stream.aclose()
            await tts.aclose()

    assert server.requests[0]["start"]["audioParameters"]["samplingRate"] == 22050
    assert events[0].frame.sample_rate == 22050


async def test_v1_rejects_wrong_ready_protocol() -> None:
    async with _RimeV1Server(ready={"protocol": 2}) as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("hello")
        stream.end_input()
        with pytest.raises(APIConnectionError, match="unsupported protocol"):
            await _collect(stream)
        await stream.aclose()
        await tts.aclose()


async def test_v1_buffers_fragments_into_complete_sentences() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("This is the first sentence. ")
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(server.wait_for_text_messages(1), timeout=0.05)

        stream.push_text("Second")
        await asyncio.wait_for(server.wait_for_text_messages(1), timeout=2)
        stream.end_input()
        await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert [request["text"] for request in server.requests if "text" in request] == [
        "This is the first sentence. ",
        "Second ",
    ]


async def test_v1_default_tokenizer_preserves_fragment_boundaries() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        try:
            stream.push_text("The price is 1.")
            stream.push_text("7 dollars. Hel")
            stream.push_text("lo world. Next")
            stream.push_text(" sentence.")
            stream.end_input()
            await _collect(stream)
        finally:
            await stream.aclose()
            await tts.aclose()

    assert [request["text"] for request in server.requests if "text" in request] == [
        "The price is 1.7 dollars. ",
        "Hello world. ",
        "Next sentence. ",
    ]


async def test_v1_uses_custom_tokenizer_behavior() -> None:
    from livekit.agents import tokenize

    async with _RimeV1Server() as server:
        tokenizer = tokenize.basic.SentenceTokenizer(
            min_sentence_len=1000,
            stream_context_len=1,
        )
        tts = _v1_tts(server, tokenizer=tokenizer)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        try:
            stream.push_text("First sentence. Second sentence.")
            stream.end_input()
            await _collect(stream)
        finally:
            await stream.aclose()
            await tts.aclose()

    assert [request["text"] for request in server.requests if "text" in request] == [
        "First sentence. Second sentence. "
    ]


async def test_v1_local_flush_drains_text_before_end() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("first")
        stream.flush()
        stream.end_input()
        await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert _payloads(server) == ["start", "text", "end"]


async def test_v1_local_flush_sends_text_without_control_message() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("first")
        stream.flush()
        try:
            await asyncio.wait_for(server.wait_for_text_messages(1), timeout=2)
        finally:
            stream.end_input()
            await _collect(stream)
            await stream.aclose()
            await tts.aclose()

    assert _payloads(server) == ["start", "text", "end"]


async def test_v1_resumes_context_after_local_flush() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        metrics = []
        tts.on("metrics_collected", metrics.append)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("first")
        stream.flush()
        await asyncio.wait_for(server.wait_for_text_messages(1), timeout=2)
        stream.push_text("second")
        stream.end_input()
        events = await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert _payloads(server) == ["start", "text", "text", "end"]
    assert [request["text"] for request in server.requests if "text" in request] == [
        "first ",
        "second ",
    ]
    assert len({request["contextId"] for request in server.requests}) == 1
    assert sum(event.is_final for event in events) == 1
    assert len(metrics) == 1
    assert metrics[0].characters_count == len("firstsecond")


async def test_v1_input_pause_can_exceed_api_timeout() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.1))
        stream.push_text("First sentence.")
        stream.flush()
        await asyncio.wait_for(server.wait_for_text_messages(1), timeout=2)

        await asyncio.sleep(0.2)

        stream.push_text("Second sentence.")
        stream.end_input()
        events = await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert _payloads(server) == ["start", "text", "text", "end"]
    assert len({request["contextId"] for request in server.requests}) == 1
    assert events[-1].is_final


async def test_v1_flush_before_first_text_sends_no_rime_message() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.flush()
        await asyncio.wait_for(server.connection_opened.wait(), timeout=2)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(server.request_received.wait(), timeout=0.05)

        stream.push_text("first")
        stream.end_input()
        await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert _payloads(server) == ["start", "text", "end"]


async def test_v1_end_input_drains_buffered_final_text_before_end() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("buffered final fragment")
        stream.end_input()
        await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert _payloads(server) == ["start", "text", "end"]
    assert server.requests[1]["text"] == "buffered final fragment "


async def test_v1_reuses_socket_and_does_not_start_empty_context() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        for text in ("one", "two"):
            stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
            stream.push_text(text)
            stream.end_input()
            await _collect(stream)
            await stream.aclose()

        empty = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        empty.end_input()
        assert await _collect(empty) == []
        await empty.aclose()
        await tts.aclose()

    assert server.connections == 1
    assert server.ready_events == 1
    assert _payloads(server).count("start") == 2


async def test_v1_overlapping_streams_use_separate_connections() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        first = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        first.push_text("First stream stays active. Pending")
        await asyncio.wait_for(server.wait_for_text_messages(1), timeout=2)

        second = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        second.push_text("Second stream completes.")
        second.end_input()
        events = await _collect(second)

        await first.aclose()
        await second.aclose()
        await tts.aclose()

    contexts_by_connection: dict[int, set[str]] = {}
    for connection, request in zip(server.request_connections, server.requests, strict=True):
        contexts_by_connection.setdefault(connection, set()).add(request["contextId"])

    assert server.connections == 2
    assert len(contexts_by_connection) == 2
    assert all(len(contexts) == 1 for contexts in contexts_by_connection.values())
    assert events[-1].is_final


async def test_v1_stream_snapshots_options() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        tts.update_options(speaker="changed")
        stream.push_text("hello")
        stream.end_input()
        await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert server.requests[0]["start"]["speaker"] == "astra"


async def test_v1_stream_snapshots_websocket_url() -> None:
    async with _RimeV1Server() as first_server, _RimeV1Server() as second_server:
        tts = _v1_tts(first_server)
        first_stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        tts.update_options(websocket_url=second_server.websocket_url)
        second_stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))

        first_stream.push_text("first")
        first_stream.end_input()
        second_stream.push_text("second")
        second_stream.end_input()
        await _collect(first_stream)
        await _collect(second_stream)
        await first_stream.aclose()
        await second_stream.aclose()
        await tts.aclose()

    assert [request["text"] for request in first_server.requests if "text" in request] == ["first "]
    assert [request["text"] for request in second_server.requests if "text" in request] == [
        "second "
    ]


async def test_v1_stream_metrics_keep_model_after_endpoint_update() -> None:
    async with _RimeV1Server() as first_server, _RimeV1Server() as second_server:
        tts = _v1_tts(first_server)
        metrics = []
        tts.on("metrics_collected", metrics.append)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("first")
        stream.flush()
        try:
            await asyncio.wait_for(first_server.wait_for_text_messages(1), timeout=2)
            mist_url = second_server.websocket_url.replace("/coda/ws", "/mist/ws")
            tts.update_options(websocket_url=mist_url)
            stream.end_input()
            await _collect(stream)
        finally:
            await stream.aclose()
            await tts.aclose()

    assert tts.model == "mistv3"
    assert len(metrics) == 1
    assert metrics[0].metadata.model_name == "coda"


async def test_v1_omits_retained_time_scale_factor_after_switch_to_mistv2() -> None:
    async with _RimeV1Server() as first_server, _RimeV1Server() as second_server:
        tts = _v1_tts(first_server, time_scale_factor=1.2)
        mistv2_url = second_server.websocket_url.replace("/coda/ws", "/mistv2/ws")
        tts.update_options(websocket_url=mistv2_url)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))

        stream.push_text("hello")
        stream.end_input()
        await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert second_server.paths == ["/mistv2/ws"]
    assert "timeScaleFactor" not in second_server.requests[0]["start"]["audioParameters"]


async def test_v1_clean_interruption_cancels_and_reuses_socket() -> None:
    async with _RimeV1Server(response_mode="no_audio") as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("Please stop this synthesis now. Pending")
        await asyncio.wait_for(server.wait_for_text_messages(1), timeout=2)
        await stream.aclose()

        second = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        second.end_input()
        await _collect(second)
        await second.aclose()
        await tts.aclose()

    assert "cancel" in _payloads(server)
    assert server.connections == 1


async def test_v1_interruption_stays_cancelled_without_retry() -> None:
    async with _RimeV1Server(response_mode="no_audio") as server:
        tts = _v1_tts(server)
        stream = tts.stream(
            conn_options=APIConnectOptions(max_retry=1, timeout=0.1, retry_interval=0)
        )
        stream.push_text("Please stop this synthesis now. Pending")
        await asyncio.wait_for(server.wait_for_text_messages(1), timeout=2)
        await asyncio.wait_for(stream.aclose(), timeout=0.5)
        await tts.aclose()

    assert _payloads(server).count("start") == 1


async def test_v1_cancels_when_start_write_is_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.plugins.rime import _websocket_v1

    async with _RimeV1Server() as server:
        original_send = _websocket_v1._send_envelope
        start_written = asyncio.Event()
        block_first_start = True

        async def _send_envelope(
            connection: _websocket_v1.Connection,
            context_id: str,
            payload: str,
            value: object,
        ) -> None:
            nonlocal block_first_start
            await original_send(connection, context_id, payload, value)
            if payload == "start" and block_first_start:
                block_first_start = False
                start_written.set()
                await asyncio.Future()

        monkeypatch.setattr(_websocket_v1, "_send_envelope", _send_envelope)
        tts = _v1_tts(server)
        interrupted = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        interrupted.push_text("Please stop this synthesis now. Pending")
        await asyncio.wait_for(start_written.wait(), timeout=2)
        await interrupted.aclose()

        next_stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        next_stream.push_text("next")
        next_stream.end_input()
        events = await _collect(next_stream)
        await next_stream.aclose()
        await tts.aclose()

    assert events[-1].is_final
    assert "cancel" in _payloads(server)
    assert server.connections == 1


async def test_v1_closes_socket_when_cancel_has_no_reply() -> None:
    async with _RimeV1Server(response_mode="no_audio", cancel_reply=False) as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.1))
        stream.push_text("Please stop this synthesis now. Pending")
        await asyncio.wait_for(server.wait_for_text_messages(1), timeout=2)
        await stream.aclose()

        second = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        second.end_input()
        await _collect(second)
        await second.aclose()
        await tts.aclose()

    assert server.connections == 2


async def test_v1_end_times_out_when_terminal_event_never_arrives() -> None:
    async with _RimeV1Server(response_mode="no_done") as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.1))
        stream.push_text("The server will not finish this context.")
        stream.end_input()
        with pytest.raises(APITimeoutError, match="after end"):
            await _collect(stream)
        await stream.aclose()

        empty = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        empty.end_input()
        assert await _collect(empty) == []
        await empty.aclose()
        await tts.aclose()

    assert server.connections == 2


async def test_v1_rejects_malformed_done_before_socket_reuse() -> None:
    async with _RimeV1Server(response_mode="malformed_done") as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("hello")
        stream.end_input()
        with pytest.raises(APIConnectionError, match="exactly one payload"):
            await _collect(stream)
        await stream.aclose()

        empty = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        empty.end_input()
        await _collect(empty)
        await empty.aclose()
        await tts.aclose()

    assert server.connections == 2


async def test_v1_rejects_malformed_cancelled_before_socket_reuse() -> None:
    async with _RimeV1Server(response_mode="malformed_cancelled") as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("Please stop this synthesis now. Pending")
        await asyncio.wait_for(server.wait_for_text_messages(1), timeout=2)
        await stream.aclose()

        empty = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        empty.end_input()
        await _collect(empty)
        await empty.aclose()
        await tts.aclose()

    assert server.connections == 2


@pytest.mark.parametrize(
    "response_mode",
    ["wrong_context", "invalid_envelope", "wrong_frame"],
)
async def test_v1_rejects_contaminated_responses(response_mode: str) -> None:
    async with _RimeV1Server(response_mode=response_mode) as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("hello")
        stream.end_input()
        with pytest.raises(APIConnectionError):
            await _collect(stream)
        await stream.aclose()
        await tts.aclose()


async def test_v1_json_rejects_invalid_base64_audio() -> None:
    async with _RimeV1Server(response_mode="invalid_base64") as server:
        tts = _v1_tts(server, websocket_protocol="json")
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("hello")
        stream.end_input()
        with pytest.raises(APIConnectionError, match="invalid Base64 audio"):
            await _collect(stream)
        await stream.aclose()
        await tts.aclose()


async def test_v1_json_rejects_invalid_done_type() -> None:
    async with _RimeV1Server(response_mode="invalid_done_type") as server:
        tts = _v1_tts(server, websocket_protocol="json")
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("hello")
        stream.end_input()
        with pytest.raises(APIConnectionError, match="malformed done event"):
            await _collect(stream)
        await stream.aclose()
        await tts.aclose()


@pytest.mark.parametrize("websocket_protocol", ["binary", "json"])
async def test_v1_rejects_started_without_request_id(websocket_protocol: str) -> None:
    async with _RimeV1Server(response_mode="missing_started_request_id") as server:
        tts = _v1_tts(server, websocket_protocol=websocket_protocol)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("hello")
        stream.end_input()
        with pytest.raises(APIConnectionError, match="malformed started event"):
            await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert server.closed_connections == 1


@pytest.mark.parametrize(
    ("kind", "status_code", "retryable"),
    [
        ("invalid_input", 400, False),
        ("unauthenticated", 401, False),
        ("permission_denied", 403, False),
        ("not_found", 404, False),
        ("resource_exhausted", 429, True),
        ("timeout", 504, True),
        ("unavailable", 503, True),
        ("unimplemented", 501, False),
        ("internal", 500, True),
    ],
)
async def test_v1_maps_context_error(kind: str, status_code: int, retryable: bool) -> None:
    async with _RimeV1Server(response_mode="error", error_kind=kind) as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("hello")
        stream.end_input()
        with pytest.raises(APIStatusError) as exc_info:
            await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert exc_info.value.status_code == status_code
    assert exc_info.value.retryable is retryable


async def test_v1_maps_connection_scoped_error() -> None:
    async with _RimeV1Server(response_mode="connection_error", error_kind="unavailable") as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("hello")
        stream.end_input()
        with pytest.raises(APIStatusError) as exc_info:
            await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert exc_info.value.status_code == 503


async def test_v1_does_not_retry_unimplemented_error() -> None:
    async with _RimeV1Server(response_mode="error", error_kind="unimplemented") as server:
        tts = _v1_tts(server)
        stream = tts.stream(
            conn_options=APIConnectOptions(max_retry=1, timeout=2, retry_interval=0)
        )
        stream.push_text("hello")
        stream.end_input()
        with pytest.raises(APIStatusError) as exc_info:
            await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert exc_info.value.status_code == 501
    assert exc_info.value.retryable is False
    assert server.connections == 1


async def test_v1_retries_before_audio() -> None:
    async with _RimeV1Server(fail_before_audio=1) as server:
        tts = _v1_tts(server)
        stream = tts.stream(
            conn_options=APIConnectOptions(max_retry=1, timeout=2, retry_interval=0)
        )
        stream.push_text("hello")
        stream.end_input()
        events = await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert events[-1].is_final
    assert server.connections == 2


async def test_v1_does_not_retry_after_partial_audio() -> None:
    async with _RimeV1Server(response_mode="partial_error") as server:
        tts = _v1_tts(server)
        stream = tts.stream(
            conn_options=APIConnectOptions(max_retry=1, timeout=2, retry_interval=0)
        )
        stream.push_text("hello")
        stream.end_input()
        with pytest.raises(APIStatusError):
            await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert server.connections == 1


async def test_v1_ready_error_does_not_expose_provider_event() -> None:
    async with _RimeV1Server(ready={"protocol": _SECRET, "providerData": _SECRET}) as server:
        tts = _v1_tts(server, websocket_protocol="json")
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("hello")
        stream.end_input()
        try:
            with pytest.raises(APIConnectionError) as exc_info:
                await _collect(stream)
            _assert_exception_is_safe(exc_info.value)
        finally:
            await stream.aclose()
            await tts.aclose()


def test_v1_context_mismatch_does_not_expose_provider_value() -> None:
    from livekit.plugins.rime import _websocket_v1

    response = proto.WebSocketResponse(context_id=_SECRET)
    with pytest.raises(APIConnectionError, match="unexpected contextId") as exc_info:
        _websocket_v1._check_context(response, "expected-context")

    _assert_exception_is_safe(exc_info.value)


@pytest.mark.parametrize(
    "websocket_url",
    [
        "wss://api.rime.ai/coda/ws",
        "ws://127.0.0.1:8080/coda/ws",
        "ws://[::1]:8080/coda/ws",
    ],
)
def test_v1_accepts_secure_or_loopback_websocket_url(websocket_url: str) -> None:
    from livekit.plugins.rime import _websocket_v1

    _websocket_v1.validate_websocket_url(websocket_url)


def test_v1_rejects_untrusted_secure_host_without_opt_in() -> None:
    from livekit.plugins.rime import _websocket_v1

    with pytest.raises(ValueError, match="trusted Rime host"):
        _websocket_v1.validate_websocket_url("wss://attacker.example/coda/ws")


def test_v1_accepts_custom_secure_host_with_opt_in() -> None:
    from livekit.plugins.rime import _websocket_v1

    _websocket_v1.validate_websocket_url(
        "wss://voice.customer.example/coda/ws", allow_custom_endpoint=True
    )


def test_v1_accepts_dedicated_rime_subdomain() -> None:
    from livekit.plugins.rime import _websocket_v1

    _websocket_v1.validate_websocket_url(
        "wss://tigerstripe-dialpad.aws-us-east-1.whiteglove.rime.ai/ws"
    )


def test_v1_rejects_lookalike_rime_host() -> None:
    from livekit.plugins.rime import _websocket_v1

    with pytest.raises(ValueError, match="trusted Rime host"):
        _websocket_v1.validate_websocket_url("wss://whiteglove.rime.ai.attacker.example/coda/ws")


@pytest.mark.parametrize(
    ("websocket_url", "model"),
    [
        ("wss://api.rime.ai/coda/ws", "coda"),
        ("wss://api.rime.ai/mist/ws", "mist"),
        ("wss://api.rime.ai/future-model/ws/?token=value", "future-model"),
    ],
)
def test_v1_reads_model_from_websocket_url(websocket_url: str, model: str) -> None:
    from livekit.plugins.rime import _websocket_v1

    assert _websocket_v1.model_from_websocket_url(websocket_url) == model


def test_v1_dedicated_websocket_url_has_no_embedded_model() -> None:
    from livekit.plugins.rime import _websocket_v1

    assert (
        _websocket_v1.model_from_websocket_url(
            "wss://tigerstripe-dialpad.aws-us-east-1.whiteglove.rime.ai/ws"
        )
        is None
    )


@pytest.mark.parametrize(
    "websocket_url",
    [
        "wss://api.rime.ai/coda",
        "wss://api.rime.ai/coda/stream",
    ],
)
def test_v1_rejects_url_not_ending_in_ws(websocket_url: str) -> None:
    from livekit.plugins.rime import _websocket_v1

    with pytest.raises(ValueError, match="end with /ws"):
        _websocket_v1.model_from_websocket_url(websocket_url)


@pytest.mark.parametrize(
    "websocket_url",
    [
        "ws://api.rime.ai/coda/ws",
        "ws://192.168.1.20/coda/ws",
        "ws://localhost:8080/coda/ws",
        "http://api.rime.ai/coda/ws",
    ],
)
def test_v1_rejects_insecure_remote_websocket_url(websocket_url: str) -> None:
    from livekit.plugins.rime import _websocket_v1

    with pytest.raises(ValueError):
        _websocket_v1.validate_websocket_url(websocket_url)


async def test_v1_connection_error_does_not_expose_transport_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.plugins.rime import _websocket_v1

    async def _fail_connect(*args: Any, **kwargs: Any) -> aiohttp.ClientWebSocketResponse:
        raise RuntimeError(f"credential-bearing transport error: {_SECRET}")

    monkeypatch.setattr(aiohttp.ClientSession, "ws_connect", _fail_connect)
    async with aiohttp.ClientSession() as session:
        with pytest.raises(APIConnectionError) as exc_info:
            await _websocket_v1.connect(
                session,
                websocket_url="wss://api.rime.ai/coda/ws",
                api_key="test-key",
                protocol="binary",
                timeout=1,
            )

    _assert_exception_is_safe(exc_info.value)
    assert exc_info.value.__cause__ is None


async def test_v1_connect_rejects_untrusted_host_before_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.plugins.rime import _websocket_v1

    transport_called = False

    async def _connect(*args: Any, **kwargs: Any) -> aiohttp.ClientWebSocketResponse:
        nonlocal transport_called
        transport_called = True
        raise AssertionError("transport must not receive credentials")

    monkeypatch.setattr(aiohttp.ClientSession, "ws_connect", _connect)
    async with aiohttp.ClientSession() as session:
        with pytest.raises(ValueError, match="trusted Rime host"):
            await _websocket_v1.connect(
                session,
                websocket_url="wss://attacker.example/coda/ws",
                api_key="test-key",
                protocol="binary",
                timeout=1,
            )

    assert transport_called is False


@pytest.mark.parametrize(
    ("payload", "value"),
    [
        ("start", proto.SynthesisRequest(text="")),
        ("text", "hello"),
        ("end", None),
    ],
)
async def test_v1_wraps_write_failures_as_safe_api_errors(
    payload: str,
    value: object,
) -> None:
    from livekit.plugins.rime import _websocket_v1

    class _FailingWebSocket:
        async def send_str(self, data: str) -> None:
            raise ConnectionResetError(f"write failed with {_SECRET}")

    connection = _websocket_v1.Connection(
        websocket=cast(aiohttp.ClientWebSocketResponse, _FailingWebSocket()),
        codec=_websocket_v1._JsonEnvelopeCodec(),
    )
    with pytest.raises(APIConnectionError) as exc_info:
        await _websocket_v1._send_envelope(connection, "context", payload, value)

    _assert_exception_is_safe(exc_info.value)
    assert exc_info.value.retryable is True
    assert exc_info.value.__cause__ is None


async def test_v1_retries_after_write_failure_before_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_send_str = aiohttp.ClientWebSocketResponse.send_str
    fail_first_text = True

    async def _send_str(
        self: aiohttp.ClientWebSocketResponse,
        data: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        nonlocal fail_first_text
        if fail_first_text and "text" in json.loads(data):
            fail_first_text = False
            raise ConnectionResetError(f"write failed with {_SECRET}")
        await original_send_str(self, data, *args, **kwargs)

    monkeypatch.setattr(aiohttp.ClientWebSocketResponse, "send_str", _send_str)

    async with _RimeV1Server() as server:
        tts = _v1_tts(server, websocket_protocol="json")
        errors: list[Any] = []
        tts.on("error", errors.append)
        stream = tts.stream(
            conn_options=APIConnectOptions(max_retry=1, timeout=2, retry_interval=0)
        )
        stream.push_text("hello")
        stream.end_input()
        try:
            events = await _collect(stream)
        finally:
            await stream.aclose()
            await tts.aclose()

    assert events[-1].is_final
    assert server.connections == 2
    assert len(errors) == 1
    assert isinstance(errors[0].error, APIConnectionError)
    assert errors[0].recoverable is True
    _assert_exception_is_safe(errors[0].error)


async def test_v1_websocket_error_does_not_expose_transport_data() -> None:
    from livekit.plugins.rime import _websocket_v1

    class _ErrorWebSocket:
        async def receive(self, *, timeout: float) -> aiohttp.WSMessage:
            return aiohttp.WSMessage(aiohttp.WSMsgType.ERROR, None, None)

        def exception(self) -> BaseException:
            return RuntimeError(f"socket failed with {_SECRET}")

    connection = _websocket_v1.Connection(
        websocket=cast(aiohttp.ClientWebSocketResponse, _ErrorWebSocket()),
        codec=_websocket_v1._JsonEnvelopeCodec(),
    )
    with pytest.raises(APIConnectionError) as exc_info:
        await _websocket_v1._receive_envelope(connection, timeout=1)

    _assert_exception_is_safe(exc_info.value)


async def test_v1_invalid_json_does_not_retain_provider_frame() -> None:
    from livekit.plugins.rime import _websocket_v1

    class _InvalidJsonWebSocket:
        async def receive(self, *, timeout: float) -> aiohttp.WSMessage:
            return aiohttp.WSMessage(
                aiohttp.WSMsgType.TEXT,
                f'{{"providerData":"{_SECRET}"',
                None,
            )

    connection = _websocket_v1.Connection(
        websocket=cast(aiohttp.ClientWebSocketResponse, _InvalidJsonWebSocket()),
        codec=_websocket_v1._JsonEnvelopeCodec(),
    )
    with pytest.raises(APIConnectionError) as exc_info:
        await _websocket_v1._receive_envelope(connection, timeout=1)

    _assert_exception_is_safe(exc_info.value)
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize(
    "error",
    [
        proto.WebSocketError(kind="invalid_input", message=_SECRET),
        proto.WebSocketError(kind=_SECRET, message="provider failure"),
        proto.WebSocketError(kind="invalid_input", message=""),
    ],
)
def test_v1_error_mapping_does_not_expose_provider_payload(
    error: proto.WebSocketError,
) -> None:
    from livekit.plugins.rime import _websocket_v1

    exc = _websocket_v1._rime_error(error, fallback_request_id="request-id")

    _assert_exception_is_safe(exc)
    assert _SECRET not in repr(exc.body)


async def test_v1_closes_idle_retired_pools_after_url_changes() -> None:
    async with (
        _RimeV1Server() as first_server,
        _RimeV1Server() as second_server,
        _RimeV1Server() as third_server,
    ):
        tts = _v1_tts(first_server)
        try:
            tts.prewarm()
            await asyncio.wait_for(first_server.connection_opened.wait(), timeout=2)

            tts.update_options(websocket_url=second_server.websocket_url)
            await asyncio.wait_for(first_server.connection_closed.wait(), timeout=1)

            tts.prewarm()
            await asyncio.wait_for(second_server.connection_opened.wait(), timeout=2)

            tts.update_options(websocket_url=third_server.websocket_url)
            await asyncio.wait_for(second_server.connection_closed.wait(), timeout=1)

            assert first_server.closed_connections == 1
            assert second_server.closed_connections == 1
        finally:
            await tts.aclose()


async def test_v1_closes_retired_pool_after_its_last_stream_finishes() -> None:
    async with _RimeV1Server() as first_server, _RimeV1Server() as second_server:
        tts = _v1_tts(first_server)
        first_stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        tts.update_options(websocket_url=second_server.websocket_url)

        first_stream.push_text("first")
        first_stream.end_input()
        try:
            events = await _collect(first_stream)
            await asyncio.wait_for(first_server.connection_closed.wait(), timeout=1)
        finally:
            await first_stream.aclose()
            await tts.aclose()

    assert events[-1].is_final
    assert first_server.closed_connections == 1
    assert second_server.connections == 0
