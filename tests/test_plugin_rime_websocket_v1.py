"""Tests for the Rime WebSocket v1 protocol and LiveKit adapter."""

from __future__ import annotations

import asyncio
import base64
import json
import traceback
from typing import Any, cast

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
)

pytestmark = pytest.mark.unit


_PCM = b"\x01\x00" * 2205
_SECRET = "customer-secret-marker"


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
    ) -> None:
        self.response_mode = response_mode
        self.ready = ready or {"protocol": 1, "languages": ["eng"]}
        self.cancel_reply = cancel_reply
        self.fail_before_audio = fail_before_audio
        self.error_kind = error_kind
        self.connections = 0
        self.ready_events = 0
        self.requests: list[dict[str, Any]] = []
        self.request_connections: list[int] = []
        self.headers: list[dict[str, str]] = []
        self.paths: list[str] = []
        self.text_received = asyncio.Event()
        self.flush_received = asyncio.Event()
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
        self.base_url = f"http://127.0.0.1:{port}/coda/v1/coda"
        self.websocket_url = f"ws://127.0.0.1:{port}/coda/v1/coda/ws"
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.session.close()
        await self._runner.cleanup()

    async def _handle(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(protocols=("rime.v1.json",))
        await ws.prepare(request)
        connection = self.connections
        self.connections += 1
        self.headers.append(dict(request.headers))
        self.paths.append(request.path)
        await ws.send_json({"ready": self.ready})
        self.ready_events += 1
        self.connection_opened.set()

        try:
            async for message in ws:
                if message.type != aiohttp.WSMsgType.TEXT:
                    continue
                envelope = json.loads(message.data)
                self.requests.append(envelope)
                self.request_connections.append(connection)
                context_id = envelope.get("contextId", "")
                if "start" in envelope:
                    await ws.send_json(
                        {
                            "contextId": context_id,
                            "started": {"requestId": f"request-{connection}"},
                        }
                    )
                elif "text" in envelope:
                    self.text_received.set()
                    if self.fail_before_audio > 0:
                        self.fail_before_audio -= 1
                        await ws.send_json(
                            {
                                "contextId": context_id,
                                "error": {
                                    "kind": "unavailable",
                                    "message": "retry later",
                                    "requestId": f"request-{connection}",
                                },
                            }
                        )
                    else:
                        await self._respond_to_text(ws, context_id)
                elif "end" in envelope:
                    if self.response_mode == "malformed_done":
                        await ws.send_json({"contextId": context_id, "done": None})
                    elif self.response_mode not in ("error", "partial_error", "no_done"):
                        await ws.send_json({"contextId": context_id, "done": {}})
                elif "cancel" in envelope and self.cancel_reply:
                    if self.response_mode == "malformed_cancelled":
                        await ws.send_json({"contextId": context_id, "cancelled": "bad"})
                    else:
                        await ws.send_json({"contextId": context_id, "cancelled": {}})
                elif "flush" in envelope:
                    self.flush_received.set()
        finally:
            self.closed_connections += 1
            self.connection_closed.set()
        return ws

    async def _respond_to_text(self, ws: web.WebSocketResponse, context_id: str) -> None:
        if self.response_mode == "normal":
            await ws.send_json({"contextId": context_id, "audio": base64.b64encode(_PCM).decode()})
        elif self.response_mode == "wrong_context":
            await ws.send_json({"contextId": "wrong", "done": {}})
        elif self.response_mode == "invalid_json":
            await ws.send_str("{")
        elif self.response_mode == "binary":
            await ws.send_bytes(b"not-json")
        elif self.response_mode == "invalid_base64":
            await ws.send_json({"contextId": context_id, "audio": "%%%"})
        elif self.response_mode == "error":
            await ws.send_json(
                {
                    "contextId": context_id,
                    "error": {"kind": self.error_kind, "message": "bad input"},
                }
            )
        elif self.response_mode == "connection_error":
            await ws.send_json({"error": {"kind": self.error_kind, "message": "connection failed"}})
        elif self.response_mode == "partial_error":
            await ws.send_json({"contextId": context_id, "audio": base64.b64encode(_PCM).decode()})
            await asyncio.sleep(0.05)
            await ws.send_json(
                {
                    "contextId": context_id,
                    "error": {"kind": "unavailable", "message": "failed late"},
                }
            )


def _v1_tts(server: _RimeV1Server, **kwargs: Any):
    from livekit.plugins.rime import TTS

    return TTS(
        api_key="test-key",
        websocket_url=server.websocket_url,
        http_session=server.session,
        **kwargs,
    )


async def _collect(stream) -> list:
    events = []
    async for event in stream:
        events.append(event)
    return events


def _payloads(server: _RimeV1Server) -> list[str]:
    return [next(key for key in request if key != "contextId") for request in server.requests]


async def test_v1_streams_audio_before_end_and_maps_start_options() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(
            server,
            lang="eng",
            sample_rate=22050,
            repetition_penalty=1.1,
            temperature=0.5,
            top_p=0.9,
            max_tokens=200,
            time_scale_factor=1.2,
        )
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        try:
            first_event = asyncio.create_task(anext(stream))
            stream.push_text("Hello from ")
            stream.push_text("LiveKit today. Next")
            first = await asyncio.wait_for(first_event, timeout=2)
            assert first.frame.data

            stream.push_text(" sentence.")
            stream.end_input()
            remaining = await _collect(stream)
        finally:
            await stream.aclose()
            await tts.aclose()

    assert remaining[-1].is_final
    assert server.paths == ["/coda/v1/coda/ws"]
    assert server.headers[0]["Authorization"] == "Bearer test-key"
    assert server.headers[0]["Sec-WebSocket-Protocol"] == "rime.v1.json"
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
        "codaParameters": {
            "repetitionPenalty": 1.1,
            "temperature": 0.5,
            "topP": 0.9,
            "maxTokens": 200,
        },
    }


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
            await asyncio.wait_for(server.text_received.wait(), timeout=0.05)

        stream.push_text("Second")
        await asyncio.wait_for(server.text_received.wait(), timeout=2)
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


async def test_v1_nonfinal_flush_does_not_replace_end() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("first")
        stream.flush()
        stream.end_input()
        await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert _payloads(server) == ["start", "text", "flush", "end"]


async def test_v1_sends_nonfinal_flush_without_more_input() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("first")
        stream.flush()
        try:
            await asyncio.wait_for(server.flush_received.wait(), timeout=0.2)
        finally:
            stream.end_input()
            await _collect(stream)
            await stream.aclose()
            await tts.aclose()

    assert _payloads(server) == ["start", "text", "flush", "end"]


async def test_v1_resumes_context_after_nonfinal_flush() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        metrics = []
        tts.on("metrics_collected", metrics.append)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("first")
        stream.flush()
        await asyncio.wait_for(server.flush_received.wait(), timeout=2)
        stream.push_text("second")
        stream.end_input()
        events = await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert _payloads(server) == ["start", "text", "flush", "text", "end"]
    assert [request["text"] for request in server.requests if "text" in request] == [
        "first ",
        "second ",
    ]
    assert len({request["contextId"] for request in server.requests}) == 1
    assert sum(event.is_final for event in events) == 1
    assert len(metrics) == 1
    assert metrics[0].characters_count == len("firstsecond")


async def test_v1_flush_pause_can_exceed_api_timeout() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.1))
        stream.push_text("First sentence.")
        stream.flush()
        await asyncio.wait_for(server.flush_received.wait(), timeout=2)

        await asyncio.sleep(0.2)

        stream.push_text("Second sentence.")
        stream.end_input()
        events = await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert _payloads(server) == ["start", "text", "flush", "text", "end"]
    assert len({request["contextId"] for request in server.requests}) == 1
    assert events[-1].is_final


async def test_v1_ignores_flush_before_first_text() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.flush()
        stream.push_text("first")
        stream.end_input()
        await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert _payloads(server) == ["start", "text", "end"]


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
        await asyncio.wait_for(server.text_received.wait(), timeout=2)

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


async def test_v1_clean_interruption_cancels_and_reuses_socket() -> None:
    async with _RimeV1Server(response_mode="no_audio") as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("Please stop this synthesis now. Pending")
        await asyncio.wait_for(server.text_received.wait(), timeout=2)
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
        await asyncio.wait_for(server.text_received.wait(), timeout=2)
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
            ws: aiohttp.ClientWebSocketResponse,
            context_id: str,
            payload: str,
            value: object,
        ) -> None:
            nonlocal block_first_start
            await original_send(ws, context_id, payload, value)
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
        await asyncio.wait_for(server.text_received.wait(), timeout=2)
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
        with pytest.raises(APIConnectionError, match="malformed done"):
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
        await asyncio.wait_for(server.text_received.wait(), timeout=2)
        await stream.aclose()

        empty = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        empty.end_input()
        await _collect(empty)
        await empty.aclose()
        await tts.aclose()

    assert server.connections == 2


@pytest.mark.parametrize(
    "response_mode",
    ["wrong_context", "invalid_json", "binary", "invalid_base64"],
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


@pytest.mark.parametrize(
    ("kind", "status_code"),
    [
        ("invalid_input", 400),
        ("unauthenticated", 401),
        ("permission_denied", 403),
        ("not_found", 404),
        ("resource_exhausted", 429),
        ("timeout", 504),
        ("unavailable", 503),
        ("unimplemented", 501),
        ("internal", 500),
    ],
)
async def test_v1_maps_context_error(kind: str, status_code: int) -> None:
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
    assert exc_info.value.retryable is (status_code >= 500 or status_code == 429)


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
        tts = _v1_tts(server)
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

    with pytest.raises(APIConnectionError, match="unexpected contextId") as exc_info:
        _websocket_v1._check_context({"contextId": _SECRET}, "expected-context")

    _assert_exception_is_safe(exc_info.value)


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
                websocket_url="wss://example.com/coda/ws",
                api_key="test-key",
                timeout=1,
            )

    _assert_exception_is_safe(exc_info.value)
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize(
    ("payload", "value"),
    [
        ("start", {}),
        ("text", "hello"),
        ("flush", {}),
        ("end", {}),
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

    ws = cast(aiohttp.ClientWebSocketResponse, _FailingWebSocket())
    with pytest.raises(APIConnectionError) as exc_info:
        await _websocket_v1._send_envelope(ws, "context", payload, value)

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
        tts = _v1_tts(server)
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

    ws = cast(aiohttp.ClientWebSocketResponse, _ErrorWebSocket())
    with pytest.raises(APIConnectionError) as exc_info:
        await _websocket_v1._receive_envelope(ws, timeout=1)

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

    ws = cast(aiohttp.ClientWebSocketResponse, _InvalidJsonWebSocket())
    with pytest.raises(APIConnectionError) as exc_info:
        await _websocket_v1._receive_envelope(ws, timeout=1)

    _assert_exception_is_safe(exc_info.value)
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize(
    "error",
    [
        {"kind": "invalid_input", "message": _SECRET},
        {"kind": _SECRET, "message": "provider failure"},
        {"kind": "invalid_input", "message": 1, "providerData": _SECRET},
    ],
)
def test_v1_error_mapping_does_not_expose_provider_payload(error: object) -> None:
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
