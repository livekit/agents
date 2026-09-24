from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Callable
from typing import Any
from unittest.mock import AsyncMock

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import APIConnectOptions, APIError
from livekit.agents.inference import STT
from livekit.agents.inference.stt import SpeechStream
from livekit.agents.stt import RecognizeStream, SpeechEventType

pytestmark = pytest.mark.unit


def _make_stt(base_url: str, session: aiohttp.ClientSession, **kwargs: Any) -> STT:
    return STT(
        model="deepgram/nova-3",
        api_key="test-key",
        api_secret="test-secret",
        base_url=base_url,
        http_session=session,
        **kwargs,
    )


@contextlib.asynccontextmanager
async def _gateway(
    handler: Callable[[web.Request], Any],
) -> AsyncIterator[tuple[str, aiohttp.ClientSession]]:
    app = web.Application()
    app.router.add_get("/stt", handler)
    server = TestServer(app)
    await server.start_server()
    session = aiohttp.ClientSession()
    try:
        yield str(server.make_url("")).rstrip("/"), session
    finally:
        await session.close()
        await server.close()


async def _final_transcripts(stream: RecognizeStream) -> list[str]:
    return [
        event.alternatives[0].text
        async for event in stream
        if event.type == SpeechEventType.FINAL_TRANSCRIPT
    ]


async def test_input_end_preserves_delayed_final_after_session_finalized() -> None:
    message_types: list[str] = []
    request_model = ""
    socket_closed = asyncio.Event()
    transcript_sent = False
    close_before_transcript = False

    async def handler(request: web.Request) -> web.WebSocketResponse:
        nonlocal request_model, close_before_transcript
        request_model = request.query.get("model", "")
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        async def send_final() -> None:
            nonlocal transcript_sent
            await asyncio.sleep(0.025)
            if not ws.closed:
                await ws.send_json(
                    {"type": "final_transcript", "transcript": "final words", "language": "en"}
                )
                transcript_sent = True

        final_task = None
        try:
            async for msg in ws:
                event = json.loads(msg.data)
                message_types.append(event["type"])
                if event["type"] == "session.finalize":
                    await ws.send_json({"type": "session.finalized"})
                    final_task = asyncio.create_task(send_final())
                if event["type"] == "session.close":
                    close_before_transcript = not transcript_sent
        finally:
            if final_task is not None:
                final_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await final_task
            socket_closed.set()
        return ws

    async with _gateway(handler) as (base_url, session):
        stt = _make_stt(base_url, session)
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
        try:
            stream.end_input()
            transcripts = await asyncio.wait_for(_final_transcripts(stream), timeout=5.0)
            await asyncio.wait_for(socket_closed.wait(), timeout=1.0)
        finally:
            await stream.aclose()

    assert request_model == "deepgram/nova-3"
    assert message_types == ["session.create", "session.finalize", "session.close"]
    assert transcripts == ["final words"]
    assert not close_before_transcript


@pytest.mark.parametrize("close_code", [1000, 1011])
async def test_socket_close_after_input_end_is_accepted(close_code: int) -> None:
    connection_count = 0

    async def handler(request: web.Request) -> web.WebSocketResponse:
        nonlocal connection_count
        connection_count += 1
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            if json.loads(msg.data)["type"] == "session.finalize":
                await ws.send_json(
                    {"type": "final_transcript", "transcript": "final words", "language": "en"}
                )
                await ws.close(code=close_code)
        return ws

    async with _gateway(handler) as (base_url, session):
        stt = _make_stt(base_url, session)
        errors: list[Exception] = []
        stt.on("error", lambda event: errors.append(event.error))
        stream = stt.stream(
            conn_options=APIConnectOptions(max_retry=3, retry_interval=0.001, timeout=1.0)
        )
        try:
            stream.end_input()
            transcripts = await asyncio.wait_for(_final_transcripts(stream), timeout=1.0)
        finally:
            await stream.aclose()

    assert connection_count == 1
    assert not errors
    assert transcripts == ["final words"]


async def test_session_closed_after_input_end_finishes_without_another_close() -> None:
    message_types: list[str] = []
    socket_closed = asyncio.Event()

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        try:
            async for msg in ws:
                event = json.loads(msg.data)
                message_types.append(event["type"])
                if event["type"] == "session.finalize":
                    await ws.send_json(
                        {"type": "final_transcript", "transcript": "final words", "language": "en"}
                    )
                    await ws.send_json({"type": "session.closed"})
        finally:
            socket_closed.set()
        return ws

    async with _gateway(handler) as (base_url, session):
        stream = _make_stt(base_url, session).stream(
            conn_options=APIConnectOptions(max_retry=0, timeout=1.0)
        )
        try:
            stream.end_input()
            transcripts = await asyncio.wait_for(_final_transcripts(stream), timeout=1.0)
            await asyncio.wait_for(socket_closed.wait(), timeout=1.0)
        finally:
            await stream.aclose()

    assert message_types == ["session.create", "session.finalize"]
    assert transcripts == ["final words"]


@pytest.mark.parametrize("close_session", [False, True], ids=["socket", "session"])
async def test_close_before_input_end_reconnects(close_session: bool) -> None:
    connection_count = 0
    second_connection = asyncio.Event()

    async def handler(request: web.Request) -> web.WebSocketResponse:
        nonlocal connection_count
        connection_count += 1
        connection = connection_count
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            event = json.loads(msg.data)
            if connection == 1 and event["type"] == "session.create":
                if close_session:
                    await ws.send_json({"type": "session.closed"})
                else:
                    await ws.close()
            if connection == 2:
                second_connection.set()
                if event["type"] == "session.finalize":
                    await ws.send_json(
                        {"type": "final_transcript", "transcript": "final words", "language": "en"}
                    )
                    await ws.send_json({"type": "session.closed"})
        return ws

    async with _gateway(handler) as (base_url, session):
        stt = _make_stt(base_url, session)
        errors: list[Exception] = []
        stt.on("error", lambda event: errors.append(event.error))
        stream = stt.stream(
            conn_options=APIConnectOptions(max_retry=1, retry_interval=0.001, timeout=1.0)
        )
        try:
            await asyncio.wait_for(second_connection.wait(), timeout=1.0)
            stream.end_input()
            transcripts = await asyncio.wait_for(_final_transcripts(stream), timeout=1.0)
        finally:
            await stream.aclose()

    assert connection_count == 2
    assert len(errors) == 1
    assert isinstance(errors[0], APIError) and errors[0].retryable
    assert transcripts == ["final words"]


@pytest.mark.parametrize("end_input", [False, True], ids=["active", "draining"])
async def test_stream_close_sends_session_close_and_closes_socket(end_input: bool) -> None:
    message_types: list[str] = []
    session_created = asyncio.Event()
    session_finalized = asyncio.Event()
    socket_closed = asyncio.Event()

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        try:
            async for msg in ws:
                event = json.loads(msg.data)
                message_types.append(event["type"])
                if event["type"] == "session.create":
                    session_created.set()
                if event["type"] == "session.finalize":
                    session_finalized.set()
        finally:
            socket_closed.set()
        return ws

    async with _gateway(handler) as (base_url, session):
        stt = _make_stt(base_url, session)
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
        try:
            await asyncio.wait_for(session_created.wait(), timeout=1.0)
            if end_input:
                stream.end_input()
                await asyncio.wait_for(session_finalized.wait(), timeout=1.0)
            await asyncio.wait_for(stream.aclose(), timeout=1.0)
            await asyncio.wait_for(socket_closed.wait(), timeout=1.0)
        finally:
            await stream.aclose()

    expected = ["session.create", "session.finalize", "session.close"]
    if end_input:
        assert message_types == expected
    else:
        # aclose closes the input channel before it cancels the stream task.
        assert message_types in (["session.create", "session.close"], expected)


class _Socket:
    def __init__(self) -> None:
        self.closed = False
        self.sent: list[str] = []
        self.messages: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()
        self.finalized = asyncio.Event()

    async def send_str(self, message: str) -> None:
        msg_type = json.loads(message)["type"]
        self.sent.append(msg_type)
        if msg_type == "session.finalize":
            self.finalized.set()

    async def receive(self) -> aiohttp.WSMessage:
        return await self.messages.get()

    async def close(self) -> None:
        self.closed = True

    def send_event(self, msg_type: str) -> None:
        self.messages.put_nowait(
            aiohttp.WSMessage(
                aiohttp.WSMsgType.TEXT,
                json.dumps({"type": msg_type, "transcript": "words", "language": "en"}),
                "",
            )
        )


@pytest.mark.virtual_time
@pytest.mark.parametrize(
    ("events", "expected_duration"),
    [
        ([], 30.0),
        ([(5.0, "interim_transcript")], 35.0),
        ([(5.0, "preflight_transcript")], 35.0),
        ([(5.0, "session.finalized")], 30.0),
        ([(5.0, "final_transcript")], 8.0),
        ([(5.0, "final_transcript"), (2.0, "interim_transcript")], 10.0),
        ([(5.0, "final_transcript"), (2.0, "preflight_transcript")], 10.0),
        ([(5.0, "final_transcript"), (2.0, "final_transcript")], 10.0),
    ],
    ids=[
        "silent",
        "interim",
        "preflight",
        "ack",
        "final",
        "final-interim",
        "final-preflight",
        "final-final",
    ],
)
async def test_input_end_waits_for_transcript_inactivity(
    monkeypatch: pytest.MonkeyPatch, events: list[tuple[float, str]], expected_duration: float
) -> None:
    socket = _Socket()
    monkeypatch.setattr(SpeechStream, "_connect_ws", AsyncMock(return_value=socket))
    async with aiohttp.ClientSession() as session:
        stream = _make_stt("http://unused", session).stream(
            conn_options=APIConnectOptions(max_retry=0)
        )

        async def send_events() -> None:
            await socket.finalized.wait()
            for delay, msg_type in events:
                await asyncio.sleep(delay)
                socket.send_event(msg_type)

        sender = asyncio.create_task(send_events())
        start = asyncio.get_running_loop().time()
        try:
            stream.end_input()
            await asyncio.wait_for(_final_transcripts(stream), timeout=60.0)
            elapsed = asyncio.get_running_loop().time() - start
        finally:
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender
            await stream.aclose()

    assert elapsed == pytest.approx(expected_duration, abs=0.01)
    assert socket.sent == ["session.finalize", "session.close"]
    assert socket.closed


@pytest.mark.virtual_time
async def test_final_before_input_end_uses_shorter_inactivity_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = _Socket()
    monkeypatch.setattr(SpeechStream, "_connect_ws", AsyncMock(return_value=socket))
    async with aiohttp.ClientSession() as session:
        stream = _make_stt("http://unused", session).stream(
            conn_options=APIConnectOptions(max_retry=0)
        )
        try:
            socket.send_event("final_transcript")
            async for event in stream:
                if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                    break
            await asyncio.sleep(40.0)
            assert not socket.closed
            start = asyncio.get_running_loop().time()
            stream.end_input()
            await asyncio.wait_for(_final_transcripts(stream), timeout=5.0)
            elapsed = asyncio.get_running_loop().time() - start
        finally:
            await stream.aclose()

    assert elapsed == pytest.approx(3.0, abs=0.01)
    assert socket.sent == ["session.finalize", "session.close"]


@pytest.mark.parametrize("error_code", [2006, 2007], ids=["connection", "inactivity"])
async def test_error_before_input_end_reconnects(error_code: int) -> None:
    connection_count = 0
    second_connection = asyncio.Event()
    audio_connections: list[int] = []

    async def handler(request: web.Request) -> web.WebSocketResponse:
        nonlocal connection_count
        connection_count += 1
        connection = connection_count
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            event = json.loads(msg.data)
            if event["type"] == "session.create":
                if connection == 1:
                    await ws.send_json(
                        {
                            "type": "error",
                            "code": error_code,
                            "message": "customer content must not reach the API error",
                        }
                    )
                else:
                    second_connection.set()
            elif event["type"] == "input_audio":
                audio_connections.append(connection)
            elif event["type"] == "session.finalize":
                await ws.send_json(
                    {"type": "final_transcript", "transcript": "final words", "language": "en"}
                )
                await ws.send_json({"type": "session.closed"})
        return ws

    async with _gateway(handler) as (base_url, session):
        stt = _make_stt(base_url, session, sample_rate=16000)
        errors: list[Exception] = []
        stt.on("error", lambda event: errors.append(event.error))
        stream = stt.stream(
            conn_options=APIConnectOptions(max_retry=1, retry_interval=0.001, timeout=1.0)
        )
        try:
            await asyncio.wait_for(second_connection.wait(), timeout=1.0)
            stream.push_frame(
                rtc.AudioFrame.create(sample_rate=16000, num_channels=1, samples_per_channel=800)
            )
            stream.end_input()
            transcripts = await asyncio.wait_for(_final_transcripts(stream), timeout=1.0)
        finally:
            await stream.aclose()

    assert connection_count == 2
    assert audio_connections == [2]
    assert transcripts == ["final words"]
    assert len(errors) == 1
    error = errors[0]
    assert isinstance(error, APIError)
    assert error.message == "LiveKit Inference STT returned an error"
    assert error.body == {"code": error_code}
    assert error.retryable is True


@pytest.mark.parametrize("error_code", [2006, 2007], ids=["connection", "inactivity"])
async def test_error_after_input_end_is_not_retried(error_code: int) -> None:
    connection_count = 0

    async def handler(request: web.Request) -> web.WebSocketResponse:
        nonlocal connection_count
        connection_count += 1
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            if json.loads(msg.data)["type"] == "session.finalize":
                await ws.send_json(
                    {
                        "type": "error",
                        "code": error_code,
                        "message": "customer content must not reach the API error",
                    }
                )
        return ws

    async with _gateway(handler) as (base_url, session):
        stt = _make_stt(base_url, session)
        errors: list[Exception] = []
        stt.on("error", lambda event: errors.append(event.error))
        stream = stt.stream(
            conn_options=APIConnectOptions(max_retry=3, retry_interval=0.001, timeout=1.0)
        )
        try:
            stream.end_input()
            with pytest.raises(APIError) as exc_info:
                await asyncio.wait_for(_final_transcripts(stream), timeout=1.0)
        finally:
            await stream.aclose()

    assert connection_count == 1
    assert errors == [exc_info.value]
    assert exc_info.value.message == "LiveKit Inference STT returned an error"
    assert exc_info.value.body == {"code": error_code}
    assert exc_info.value.retryable is False
