from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any
from unittest.mock import AsyncMock

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, APIError
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


async def _wait_until_stream_is_live(stream: RecognizeStream, *, timeout: float = 5.0) -> None:
    """Block until the stream's session is connected."""
    for _ in range(int(timeout / 0.02)):
        if stream._ws is not None and not stream._ws.closed:
            return
        await asyncio.sleep(0.02)
    raise AssertionError("stream did not connect in time")


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


async def test_flux_language_hint_reaches_session_create() -> None:
    extra: dict[str, Any] = {}

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            event = json.loads(msg.data)
            if event["type"] == "session.create":
                extra.update(event["settings"]["extra"])
            elif event["type"] == "session.finalize":
                await ws.send_json(
                    {"type": "final_transcript", "transcript": "hola", "language": "es"}
                )
                await ws.close()
        return ws

    async with _gateway(handler) as (base_url, session):
        stt = STT(
            model="deepgram/flux-general-multi",
            api_key="test-key",
            api_secret="test-secret",
            base_url=base_url,
            http_session=session,
            extra_kwargs={"language_hint": ["en", "es"]},
        )
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
        try:
            stream.end_input()
            await asyncio.wait_for(_final_transcripts(stream), timeout=5.0)
        finally:
            await stream.aclose()

    assert extra["language_hint"] == ["en", "es"]


async def test_flux_language_hint_is_not_language_gated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Flux takes a hint in any language: the gateway gates a hint on the model
    (flux-general-multi), not on the session language."""
    extra: dict[str, Any] = {}

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            event = json.loads(msg.data)
            if event["type"] == "session.create":
                extra.update(event["settings"]["extra"])
            elif event["type"] == "session.finalize":
                await ws.send_json(
                    {"type": "final_transcript", "transcript": "hola", "language": "es"}
                )
                await ws.close()
        return ws

    with caplog.at_level(logging.WARNING):
        async with _gateway(handler) as (base_url, session):
            stt = STT(
                model="deepgram/flux-general-multi",
                language="en",
                api_key="test-key",
                api_secret="test-secret",
                base_url=base_url,
                http_session=session,
                extra_kwargs={"language_hint": ["en", "es"]},
            )
            stream = stt.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
            try:
                stream.end_input()
                await asyncio.wait_for(_final_transcripts(stream), timeout=5.0)
            finally:
                await stream.aclose()

    assert extra["language_hint"] == ["en", "es"]
    assert "removed `language_hint`" not in caplog.text


async def test_nova3_language_hint_reaches_session_create() -> None:
    extra: dict[str, Any] = {}
    session_language = ""

    async def handler(request: web.Request) -> web.WebSocketResponse:
        nonlocal session_language
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            event = json.loads(msg.data)
            if event["type"] == "session.create":
                extra.update(event["settings"]["extra"])
                session_language = event["settings"]["language"]
            elif event["type"] == "session.finalize":
                await ws.send_json(
                    {"type": "final_transcript", "transcript": "hola", "language": "es"}
                )
                await ws.close()
        return ws

    async with _gateway(handler) as (base_url, session):
        stt = STT(
            model="deepgram/nova-3",
            language="multi",
            api_key="test-key",
            api_secret="test-secret",
            base_url=base_url,
            http_session=session,
            extra_kwargs={"language_hint": ["en", "es"]},
        )
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
        try:
            stream.end_input()
            await asyncio.wait_for(_final_transcripts(stream), timeout=5.0)
        finally:
            await stream.aclose()

    assert extra["language_hint"] == ["en", "es"]
    assert session_language == "multi"


async def test_nova3_language_hint_is_not_sent_outside_multi_mode(
    caplog: pytest.LogCaptureFixture,
) -> None:
    extra: dict[str, Any] = {}

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            event = json.loads(msg.data)
            if event["type"] == "session.create":
                extra.update(event["settings"]["extra"])
            elif event["type"] == "session.finalize":
                await ws.send_json(
                    {"type": "final_transcript", "transcript": "hola", "language": "es"}
                )
                await ws.close()
        return ws

    with caplog.at_level(logging.WARNING):
        async with _gateway(handler) as (base_url, session):
            stt = STT(
                model="deepgram/nova-3",
                language="en",
                api_key="test-key",
                api_secret="test-secret",
                base_url=base_url,
                http_session=session,
                extra_kwargs={"language_hint": ["en", "es"]},
            )
            stream = stt.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
            try:
                stream.end_input()
                await asyncio.wait_for(_final_transcripts(stream), timeout=5.0)
            finally:
                await stream.aclose()

    # the gateway forwards a hint only in multi mode, so sending it would be dead weight
    assert "language_hint" not in extra
    assert "removed `language_hint`" in caplog.text
    assert "model 'deepgram/nova-3'" in caplog.text
    assert "language is 'en'" in caplog.text
    # the suppressed send lives on the payload, never on the stored options
    assert stream._opts.extra_kwargs["language_hint"] == ["en", "es"]
    assert stt._opts.extra_kwargs["language_hint"] == ["en", "es"]


async def test_a_later_multi_mode_sends_the_hint_again(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A hint this connection could not carry stays on the stream, so the next connection
    under a language that can use it sends it again."""
    created: list[dict[str, Any]] = []
    sockets: list[web.WebSocketResponse] = []
    updated = asyncio.Event()
    reconnected = asyncio.Event()

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        sockets.append(ws)
        async for msg in ws:
            event = json.loads(msg.data)
            if event["type"] == "session.create":
                created.append(event)
                if len(created) == 2:
                    reconnected.set()
            elif event["type"] == "session.update":
                updated.set()
        return ws

    with caplog.at_level(logging.WARNING):
        async with _gateway(handler) as (base_url, session):
            stt = STT(
                model="deepgram/nova-3",
                language="en",
                api_key="test-key",
                api_secret="test-secret",
                base_url=base_url,
                http_session=session,
                extra_kwargs={"language_hint": ["en", "es"]},
            )
            stream = stt.stream(
                conn_options=APIConnectOptions(max_retry=1, retry_interval=0.001, timeout=1.0)
            )
            try:
                await _wait_until_stream_is_live(stream)
                stt.update_options(language="multi")
                await asyncio.wait_for(updated.wait(), timeout=1.0)
                # a drop of the socket rebuilds the connection on the new language
                await sockets[0].close()
                await asyncio.wait_for(reconnected.wait(), timeout=1.0)
            finally:
                await stream.aclose()

    assert len(created) == 2
    assert "language_hint" not in created[0]["settings"]["extra"]
    assert created[1]["settings"]["language"] == "multi"
    assert created[1]["settings"]["extra"]["language_hint"] == ["en", "es"]
    # the caller's hint survived the suppressed send, on both copies
    assert stream._opts.extra_kwargs["language_hint"] == ["en", "es"]
    assert stt._opts.extra_kwargs["language_hint"] == ["en", "es"]
    assert caplog.text.count("removed `language_hint`") == 1


async def test_a_stream_language_keeps_the_hint_the_parent_would_drop() -> None:
    """The guard reads the stream's effective language: `stream(language=...)` can put a
    session in multi mode that the STT was not built in."""
    extra: dict[str, Any] = {}

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            event = json.loads(msg.data)
            if event["type"] == "session.create":
                extra.update(event["settings"]["extra"])
            elif event["type"] == "session.finalize":
                await ws.send_json(
                    {"type": "final_transcript", "transcript": "hola", "language": "es"}
                )
                await ws.close()
        return ws

    async with _gateway(handler) as (base_url, session):
        stt = STT(
            model="deepgram/nova-3",
            language="en",
            api_key="test-key",
            api_secret="test-secret",
            base_url=base_url,
            http_session=session,
            extra_kwargs={"language_hint": ["en", "es"]},
        )
        stream = stt.stream(
            language="multi", conn_options=APIConnectOptions(max_retry=0, timeout=1.0)
        )
        try:
            stream.end_input()
            await asyncio.wait_for(_final_transcripts(stream), timeout=5.0)
        finally:
            await stream.aclose()

    assert extra["language_hint"] == ["en", "es"]


async def test_a_live_update_drops_a_hint_outside_multi_mode(
    caplog: pytest.LogCaptureFixture,
) -> None:
    updates: list[dict[str, Any]] = []
    updated = asyncio.Event()

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            event = json.loads(msg.data)
            if event["type"] == "session.update":
                updates.append(event)
                updated.set()
        return ws

    with caplog.at_level(logging.WARNING):
        async with _gateway(handler) as (base_url, session):
            stt = STT(
                model="deepgram/nova-3",
                language="en",
                api_key="test-key",
                api_secret="test-secret",
                base_url=base_url,
                http_session=session,
            )
            stream = stt.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
            try:
                await _wait_until_stream_is_live(stream)
                stt.update_options(extra={"language_hint": ["en", "es"]})
                await asyncio.wait_for(updated.wait(), timeout=1.0)
                updated.clear()
                # a hint arriving again is dropped silently: the stream already reported it
                stt.update_options(extra={"language_hint": ["en", "es"]})
                await asyncio.wait_for(updated.wait(), timeout=1.0)
            finally:
                await stream.aclose()

    # the update path builds its own extra, so it needs the guard as much as session.create
    assert len(updates) == 2
    assert all("language_hint" not in frame["settings"]["extra"] for frame in updates)
    assert caplog.text.count("removed `language_hint`") == 1


async def test_a_live_update_into_multi_mode_keeps_the_hint(
    caplog: pytest.LogCaptureFixture,
) -> None:
    updates: list[dict[str, Any]] = []
    updated = asyncio.Event()

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            event = json.loads(msg.data)
            if event["type"] == "session.update":
                updates.append(event)
                updated.set()
        return ws

    with caplog.at_level(logging.WARNING):
        async with _gateway(handler) as (base_url, session):
            stt = STT(
                model="deepgram/nova-3",
                language="en",
                api_key="test-key",
                api_secret="test-secret",
                base_url=base_url,
                http_session=session,
            )
            stream = stt.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
            try:
                await _wait_until_stream_is_live(stream)
                # switching into multi mode and adding hints in one call is legitimate
                stt.update_options(language="multi", extra={"language_hint": ["en", "es"]})
                await asyncio.wait_for(updated.wait(), timeout=1.0)
            finally:
                await stream.aclose()

    assert len(updates) == 1
    assert updates[0]["settings"]["language"] == "multi"
    assert updates[0]["settings"]["extra"]["language_hint"] == ["en", "es"]
    assert "removed `language_hint`" not in caplog.text


async def test_linden_session_keyterms_reach_session_create() -> None:
    extra: dict[str, Any] = {}

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            event = json.loads(msg.data)
            if event["type"] == "session.create":
                extra.update(event["settings"]["extra"])
            elif event["type"] == "session.finalize":
                await ws.send_json({"type": "final_transcript", "transcript": "hi"})
                await ws.close()
        return ws

    user_entry = {"content": "gnocchi", "sounds_like": ["nyohki"]}
    async with _gateway(handler) as (base_url, session):
        stt = STT(
            model="speechmatics/linden-1",
            api_key="test-key",
            api_secret="test-secret",
            base_url=base_url,
            http_session=session,
            extra_kwargs={"additional_vocab": [user_entry]},
        )
        assert stt.capabilities.keyterms is True
        stt._update_session_keyterms(["gnocchi", "LiveKit"])
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
        try:
            stream.end_input()
            await asyncio.wait_for(_final_transcripts(stream), timeout=5.0)
        finally:
            await stream.aclose()

    # the user's entry keeps its sounds_like; a session term it already covers is not duplicated
    assert extra["additional_vocab"] == [user_entry, {"content": "LiveKit"}]


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

    async def ping(self) -> None:
        self.messages.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.PONG, b"", ""))

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
    ("message", "progress"),
    [
        ("start_of_speech", True),
        ("interim_transcript", True),
        ("preflight_transcript", True),
        ("final_transcript", True),
        ("session.finalized", True),
        ("session.created", False),
        ("unknown", False),
        ("pong", False),
    ],
)
async def test_provider_events_reset_retries_but_setup_and_pongs_do_not(
    monkeypatch: pytest.MonkeyPatch, message: str, progress: bool
) -> None:
    processed = asyncio.Event()

    class Socket(_Socket):
        async def receive(self) -> aiohttp.WSMessage:
            if self.messages.empty():
                processed.set()
            return await super().receive()

    socket = Socket()
    if message == "pong":
        socket.messages.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.PONG, b"", ""))
    else:
        socket.send_event(message)
    monkeypatch.setattr(SpeechStream, "_connect_ws", AsyncMock(return_value=socket))
    async with aiohttp.ClientSession() as session:
        stream = _make_stt("http://unused", session).stream()
        stream._num_retries = 2
        try:
            await asyncio.wait_for(processed.wait(), 1)
            assert stream._num_retries == (0 if progress else 2)
        finally:
            await stream.aclose()


@pytest.mark.virtual_time
@pytest.mark.parametrize("message", ["error", "session.closed"])
async def test_provider_failure_events_do_not_reset_retries(
    monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    socket = _Socket()
    socket.send_event(message)
    monkeypatch.setattr(SpeechStream, "_connect_ws", AsyncMock(return_value=socket))
    async with aiohttp.ClientSession() as session:
        stream = _make_stt("http://unused", session).stream()
        stream._num_retries = 3
        try:
            with pytest.raises(APIError):
                await asyncio.wait_for(stream._task, 1)
            assert stream._num_retries == 3
        finally:
            await stream.aclose()


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


async def test_half_open_socket_raises_retryable_error() -> None:

    async def handler(request: web.Request) -> web.WebSocketResponse:
        # accepts the session, then never sends anything or answers a ping
        ws = web.WebSocketResponse(autoping=False)
        await ws.prepare(request)
        async for _ in ws:
            pass
        return ws

    async with _gateway(handler) as (base_url, session):
        stream = _make_stt(base_url, session).stream(
            conn_options=APIConnectOptions(max_retry=0, timeout=1.0, liveness_timeout=0.15)
        )
        try:
            with pytest.raises(APIConnectionError) as exc_info:
                await asyncio.wait_for(_final_transcripts(stream), timeout=5.0)
        finally:
            await stream.aclose()

    assert exc_info.value.retryable is True
    assert isinstance(exc_info.value.__cause__, aiohttp.ServerTimeoutError)
