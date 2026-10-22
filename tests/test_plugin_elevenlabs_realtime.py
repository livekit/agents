from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIError, llm
from livekit.agents.types import APIConnectOptions
from livekit.plugins.elevenlabs import realtime
from livekit.plugins.elevenlabs.realtime import realtime_model as _realtime_model
from livekit.plugins.elevenlabs.realtime.realtime_model import _close_error

pytestmark = pytest.mark.unit

KEY = "sk_test"
LONG_CALL_ID = "call_" + "x" * 40


class FakeServer:
    def __init__(self) -> None:
        self.received: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.requests: list[web.Request] = []
        self.connected = asyncio.Event()
        self.ws: web.WebSocketResponse | None = None
        self.http: aiohttp.ClientSession | None = None

    async def handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.requests.append(request)
        self.ws = ws
        self.connected.set()
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self.received.put(json.loads(msg.data))
        return ws

    async def next(self, event_type: str) -> dict[str, Any]:
        while True:
            event = await asyncio.wait_for(self.received.get(), 2)
            if event["type"] == event_type:
                return event

    async def send(self, event: dict[str, Any]) -> None:
        assert self.ws is not None
        await self.ws.send_json(event)

    async def close(self, code: int, reason: str) -> None:
        assert self.ws is not None
        await self.ws.close(code=code, message=reason.encode())

    async def reconnected(self) -> None:
        self.connected.clear()
        await asyncio.wait_for(self.connected.wait(), 2)


@pytest.fixture
async def server() -> AsyncIterator[tuple[FakeServer, str]]:
    fake = FakeServer()
    app = web.Application()
    app.router.add_get("/v1/convai/realtime", fake.handler)
    test_server = TestServer(app)
    await test_server.start_server()
    fake.http = aiohttp.ClientSession()
    try:
        yield fake, f"ws://{test_server.host}:{test_server.port}/v1/convai/realtime"
    finally:
        await fake.http.close()
        await test_server.close()


async def _connect(
    url: str, fake: FakeServer, **kwargs: Any
) -> tuple[realtime.RealtimeModel, realtime.RealtimeSession, dict[str, Any]]:
    model = realtime.RealtimeModel(api_key=KEY, base_url=url, http_session=fake.http, **kwargs)
    session = model.session()
    first = await fake.next("session.update")
    return model, session, first


async def _wait_closed(session: realtime.RealtimeSession) -> None:
    await asyncio.wait([session._main_atask], timeout=2)
    assert session._main_atask.done()


def _response_created(response_id: str) -> dict[str, Any]:
    return {
        "type": "response.created",
        "event_id": f"ev_{response_id}",
        "response": {"id": response_id, "object": "realtime.response", "status": "in_progress"},
    }


def _response_done(response_id: str) -> dict[str, Any]:
    return {
        "type": "response.done",
        "event_id": f"ev_done_{response_id}",
        "response": {"id": response_id, "status": "completed", "output": [], "usage": None},
    }


def test_model_requires_an_api_key() -> None:
    with patch.dict("os.environ", {}, clear=True), pytest.raises(ValueError, match="ELEVEN"):
        realtime.RealtimeModel()


def test_model_reads_key_from_env_and_reports_its_capabilities() -> None:
    with patch.dict("os.environ", {"ELEVEN_API_KEY": KEY}):
        model = realtime.RealtimeModel()
    assert model.provider == "ElevenLabs"
    assert model.model == "eleven_realtime_v1_mini"
    caps = model.capabilities
    assert caps.turn_detection and caps.user_transcription and caps.auto_tool_reply_generation
    assert caps.mutable_instructions and caps.mutable_tools
    assert not caps.mutable_chat_context
    assert not caps.can_disable_turn_detection
    assert not caps.per_response_tool_choice
    assert model._opts.max_session_duration is None


async def test_connects_with_api_key_header_and_shaped_session(
    server: tuple[FakeServer, str],
) -> None:
    fake, url = server
    _, session, first = await _connect(
        url, fake, model="eleven_realtime_v1_max", language="pt-BR", max_duration_seconds=1800
    )
    try:
        request = fake.requests[0]
        assert request.headers["xi-api-key"] == KEY
        assert "Authorization" not in request.headers
        assert request.query_string == ""

        shaped = first["session"]
        assert shaped["model"] == "eleven_realtime_v1_max"
        assert shaped["max_duration_seconds"] == 1800
        for key in ("truncation", "reasoning", "tracing"):
            assert key not in shaped
        assert shaped["audio"]["input"]["transcription"] == {"language": "pt-BR"}
        assert shaped["audio"]["input"]["turn_detection"]["type"] == "semantic_vad"
        assert "noise_reduction" not in shaped["audio"]["input"]
        assert shaped["audio"]["output"] == {
            "format": {"type": "audio/pcm", "rate": 24000},
            "voice": "brian",
        }
    finally:
        await session.aclose()


async def test_transcription_is_omitted_without_a_language(server: tuple[FakeServer, str]) -> None:
    fake, url = server
    _, session, first = await _connect(url, fake)
    try:
        assert "transcription" not in first["session"]["audio"]["input"]
        assert "max_duration_seconds" not in first["session"]
    finally:
        await session.aclose()


async def test_tools_update_leaves_the_locked_model_alone(server: tuple[FakeServer, str]) -> None:
    fake, url = server
    _, session, _ = await _connect(url, fake)

    @llm.function_tool
    async def lookup(city: str) -> str:
        """Look up the weather."""
        return "sunny"

    try:
        await session.update_tools([lookup])
        update = await fake.next("session.update")
        assert "model" not in update["session"]
        assert update["session"]["tools"][0]["name"] == "lookup"
    finally:
        await session.aclose()


async def test_locked_options_are_not_sent_once_the_conversation_started(
    server: tuple[FakeServer, str], caplog: pytest.LogCaptureFixture
) -> None:
    fake, url = server
    _, session, _ = await _connect(url, fake)
    try:
        session.send_event({"type": "input_audio_buffer.append", "audio": ""})
        with caplog.at_level(logging.WARNING):
            session.update_options(voice="lily")
        session.send_event({"type": "input_audio_buffer.clear"})

        await fake.next("input_audio_buffer.append")
        assert (await fake.received.get())["type"] == "input_audio_buffer.clear"
        assert "locks these options" in caplog.text
    finally:
        await session.aclose()


async def test_unlabeled_response_resolves_the_pending_reply(
    server: tuple[FakeServer, str],
) -> None:
    fake, url = server
    _, session, _ = await _connect(url, fake)
    try:
        fut = session.generate_reply()
        await fake.next("response.create")
        await fake.send(_response_created("resp_1"))

        generation = await asyncio.wait_for(fut, 2)
        assert generation.user_initiated
        assert generation.response_id == "resp_1"
    finally:
        await session.aclose()


async def test_tool_continuation_is_not_credited_to_a_pending_reply(
    server: tuple[FakeServer, str],
) -> None:
    fake, url = server
    _, session, _ = await _connect(url, fake)
    try:
        session.send_event(
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": "c1", "output": "ok"},
            }
        )
        fut = session.generate_reply()
        await fake.next("response.create")
        await fake.send(_response_created("resp_continuation"))
        await fake.send(_response_done("resp_continuation"))
        await fake.send(_response_created("resp_reply"))

        generation = await asyncio.wait_for(fut, 2)
        assert generation.response_id == "resp_reply"
    finally:
        await session.aclose()


async def test_audio_tags_are_stripped_from_the_transcript(
    server: tuple[FakeServer, str],
) -> None:
    fake, url = server
    _, session, _ = await _connect(url, fake)
    try:
        fut = session.generate_reply()
        await fake.next("response.create")
        await fake.send(_response_created("resp_1"))
        generation = await asyncio.wait_for(fut, 2)
        await fake.send(
            {
                "type": "response.output_item.added",
                "event_id": "ev_item",
                "response_id": "resp_1",
                "output_index": 0,
                "item": {"id": "item_1", "type": "message", "role": "assistant", "content": []},
            }
        )
        await fake.send(
            {
                "type": "response.output_audio_transcript.delta",
                "event_id": "ev_delta",
                "response_id": "resp_1",
                "item_id": "item_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "[cheerful] Hello  [laughs] there",
            }
        )

        message = await asyncio.wait_for(generation.message_stream.__anext__(), 2)
        delta = await asyncio.wait_for(message.text_stream.__anext__(), 2)
        assert delta == " Hello there"
    finally:
        await session.aclose()


async def test_chat_ctx_sync_sends_only_items_the_server_confirms(
    server: tuple[FakeServer, str],
) -> None:
    fake, url = server
    _, session, _ = await _connect(url, fake)
    try:
        session._el_on_server_event(
            {
                "type": "response.output_item.done",
                "item": {"type": "function_call", "call_id": LONG_CALL_ID, "name": "lookup"},
            }
        )
        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(role="system", content="be brief")
        chat_ctx.add_message(role="user", content="hi")
        chat_ctx.items.append(llm.FunctionCall(call_id=LONG_CALL_ID, name="lookup", arguments="{}"))
        chat_ctx.items.append(
            llm.FunctionCallOutput(call_id=LONG_CALL_ID, name="lookup", output="ok", is_error=False)
        )
        chat_ctx.items.append(
            llm.FunctionCallOutput(
                call_id="call_unknown", name="lookup", output="x", is_error=False
            )
        )

        events = session._create_update_chat_ctx_events(chat_ctx)

        assert [ev.type for ev in events] == ["conversation.item.create"] * 2
        user, output = (ev.item for ev in events)
        assert user.type == "message" and user.role == "user"
        assert output.type == "function_call_output"
        assert output.call_id == LONG_CALL_ID
    finally:
        await session.aclose()


async def test_late_tool_output_is_dropped(
    server: tuple[FakeServer, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, url = server
    _, session, _ = await _connect(url, fake)
    try:
        session._el_on_server_event(
            {
                "type": "response.output_item.done",
                "item": {"type": "function_call", "call_id": "call_1", "name": "lookup"},
            }
        )
        monkeypatch.setattr(_realtime_model, "TOOL_OUTPUT_DEADLINE", -1.0)
        chat_ctx = llm.ChatContext(
            [llm.FunctionCallOutput(call_id="call_1", name="lookup", output="ok", is_error=False)]
        )
        assert session._create_update_chat_ctx_events(chat_ctx) == []
    finally:
        await session.aclose()


async def test_truncating_unplayed_audio_sends_nothing(server: tuple[FakeServer, str]) -> None:
    fake, url = server
    _, session, _ = await _connect(url, fake)
    try:
        session.truncate(message_id="item_1", modalities=["audio"], audio_end_ms=0)
        session.send_event({"type": "input_audio_buffer.clear"})
        assert (await fake.received.get())["type"] == "input_audio_buffer.clear"
    finally:
        await session.aclose()


async def test_idle_session_sends_keepalive_audio(
    server: tuple[FakeServer, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, url = server
    monkeypatch.setattr(_realtime_model, "KEEPALIVE_INTERVAL", 0.05)
    _, session, _ = await _connect(url, fake)
    try:
        keepalive = await fake.next("input_audio_buffer.append")
        assert keepalive["audio"] == ""
    finally:
        await session.aclose()


async def test_auth_close_is_fatal(server: tuple[FakeServer, str]) -> None:
    fake, url = server
    _, session, _ = await _connect(
        url, fake, conn_options=APIConnectOptions(max_retry=2, retry_interval=0.01)
    )
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)
    await fake.close(4001, "Authentication required")
    await _wait_closed(session)
    assert len(fake.requests) == 1
    assert [e.recoverable for e in errors] == [False]
    with pytest.raises(APIError, match="4001"):
        await session.aclose()


async def test_capacity_close_reconnects(server: tuple[FakeServer, str]) -> None:
    fake, url = server
    _, session, _ = await _connect(
        url, fake, conn_options=APIConnectOptions(max_retry=2, retry_interval=0.01)
    )
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)
    try:
        await fake.close(4300, "Workspace concurrency limit reached")
        await fake.reconnected()
        await fake.next("session.update")
        assert [e.recoverable for e in errors] == [True]
    finally:
        await session.aclose()


@pytest.mark.parametrize(
    ("code", "reason", "retryable"),
    [
        (1000, "Max call duration exceeded", False),
        (1002, "No user message received for a long period of time", True),
        (1002, "Quota exceeded", False),
        (1002, "No access to the voice", False),
        (1008, "Invalid message", False),
        (1011, "Unexpected server error", True),
        (4001, "Authentication required", False),
        (4003, "Realtime API not enabled", False),
        (4300, "Agent concurrency limit reached", True),
        (4300, "Daily call limit reached", False),
    ],
)
def test_close_codes_map_to_retryability(code: int, reason: str, retryable: bool) -> None:
    error = _close_error(code, reason)
    assert error.retryable is retryable
    assert error.body == {"code": code, "reason": reason}
