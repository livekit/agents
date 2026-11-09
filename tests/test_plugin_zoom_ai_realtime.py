"""Hermetic tests for the Zoom Voice Agent realtime model.

A local aiohttp server stands in for Zoom Voice Agent and speaks the live-session protocol
(``voice-agent.zoom.v2`` subprotocol, server events in ``{"type", "payload"}`` envelopes).

Docs: https://developers.zoom.us/docs/ai-services/voice-agent/live-mode/
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError, llm
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.plugins import zoom_ai
from livekit.plugins.zoom_ai.realtime import RealtimeSession

# The fake server and HTTP session are fixtures bound to a single task.
pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

API_KEY = "test-key"
FAST_RETRY = APIConnectOptions(max_retry=2, retry_interval=0.01, timeout=5)
NO_RETRY = APIConnectOptions(max_retry=0, retry_interval=0.01, timeout=5)


# ---------------------------------------------------------------------------
# Fake Zoom Voice Agent server
# ---------------------------------------------------------------------------


@dataclass
class Conn:
    ws: web.WebSocketResponse
    headers: dict[str, str]
    protocol: str | None
    events: list[dict[str, Any]] = field(default_factory=list)
    audio: bytearray = field(default_factory=bytearray)

    def of_type(self, event_type: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["type"] == event_type]

    async def emit(self, event_type: str, **payload: Any) -> None:
        await self.ws.send_json({"type": event_type, "payload": {"type": event_type, **payload}})


Hook = Callable[[Conn, dict[str, Any]], Awaitable[None]]


class FakeVoiceAgent:
    def __init__(self) -> None:
        self.conns: list[Conn] = []
        self.on_event: Hook | None = None
        self.reject_update: str | None = None
        self.status = 101
        self._new_conn = asyncio.Event()

    async def handler(self, request: web.Request) -> web.StreamResponse:
        if self.status != 101:
            return web.Response(status=self.status)
        ws = web.WebSocketResponse(protocols=("voice-agent.zoom.v2",))
        await ws.prepare(request)
        conn = Conn(ws=ws, headers=dict(request.headers), protocol=ws.ws_protocol)
        self.conns.append(conn)
        self._new_conn.set()
        await conn.emit("session.created", session={"id": "sess_1"})
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            event = json.loads(msg.data)
            conn.events.append(event)
            if event["type"] == "input_audio_buffer.append":
                conn.audio.extend(base64.b64decode(event["audio"]))
            elif event["type"] == "session.update":
                if self.reject_update:
                    await ws.send_json(
                        {
                            "type": "error",
                            "payload": {
                                "type": "error",
                                "error": {
                                    "type": "invalid_request_error",
                                    "message": self.reject_update,
                                },
                            },
                        }
                    )
                    continue
                await conn.emit("session.updated", session={"id": "sess_1", **event["session"]})
            if self.on_event:
                await self.on_event(conn, event)
        return ws

    async def conn(self, index: int = 0, timeout: float = 2.0) -> Conn:
        async def _wait() -> Conn:
            while len(self.conns) <= index:
                self._new_conn.clear()
                await self._new_conn.wait()
            return self.conns[index]

        return await asyncio.wait_for(_wait(), timeout)


async def wait_for(predicate: Callable[[], Any], timeout: float = 2.0) -> None:
    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(_poll(), timeout)


@pytest.fixture
async def server() -> AsyncIterator[tuple[FakeVoiceAgent, str]]:
    fake = FakeVoiceAgent()
    app = web.Application()
    app.router.add_get("/live", fake.handler)
    test_server = TestServer(app)
    await test_server.start_server()
    yield fake, f"ws://{test_server.host}:{test_server.port}/live"
    await test_server.close()


@pytest.fixture
async def http() -> AsyncIterator[aiohttp.ClientSession]:
    async with aiohttp.ClientSession() as session:
        yield session


def make_model(
    url: str, http: aiohttp.ClientSession, **kwargs: Any
) -> zoom_ai.realtime.RealtimeModel:
    kwargs.setdefault("conn_options", NO_RETRY)
    return zoom_ai.realtime.RealtimeModel(api_key=API_KEY, url=url, http_session=http, **kwargs)


@llm.function_tool
async def get_order_status(order_id: str) -> str:
    """Get the current status of an order.

    Args:
        order_id: The order number.
    """
    return "shipped"


async def start(
    model: zoom_ai.realtime.RealtimeModel,
    *,
    instructions: str = "You are a helpful agent.",
    chat_ctx: llm.ChatContext | None = None,
    tools: list[llm.Tool] | None = None,
) -> RealtimeSession:
    session = model.session()
    await session._update_session(
        instructions=instructions,
        chat_ctx=chat_ctx or llm.ChatContext.empty(),
        tools=tools or [],
    )
    return session


async def collect(gen: llm.GenerationCreatedEvent) -> tuple[str, int, list[llm.FunctionCall]]:
    text = ""
    samples = 0
    async for msg in gen.message_stream:

        async def _text(m: llm.MessageGeneration = msg) -> str:
            return "".join([t async for t in m.text_stream])

        async def _audio(m: llm.MessageGeneration = msg) -> int:
            return sum([f.samples_per_channel async for f in m.audio_stream])

        t, n = await asyncio.gather(_text(), _audio())
        text += t
        samples += n
    calls = [c async for c in gen.function_stream]
    return text, samples, calls


async def reply(conn: Conn, *, response_id: str = "resp_1", item_id: str = "item_1") -> None:
    """Script one spoken agent response."""
    await conn.emit("response.created", response={"id": response_id, "status": "in_progress"})
    await conn.emit(
        "response.output_item.added",
        response_id=response_id,
        item={"id": item_id, "type": "message"},
    )
    await conn.emit(
        "response.output_audio_transcript.delta",
        response_id=response_id,
        item_id=item_id,
        delta="Hello there.",
    )
    pcm = b"\x01\x00" * 1600  # 100 ms at 16 kHz
    await conn.emit(
        "response.output_audio.delta",
        response_id=response_id,
        item_id=item_id,
        delta=base64.b64encode(pcm).decode(),
    )
    await conn.emit("response.output_audio.done", response_id=response_id, item_id=item_id)
    await conn.emit(
        "response.done",
        response={
            "id": response_id,
            "status": "completed",
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        },
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZOOM_AI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ZOOM_AI_API_KEY"):
        zoom_ai.realtime.RealtimeModel()
    monkeypatch.setenv("ZOOM_AI_API_KEY", "from-env")
    assert zoom_ai.realtime.RealtimeModel()._opts.api_key == "from-env"


def test_bring_your_own_provider_needs_headers() -> None:
    with pytest.raises(ValueError, match="provider_headers"):
        zoom_ai.realtime.RealtimeModel(api_key=API_KEY, llm_provider="openai", model="gpt-4.1")


def test_model_metadata_and_capabilities() -> None:
    model = zoom_ai.realtime.RealtimeModel(api_key=API_KEY)
    assert model.model == "gemma-4"
    assert model.provider == "Zoom"
    caps = model.capabilities
    assert caps.turn_detection and caps.user_transcription and caps.audio_output
    assert caps.message_truncation
    assert not caps.auto_tool_reply_generation
    assert not caps.mutable_chat_context
    assert caps.mutable_instructions and caps.mutable_tools


# ---------------------------------------------------------------------------
# Session setup
# ---------------------------------------------------------------------------


async def test_first_event_is_one_complete_session_update(
    server: tuple[FakeVoiceAgent, str], http: aiohttp.ClientSession
) -> None:
    fake, url = server
    model = make_model(
        url,
        http,
        voice="tony",
        vocabulary=["LiveKit"],
        reasoning_effort="low",
        tool_choice="auto",
    )
    session = model.session()
    await asyncio.sleep(0.05)
    assert not fake.conns, "connects only once the agent has configured the session"

    await session._update_session(
        instructions="Be brief.", chat_ctx=llm.ChatContext.empty(), tools=[get_order_status]
    )
    conn = await fake.conn()
    await wait_for(lambda: conn.events)

    assert conn.protocol == "voice-agent.zoom.v2"
    assert conn.headers["Authorization"] == f"Bearer {API_KEY}"
    assert conn.headers["x-zm-trackingid"].startswith("zoom_ai_community_livekit_")
    first = conn.events[0]
    assert first["type"] == "session.update"
    s = first["session"]
    assert s["listen"] == {
        "format": {"type": "pcm16", "rate": 16000},
        "transcription": {"vocabulary": {"phrases": ["LiveKit"]}},
    }
    assert s["think"]["provider"] == {"type": "zoom", "model": "gemma-4"}
    assert s["think"]["instructions"] == "Be brief."
    assert s["think"]["tool_choice"] == "auto"
    assert s["think"]["reasoning"] == {"effort": "low"}
    (tool,) = s["think"]["tools"]
    assert tool["type"] == "function"
    assert tool["name"] == "get_order_status"
    assert tool["parameters"]["properties"]["order_id"]["type"] == "string"
    assert s["speak"] == {"format": {"type": "pcm16"}, "voice": "tony"}
    assert len(conn.of_type("session.update")) == 1

    await wait_for(lambda: session.session_id == "sess_1")
    await session.aclose()


async def test_bring_your_own_key_is_sent_in_provider(
    server: tuple[FakeVoiceAgent, str], http: aiohttp.ClientSession
) -> None:
    fake, url = server
    model = make_model(
        url,
        http,
        llm_provider="anthropic",
        model="claude-sonnet-5-5",
        provider_headers={"X-Api-Key": "sk-ant"},
        provider_metadata={"context_window": 200000},
    )
    session = await start(model)
    conn = await fake.conn()
    await wait_for(lambda: conn.events)
    assert conn.events[0]["session"]["think"]["provider"] == {
        "type": "anthropic",
        "model": "claude-sonnet-5-5",
        "headers": {"X-Api-Key": "sk-ant"},
        "metadata": {"context_window": 200000},
    }
    await session.aclose()


async def test_audio_is_resampled_to_16k_pcm(
    server: tuple[FakeVoiceAgent, str], http: aiohttp.ClientSession
) -> None:
    fake, url = server
    session = await start(make_model(url, http))
    frame = rtc.AudioFrame(
        data=b"\x00\x00" * 4800, sample_rate=48000, num_channels=1, samples_per_channel=4800
    )
    for _ in range(5):  # 500 ms
        session.push_audio(frame)
    session.commit_audio()
    conn = await fake.conn()
    await wait_for(lambda: len(conn.audio) >= 2 * 16000 * 4 // 10)
    assert conn.of_type("session.update")[0] is conn.events[0]
    # no commit/clear events: Zoom ends turns itself
    assert not conn.of_type("input_audio_buffer.commit")
    await session.aclose()


async def test_rejected_configuration_is_fatal(
    server: tuple[FakeVoiceAgent, str], http: aiohttp.ClientSession
) -> None:
    fake, url = server
    fake.reject_update = "Unknown voice: bogus"
    session = await start(make_model(url, http, voice="bogus"))
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)
    await wait_for(lambda: session._main_atask.done())
    assert errors and not errors[-1].recoverable
    assert "Unknown voice" in str(errors[-1].error)
    await session.aclose()


async def test_handshake_rejection_is_not_retried(
    server: tuple[FakeVoiceAgent, str], http: aiohttp.ClientSession
) -> None:
    fake, url = server
    fake.status = 401
    session = await start(make_model(url, http, conn_options=FAST_RETRY))
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)
    await wait_for(lambda: session._main_atask.done())
    assert len(errors) == 1 and not errors[0].recoverable
    assert isinstance(errors[0].error, APIStatusError)
    assert errors[0].error.status_code == 401
    assert API_KEY not in repr(errors[0].error)
    await session.aclose()


async def test_history_before_the_conversation_goes_into_instructions(
    server: tuple[FakeVoiceAgent, str], http: aiohttp.ClientSession
) -> None:
    fake, url = server
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="My order is 123.")
    chat_ctx.add_message(role="assistant", content="Let me transfer you.")
    session = await start(make_model(url, http), instructions="You are Nova.", chat_ctx=chat_ctx)
    conn = await fake.conn()
    await wait_for(lambda: conn.events)
    instructions = conn.events[0]["session"]["think"]["instructions"]
    assert instructions.startswith("You are Nova.\n\n")
    assert "user: My order is 123." in instructions
    assert "assistant: Let me transfer you." in instructions
    # the history counts as part of the session, so a handoff can reuse it
    assert [i.id for i in session.chat_ctx.items] == [i.id for i in chat_ctx.items]
    await session.aclose()


async def test_instruction_and_option_updates_resend_the_session(
    server: tuple[FakeVoiceAgent, str], http: aiohttp.ClientSession
) -> None:
    fake, url = server
    model = make_model(url, http, reasoning_effort="low")
    session = await start(model)
    conn = await fake.conn()
    await session.update_instructions("New instructions.")
    model.update_options(voice="ava")
    await wait_for(lambda: len(conn.of_type("session.update")) == 3)
    updates = conn.of_type("session.update")
    assert updates[1]["session"]["think"]["instructions"] == "New instructions."
    assert updates[2]["session"]["speak"]["voice"] == "ava"
    # reasoning is set in the connection's first update only
    assert "reasoning" not in updates[1]["session"]["think"]
    await session.aclose()


async def test_cleared_settings_are_reset_on_later_updates(
    server: tuple[FakeVoiceAgent, str], http: aiohttp.ClientSession
) -> None:
    fake, url = server
    model = make_model(url, http, vocabulary=["LiveKit"])
    session = await start(model)
    conn = await fake.conn()
    session.update_options(tool_choice="required")
    session.update_options(tool_choice=None)  # what AgentActivity does after a forced turn
    model.update_options(vocabulary=None)
    await session.update_instructions("")
    await wait_for(lambda: len(conn.of_type("session.update")) == 5)
    first, required, cleared, no_vocab, no_instructions = conn.of_type("session.update")

    # the first update leaves unset options out, so Zoom's defaults apply
    assert "tool_choice" not in first["session"]["think"]
    assert "response_token_limit" not in first["session"]["think"]
    assert required["session"]["think"]["tool_choice"] == "required"
    # once sent, clearing a setting sends its reset value instead of leaving it out
    assert cleared["session"]["think"]["tool_choice"] == "auto"
    assert no_vocab["session"]["listen"]["transcription"] == {"vocabulary": {"phrases": []}}
    assert no_instructions["session"]["think"]["instructions"] == ""
    # a setting never sent on this connection stays out
    assert "response_token_limit" not in no_instructions["session"]["think"]
    await session.aclose()


async def test_history_replaced_before_the_conversation_starts(
    server: tuple[FakeVoiceAgent, str], http: aiohttp.ClientSession
) -> None:
    fake, url = server
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="My order is 123.")
    session = await start(make_model(url, http), instructions="You are Nova.", chat_ctx=chat_ctx)
    conn = await fake.conn()
    await wait_for(lambda: conn.events)
    assert "My order is 123." in conn.events[0]["session"]["think"]["instructions"]

    await session.update_chat_ctx(chat_ctx.copy())  # unchanged: nothing to send
    await session.update_chat_ctx(llm.ChatContext.empty())  # cleared
    await wait_for(lambda: len(conn.of_type("session.update")) >= 2)
    await asyncio.sleep(0.05)
    updates = conn.of_type("session.update")
    assert len(updates) == 2
    assert updates[1]["session"]["think"]["instructions"] == "You are Nova."
    assert session.chat_ctx.items == []
    await session.aclose()


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


async def test_generate_reply_streams_audio_and_transcript(
    server: tuple[FakeVoiceAgent, str], http: aiohttp.ClientSession
) -> None:
    fake, url = server

    async def on_event(conn: Conn, event: dict[str, Any]) -> None:
        if event["type"] == "response.create":
            await reply(conn)

    fake.on_event = on_event
    session = await start(make_model(url, http))
    metrics: list[RealtimeModelMetrics] = []
    session.on(
        "metrics_collected",
        lambda m: (
            metrics.append(m) if isinstance(m, RealtimeModelMetrics) and m.request_id else None
        ),
    )

    gen = await asyncio.wait_for(session.generate_reply(), 2)
    assert gen.user_initiated
    assert gen.response_id == "resp_1"
    text, samples, calls = await asyncio.wait_for(collect(gen), 2)
    assert text == "Hello there."
    assert samples == 1600
    assert calls == []

    await wait_for(lambda: metrics)
    assert metrics[0].request_id == "resp_1"
    assert metrics[0].output_tokens == 5
    assert metrics[0].metadata and metrics[0].metadata.model_provider == "Zoom"
    (item,) = session.chat_ctx.items
    assert isinstance(item, llm.ChatMessage)
    assert item.role == "assistant" and item.text_content == "Hello there."
    await session.aclose()


async def test_server_started_response_is_not_user_initiated(
    server: tuple[FakeVoiceAgent, str], http: aiohttp.ClientSession
) -> None:
    fake, url = server
    session = await start(make_model(url, http))
    gens: list[llm.GenerationCreatedEvent] = []
    started: list[llm.InputSpeechStartedEvent] = []
    transcripts: list[llm.InputTranscriptionCompleted] = []
    session.on("generation_created", gens.append)
    session.on("input_speech_started", started.append)
    session.on("input_audio_transcription_completed", transcripts.append)

    conn = await fake.conn()
    await conn.emit("input_audio_buffer.speech_started", item_id="user_1")
    await conn.emit("input_audio_buffer.speech_stopped", item_id="user_1")
    await conn.emit(
        "conversation.item.input_audio_transcription.completed",
        item_id="user_1",
        transcript="Where is my order?",
    )
    await reply(conn)
    await wait_for(lambda: gens and transcripts)
    assert started
    assert not gens[0].user_initiated
    assert transcripts[0].transcript == "Where is my order?"
    assert transcripts[0].item_id == "user_1"
    assert transcripts[0].turn_started_at is not None
    await asyncio.wait_for(collect(gens[0]), 2)
    roles = [(i.role, i.text_content) for i in session.chat_ctx.items if i.type == "message"]
    assert roles == [("user", "Where is my order?"), ("assistant", "Hello there.")]
    await session.aclose()


async def test_reply_waits_for_the_active_response(
    server: tuple[FakeVoiceAgent, str], http: aiohttp.ClientSession
) -> None:
    fake, url = server
    session = await start(make_model(url, http))
    conn = await fake.conn()
    await conn.emit("response.created", response={"id": "resp_auto"})
    await wait_for(lambda: session._active_response_id == "resp_auto")

    fut = session.generate_reply()
    await asyncio.sleep(0.05)
    assert not conn.of_type("response.create"), "only one response may be active"

    await conn.emit("response.done", response={"id": "resp_auto", "status": "completed"})
    await wait_for(lambda: conn.of_type("response.create"))
    await reply(conn, response_id="resp_2")
    gen = await asyncio.wait_for(fut, 2)
    assert gen.user_initiated and gen.response_id == "resp_2"
    await session.aclose()


ACTIVE_RESPONSE_ERROR = {
    "type": "error",
    "payload": {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "Conversation already has an active response",
        },
    },
}


async def test_active_response_error_retries_after_response_done(
    server: tuple[FakeVoiceAgent, str], http: aiohttp.ClientSession
) -> None:
    fake, url = server
    session = await start(make_model(url, http))
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)
    conn = await fake.conn()
    fut = session.generate_reply()
    await wait_for(lambda: conn.of_type("response.create"))
    # the server was starting its own response and refuses ours before announcing it
    await conn.ws.send_json(ACTIVE_RESPONSE_ERROR)
    await conn.emit("response.created", response={"id": "resp_auto"})
    await conn.emit("response.done", response={"id": "resp_auto", "status": "completed"})
    await wait_for(lambda: len(conn.of_type("response.create")) == 2)
    await reply(conn, response_id="resp_2")
    gen = await asyncio.wait_for(fut, 2)
    assert gen.response_id == "resp_2"
    assert not errors
    await session.aclose()


async def test_server_response_racing_the_request_answers_it(
    server: tuple[FakeVoiceAgent, str], http: aiohttp.ClientSession
) -> None:
    fake, url = server
    session = await start(make_model(url, http))
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)
    conn = await fake.conn()
    fut = session.generate_reply()
    await wait_for(lambda: conn.of_type("response.create"))
    # the server's own response is announced first, then ours is refused
    await conn.emit("response.created", response={"id": "resp_auto"})
    await conn.ws.send_json(ACTIVE_RESPONSE_ERROR)
    gen = await asyncio.wait_for(fut, 2)
    assert gen.response_id == "resp_auto"
    await conn.emit("response.done", response={"id": "resp_auto", "status": "completed"})
    await asyncio.sleep(0.05)
    assert len(conn.of_type("response.create")) == 1
    assert not errors
    await session.aclose()


async def test_interrupt_cancels_and_truncate_aligns_history(
    server: tuple[FakeVoiceAgent, str], http: aiohttp.ClientSession
) -> None:
    fake, url = server
    session = await start(make_model(url, http))
    conn = await fake.conn()
    await conn.emit("response.created", response={"id": "resp_9"})
    await wait_for(lambda: session._active_response_id == "resp_9")
    session.interrupt()
    session.truncate(message_id="item_9", modalities=["audio", "text"], audio_end_ms=1500)
    await wait_for(lambda: conn.of_type("conversation.item.truncate"))
    assert conn.of_type("response.cancel") == [{"type": "response.cancel", "response_id": "resp_9"}]
    assert conn.of_type("conversation.item.truncate") == [
        {
            "type": "conversation.item.truncate",
            "item_id": "item_9",
            "content_index": 0,
            "audio_end_ms": 1500,
        }
    ]
    await session.aclose()


async def test_interrupt_before_creation_discards_the_response(
    server: tuple[FakeVoiceAgent, str], http: aiohttp.ClientSession
) -> None:
    fake, url = server
    session = await start(make_model(url, http))
    gens: list[llm.GenerationCreatedEvent] = []
    session.on("generation_created", gens.append)
    conn = await fake.conn()
    fut = session.generate_reply()
    await wait_for(lambda: conn.of_type("response.create"))
    session.interrupt()
    await conn.emit("response.created", response={"id": "resp_late"})
    with pytest.raises(llm.RealtimeError):
        await asyncio.wait_for(fut, 2)
    await wait_for(lambda: conn.of_type("response.cancel"))
    assert conn.of_type("response.cancel")[0]["response_id"] == "resp_late"
    assert not gens
    await session.aclose()


# ---------------------------------------------------------------------------
# Function calling
# ---------------------------------------------------------------------------


async def test_function_call_and_result_round_trip(
    server: tuple[FakeVoiceAgent, str], http: aiohttp.ClientSession
) -> None:
    fake, url = server
    session = await start(make_model(url, http), tools=[get_order_status])
    gens: list[llm.GenerationCreatedEvent] = []
    session.on("generation_created", gens.append)
    conn = await fake.conn()
    await conn.emit(
        "conversation.item.input_audio_transcription.completed",
        item_id="user_1",
        transcript="Status of order 123?",
    )
    await conn.emit("response.created", response={"id": "resp_fc"})
    await conn.emit(
        "response.function_call_arguments.done",
        event_id="event_abc123",
        response_id="resp_fc",
        item_id="item_fc",
        output_index=0,
        call_id="call_abc123",
        name="get_order_status",
        arguments='{"order_id":"123"}',
    )
    await conn.emit("response.done", response={"id": "resp_fc", "status": "completed"})
    await wait_for(lambda: gens)
    _, _, calls = await asyncio.wait_for(collect(gens[0]), 2)
    (call,) = calls
    assert (call.call_id, call.name, call.arguments) == (
        "call_abc123",
        "get_order_status",
        '{"order_id":"123"}',
    )

    # what AgentActivity does once the tool has run
    chat_ctx = session.chat_ctx.copy()
    chat_ctx.items.append(
        llm.FunctionCallOutput(
            call_id="call_abc123", name="get_order_status", output="shipped", is_error=False
        )
    )
    await session.update_chat_ctx(chat_ctx)
    session.interrupt()  # no active response: nothing to cancel
    fut = session.generate_reply(tool_choice="none")

    await wait_for(lambda: conn.of_type("response.create"))
    assert conn.of_type("conversation.item.create") == [
        {
            "type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": "call_abc123", "output": "shipped"},
        }
    ]
    assert not conn.of_type("response.cancel")
    item_create = conn.events.index(conn.of_type("conversation.item.create")[0])
    assert item_create < conn.events.index(conn.of_type("response.create")[0])
    await reply(conn, response_id="resp_after")
    assert (await asyncio.wait_for(fut, 2)).response_id == "resp_after"
    await session.aclose()


async def test_messages_cannot_be_added_mid_conversation(
    server: tuple[FakeVoiceAgent, str], http: aiohttp.ClientSession
) -> None:
    fake, url = server
    session = await start(make_model(url, http))
    conn = await fake.conn()
    await conn.emit(
        "conversation.item.input_audio_transcription.completed", item_id="u1", transcript="Hi"
    )
    await wait_for(lambda: session.chat_ctx.items)
    chat_ctx = session.chat_ctx.copy()
    chat_ctx.add_message(role="user", content="typed text")
    await session.update_chat_ctx(chat_ctx)
    await asyncio.sleep(0.05)
    assert not conn.of_type("conversation.item.create")
    assert len(conn.of_type("session.update")) == 1
    assert len(session.chat_ctx.items) == 1
    await session.aclose()


# ---------------------------------------------------------------------------
# Reconnection
# ---------------------------------------------------------------------------


async def test_reconnects_with_the_conversation_after_the_server_closes(
    server: tuple[FakeVoiceAgent, str], http: aiohttp.ClientSession
) -> None:
    fake, url = server
    session = await start(make_model(url, http, conn_options=FAST_RETRY), instructions="Be kind.")
    reconnected: list[llm.RealtimeSessionReconnectedEvent] = []
    session.on("session_reconnected", reconnected.append)
    conn = await fake.conn()
    await conn.emit(
        "conversation.item.input_audio_transcription.completed",
        item_id="u1",
        transcript="My name is Sam.",
    )
    await reply(conn)
    await wait_for(lambda: len(session.chat_ctx.items) == 2)

    await conn.ws.close()  # e.g. the 60-minute session limit
    conn2 = await fake.conn(1)
    await wait_for(lambda: conn2.events and reconnected)
    first = conn2.events[0]
    assert first["type"] == "session.update"
    instructions = first["session"]["think"]["instructions"]
    assert instructions.startswith("Be kind.\n\n")
    assert "user: My name is Sam." in instructions
    assert "assistant: Hello there." in instructions
    await session.aclose()


async def test_aclose_before_configuration_never_connects(
    server: tuple[FakeVoiceAgent, str], http: aiohttp.ClientSession
) -> None:
    fake, url = server
    session = make_model(url, http).session()
    await asyncio.wait_for(session.aclose(), 2)
    assert not fake.conns


# ---------------------------------------------------------------------------
# Driven by AgentSession
# ---------------------------------------------------------------------------


async def test_agent_session_runs_a_tool_turn_end_to_end(
    server: tuple[FakeVoiceAgent, str], http: aiohttp.ClientSession
) -> None:
    from livekit.agents import Agent, AgentSession, function_tool

    from .fake_io import FakeAudioOutput

    fake, url = server
    tool_args: list[str] = []
    creates = 0

    async def on_event(conn: Conn, event: dict[str, Any]) -> None:
        nonlocal creates
        if event["type"] != "response.create":
            return
        creates += 1
        if creates == 1:
            await conn.emit("response.created", response={"id": "resp_fc"})
            await conn.emit(
                "response.function_call_arguments.done",
                response_id="resp_fc",
                item_id="item_fc",
                call_id="call_1",
                name="get_order_status",
                arguments='{"order_id":"123"}',
            )
            await conn.emit("response.done", response={"id": "resp_fc", "status": "completed"})
        else:
            await reply(conn, response_id="resp_answer", item_id="item_answer")

    fake.on_event = on_event

    class OrderAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You track orders.")

        @function_tool
        async def get_order_status(self, order_id: str) -> str:
            """Get the current status of an order.

            Args:
                order_id: The order number.
            """
            tool_args.append(order_id)
            return "shipped"

    async with AgentSession(llm=make_model(url, http)) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(OrderAgent())
        handle = session.generate_reply()
        await asyncio.wait_for(handle.wait_for_playout(), 10)
        conn = await fake.conn()
        await wait_for(lambda: creates == 2, timeout=5)
        await wait_for(
            lambda: any(
                i.type == "message" and i.role == "assistant" and i.text_content == "Hello there."
                for i in session.history.items
            ),
            timeout=5,
        )

    assert tool_args == ["123"]
    (update, *_) = conn.of_type("session.update")
    assert update["session"]["think"]["instructions"].startswith("You track orders.")
    assert [t["name"] for t in update["session"]["think"]["tools"]] == ["get_order_status"]
    assert conn.of_type("conversation.item.create") == [
        {
            "type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": "call_1", "output": "shipped"},
        }
    ]
    # the tool reply is requested once the result is in
    types = [e["type"] for e in conn.events]
    second_create = [n for n, t in enumerate(types) if t == "response.create"][1]
    assert types.index("conversation.item.create") < second_create
