"""Azure Voice Live realtime session tests, against a local fake of the Voice Live websocket."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import sys
import time
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from typing import Any, TypeVar

import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestServer
from azure.ai.voicelive.models import (
    AudioInputTranscriptionOptions,
    Response,
    ServerEventResponseCreated,
)
from azure.core.credentials import AccessToken

from livekit import rtc
from livekit.agents import Agent, AgentSession, APIConnectOptions, function_tool, llm
from livekit.plugins import azure as azure_plugin
from livekit.plugins.azure.realtime import RealtimeModel, RealtimeSession, realtime_model

from .fake_io import FakeAudioOutput

pytestmark = pytest.mark.unit

T = TypeVar("T")

# 20ms of 24kHz mono PCM16
_PCM_20MS = b"\x01\x00" * 480


class _Connection:
    def __init__(self, ws: web.WebSocketResponse, index: int) -> None:
        self.ws = ws
        self.index = index
        self.events: list[dict[str, Any]] = []
        self.item_ids: list[str] = []
        # bytes in the input audio buffer
        self.input_audio = 0

    async def send(self, event_type: str, **fields: Any) -> None:
        if not self.ws.closed:
            await self.ws.send_json({"type": event_type, "event_id": f"evt_{event_type}", **fields})


ResponseHandler = Callable[[_Connection, dict[str, Any], str], Awaitable[None]]


class _FakeVoiceLive:
    """Serves the Voice Live websocket protocol, answering the way the service does."""

    def __init__(self) -> None:
        self.endpoint = ""
        self.connections: list[_Connection] = []
        self.headers: list[dict[str, str]] = []
        self.on_response: ResponseHandler = _reply
        self.reject_items = False
        self.answer_items = True
        # close the socket instead of answering the next conversation.item.create
        self.drop_on_item_create = False
        # close the socket instead of answering the next input_audio_buffer.commit
        self.drop_on_commit = False
        # connections from this index on close on their first conversation.item.create
        self.drop_item_creates_from: int | None = None
        self.accept_gate: asyncio.Event | None = None
        self._responses = 0

    @property
    def events(self) -> list[dict[str, Any]]:
        return [event for conn in self.connections for event in conn.events]

    def sent(self, event_type: str) -> list[dict[str, Any]]:
        return [event for event in self.events if event["type"] == event_type]

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        if self.accept_gate is not None:
            await self.accept_gate.wait()

        self.headers.append(dict(request.headers))
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        conn = _Connection(ws, len(self.connections))
        self.connections.append(conn)
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            event = json.loads(msg.data)
            conn.events.append(event)
            await self._answer(conn, event)
        return ws

    async def _answer(self, conn: _Connection, event: dict[str, Any]) -> None:
        if event["type"] == "session.update":
            await conn.send("session.updated", session={"id": f"sess_{conn.index}"})
        elif event["type"] == "conversation.item.create":
            if self.drop_on_item_create or (
                self.drop_item_creates_from is not None
                and conn.index >= self.drop_item_creates_from
            ):
                self.drop_on_item_create = False
                await conn.ws.close()
                return
            if not self.answer_items:
                return
            if self.reject_items:
                await conn.send(
                    "error",
                    error={
                        "type": "invalid_request_error",
                        "code": "invalid_value",
                        "message": "item rejected",
                        "event_id": event.get("event_id"),
                    },
                )
                return
            item = event["item"]
            anchor = event.get("previous_item_id")
            if anchor == "root":
                # the item opens the conversation, which names no predecessor for it
                conn.item_ids.insert(0, item["id"])
                previous = None
            else:
                previous = anchor or (conn.item_ids[-1] if conn.item_ids else None)
                index = conn.item_ids.index(previous) + 1 if previous in conn.item_ids else 0
                conn.item_ids.insert(index, item["id"])
            await conn.send("conversation.item.created", previous_item_id=previous, item=item)
        elif event["type"] == "response.create":
            self._responses += 1
            await self.on_response(conn, event, f"resp_{self._responses}")
        elif event["type"] == "conversation.item.delete":
            await conn.send("conversation.item.deleted", item_id=event["item_id"])
        elif event["type"] == "input_audio_buffer.append":
            conn.input_audio += len(base64.b64decode(event["audio"]))
        elif event["type"] == "input_audio_buffer.commit":
            if self.drop_on_commit:
                self.drop_on_commit = False
                await conn.ws.close()
                return
            # 100ms of 24kHz PCM16
            if conn.input_audio < 4800:
                await conn.send(
                    "error",
                    error={
                        "type": "invalid_request_error",
                        "code": "input_audio_buffer_commit_empty",
                        "message": "buffer too small",
                        "event_id": event.get("event_id"),
                    },
                )
                return
            await self.commit(conn)
        elif event["type"] == "input_audio_buffer.clear":
            conn.input_audio = 0
            await conn.send("input_audio_buffer.cleared")

    async def commit(self, conn: _Connection, *, keep: int = 0) -> None:
        """Commit the input audio buffer, as requested or as the turn detection of Azure does.

        `keep` is the audio that arrived past the commit boundary, which stays buffered.
        """
        conn.input_audio = keep
        await conn.send(
            "input_audio_buffer.committed",
            previous_item_id=conn.item_ids[-1] if conn.item_ids else None,
            item_id=f"item_turn_{len(self.sent('input_audio_buffer.append'))}",
        )

    async def close(self) -> None:
        for conn in self.connections:
            await conn.ws.close()


def _metadata(event: dict[str, Any]) -> dict[str, str] | None:
    return event.get("response", {}).get("metadata")


async def _created(conn: _Connection, response_id: str, metadata: dict[str, str] | None) -> None:
    await conn.send(
        "response.created",
        response={
            "id": response_id,
            "object": "realtime.response",
            "status": "in_progress",
            "output": [],
            "metadata": metadata,
        },
    )


async def _add_message(
    conn: _Connection, response_id: str, item_id: str, *, part_type: str = "audio"
) -> None:
    item = {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "status": "in_progress",
        "content": [],
    }
    await conn.send(
        "response.output_item.added", response_id=response_id, output_index=0, item=item
    )
    await conn.send(
        "conversation.item.created",
        previous_item_id=conn.item_ids[-1] if conn.item_ids else None,
        item=item,
    )
    conn.item_ids.append(item_id)
    await conn.send(
        "response.content_part.added",
        response_id=response_id,
        item_id=item_id,
        output_index=0,
        content_index=0,
        part={"type": part_type, "transcript": ""} if part_type == "audio" else {"type": "text"},
    )


async def _audio(conn: _Connection, response_id: str, item_id: str, count: int = 1) -> None:
    for _ in range(count):
        await conn.send(
            "response.audio.delta",
            response_id=response_id,
            item_id=item_id,
            output_index=0,
            content_index=0,
            delta=base64.b64encode(_PCM_20MS).decode(),
        )


async def _transcript(conn: _Connection, response_id: str, item_id: str, text: str) -> None:
    await conn.send(
        "response.audio_transcript.delta",
        response_id=response_id,
        item_id=item_id,
        output_index=0,
        content_index=0,
        delta=text,
    )


async def _text(conn: _Connection, response_id: str, item_id: str, text: str) -> None:
    await conn.send(
        "response.text.delta",
        response_id=response_id,
        item_id=item_id,
        output_index=0,
        content_index=0,
        delta=text,
    )


async def _done(conn: _Connection, response_id: str, status: str = "completed") -> None:
    await conn.send(
        "response.done",
        response={
            "id": response_id,
            "object": "realtime.response",
            "status": status,
            "output": [],
            "usage": {
                "total_tokens": 12,
                "input_tokens": 5,
                "output_tokens": 7,
                "input_token_details": {"text_tokens": 5, "audio_tokens": 0, "cached_tokens": 0},
                "output_token_details": {"text_tokens": 2, "audio_tokens": 5},
            },
        },
    )


async def _reply(conn: _Connection, event: dict[str, Any], response_id: str) -> None:
    item_id = f"item_{response_id}"
    await _created(conn, response_id, _metadata(event))
    await _add_message(conn, response_id, item_id)
    await _audio(conn, response_id, item_id)
    await _transcript(conn, response_id, item_id, "Hello!")
    await _done(conn, response_id)


async def _ignore(conn: _Connection, event: dict[str, Any], response_id: str) -> None:
    pass


@pytest.fixture
async def voice_live() -> AsyncIterator[_FakeVoiceLive]:
    fake = _FakeVoiceLive()
    app = web.Application()
    app.router.add_get("/voice-live/realtime", fake.handle)
    server = TestServer(app, host="127.0.0.1")
    await server.start_server()
    fake.endpoint = f"http://127.0.0.1:{server.port}"
    try:
        yield fake
    finally:
        if fake.accept_gate is not None:
            fake.accept_gate.set()
        await fake.close()
        await server.close()


def _model(fake: _FakeVoiceLive, **kwargs: Any) -> RealtimeModel:
    kwargs.setdefault("api_key", "test-key")
    return RealtimeModel(
        endpoint=fake.endpoint,
        conn_options=APIConnectOptions(max_retry=3, retry_interval=0.0, timeout=5.0),
        **kwargs,
    )


@contextlib.asynccontextmanager
async def _session(fake: _FakeVoiceLive, **kwargs: Any) -> AsyncIterator[RealtimeSession]:
    model = _model(fake, **kwargs)
    session = model.session()
    try:
        yield session
    finally:
        await session.aclose()
        await model.aclose()


async def _wait_until(predicate: Callable[[], object], timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


async def _connected(fake: _FakeVoiceLive, index: int = 0) -> _Connection:
    await _wait_until(
        lambda: (
            len(fake.connections) > index
            and any(e["type"] == "session.update" for e in fake.connections[index].events)
        )
    )
    return fake.connections[index]


async def _collect(stream: AsyncIterable[T], timeout: float = 5.0) -> list[T]:
    async def _read() -> list[T]:
        return [value async for value in stream]

    return await asyncio.wait_for(_read(), timeout)


async def _first(stream: AsyncIterable[T]) -> T:
    async def _read() -> T:
        async for value in stream:
            return value
        raise AssertionError("stream ended")

    return await asyncio.wait_for(_read(), 5.0)


@llm.function_tool
async def lookup(query: str) -> str:
    """Look something up.

    Args:
        query: What to look up.
    """
    return query


async def test_session_honors_the_realtime_interface(voice_live: _FakeVoiceLive) -> None:
    model = _model(voice_live)
    assert model.capabilities.can_disable_turn_detection is False
    assert model.capabilities.per_response_tool_choice is True

    # manual turn taking is unsupported, the flag is accepted and server VAD stays on
    session = model.session(turn_detection_disabled=True)
    try:
        conn = await _connected(voice_live)
        assert conn.events[0]["session"]["turn_detection"]["type"] == "server_vad"
    finally:
        await session.aclose()


async def test_initial_configuration(voice_live: _FakeVoiceLive) -> None:
    transcription = AudioInputTranscriptionOptions(model="whisper-1", language="en-US")
    async with _session(voice_live, input_audio_transcription=transcription) as session:
        await session.update_instructions("Be helpful.")
        conn = await _connected(voice_live)

        config = conn.events[0]["session"]
        assert config["instructions"] == "Be helpful."
        assert config["voice"] == {
            "type": "azure-standard",
            "name": "en-US-AvaMultilingualNeural",
            "locale": "en-US",
        }
        assert config["input_audio_transcription"]["language"] == "en-US"
        assert config["tool_choice"] == "auto"
        # no tools yet: the field is left out rather than sent as null
        assert "tools" not in config
        assert voice_live.headers[0]["api-key"] == "test-key"

    languages = AudioInputTranscriptionOptions(model="whisper-1", language="en,zh")
    async with _session(voice_live, input_audio_transcription=languages):
        conn = await _connected(voice_live, index=1)
        # a voice can't be pinned to a list of languages
        assert "locale" not in conn.events[0]["session"]["voice"]


async def test_turn_detection_and_transcription_can_be_disabled(
    voice_live: _FakeVoiceLive,
) -> None:
    async with _session(voice_live, turn_detection=None, input_audio_transcription=None) as session:
        assert session.realtime_model.capabilities.turn_detection is False
        assert session.realtime_model.capabilities.user_transcription is False

        config = (await _connected(voice_live)).events[0]["session"]
        assert config["turn_detection"] is None
        assert config["input_audio_transcription"] is None


@pytest.mark.parametrize(
    ("model", "transcription_model"),
    [
        # whisper-1 is documented for gpt-realtime and gpt-realtime-mini, the other realtime
        # models keep it as the previous default
        ("gpt-realtime", "whisper-1"),
        ("gpt-realtime-mini", "whisper-1"),
        ("gpt-realtime-1.5", "whisper-1"),
        ("azure-realtime", "whisper-1"),
        # documented to transcribe with azure-speech, not whisper-1
        ("gpt-4o", "azure-speech"),
        ("gpt-4.1", "azure-speech"),
        ("gpt-5-mini", "azure-speech"),
        ("phi4-mm-realtime", "azure-speech"),
    ],
)
async def test_default_transcription_depends_on_the_model(
    voice_live: _FakeVoiceLive, model: str, transcription_model: str
) -> None:
    async with _session(voice_live, model=model) as session:
        assert session.realtime_model.capabilities.user_transcription is True

        config = (await _connected(voice_live)).events[0]["session"]
        assert config["input_audio_transcription"] == {"model": transcription_model}


async def test_default_transcription_follows_the_model_from_env(
    voice_live: _FakeVoiceLive, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AZURE_VOICE_LIVE_MODEL", "gpt-4.1")
    async with _session(voice_live) as session:
        assert session.realtime_model.model == "gpt-4.1"

        config = (await _connected(voice_live)).events[0]["session"]
        assert config["input_audio_transcription"] == {"model": "azure-speech"}

    # a given configuration is sent as is, whatever the model
    whisper = AudioInputTranscriptionOptions(model="whisper-1")
    async with _session(voice_live, input_audio_transcription=whisper):
        config = (await _connected(voice_live, index=1)).events[0]["session"]
        assert config["input_audio_transcription"] == {"model": "whisper-1"}


async def test_generate_reply_ignores_server_initiated_responses(
    voice_live: _FakeVoiceLive,
) -> None:
    async def on_response(conn: _Connection, event: dict[str, Any], response_id: str) -> None:
        # server-side turn detection answers the user before the requested reply is created
        await _created(conn, "resp_vad", None)
        await _done(conn, "resp_vad")
        await _reply(conn, event, response_id)

    voice_live.on_response = on_response
    async with _session(voice_live) as session:
        generations: list[llm.GenerationCreatedEvent] = []
        session.on("generation_created", generations.append)

        generation = await asyncio.wait_for(session.generate_reply(), 5)

        assert generation.response_id == "resp_1"
        assert generation.user_initiated
        assert [(g.response_id, g.user_initiated) for g in generations] == [
            ("resp_vad", False),
            ("resp_1", True),
        ]


async def test_generate_reply_forwards_the_request_options(voice_live: _FakeVoiceLive) -> None:
    async with _session(voice_live) as session:
        await asyncio.wait_for(
            session.generate_reply(
                instructions="Be brief.",
                tool_choice={"type": "function", "function": {"name": "lookup"}},
                tools=[lookup],
            ),
            5,
        )

        request = voice_live.sent("response.create")[0]
        assert request["additional_instructions"] == "Be brief."
        assert request["response"]["tool_choice"] == "lookup"
        assert [tool["name"] for tool in request["response"]["tools"]] == ["lookup"]
        assert request["response"]["metadata"] == {"client_event_id": request["event_id"]}


async def test_session_updates_do_not_discard_pending_replies(voice_live: _FakeVoiceLive) -> None:
    async def on_response(conn: _Connection, event: dict[str, Any], response_id: str) -> None:
        # e.g. the tool_choice change the agent makes around a tool reply
        await conn.send("session.updated", session={"id": "sess_changed"})
        await conn.send("session.updated", session={})
        await _reply(conn, event, response_id)

    voice_live.on_response = on_response
    async with _session(voice_live) as session:
        reconnected: list[llm.RealtimeSessionReconnectedEvent] = []
        session.on("session_reconnected", reconnected.append)

        session.update_options(tool_choice="none")
        generation = await asyncio.wait_for(session.generate_reply(), 5)

        assert generation.user_initiated
        assert reconnected == []
        assert voice_live.sent("session.update")[1]["session"] == {"tool_choice": "none"}


async def test_timed_out_reply_is_cancelled_when_it_arrives(
    voice_live: _FakeVoiceLive, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(realtime_model, "_GENERATE_REPLY_TIMEOUT", 0.2)
    voice_live.on_response = _ignore
    async with _session(voice_live) as session:
        generations: list[llm.GenerationCreatedEvent] = []
        session.on("generation_created", generations.append)

        with pytest.raises(llm.RealtimeError, match="timed out"):
            await session.generate_reply()

        conn = voice_live.connections[0]
        request = voice_live.sent("response.create")[0]
        await _created(conn, "resp_late", _metadata(request))
        await _add_message(conn, "resp_late", "item_late")
        await _audio(conn, "resp_late", "item_late")

        await _wait_until(lambda: voice_live.sent("response.cancel"))
        assert voice_live.sent("response.cancel")[0]["response_id"] == "resp_late"
        await _done(conn, "resp_late", status="cancelled")

        # nobody heard the discarded response, it leaves the conversation
        await _wait_until(lambda: voice_live.sent("conversation.item.delete"))
        assert voice_live.sent("conversation.item.delete")[0]["item_id"] == "item_late"
        assert session.chat_ctx.get_by_id("item_late") is None

        # a later reply is unaffected by the discarded one
        voice_live.on_response = _reply
        monkeypatch.setattr(realtime_model, "_GENERATE_REPLY_TIMEOUT", 5.0)
        generation = await asyncio.wait_for(session.generate_reply(), 5)
        assert [g.response_id for g in generations] == [generation.response_id]


async def test_discarded_reply_created_before_its_output_is_announced(
    voice_live: _FakeVoiceLive, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(realtime_model, "_GENERATE_REPLY_TIMEOUT", 0.2)
    voice_live.on_response = _ignore
    async with _session(voice_live) as session:
        added: list[llm.RemoteItemAddedEvent] = []
        session.on("remote_item_added", added.append)
        received: list[Any] = []
        session.on("azure_server_event_received", received.append)

        with pytest.raises(llm.RealtimeError, match="timed out"):
            await session.generate_reply()

        conn = voice_live.connections[0]
        request = voice_live.sent("response.create")[0]
        await _created(conn, "resp_late", _metadata(request))
        reply = {"id": "item_late", "type": "message", "role": "assistant", "content": []}
        # the item is created before it's announced as an output of the discarded response
        await conn.send("conversation.item.created", previous_item_id=None, item=reply)
        # the user speaks meanwhile, the turn isn't part of the discarded response
        turn = {"id": "item_turn", "type": "message", "role": "user", "content": []}
        await conn.send("conversation.item.created", previous_item_id="item_late", item=turn)
        await conn.send(
            "response.output_item.added", response_id="resp_late", output_index=0, item=reply
        )
        await _done(conn, "resp_late", status="cancelled")
        await _wait_until(lambda: any(e.type == "response.done" for e in received))

        # every request the discarded response caused has reached Azure
        session.clear_audio()
        await _wait_until(lambda: voice_live.sent("input_audio_buffer.clear"))

        assert [e["item_id"] for e in voice_live.sent("conversation.item.delete")] == ["item_late"]
        assert [ev.item.id for ev in added] == ["item_turn"]
        assert [item.id for item in session.chat_ctx.items] == ["item_turn"]


async def test_cancelled_reply_is_discarded(voice_live: _FakeVoiceLive) -> None:
    voice_live.on_response = _ignore
    async with _session(voice_live) as session:
        generations: list[llm.GenerationCreatedEvent] = []
        session.on("generation_created", generations.append)

        fut = session.generate_reply()
        await _wait_until(lambda: voice_live.sent("response.create"))
        request = voice_live.sent("response.create")[0]

        # the agent gives up on the reply, e.g. the user interrupted, and Azure creates the
        # response before the cancellation's callbacks even ran
        fut.cancel()
        session._handle_server_event(
            ServerEventResponseCreated(
                response=Response(id="resp_late", status="in_progress", metadata=_metadata(request))
            )
        )

        await _wait_until(lambda: voice_live.sent("response.cancel"))
        assert voice_live.sent("response.cancel")[0]["response_id"] == "resp_late"
        assert generations == []


async def test_timed_out_reply_is_never_sent(
    voice_live: _FakeVoiceLive, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(realtime_model, "_GENERATE_REPLY_TIMEOUT", 0.1)
    voice_live.accept_gate = asyncio.Event()
    async with _session(voice_live) as session:
        # the connection is still being established when the reply times out
        with pytest.raises(llm.RealtimeError, match="timed out"):
            await session.generate_reply()

        voice_live.accept_gate.set()
        await _connected(voice_live)
        session.clear_audio()
        await _wait_until(lambda: voice_live.sent("input_audio_buffer.clear"))

        assert voice_live.sent("response.create") == []


async def test_audio_is_not_dropped_while_the_reply_waits(voice_live: _FakeVoiceLive) -> None:
    async def on_response(conn: _Connection, event: dict[str, Any], response_id: str) -> None:
        item_id = f"item_{response_id}"
        await _created(conn, response_id, _metadata(event))
        await _add_message(conn, response_id, item_id)
        await _audio(conn, response_id, item_id, count=200)
        await _done(conn, response_id)

    voice_live.on_response = on_response
    async with _session(voice_live) as session:
        metrics: list[Any] = []
        session.on("metrics_collected", metrics.append)

        generation = await asyncio.wait_for(session.generate_reply(), 5)
        # nothing reads the reply until the whole response has streamed in
        await _wait_until(lambda: metrics)

        message = await _first(generation.message_stream)
        frames = await _collect(message.audio_stream)
        assert len(frames) == 200
        assert b"".join(bytes(frame.data) for frame in frames) == _PCM_20MS * 200
        assert await message.modalities == ["audio", "text"]

        assert metrics[0].request_id == generation.response_id
        assert metrics[0].output_tokens == 7
        assert metrics[0].cancelled is False


def _pcm_frame(data: bytes = _PCM_20MS) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=data, sample_rate=24000, num_channels=1, samples_per_channel=len(data) // 2
    )


def _input_audio(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in events if e["type"].startswith("input_audio_buffer.")]


async def test_turns_under_100ms_are_cleared_instead_of_committed(
    voice_live: _FakeVoiceLive,
) -> None:
    async with _session(voice_live) as session:
        errors: list[llm.RealtimeModelError] = []
        session.on("error", errors.append)

        # 80ms can't be committed, it's cleared so it doesn't merge with the next turn
        for _ in range(4):
            session.push_audio(_pcm_frame())
        session.commit_audio()
        # a turn of 120ms follows, then there's nothing left to commit
        for _ in range(6):
            session.push_audio(_pcm_frame())
        session.commit_audio()
        session.commit_audio()
        session.clear_audio()

        await _wait_until(lambda: voice_live.sent("input_audio_buffer.clear")[1:])
        events = _input_audio(voice_live.events)
        assert [e["type"] for e in events] == [
            "input_audio_buffer.append",
            "input_audio_buffer.clear",
            "input_audio_buffer.append",
            "input_audio_buffer.append",
            "input_audio_buffer.commit",
            "input_audio_buffer.clear",
        ]
        committed = b"".join(base64.b64decode(e["audio"]) for e in events[2:4])
        assert committed == _PCM_20MS * 6
        assert errors == []


async def test_audio_of_a_lost_connection_is_sent_again(voice_live: _FakeVoiceLive) -> None:
    async with _session(voice_live) as session:
        first = await _connected(voice_live)
        for _ in range(10):
            session.push_audio(_pcm_frame())
        await _wait_until(lambda: len(voice_live.sent("input_audio_buffer.append")) == 2)

        # the socket drops mid-turn, and the turn ends while reconnecting
        await first.ws.close()
        session.commit_audio()

        second = await _connected(voice_live, 1)
        await _wait_until(lambda: _input_audio(second.events)[2:])
        events = _input_audio(second.events)
        assert [e["type"] for e in events] == [
            "input_audio_buffer.append",
            "input_audio_buffer.append",
            "input_audio_buffer.commit",
        ]
        assert b"".join(base64.b64decode(e["audio"]) for e in events[:2]) == _PCM_20MS * 10


async def test_unconfirmed_commits_are_committed_again(voice_live: _FakeVoiceLive) -> None:
    voice_live.drop_on_commit = True
    async with _session(voice_live) as session:
        # the socket drops before Azure confirms the turn, the reply must still follow it
        for _ in range(10):
            session.push_audio(_pcm_frame())
        session.commit_audio()
        generation = await asyncio.wait_for(session.generate_reply(), 5)
        assert generation.user_initiated

        first, second = voice_live.connections
        events = [e for e in second.events if e["type"] != "session.update"]
        assert [e["type"] for e in events] == [
            "input_audio_buffer.append",
            "input_audio_buffer.append",
            "input_audio_buffer.commit",
            "response.create",
        ]
        assert b"".join(base64.b64decode(e["audio"]) for e in events[:2]) == _PCM_20MS * 10
        lost = [e for e in first.events if e["type"] == "input_audio_buffer.commit"]
        assert [e["event_id"] for e in lost] == [events[2]["event_id"]]


async def test_committed_audio_is_not_sent_again(voice_live: _FakeVoiceLive) -> None:
    async with _session(voice_live) as session:
        received: list[Any] = []
        session.on("azure_server_event_received", received.append)

        def committed() -> int:
            return sum(e.type == "input_audio_buffer.committed" for e in received)

        first = await _connected(voice_live)
        for _ in range(10):
            session.push_audio(_pcm_frame())
        session.commit_audio()
        await _wait_until(lambda: committed() == 1)

        # the turn detection of Azure commits the next turn
        for _ in range(10):
            session.push_audio(_pcm_frame())
        await _wait_until(lambda: len(voice_live.sent("input_audio_buffer.append")) == 4)
        await voice_live.commit(first)
        await _wait_until(lambda: committed() == 2)

        # only the turn in progress is sent again
        for _ in range(5):
            session.push_audio(_pcm_frame())
        await _wait_until(lambda: len(voice_live.sent("input_audio_buffer.append")) == 5)
        await first.ws.close()

        second = await _connected(voice_live, 1)
        session.clear_audio()
        await _wait_until(lambda: _input_audio(second.events)[1:])
        events = _input_audio(second.events)
        assert [e["type"] for e in events] == [
            "input_audio_buffer.append",
            "input_audio_buffer.clear",
        ]
        assert base64.b64decode(events[0]["audio"]) == _PCM_20MS * 5


async def test_automatic_commit_leaves_the_next_turn_in_the_buffer(
    voice_live: _FakeVoiceLive,
) -> None:
    async with _session(voice_live) as session:
        errors: list[llm.RealtimeModelError] = []
        received: list[Any] = []
        session.on("error", errors.append)
        session.on("azure_server_event_received", received.append)

        conn = await _connected(voice_live)
        # the turn detection of Azure commits a 200ms turn without being asked to
        for _ in range(10):
            session.push_audio(_pcm_frame())
        await _wait_until(lambda: len(voice_live.sent("input_audio_buffer.append")) == 2)
        await conn.send("input_audio_buffer.speech_started", audio_start_ms=0, item_id="item_turn")
        await conn.send("input_audio_buffer.speech_stopped", audio_end_ms=200, item_id="item_turn")
        await voice_live.commit(conn)
        await _wait_until(lambda: any(e.type == "input_audio_buffer.committed" for e in received))

        # the 80ms that follow are a turn of their own, too short to commit
        for _ in range(4):
            session.push_audio(_pcm_frame())
        session.commit_audio()

        await _wait_until(lambda: voice_live.sent("input_audio_buffer.clear"))
        await asyncio.sleep(0.05)
        events = _input_audio(voice_live.events)
        assert [e["type"] for e in events] == [
            "input_audio_buffer.append",
            "input_audio_buffer.append",
            "input_audio_buffer.append",
            "input_audio_buffer.clear",
        ]
        assert base64.b64decode(events[2]["audio"]) == _PCM_20MS * 4
        assert errors == []


async def test_speech_stop_consumes_the_audio_azure_reports(
    voice_live: _FakeVoiceLive,
) -> None:
    async with _session(voice_live) as session:
        received: list[Any] = []
        session.on("azure_server_event_received", received.append)

        conn = await _connected(voice_live)
        # 200ms of a turn, then 100ms of the next one goes out before the speech stop,
        # which travels back while the microphone keeps streaming, reaches the client
        for _ in range(15):
            session.push_audio(_pcm_frame())
        await _wait_until(lambda: len(voice_live.sent("input_audio_buffer.append")) == 3)
        await conn.send("input_audio_buffer.speech_started", audio_start_ms=0, item_id="item_turn")
        await conn.send("input_audio_buffer.speech_stopped", audio_end_ms=200, item_id="item_turn")
        await voice_live.commit(conn, keep=len(_PCM_20MS) * 5)
        await _wait_until(lambda: any(e.type == "input_audio_buffer.committed" for e in received))

        # the 100ms Azure left in the buffer is the next turn: a new connection starts
        # from it, whole, and it is long enough to commit rather than be dropped
        await conn.ws.close()
        second = await _connected(voice_live, 1)
        await _wait_until(lambda: _input_audio(second.events))
        session.commit_audio()

        await _wait_until(lambda: _input_audio(second.events)[1:])
        events = _input_audio(second.events)
        assert [e["type"] for e in events] == [
            "input_audio_buffer.append",
            "input_audio_buffer.commit",
        ]
        assert base64.b64decode(events[0]["audio"]) == _PCM_20MS * 5


async def test_audio_sent_again_is_bounded(
    voice_live: _FakeVoiceLive, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(realtime_model, "_MAX_RESENT_AUDIO_CHUNKS", 2)
    async with _session(voice_live) as session:
        first = await _connected(voice_live)
        chunks = [bytes([i, 0]) * 2400 for i in range(5)]
        for chunk in chunks:
            session.push_audio(_pcm_frame(chunk))
        await _wait_until(lambda: len(voice_live.sent("input_audio_buffer.append")) == 5)
        await first.ws.close()

        second = await _connected(voice_live, 1)
        session.commit_audio()
        await _wait_until(lambda: voice_live.sent("input_audio_buffer.commit"))
        events = _input_audio(second.events)
        assert [base64.b64decode(e["audio"]) for e in events[:-1]] == chunks[-2:]
        assert events[-1]["type"] == "input_audio_buffer.commit"


async def test_closing_before_the_session_started_settles_requests(
    voice_live: _FakeVoiceLive,
) -> None:
    model = _model(voice_live)
    session = model.session()
    reply = session.generate_reply()
    await session.aclose()

    with pytest.raises(llm.RealtimeError, match="closed"):
        await reply


async def test_audio_appends_and_commit_keep_their_order(voice_live: _FakeVoiceLive) -> None:
    async with _session(voice_live) as session:
        frame = rtc.AudioFrame(
            data=b"\x01\x00" * 240, sample_rate=24000, num_channels=1, samples_per_channel=240
        )
        # 250ms: two 100ms chunks, and 50ms still buffered when the turn is committed
        for _ in range(25):
            session.push_audio(frame)
        session.commit_audio()

        await _wait_until(lambda: voice_live.sent("input_audio_buffer.commit"))
        audio_events = [e for e in voice_live.events if e["type"].startswith("input_audio_buffer")]
        assert [e["type"] for e in audio_events] == [
            "input_audio_buffer.append",
            "input_audio_buffer.append",
            "input_audio_buffer.append",
            "input_audio_buffer.commit",
        ]
        pushed = b"".join(base64.b64decode(e["audio"]) for e in audio_events[:3])
        assert pushed == b"\x01\x00" * 240 * 25


async def test_connection_loss_settles_the_reply_and_reconnects(
    voice_live: _FakeVoiceLive,
) -> None:
    async def on_response(conn: _Connection, event: dict[str, Any], response_id: str) -> None:
        if response_id == "resp_1":
            item_id = f"item_{response_id}"
            await _created(conn, response_id, _metadata(event))
            await _add_message(conn, response_id, item_id)
            await _audio(conn, response_id, item_id)
        elif conn.index == 0:
            # the socket drops mid-reply, while another reply is pending
            await conn.ws.close()
        else:
            await _reply(conn, event, response_id)

    voice_live.on_response = on_response
    async with _session(voice_live) as session:
        reconnected: list[llm.RealtimeSessionReconnectedEvent] = []
        session.on("session_reconnected", reconnected.append)

        generation = await asyncio.wait_for(session.generate_reply(), 5)
        message = await _first(generation.message_stream)

        pending = session.generate_reply()

        # every stream of the interrupted reply ends instead of waiting for a response.done
        assert len(await _collect(message.audio_stream)) == 1
        assert await _collect(message.text_stream) == []
        assert await _collect(generation.message_stream) == []
        assert await _collect(generation.function_stream) == []

        # the pending reply is requested again from the new connection
        answered = await asyncio.wait_for(pending, 5)
        assert answered.user_initiated
        assert reconnected

        first, second = (conn.events for conn in voice_live.connections)
        lost = [e for e in first if e["type"] == "response.create"][-1]
        resent = [e for e in second if e["type"] == "response.create"]
        assert [e["event_id"] for e in resent] == [lost["event_id"]]


async def test_unconfirmed_items_are_created_again_after_a_reconnection(
    voice_live: _FakeVoiceLive,
) -> None:
    async with _session(voice_live) as session:
        chat_ctx = llm.ChatContext.empty()
        typed = chat_ctx.add_message(role="user", content="typed while the socket dropped")

        # the socket drops before Azure confirms the item: it's created on the new connection
        voice_live.drop_on_item_create = True
        await asyncio.wait_for(session.update_chat_ctx(chat_ctx), 5)

        assert [i.id for i in session.chat_ctx.items] == [typed.id]
        first, second = (conn.events for conn in voice_live.connections)
        lost = [e for e in first if e["type"] == "conversation.item.create"]
        resent = [e for e in second if e["type"] == "conversation.item.create"]
        assert [e["event_id"] for e in resent] == [e["event_id"] for e in lost]


async def test_unconfirmed_requests_survive_a_failed_reconnection(
    voice_live: _FakeVoiceLive, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure = RealtimeSession._create_session_config
    configured = 0

    def flaky_configure(session: RealtimeSession) -> Any:
        nonlocal configured
        configured += 1
        if configured == 2:
            raise RuntimeError("configuration failed")
        return configure(session)

    monkeypatch.setattr(RealtimeSession, "_create_session_config", flaky_configure)
    async with _session(voice_live) as session:
        chat_ctx = llm.ChatContext.empty()
        typed = chat_ctx.add_message(role="user", content="hello?")

        # the first reconnection fails too, before anything could be sent again
        voice_live.drop_on_item_create = True
        await asyncio.wait_for(session.update_chat_ctx(chat_ctx), 5)

        assert [i.id for i in session.chat_ctx.items] == [typed.id]
        assert len(voice_live.connections) == 3
        resent = [e for e in voice_live.connections[2].events if e.get("item")]
        assert [e["item"]["id"] for e in resent] == [typed.id]


async def test_reconnect_replays_tool_calls_and_system_messages(
    voice_live: _FakeVoiceLive,
) -> None:
    async def on_response(conn: _Connection, event: dict[str, Any], response_id: str) -> None:
        await _created(conn, response_id, _metadata(event))
        for call_id, arguments in (("call_done", '{"query": "rain"}'), ("call_cut", None)):
            item = {
                "id": f"item_{call_id}",
                "type": "function_call",
                "call_id": call_id,
                "name": "lookup",
                "arguments": "",
                "status": "in_progress",
            }
            await conn.send(
                "response.output_item.added", response_id=response_id, output_index=0, item=item
            )
            await conn.send(
                "conversation.item.created", previous_item_id=conn.item_ids[-1], item=item
            )
            conn.item_ids.append(item["id"])
            if arguments is not None:
                await conn.send(
                    "response.function_call_arguments.done",
                    response_id=response_id,
                    item_id=item["id"],
                    output_index=0,
                    call_id=call_id,
                    name="lookup",
                    arguments=arguments,
                )
        # the second call is cut off before its arguments are complete
        await _done(conn, response_id, status="cancelled")

    voice_live.on_response = on_response
    async with _session(voice_live) as session:
        chat_ctx = llm.ChatContext.empty()
        chat_ctx.add_message(role="system", content="Speak formally.")
        await session.update_chat_ctx(chat_ctx)

        generation = await asyncio.wait_for(session.generate_reply(), 5)
        [call] = await _collect(generation.function_stream)
        chat_ctx = session.chat_ctx.copy()
        chat_ctx.items.append(
            llm.FunctionCallOutput(call_id=call.call_id, name="lookup", output="42", is_error=False)
        )
        await session.update_chat_ctx(chat_ctx)

        await voice_live.connections[0].ws.close()
        new_conn = await _connected(voice_live, index=1)
        await _wait_until(lambda: len(new_conn.events) >= 4)

        replayed = [e["item"] for e in new_conn.events[1:4]]
        assert [(item["type"], item.get("role"), item.get("call_id")) for item in replayed] == [
            ("message", "system", None),
            ("function_call", None, "call_done"),
            ("function_call_output", None, "call_done"),
        ]
        assert replayed[1]["arguments"] == '{"query": "rain"}'
        assert session.chat_ctx.get_by_id("item_call_cut") is None


async def test_reconnect_replays_the_generated_conversation(voice_live: _FakeVoiceLive) -> None:
    transcription = AudioInputTranscriptionOptions(model="whisper-1")
    async with _session(voice_live, input_audio_transcription=transcription) as session:
        added: list[llm.RemoteItemAddedEvent] = []
        transcripts: list[llm.InputTranscriptionCompleted] = []
        stopped: list[llm.InputSpeechStoppedEvent] = []
        generations: list[llm.GenerationCreatedEvent] = []
        session.on("remote_item_added", added.append)
        session.on("input_audio_transcription_completed", transcripts.append)
        session.on("input_speech_stopped", stopped.append)
        session.on("generation_created", generations.append)

        conn = await _connected(voice_live)
        # a user turn detected and answered by the server
        await conn.send("input_audio_buffer.speech_started", audio_start_ms=0, item_id="item_user")
        await conn.send("input_audio_buffer.speech_stopped", audio_end_ms=900, item_id="item_user")
        await conn.send(
            "conversation.item.created",
            previous_item_id=None,
            item={
                "id": "item_user",
                "type": "message",
                "role": "user",
                "status": "completed",
                "content": [{"type": "input_audio", "transcript": None}],
            },
        )
        conn.item_ids.append("item_user")
        await conn.send(
            "conversation.item.input_audio_transcription.completed",
            item_id="item_user",
            content_index=0,
            transcript="What's the weather?",
        )
        await _created(conn, "resp_vad", None)
        await _add_message(conn, "resp_vad", "item_assistant")
        await _transcript(conn, "resp_vad", "item_assistant", "It is sunny.")
        await _audio(conn, "resp_vad", "item_assistant")
        await _done(conn, "resp_vad")

        await _wait_until(
            lambda: any(
                i.type == "message" and i.content == ["It is sunny."]
                for i in session.chat_ctx.items
            )
        )
        items = session.chat_ctx.items
        assert [(i.id, i.role, i.content) for i in items if i.type == "message"] == [
            ("item_user", "user", ["What's the weather?"]),
            ("item_assistant", "assistant", ["It is sunny."]),
        ]
        assert [(e.previous_item_id, e.item.id) for e in added] == [
            (None, "item_user"),
            ("item_user", "item_assistant"),
        ]
        assert [t.transcript for t in transcripts] == ["What's the weather?"]
        assert stopped[0].user_transcription_enabled
        assert [(g.response_id, g.user_initiated) for g in generations] == [("resp_vad", False)]

        await conn.ws.close()
        new_conn = await _connected(voice_live, index=1)
        await _wait_until(lambda: len(new_conn.events) >= 3)

        # the new Azure session starts empty, the whole conversation is sent again
        assert new_conn.events[0]["type"] == "session.update"
        replayed = new_conn.events[1:3]
        assert [(e["type"], e["item"]["id"]) for e in replayed] == [
            ("conversation.item.create", "item_user"),
            ("conversation.item.create", "item_assistant"),
        ]
        assert replayed[0]["item"]["content"] == [
            {"type": "input_text", "text": "What's the weather?"}
        ]
        assert replayed[1]["item"]["content"] == [{"type": "text", "text": "It is sunny."}]
        assert [i.id for i in session.chat_ctx.items] == ["item_user", "item_assistant"]


async def test_reconnect_replays_an_interrupted_reply(voice_live: _FakeVoiceLive) -> None:
    async def on_response(conn: _Connection, event: dict[str, Any], response_id: str) -> None:
        # the reply streams, but never reaches its response.done
        await _created(conn, response_id, _metadata(event))
        await _add_message(conn, response_id, "item_reply")
        await _audio(conn, response_id, "item_reply")
        await _transcript(conn, response_id, "item_reply", "The answer is forty")

    voice_live.on_response = on_response
    async with _session(voice_live) as session:
        received: list[Any] = []
        session.on("azure_server_event_received", received.append)

        first = await _connected(voice_live)
        await asyncio.wait_for(session.generate_reply(), 5)
        await _wait_until(
            lambda: any(e.type == "response.audio_transcript.delta" for e in received)
        )

        # the socket drops while the reply is still streaming
        await first.ws.close()

        second = await _connected(voice_live, 1)
        await _wait_until(lambda: [e for e in second.events if e["type"] != "session.update"])
        replayed = [e for e in second.events if e["type"] == "conversation.item.create"]
        assert [(e["item"]["id"], e["item"]["content"]) for e in replayed] == [
            ("item_reply", [{"type": "text", "text": "The answer is forty"}])
        ]
        assert [(i.id, i.content) for i in session.chat_ctx.items] == [
            ("item_reply", ["The answer is forty"])
        ]


async def test_reconnect_restores_the_session_configuration(voice_live: _FakeVoiceLive) -> None:
    async with _session(voice_live) as session:
        conn = await _connected(voice_live)
        await session.update_instructions("Talk like a pirate.")
        await session.update_tools([lookup])
        session.update_options(tool_choice="required")
        await _wait_until(lambda: len(conn.events) == 4)

        await conn.ws.close()
        new_conn = await _connected(voice_live, index=1)

        config = new_conn.events[0]["session"]
        assert config["instructions"] == "Talk like a pirate."
        assert [tool["name"] for tool in config["tools"]] == ["lookup"]
        assert config["tool_choice"] == "required"


async def test_reconnect_keeps_empty_instructions(voice_live: _FakeVoiceLive) -> None:
    async with _session(voice_live) as session:
        conn = await _connected(voice_live)
        await session.update_instructions("")
        await _wait_until(lambda: len(conn.events) == 2)
        assert conn.events[1]["session"]["instructions"] == ""

        await conn.ws.close()
        new_conn = await _connected(voice_live, index=1)

        assert new_conn.events[0]["session"]["instructions"] == ""


async def test_agent_session_answers_after_a_tool_call(voice_live: _FakeVoiceLive) -> None:
    lookups: list[str] = []

    class WeatherAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You tell the weather.")

        @function_tool
        async def get_weather(self, location: str) -> str:
            """Get the weather for a location.

            Args:
                location: The city to get the weather for.
            """
            lookups.append(location)
            return "sunny"

    async def on_response(conn: _Connection, event: dict[str, Any], response_id: str) -> None:
        await _created(conn, response_id, _metadata(event))
        if response_id == "resp_1":
            item = {
                "id": "item_call",
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "",
                "status": "in_progress",
            }
            await conn.send(
                "response.output_item.added", response_id=response_id, output_index=0, item=item
            )
            await conn.send(
                "conversation.item.created", previous_item_id=conn.item_ids[-1], item=item
            )
            conn.item_ids.append("item_call")
            await conn.send(
                "response.function_call_arguments.done",
                response_id=response_id,
                item_id="item_call",
                output_index=0,
                call_id="call_1",
                name="get_weather",
                arguments='{"location": "Tokyo"}',
            )
        else:
            item_id = f"item_{response_id}"
            await _add_message(conn, response_id, item_id)
            await _transcript(conn, response_id, item_id, "It is sunny in Tokyo.")
            await _audio(conn, response_id, item_id)
        await _done(conn, response_id)

    voice_live.on_response = on_response
    async with AgentSession(llm=_model(voice_live)) as session:
        errors: list[Any] = []
        session.on("error", errors.append)
        session.output.audio = FakeAudioOutput()
        await session.start(WeatherAgent())

        session.generate_reply(user_input="What's the weather in Tokyo?")
        await _wait_until(
            lambda: any(
                item.type == "message" and item.text_content == "It is sunny in Tokyo."
                for item in session.history.items
            ),
            timeout=10,
        )

        assert lookups == ["Tokyo"]
        assert errors == []

        creates = [e["item"] for e in voice_live.sent("conversation.item.create")]
        assert creates[0]["content"] == [
            {"type": "input_text", "text": "What's the weather in Tokyo?"}
        ]
        output = next(
            e
            for e in voice_live.sent("conversation.item.create")
            if e["item"]["type"] == "function_call_output"
        )
        assert output["previous_item_id"] == "item_call"
        assert output["item"]["call_id"] == "call_1"

        # the tool reply picks its tool_choice per response, without reconfiguring the session
        replies = voice_live.sent("response.create")
        assert len(replies) == 2
        assert replies[1]["response"]["tool_choice"] == "auto"
        assert not any("tool_choice" in e["session"] for e in voice_live.sent("session.update")[1:])


async def test_update_chat_ctx_reports_rejected_items(voice_live: _FakeVoiceLive) -> None:
    async with _session(voice_live) as session:
        errors: list[llm.RealtimeModelError] = []
        session.on("error", errors.append)

        chat_ctx = llm.ChatContext.empty()
        first = chat_ctx.add_message(role="user", content="first")

        voice_live.reject_items = True
        with pytest.raises(llm.RealtimeError, match="item rejected"):
            await session.update_chat_ctx(chat_ctx)
        assert session.chat_ctx.items == []

        voice_live.reject_items = False
        await session.update_chat_ctx(chat_ctx)
        assert [i.id for i in session.chat_ctx.items] == [first.id]

        chat_ctx = session.chat_ctx.copy()
        second = chat_ctx.add_message(role="user", content="second")
        output = llm.FunctionCallOutput(call_id="call_1", output="42", is_error=False)
        chat_ctx.items.append(output)
        await session.update_chat_ctx(chat_ctx)

        creates = voice_live.sent("conversation.item.create")
        assert [(e["item"]["id"], e.get("previous_item_id")) for e in creates[-2:]] == [
            (second.id, first.id),
            (output.id, second.id),
        ]
        assert [i.id for i in session.chat_ctx.items] == [first.id, second.id, output.id]
        # the rejection surfaced through update_chat_ctx, not as a session error
        assert errors == []


async def test_update_chat_ctx_inserts_a_prepended_item_at_the_root(
    voice_live: _FakeVoiceLive,
) -> None:
    async with _session(voice_live) as session:
        conn = await _connected(voice_live)
        chat_ctx = llm.ChatContext.empty()
        second = chat_ctx.add_message(role="user", content="second")
        await session.update_chat_ctx(chat_ctx)

        # the caller puts a message before the history Azure already holds
        chat_ctx = session.chat_ctx.copy()
        first = llm.ChatMessage(role="user", content=["first"])
        chat_ctx.items.insert(0, first)
        await session.update_chat_ctx(chat_ctx)

        creates = [e for e in conn.events if e["type"] == "conversation.item.create"]
        assert [(e["item"]["id"], e.get("previous_item_id")) for e in creates] == [
            (second.id, "root"),
            (first.id, "root"),
        ]
        # the mirror agrees with the order Azure holds
        assert conn.item_ids == [first.id, second.id]
        assert [i.id for i in session.chat_ctx.items] == [first.id, second.id]


async def test_update_chat_ctx_appends_a_context_that_drops_history(
    voice_live: _FakeVoiceLive,
) -> None:
    async with _session(voice_live) as session:
        conn = await _connected(voice_live)
        chat_ctx = llm.ChatContext.empty()
        first = chat_ctx.add_message(role="user", content="first")
        await session.update_chat_ctx(chat_ctx)

        # the caller replaces the history Azure holds, which Azure never deletes: opening
        # the conversation with the summary would leave the stale turn the newest one
        summarized = llm.ChatContext.empty()
        summary = summarized.add_message(role="user", content="a summary")
        await session.update_chat_ctx(summarized)

        creates = [e for e in conn.events if e["type"] == "conversation.item.create"]
        assert [(e["item"]["id"], e.get("previous_item_id")) for e in creates] == [
            (first.id, "root"),
            (summary.id, None),
        ]
        assert conn.item_ids == [first.id, summary.id]
        assert [i.id for i in session.chat_ctx.items] == [first.id, summary.id]


async def test_prepended_item_is_mirrored_after_a_late_confirmation(
    voice_live: _FakeVoiceLive, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(realtime_model, "_UPDATE_CHAT_CTX_TIMEOUT", 0.2)
    async with _session(voice_live) as session:
        conn = await _connected(voice_live)
        chat_ctx = llm.ChatContext.empty()
        second = chat_ctx.add_message(role="user", content="second")
        await session.update_chat_ctx(chat_ctx)

        # the caller gives up before Azure confirms the item it put first
        voice_live.answer_items = False
        chat_ctx = session.chat_ctx.copy()
        first = llm.ChatMessage(role="user", content=["first"])
        chat_ctx.items.insert(0, first)
        with pytest.raises(llm.RealtimeError, match="timed out"):
            await session.update_chat_ctx(chat_ctx)

        # the confirmation still arrives: the mirror keeps the order Azure holds
        await conn.send(
            "conversation.item.created",
            previous_item_id=None,
            item={
                "id": first.id,
                "type": "message",
                "role": "user",
                "status": "completed",
                "content": [{"type": "input_text", "text": "first"}],
            },
        )
        await _wait_until(lambda: len(session.chat_ctx.items) == 2)
        assert [i.id for i in session.chat_ctx.items] == [first.id, second.id]


async def test_update_chat_ctx_times_out_without_confirmation(
    voice_live: _FakeVoiceLive, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(realtime_model, "_UPDATE_CHAT_CTX_TIMEOUT", 0.2)
    voice_live.answer_items = False
    async with _session(voice_live) as session:
        chat_ctx = llm.ChatContext.empty()
        chat_ctx.add_message(role="user", content="hello")
        with pytest.raises(llm.RealtimeError, match="timed out"):
            await session.update_chat_ctx(chat_ctx)
        assert session.chat_ctx.items == []


@pytest.mark.parametrize("outcome", ["confirmed", "rejected"])
async def test_retried_item_waits_for_the_pending_creation(
    voice_live: _FakeVoiceLive, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    monkeypatch.setattr(realtime_model, "_UPDATE_CHAT_CTX_TIMEOUT", 0.2)
    voice_live.answer_items = False
    async with _session(voice_live) as session:
        chat_ctx = llm.ChatContext.empty()
        message = chat_ctx.add_message(role="user", content="hello")
        with pytest.raises(llm.RealtimeError, match="timed out"):
            await session.update_chat_ctx(chat_ctx)

        # the retry joins the creation still in flight instead of reporting success
        monkeypatch.setattr(realtime_model, "_UPDATE_CHAT_CTX_TIMEOUT", 5.0)
        retry = asyncio.ensure_future(session.update_chat_ctx(chat_ctx))
        await asyncio.sleep(0.1)
        assert not retry.done()

        conn = voice_live.connections[0]
        [create] = voice_live.sent("conversation.item.create")
        if outcome == "confirmed":
            await conn.send("conversation.item.created", previous_item_id=None, item=create["item"])
            await asyncio.wait_for(retry, 5)
            assert [i.id for i in session.chat_ctx.items] == [message.id]
        else:
            await conn.send(
                "error",
                error={
                    "type": "invalid_request_error",
                    "message": "item rejected",
                    "event_id": create["event_id"],
                },
            )
            with pytest.raises(llm.RealtimeError, match="item rejected"):
                await asyncio.wait_for(retry, 5)
            assert session.chat_ctx.items == []

        assert len(voice_live.sent("conversation.item.create")) == 1


async def test_failing_replays_exhaust_the_retries(voice_live: _FakeVoiceLive) -> None:
    model = RealtimeModel(
        endpoint=voice_live.endpoint,
        api_key="test-key",
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0.0, timeout=5.0),
    )
    session = model.session()
    try:
        errors: list[llm.RealtimeModelError] = []
        session.on("error", errors.append)

        chat_ctx = llm.ChatContext.empty()
        chat_ctx.add_message(role="user", content="hello")
        await session.update_chat_ctx(chat_ctx)

        # every new connection is configured, then drops while replaying the conversation
        voice_live.drop_item_creates_from = 1
        await voice_live.connections[0].ws.close()

        await _wait_until(lambda: any(not e.recoverable for e in errors))
        assert len(voice_live.connections) == 2
    finally:
        await session.aclose()


async def test_options_and_tools_reach_the_session(voice_live: _FakeVoiceLive) -> None:
    async with _session(voice_live) as session:
        session.update_options(tool_choice="required")
        session.update_options(tool_choice="required")
        await session.update_tools([lookup])
        await session.update_tools([])

        await _wait_until(lambda: len(voice_live.sent("session.update")) == 4)
        updates = [e["session"] for e in voice_live.sent("session.update")]
        assert updates[1] == {"tool_choice": "required"}
        assert [tool["name"] for tool in updates[2]["tools"]] == ["lookup"]
        # an empty list clears the tools of the session
        assert updates[3] == {"tools": []}


async def test_function_call_arguments_stay_out_of_logs(
    voice_live: _FakeVoiceLive, caplog: pytest.LogCaptureFixture
) -> None:
    arguments = '{"query": "my password is hunter-two"}'

    async def on_response(conn: _Connection, event: dict[str, Any], response_id: str) -> None:
        item = {
            "id": "item_call",
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": "",
            "status": "in_progress",
        }
        await _created(conn, response_id, _metadata(event))
        await conn.send(
            "response.output_item.added", response_id=response_id, output_index=0, item=item
        )
        await conn.send("conversation.item.created", previous_item_id=None, item=item)
        await conn.send(
            "response.function_call_arguments.delta",
            response_id=response_id,
            item_id="item_call",
            output_index=0,
            call_id="call_1",
            delta=arguments,
        )
        await conn.send(
            "response.function_call_arguments.done",
            response_id=response_id,
            item_id="item_call",
            output_index=0,
            call_id="call_1",
            name="lookup",
            arguments=arguments,
        )
        await _done(conn, response_id)

    caplog.set_level(logging.DEBUG, logger="livekit.plugins.azure")
    voice_live.on_response = on_response
    async with _session(voice_live) as session:
        generation = await asyncio.wait_for(session.generate_reply(), 5)
        calls = await _collect(generation.function_stream)

        assert [(c.id, c.call_id, c.name, c.arguments) for c in calls] == [
            ("item_call", "call_1", "lookup", arguments)
        ]
        mirrored = session.chat_ctx.items[0]
        assert mirrored.type == "function_call" and mirrored.arguments == arguments

    plugin_logs = [repr(r.__dict__) for r in caplog.records if r.name.startswith("livekit.plugins")]
    assert plugin_logs
    assert not any("hunter-two" in record for record in plugin_logs)


async def test_error_events_are_reported(voice_live: _FakeVoiceLive) -> None:
    async with _session(voice_live) as session:
        errors: list[llm.RealtimeModelError] = []
        session.on("error", errors.append)

        conn = await _connected(voice_live)
        await conn.send(
            "error",
            error={
                "type": "invalid_request_error",
                "code": "response_cancel_not_active",
                "message": "Cancellation failed: no active response found",
            },
        )
        await conn.send("error", error={"type": "server_error", "message": "boom"})

        await _wait_until(lambda: errors)
        await asyncio.sleep(0.05)
        assert len(errors) == 1
        assert errors[0].recoverable
        assert "boom" in str(errors[0].error)


async def test_error_messages_stay_out_of_logs(
    voice_live: _FakeVoiceLive, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="livekit.plugins.azure")
    async with _session(voice_live) as session:
        errors: list[llm.RealtimeModelError] = []
        session.on("error", errors.append)

        conn = await _connected(voice_live)
        # Azure quotes the request it rejected, which carries what the user said
        await conn.send(
            "error",
            error={
                "type": "invalid_request_error",
                "code": "response_cancel_not_active",
                "message": "no active response for 'my password is hunter-two'",
            },
        )
        await conn.send(
            "error",
            error={
                "type": "invalid_request_error",
                "code": "invalid_value",
                "message": "invalid item: 'my password is hunter-two'",
            },
        )
        await _wait_until(lambda: errors)

    plugin_logs = [r for r in caplog.records if r.name.startswith("livekit.plugins")]
    assert plugin_logs
    for record in plugin_logs:
        # only a `lk.pii.`-marked field may carry it, redaction drops those whole
        unmarked = {k: v for k, v in vars(record).items() if "pii" not in k.split(".")}
        assert "hunter-two" not in record.getMessage()
        assert "hunter-two" not in repr(unmarked)
    assert any("hunter-two" in str(vars(r).get("lk.pii.error_message", "")) for r in plugin_logs)
    # the app still sees the message, it is the logs that cannot carry it
    assert "hunter-two" in str(errors[0].error)


async def test_speech_start_leaves_interruption_to_the_agent(voice_live: _FakeVoiceLive) -> None:
    async def on_response(conn: _Connection, event: dict[str, Any], response_id: str) -> None:
        await _created(conn, response_id, _metadata(event))
        await _add_message(conn, response_id, "item_reply")
        await _audio(conn, response_id, "item_reply")

    voice_live.on_response = on_response
    async with _session(voice_live) as session:
        started: list[llm.InputSpeechStartedEvent] = []
        session.on("input_speech_started", started.append)
        await asyncio.wait_for(session.generate_reply(), 5)

        conn = voice_live.connections[0]
        await conn.send("input_audio_buffer.speech_started", audio_start_ms=0, item_id="item_x")
        await _wait_until(lambda: started)
        session.clear_audio()
        await _wait_until(lambda: voice_live.sent("input_audio_buffer.clear"))
        assert voice_live.sent("response.cancel") == []

        # the agent decides to interrupt
        session.interrupt()
        await _wait_until(lambda: voice_live.sent("response.cancel"))


async def test_text_only_session(voice_live: _FakeVoiceLive) -> None:
    async def on_response(conn: _Connection, event: dict[str, Any], response_id: str) -> None:
        await _created(conn, response_id, _metadata(event))
        await _add_message(conn, response_id, "item_text", part_type="text")
        await _text(conn, response_id, "item_text", "Hi there")
        await _done(conn, response_id)

    voice_live.on_response = on_response
    async with _session(voice_live, modalities=["text"]) as session:
        generation = await asyncio.wait_for(session.generate_reply(), 5)
        message = await _first(generation.message_stream)

        assert await message.modalities == ["text"]
        assert await _collect(message.audio_stream) == []
        assert "".join(await _collect(message.text_stream)) == "Hi there"
        assert voice_live.connections[0].events[0]["session"]["modalities"] == ["text"]


async def test_text_fallback_of_an_audio_session_reports_ttft(
    voice_live: _FakeVoiceLive,
) -> None:
    async def on_response(conn: _Connection, event: dict[str, Any], response_id: str) -> None:
        await _created(conn, response_id, _metadata(event))
        await _add_message(conn, response_id, "item_text", part_type="text")
        await _text(conn, response_id, "item_text", "Hi there")
        await _done(conn, response_id)

    voice_live.on_response = on_response
    async with _session(voice_live) as session:
        metrics: list[Any] = []
        session.on("metrics_collected", metrics.append)

        generation = await asyncio.wait_for(session.generate_reply(), 5)
        message = await _first(generation.message_stream)

        assert await message.modalities == ["text"]
        assert "".join(await _collect(message.text_stream)) == "Hi there"
        await _wait_until(lambda: metrics)
        assert metrics[0].ttft >= 0


async def test_default_credential_is_reused_and_closed(
    voice_live: _FakeVoiceLive, monkeypatch: pytest.MonkeyPatch
) -> None:
    credentials: list[_FakeCredential] = []

    class _FakeCredential:
        def __init__(self) -> None:
            self.closed = 0
            credentials.append(self)

        async def get_token(self, *scopes: str, **kwargs: Any) -> AccessToken:
            return AccessToken("entra-token", int(time.time()) + 3600)

        async def close(self) -> None:
            self.closed += 1

        async def __aenter__(self) -> _FakeCredential:
            return self

        async def __aexit__(self, *args: object) -> None:
            await self.close()

    monkeypatch.setattr(realtime_model, "DefaultAzureCredential", _FakeCredential)
    model = _model(voice_live, api_key=None, use_default_credential=True)
    session = model.session()
    try:
        conn = await _connected(voice_live)
        await conn.ws.close()
        await _connected(voice_live, index=1)
    finally:
        await session.aclose()

    assert len(credentials) == 1
    assert credentials[0].closed == 1
    assert all("entra-token" in headers["Authorization"] for headers in voice_live.headers)


async def test_exhausted_retries_close_the_session() -> None:
    model = RealtimeModel(
        endpoint="http://127.0.0.1:1",
        api_key="test-key",
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0.0, timeout=1.0),
    )
    session = model.session()
    try:
        errors: list[llm.RealtimeModelError] = []
        session.on("error", errors.append)
        await _wait_until(lambda: any(not e.recoverable for e in errors))

        with pytest.raises(llm.RealtimeError, match="closed"):
            await session.generate_reply()
        with pytest.raises(llm.RealtimeError, match="closed"):
            chat_ctx = llm.ChatContext.empty()
            chat_ctx.add_message(role="user", content="hello")
            await session.update_chat_ctx(chat_ctx)
    finally:
        await session.aclose()


def test_realtime_requires_the_optional_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "livekit.plugins.azure.realtime", None)
    with pytest.raises(ImportError, match=r"livekit-plugins-azure\[realtime\]"):
        azure_plugin.__getattr__("realtime")
