from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlparse

import aiohttp
import pytest
from openai.types.realtime import (
    AudioTranscription,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
)

from livekit.agents import APIConnectionError, APIError, llm
from livekit.agents.types import APIConnectOptions
from livekit.plugins.openai.realtime import (
    InferenceRealtimeModel,
    inference_realtime_model as inference_realtime,
)
from livekit.plugins.openai.realtime.realtime_model import (
    RealtimeModel,
    RealtimeSession,
)

pytestmark = pytest.mark.unit


@llm.function_tool
async def lookup_weather(city: str) -> str:
    """Look up weather for a city."""
    return city


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.incoming: asyncio.Queue[SimpleNamespace] = asyncio.Queue()
        self.send_gate: asyncio.Event | None = None
        self.send_error: Exception | None = None
        self.send_started = asyncio.Event()
        self.close_gate: asyncio.Event | None = None
        self.close_started = asyncio.Event()
        self.closed = False

    async def send_str(self, data: str) -> None:
        if self.send_gate is not None:
            self.send_started.set()
            await self.send_gate.wait()
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(json.loads(data))

    async def receive(self) -> SimpleNamespace:
        return await self.incoming.get()

    async def close(self) -> None:
        self.close_started.set()
        if self.close_gate is not None:
            await self.close_gate.wait()
        self.closed = True

    def push_server_event(self, event: dict[str, Any]) -> None:
        self.incoming.put_nowait(
            SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(event))
        )

    def disconnect(self) -> None:
        self.incoming.put_nowait(SimpleNamespace(type=aiohttp.WSMsgType.CLOSED))


class _FakeHTTPSession:
    def __init__(self) -> None:
        self.connections: list[tuple[str, dict[str, str], _FakeWebSocket]] = []
        self.next_send_gate: asyncio.Event | None = None

    async def ws_connect(self, *, url: str, headers: dict[str, str]) -> _FakeWebSocket:
        ws = _FakeWebSocket()
        ws.send_gate = self.next_send_gate
        self.next_send_gate = None
        self.connections.append((url, headers, ws))
        return ws


async def _wait_for(predicate: Callable[[], bool], *, timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.fixture
def paused_realtime_main(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _paused_main(self: RealtimeSession) -> None:
        await self._msg_ch._close_ev.wait()

    monkeypatch.setattr(RealtimeSession, "_main_task", _paused_main)


async def test_connection_refreshes_token_url_and_headers(
    monkeypatch: pytest.MonkeyPatch,
    paused_realtime_main: None,
) -> None:
    tokens = iter(("token-one", "token-two"))
    monkeypatch.setattr(
        inference_realtime,
        "create_access_token",
        lambda key, secret: f"{next(tokens)}:{key}:{secret}",
    )
    monkeypatch.setattr(
        inference_realtime,
        "get_inference_headers",
        lambda *, inference_class: {"X-Test-Class": inference_class or ""},
    )
    http_session = _FakeHTTPSession()
    model = InferenceRealtimeModel(
        "xai/grok-voice-think-fast-2.0",
        provider="xai",
        base_url="https://inference.example/v1",
        api_key="key",
        api_secret="secret",
        inference_class="priority",
        http_session=http_session,  # type: ignore[arg-type]
    )
    session = model.session()

    await session._create_ws_conn()
    await session._create_ws_conn()

    assert len(http_session.connections) == 2
    parsed = urlparse(http_session.connections[0][0])
    assert parsed.scheme == "wss"
    assert parsed.path == "/v1/realtime"
    assert parse_qs(parsed.query) == {"model": ["xai/grok-voice-think-fast-2.0"]}
    assert [headers["Authorization"] for _, headers, _ in http_session.connections] == [
        "Bearer token-one:key:secret",
        "Bearer token-two:key:secret",
    ]
    assert http_session.connections[0][1]["X-Test-Class"] == "priority"
    assert http_session.connections[0][1]["X-LiveKit-Inference-Provider"] == "xai"
    assert all(
        "X-LiveKit-Realtime-Failover-Protocol" not in headers
        for _, headers, _ in http_session.connections
    )
    await session.aclose()


def test_xai_catalog_model_is_accepted() -> None:
    model = InferenceRealtimeModel(
        "xai/grok-voice-think-fast-2.0",
        api_key="key",
        api_secret="secret",
    )

    assert model._opts.model == "xai/grok-voice-think-fast-2.0"


@pytest.mark.parametrize(
    ("model_name", "voice", "expected"),
    [
        ("openai/gpt-realtime", None, "marin"),
        ("xai/grok-voice-think-fast-2.0", None, "eve"),
        ("xai/grok-voice-think-fast-2.0", "Ara", "Ara"),
    ],
)
def test_model_specific_default_voice(
    model_name: str,
    voice: str | None,
    expected: str,
) -> None:
    kwargs = {} if voice is None else {"voice": voice}
    model = InferenceRealtimeModel(
        model_name,
        api_key="key",
        api_secret="secret",
        **kwargs,
    )

    assert model._opts.voice == expected


def test_credentials_follow_inference_environment_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LIVEKIT_INFERENCE_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_INFERENCE_API_SECRET", raising=False)
    monkeypatch.setenv("LIVEKIT_API_KEY", "environment-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "environment-secret")
    monkeypatch.setenv("LIVEKIT_INFERENCE_URL", "https://custom-inference.example/v1")

    model = InferenceRealtimeModel("openai/gpt-realtime")

    assert model._inference_opts.api_key == "environment-key"
    assert model._inference_opts.api_secret == "environment-secret"
    assert model._opts.base_url == "https://custom-inference.example/v1"


async def test_initial_event_is_ga_session_update(
    paused_realtime_main: None,
) -> None:
    model = InferenceRealtimeModel("openai/gpt-realtime", api_key="key", api_secret="secret")
    session = model.session()

    event = session._msg_ch.recv_nowait()
    dumped = event.model_dump(exclude_unset=True) if hasattr(event, "model_dump") else event
    assert dumped["type"] == "session.update"
    assert dumped["session"]["type"] == "realtime"
    assert "model" not in dumped["session"]
    assert dumped["type"] != "session.create"
    await session.aclose()


async def test_consumes_gateway_normalized_openai_transcription_events(
    paused_realtime_main: None,
) -> None:
    model = InferenceRealtimeModel(
        "xai/grok-voice-think-fast-2.0",
        api_key="key",
        api_secret="secret",
    )
    session = model.session()
    transcripts: list[llm.InputTranscriptionCompleted] = []
    session.on("input_audio_transcription_completed", transcripts.append)

    session._handle_conversion_item_input_audio_transcription_delta(
        ConversationItemInputAudioTranscriptionDeltaEvent.construct(
            type="conversation.item.input_audio_transcription.delta",
            event_id="delta-event",
            item_id="user-item",
            content_index=0,
            delta="hello ",
        )
    )
    session._handle_conversion_item_input_audio_transcription_completed(
        ConversationItemInputAudioTranscriptionCompletedEvent.construct(
            type="conversation.item.input_audio_transcription.completed",
            event_id="completed-event",
            item_id="user-item",
            content_index=0,
            transcript="hello world",
        )
    )

    assert [(event.transcript, event.is_final) for event in transcripts] == [
        ("hello ", False),
        ("hello world", True),
    ]
    await session.aclose()


async def test_gateway_disconnect_reauthenticates_and_replays_committed_state_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokens = iter(("first-token", "second-token"))
    monkeypatch.setattr(
        inference_realtime,
        "create_access_token",
        lambda _key, _secret: next(tokens),
    )
    http_session = _FakeHTTPSession()
    model = InferenceRealtimeModel(
        "openai/gpt-realtime",
        provider="openai",
        base_url="https://inference.example/v1",
        api_key="key",
        api_secret="secret",
        http_session=http_session,  # type: ignore[arg-type]
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0, timeout=1),
    )
    session = model.session()
    reconnected = asyncio.Event()
    session.on("session_reconnected", lambda _: reconnected.set())

    await _wait_for(lambda: len(http_session.connections) == 1)
    first_ws = http_session.connections[0][2]
    await _wait_for(lambda: len(first_ws.sent) == 1)

    await session.update_instructions("Keep answers short.")
    await session.update_tools([lookup_weather])
    await _wait_for(lambda: len(first_ws.sent) == 3)

    first_ws.push_server_event(
        {
            "type": "conversation.item.added",
            "event_id": "committed-event",
            "previous_item_id": None,
            "item": {
                "id": "committed-user",
                "object": "realtime.item",
                "type": "message",
                "status": "completed",
                "role": "user",
                "content": [{"type": "input_text", "text": "committed message"}],
            },
        }
    )
    await _wait_for(lambda: session.chat_ctx.get_by_id("committed-user") is not None)

    first_ws.send_gate = asyncio.Event()
    session._pushed_duration_s = 0.5
    session.send_event({"type": "input_audio_buffer.append", "audio": "stale-audio"})
    await first_ws.send_started.wait()
    stale_response = session.generate_reply()
    uncommitted_ctx = session.chat_ctx.copy()
    uncommitted_ctx.add_message(id="uncommitted-user", role="user", content="not committed")
    stale_chat_update = asyncio.create_task(session.update_chat_ctx(uncommitted_ctx))
    await asyncio.sleep(0)
    assert not stale_chat_update.done()

    first_ws.disconnect()
    await _wait_for(lambda: len(http_session.connections) == 2)
    await asyncio.wait_for(reconnected.wait(), timeout=2)
    second_ws = http_session.connections[1][2]

    assert [headers["Authorization"] for _, headers, _ in http_session.connections] == [
        "Bearer first-token",
        "Bearer second-token",
    ]
    assert [event["type"] for event in second_ws.sent] == [
        "session.update",
        "session.update",
        "conversation.item.create",
    ]
    assert second_ws.sent[0]["session"]["instructions"] == "Keep answers short."
    assert second_ws.sent[1]["session"]["tools"][0]["name"] == "lookup_weather"
    assert second_ws.sent[2]["item"]["id"] == "committed-user"
    assert all(
        event.get("audio") != "stale-audio"
        and event["type"] != "response.create"
        and event.get("item", {}).get("id") != "uncommitted-user"
        for event in second_ws.sent
    )
    assert session._pushed_duration_s == 0
    with pytest.raises(llm.RealtimeError, match="discarded due to session reconnection"):
        await stale_response
    await stale_chat_update

    session.send_event({"type": "input_audio_buffer.append", "audio": "new-audio"})
    session.send_event({"type": "response.create", "event_id": "new-response", "response": {}})
    await _wait_for(lambda: len(second_ws.sent) == 5)
    assert [event["type"] for event in second_ws.sent[-2:]] == [
        "input_audio_buffer.append",
        "response.create",
    ]
    assert second_ws.sent[-2]["audio"] == "new-audio"
    assert second_ws.sent[-1]["event_id"] == "new-response"

    await session.aclose()


async def test_events_queued_during_replay_are_filtered_before_live_sends() -> None:
    http_session = _FakeHTTPSession()
    model = InferenceRealtimeModel(
        "openai/gpt-realtime",
        base_url="https://inference.example/v1",
        api_key="key",
        api_secret="secret",
        http_session=http_session,  # type: ignore[arg-type]
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0, timeout=1),
    )
    session = model.session()
    reconnected = asyncio.Event()
    session.on("session_reconnected", lambda _: reconnected.set())

    await _wait_for(lambda: len(http_session.connections) == 1)
    first_ws = http_session.connections[0][2]
    await _wait_for(lambda: len(first_ws.sent) == 1)

    replay_gate = asyncio.Event()
    http_session.next_send_gate = replay_gate
    first_ws.disconnect()
    await _wait_for(lambda: len(http_session.connections) == 2)
    second_ws = http_session.connections[1][2]
    await second_ws.send_started.wait()

    session.send_event({"type": "input_audio_buffer.append", "audio": "replay-window-audio"})
    stale_response = session.generate_reply()
    instructions_update = asyncio.create_task(
        session.update_instructions("Instructions updated during reconnect.")
    )
    await instructions_update
    assert second_ws.sent == []

    replay_gate.set()
    await asyncio.wait_for(reconnected.wait(), timeout=2)
    await _wait_for(lambda: len(second_ws.sent) == 2)

    assert [event["type"] for event in second_ws.sent] == [
        "session.update",
        "session.update",
    ]
    assert second_ws.sent[-1]["session"]["instructions"] == (
        "Instructions updated during reconnect."
    )
    assert all(
        event.get("audio") != "replay-window-audio" and event["type"] != "response.create"
        for event in second_ws.sent
    )
    with pytest.raises(llm.RealtimeError, match="discarded due to session reconnection"):
        await stale_response

    await session.aclose()


async def test_close_during_replay_drops_preserved_update_and_closes_socket() -> None:
    http_session = _FakeHTTPSession()
    model = InferenceRealtimeModel(
        "openai/gpt-realtime",
        base_url="https://inference.example/v1",
        api_key="key",
        api_secret="secret",
        http_session=http_session,  # type: ignore[arg-type]
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0, timeout=1),
    )
    session = model.session()

    await _wait_for(lambda: len(http_session.connections) == 1)
    first_ws = http_session.connections[0][2]
    await _wait_for(lambda: len(first_ws.sent) == 1)

    replay_gate = asyncio.Event()
    http_session.next_send_gate = replay_gate
    first_ws.disconnect()
    await _wait_for(lambda: len(http_session.connections) == 2)
    second_ws = http_session.connections[1][2]
    await second_ws.send_started.wait()

    await session.update_instructions("Do not requeue after close.")
    close_task = asyncio.create_task(session.aclose())
    await _wait_for(lambda: session._msg_ch.closed)
    replay_gate.set()

    await asyncio.wait_for(close_task, timeout=2)
    assert second_ws.closed
    assert [event["type"] for event in second_ws.sent] == ["session.update"]


async def test_fatal_replay_marks_closing_before_update_waiters_resume() -> None:
    http_session = _FakeHTTPSession()
    model = InferenceRealtimeModel(
        "openai/gpt-realtime",
        base_url="https://inference.example/v1",
        api_key="key",
        api_secret="secret",
        http_session=http_session,  # type: ignore[arg-type]
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0, timeout=1),
    )
    session = model.session()

    await _wait_for(lambda: len(http_session.connections) == 1)
    first_ws = http_session.connections[0][2]
    await _wait_for(lambda: len(first_ws.sent) == 1)

    replay_gate = asyncio.Event()
    http_session.next_send_gate = replay_gate
    first_ws.disconnect()
    await _wait_for(lambda: len(http_session.connections) == 2)
    second_ws = http_session.connections[1][2]
    await second_ws.send_started.wait()

    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="must fail with the session")
    chat_update = asyncio.create_task(session.update_chat_ctx(chat_ctx))
    tools_update = asyncio.create_task(session.update_tools([lookup_weather]))
    await asyncio.sleep(0)
    assert not chat_update.done()
    assert not tools_update.done()

    close_gate = asyncio.Event()
    second_ws.close_gate = close_gate
    second_ws.send_error = RuntimeError("replay write failed")
    replay_gate.set()
    await second_ws.close_started.wait()

    assert session._closing
    assert not second_ws.closed
    assert not session._main_atask.done()
    with pytest.raises(llm.RealtimeError, match="session closed"):
        await asyncio.wait_for(chat_update, timeout=0.2)
    with pytest.raises(llm.RealtimeError, match="session closed"):
        await asyncio.wait_for(tools_update, timeout=0.2)

    close_gate.set()
    with pytest.raises(APIConnectionError, match="connection failed after 1 attempts"):
        await session._main_atask
    assert second_ws.closed

    for _ in range(2):
        with pytest.raises(APIConnectionError, match="connection failed after 1 attempts"):
            await session.aclose()


async def test_retryable_replay_failure_does_not_mark_session_closing() -> None:
    http_session = _FakeHTTPSession()
    model = InferenceRealtimeModel(
        "openai/gpt-realtime",
        base_url="https://inference.example/v1",
        api_key="key",
        api_secret="secret",
        http_session=http_session,  # type: ignore[arg-type]
        conn_options=APIConnectOptions(max_retry=2, retry_interval=0, timeout=1),
    )
    session = model.session()
    reconnected = asyncio.Event()
    session.on("session_reconnected", lambda _: reconnected.set())

    await _wait_for(lambda: len(http_session.connections) == 1)
    first_ws = http_session.connections[0][2]
    await _wait_for(lambda: len(first_ws.sent) == 1)

    replay_gate = asyncio.Event()
    http_session.next_send_gate = replay_gate
    first_ws.disconnect()
    await _wait_for(lambda: len(http_session.connections) == 2)
    second_ws = http_session.connections[1][2]
    await second_ws.send_started.wait()
    second_ws.send_error = RuntimeError("transient replay write failure")
    replay_gate.set()

    await _wait_for(lambda: len(http_session.connections) == 3)
    await asyncio.wait_for(reconnected.wait(), timeout=2)
    third_ws = http_session.connections[2][2]

    assert not session._closing
    assert second_ws.closed
    assert [event["type"] for event in third_ws.sent] == ["session.update"]

    await session.aclose()


async def test_xai_say_queued_during_replay_is_discarded_and_settled() -> None:
    from livekit.plugins.xai.realtime import RealtimeModel as XAIRealtimeModel

    http_session = _FakeHTTPSession()
    model = XAIRealtimeModel(
        api_key="key",
        base_url="https://xai.example/v1",
        http_session=http_session,  # type: ignore[arg-type]
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0, timeout=1),
    )
    session = model.session()
    reconnected = asyncio.Event()
    session.on("session_reconnected", lambda _: reconnected.set())

    await _wait_for(lambda: len(http_session.connections) == 1)
    first_ws = http_session.connections[0][2]
    await _wait_for(lambda: len(first_ws.sent) == 1)

    replay_gate = asyncio.Event()
    http_session.next_send_gate = replay_gate
    first_ws.disconnect()
    await _wait_for(lambda: len(http_session.connections) == 2)
    second_ws = http_session.connections[1][2]
    await second_ws.send_started.wait()

    say_future = session.say("Do not replay this.")
    await _wait_for(lambda: len(session._pending_say_event_ids) == 1)
    say_event_id = session._pending_say_event_ids[0]
    assert say_event_id in session._response_created_futures

    replay_gate.set()
    await asyncio.wait_for(reconnected.wait(), timeout=2)
    with pytest.raises(llm.RealtimeError, match="discarded due to session reconnection"):
        await say_future
    await _wait_for(lambda: not session._say_tasks)

    assert [event["type"] for event in second_ws.sent] == ["session.update"]
    assert say_event_id not in session._response_created_futures
    assert say_event_id not in session._pending_say_event_ids

    await session.aclose()


async def test_reconnect_cancels_streaming_xai_say_collection() -> None:
    from livekit.plugins.xai.realtime import RealtimeModel as XAIRealtimeModel

    collection_started = asyncio.Event()
    collection_cancelled = asyncio.Event()

    async def streaming_text() -> AsyncIterator[str]:
        try:
            collection_started.set()
            await asyncio.Event().wait()
            yield "unreachable"
        finally:
            collection_cancelled.set()

    http_session = _FakeHTTPSession()
    model = XAIRealtimeModel(
        api_key="key",
        base_url="https://xai.example/v1",
        http_session=http_session,  # type: ignore[arg-type]
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0, timeout=1),
    )
    session = model.session()
    reconnected = asyncio.Event()
    session.on("session_reconnected", lambda _: reconnected.set())

    await _wait_for(lambda: len(http_session.connections) == 1)
    first_ws = http_session.connections[0][2]
    await _wait_for(lambda: len(first_ws.sent) == 1)

    say_future = session.say(streaming_text())
    await collection_started.wait()
    assert session._say_tasks
    assert session._response_created_futures

    first_ws.disconnect()
    await _wait_for(lambda: len(http_session.connections) == 2)
    await asyncio.wait_for(reconnected.wait(), timeout=2)
    await asyncio.wait_for(collection_cancelled.wait(), timeout=2)
    with pytest.raises(llm.RealtimeError, match="discarded due to session reconnection"):
        await say_future
    await _wait_for(lambda: not session._say_tasks)

    assert not session._response_created_futures
    assert not session._pending_say_event_ids
    assert [event["type"] for event in http_session.connections[1][2].sent] == ["session.update"]

    await session.aclose()


async def test_cancelled_in_flight_xai_say_cannot_steal_next_response() -> None:
    from livekit.plugins.xai.realtime import RealtimeModel as XAIRealtimeModel

    http_session = _FakeHTTPSession()
    model = XAIRealtimeModel(
        api_key="key",
        base_url="https://xai.example/v1",
        http_session=http_session,  # type: ignore[arg-type]
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0, timeout=1),
    )
    session = model.session()
    reconnected = asyncio.Event()
    session.on("session_reconnected", lambda _: reconnected.set())

    await _wait_for(lambda: len(http_session.connections) == 1)
    first_ws = http_session.connections[0][2]
    await _wait_for(lambda: len(first_ws.sent) == 1)

    first_ws.send_gate = asyncio.Event()
    cancelled_say = session.say("Discard this cancelled say.")
    await _wait_for(lambda: len(session._pending_say_event_ids) == 1)
    await first_ws.send_started.wait()
    cancelled_event_id = session._pending_say_event_ids[0]
    cancelled_say.cancel()
    await _wait_for(lambda: cancelled_event_id not in session._response_created_futures)
    assert cancelled_event_id in session._discarded_event_ids

    first_ws.disconnect()
    await _wait_for(lambda: len(http_session.connections) == 2)
    second_ws = http_session.connections[1][2]
    await asyncio.wait_for(reconnected.wait(), timeout=2)
    await _wait_for(lambda: cancelled_event_id not in session._pending_say_event_ids)
    assert not session._discarded_event_ids
    assert [event["type"] for event in second_ws.sent] == ["session.update"]
    with pytest.raises(asyncio.CancelledError):
        await cancelled_say

    next_say = session.say("Speak this one.")
    await _wait_for(
        lambda: any(
            event.get("item", {}).get("type") == "force_message" for event in second_ws.sent
        )
    )
    next_event_id = session._pending_say_event_ids[0]
    second_ws.push_server_event(
        {
            "type": "response.created",
            "event_id": "server-response-created",
            "response": {
                "id": "second-response",
                "object": "realtime.response",
                "output": [],
            },
        }
    )
    generation = await asyncio.wait_for(next_say, timeout=2)
    await _wait_for(lambda: not session._say_tasks)

    assert generation.response_id == "second-response"
    assert cancelled_event_id != next_event_id
    assert not session._response_created_futures
    assert not session._pending_say_event_ids

    await session.aclose()


async def test_updates_waiting_on_replay_locks_fail_when_session_closes(
    paused_realtime_main: None,
) -> None:
    model = InferenceRealtimeModel(
        "openai/gpt-realtime",
        api_key="key",
        api_secret="secret",
    )
    session = model.session()
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="must not be queued")

    await session._update_chat_ctx_lock.acquire()
    await session._update_fnc_ctx_lock.acquire()
    chat_update = asyncio.create_task(session.update_chat_ctx(chat_ctx))
    tools_update = asyncio.create_task(session.update_tools([lookup_weather]))
    await asyncio.sleep(0)
    assert not chat_update.done()
    assert not tools_update.done()

    await session.aclose()
    session._update_chat_ctx_lock.release()
    session._update_fnc_ctx_lock.release()

    with pytest.raises(llm.RealtimeError, match="session closed"):
        await asyncio.wait_for(chat_update, timeout=0.2)
    with pytest.raises(llm.RealtimeError, match="session closed"):
        await asyncio.wait_for(tools_update, timeout=0.2)


@pytest.mark.parametrize("provider", ["openai", "azure", "xai"])
async def test_shared_reconnect_replays_state_for_direct_providers(provider: str) -> None:
    http_session = _FakeHTTPSession()
    conn_options = APIConnectOptions(max_retry=1, retry_interval=0, timeout=1)
    if provider == "openai":
        model = RealtimeModel(
            api_key="key",
            base_url="https://openai.example/v1",
            http_session=http_session,  # type: ignore[arg-type]
            conn_options=conn_options,
        )
    elif provider == "azure":
        model = RealtimeModel(
            api_key="key",
            azure_deployment="deployment",
            api_version="2025-04-01-preview",
            base_url="https://azure.example/openai",
            http_session=http_session,  # type: ignore[arg-type]
            conn_options=conn_options,
        )
    else:
        from livekit.plugins.xai.realtime import RealtimeModel as XAIRealtimeModel

        model = XAIRealtimeModel(
            api_key="key",
            base_url="https://xai.example/v1",
            http_session=http_session,  # type: ignore[arg-type]
            conn_options=conn_options,
        )

    session = model.session()
    reconnected = asyncio.Event()
    session.on("session_reconnected", lambda _: reconnected.set())
    await _wait_for(lambda: len(http_session.connections) == 1)
    first_ws = http_session.connections[0][2]
    await _wait_for(lambda: len(first_ws.sent) == 1)
    session._remote_chat_ctx.insert(
        None,
        llm.ChatMessage(
            id=f"{provider}-committed",
            role="user",
            content=["committed"],
        ),
    )

    first_ws.disconnect()
    await _wait_for(lambda: len(http_session.connections) == 2)
    await asyncio.wait_for(reconnected.wait(), timeout=2)
    second_ws = http_session.connections[1][2]

    assert [event["type"] for event in second_ws.sent] == [
        "session.update",
        "conversation.item.create",
    ]
    assert second_ws.sent[1]["item"]["id"] == f"{provider}-committed"

    await session.aclose()


def _server_error(event_id: str, code: str) -> dict[str, Any]:
    return {
        "type": "error",
        "event_id": f"server-error-{event_id}",
        "error": {
            "type": "invalid_request_error",
            "code": code,
            "message": code.replace("_", " "),
            "event_id": event_id,
        },
    }


@pytest.mark.parametrize("dynamic_update", [False, True], ids=["initial", "dynamic-update"])
async def test_unsupported_transcription_model_is_terminal_without_reconnect(
    dynamic_update: bool,
) -> None:
    http_session = _FakeHTTPSession()
    transcription = AudioTranscription(model="unpriced-transcription-model")
    model = InferenceRealtimeModel(
        "openai/gpt-realtime",
        base_url="https://inference.example/v1",
        api_key="key",
        api_secret="secret",
        input_audio_transcription=None if dynamic_update else transcription,
        http_session=http_session,  # type: ignore[arg-type]
        conn_options=APIConnectOptions(max_retry=2, retry_interval=0, timeout=1),
    )
    session = model.session()
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)

    await _wait_for(lambda: len(http_session.connections) == 1)
    ws = http_session.connections[0][2]
    await _wait_for(lambda: len(ws.sent) == 1)
    if dynamic_update:
        model.update_options(input_audio_transcription=transcription)
        await _wait_for(lambda: len(ws.sent) == 2)

    rejected_event = ws.sent[-1]
    assert rejected_event["type"] == "session.update"
    assert rejected_event["session"]["audio"]["input"]["transcription"]["model"] == (
        "unpriced-transcription-model"
    )
    ws.push_server_event(
        _server_error(rejected_event["event_id"], "unsupported_transcription_model")
    )

    await _wait_for(session._main_atask.done)
    with pytest.raises(APIError) as exc_info:
        await session._main_atask

    assert exc_info.value.retryable is False
    assert len(http_session.connections) == 1
    assert len(errors) == 1
    assert errors[0].recoverable is False
    assert isinstance(errors[0].error, APIError)


@pytest.mark.parametrize(
    "code",
    ["unsupported_audio_transport", "unsupported_audio_format", "invalid_audio_payload"],
)
async def test_terminal_gateway_audio_error_does_not_reconnect(code: str) -> None:
    http_session = _FakeHTTPSession()
    model = InferenceRealtimeModel(
        "xai/grok-voice-think-fast-2.0",
        base_url="https://inference.example/v1",
        api_key="key",
        api_secret="secret",
        http_session=http_session,  # type: ignore[arg-type]
        conn_options=APIConnectOptions(max_retry=2, retry_interval=0, timeout=1),
    )
    session = model.session()
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)

    await _wait_for(lambda: len(http_session.connections) == 1)
    ws = http_session.connections[0][2]
    await _wait_for(lambda: len(ws.sent) == 1)
    ws.push_server_event(_server_error(ws.sent[0]["event_id"], code))

    await _wait_for(session._main_atask.done)
    with pytest.raises(APIError) as exc_info:
        await session._main_atask

    assert exc_info.value.retryable is False
    assert len(http_session.connections) == 1
    assert len(errors) == 1
    assert errors[0].recoverable is False


@pytest.mark.parametrize("provider", ["openai", "azure", "xai"])
async def test_direct_provider_invalid_request_stays_recoverable(provider: str) -> None:
    http_session = _FakeHTTPSession()
    conn_options = APIConnectOptions(max_retry=1, retry_interval=0, timeout=1)
    if provider == "openai":
        model = RealtimeModel(
            api_key="key",
            base_url="https://openai.example/v1",
            http_session=http_session,  # type: ignore[arg-type]
            conn_options=conn_options,
        )
    elif provider == "azure":
        model = RealtimeModel(
            api_key="key",
            azure_deployment="deployment",
            api_version="2025-04-01-preview",
            base_url="https://azure.example/openai",
            http_session=http_session,  # type: ignore[arg-type]
            conn_options=conn_options,
        )
    else:
        from livekit.plugins.xai.realtime import RealtimeModel as XAIRealtimeModel

        model = XAIRealtimeModel(
            api_key="key",
            base_url="https://xai.example/v1",
            http_session=http_session,  # type: ignore[arg-type]
            conn_options=conn_options,
        )

    session = model.session()
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)
    await _wait_for(lambda: len(http_session.connections) == 1)
    ws = http_session.connections[0][2]
    await _wait_for(lambda: len(ws.sent) == 1)

    ws.push_server_event(_server_error(ws.sent[0]["event_id"], "invalid_request"))
    await _wait_for(lambda: len(errors) == 1)
    await session.update_instructions("The socket remains usable.")
    await _wait_for(lambda: len(ws.sent) == 2)

    assert errors[0].recoverable is True
    assert not session._main_atask.done()
    assert len(http_session.connections) == 1
    await session.aclose()


async def test_rejected_image_is_not_committed_or_replayed() -> None:
    http_session = _FakeHTTPSession()
    model = InferenceRealtimeModel(
        "openai/gpt-realtime",
        base_url="https://inference.example/v1",
        api_key="key",
        api_secret="secret",
        http_session=http_session,  # type: ignore[arg-type]
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0, timeout=1),
    )
    session = model.session()
    reconnected = asyncio.Event()
    session.on("session_reconnected", lambda _: reconnected.set())

    await _wait_for(lambda: len(http_session.connections) == 1)
    first_ws = http_session.connections[0][2]
    await _wait_for(lambda: len(first_ws.sent) == 1)

    image_ctx = llm.ChatContext.empty()
    image_ctx.add_message(
        id="rejected-image",
        role="user",
        content=[llm.ImageContent(image="data:image/png;base64,aW1hZ2U=")],
    )
    image_update = asyncio.create_task(session.update_chat_ctx(image_ctx))
    await _wait_for(lambda: len(first_ws.sent) == 2)
    image_event = first_ws.sent[-1]
    assert image_event["item"]["content"][0]["type"] == "input_image"

    first_ws.push_server_event(_server_error(image_event["event_id"], "unsupported_image_input"))
    await asyncio.wait_for(image_update, timeout=0.5)

    assert session.chat_ctx.get_by_id("rejected-image") is None
    assert len(http_session.connections) == 1
    assert not session._main_atask.done()

    text_ctx = llm.ChatContext.empty()
    text_ctx.add_message(id="valid-text", role="user", content="hello")
    text_update = asyncio.create_task(session.update_chat_ctx(text_ctx))
    await _wait_for(lambda: len(first_ws.sent) == 3)
    text_event = first_ws.sent[-1]
    first_ws.push_server_event(
        {
            "type": "conversation.item.added",
            "event_id": "valid-text-added",
            "previous_item_id": None,
            "item": text_event["item"],
        }
    )
    await asyncio.wait_for(text_update, timeout=0.5)

    assert session.chat_ctx.get_by_id("rejected-image") is None
    assert session.chat_ctx.get_by_id("valid-text") is not None
    assert len(http_session.connections) == 1

    first_ws.disconnect()
    await _wait_for(lambda: len(http_session.connections) == 2)
    await asyncio.wait_for(reconnected.wait(), timeout=2)
    replayed_events = http_session.connections[1][2].sent

    assert all(event.get("item", {}).get("id") != "rejected-image" for event in replayed_events)
    assert any(event.get("item", {}).get("id") == "valid-text" for event in replayed_events)
    await session.aclose()
