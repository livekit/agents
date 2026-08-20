from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from livekit.agents import APIConnectionError, llm
from livekit.plugins.openai.realtime import (
    InferenceRealtimeModel,
    inference_realtime_model as inference_realtime,
)
from livekit.plugins.openai.realtime.realtime_model import (
    RealtimeModel,
    RealtimeSession,
)

pytestmark = pytest.mark.unit
_REPLAY_TIMEOUT_MS = 15_000


def _failover_event(
    *,
    protocol_version: object = 1,
    replay_timeout_ms: object = 15_000,
    context_lost: object = True,
) -> dict:
    return {
        "type": "livekit.session.failover",
        "protocol_version": protocol_version,
        "replay_timeout_ms": replay_timeout_ms,
        "context_lost": context_lost,
    }


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_str(self, data: str) -> None:
        self.sent.append(json.loads(data))


class _BlockingFirstSendWebSocket(_FakeWebSocket):
    def __init__(self) -> None:
        super().__init__()
        self.first_send_started = asyncio.Event()
        self.release_first_send = asyncio.Event()

    async def send_str(self, data: str) -> None:
        if not self.sent:
            self.first_send_started.set()
            await self.release_first_send.wait()
        await super().send_str(data)


class _FakeHTTPSession:
    def __init__(self) -> None:
        self.connections: list[tuple[str, dict[str, str]]] = []

    async def ws_connect(self, *, url: str, headers: dict[str, str]) -> _FakeWebSocket:
        self.connections.append((url, headers))
        return _FakeWebSocket()


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
        "openai/gpt-realtime",
        provider="openai",
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
    assert parse_qs(parsed.query) == {"model": ["openai/gpt-realtime"]}
    assert [headers["Authorization"] for _, headers in http_session.connections] == [
        "Bearer token-one:key:secret",
        "Bearer token-two:key:secret",
    ]
    assert http_session.connections[0][1]["X-Test-Class"] == "priority"
    assert http_session.connections[0][1]["X-LiveKit-Inference-Provider"] == "openai"
    assert [
        headers["X-LiveKit-Realtime-Failover-Protocol"] for _, headers in http_session.connections
    ] == ["1", "1"]
    await session.aclose()


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
    assert dumped["type"] != "session.create"
    await session.aclose()


async def test_gateway_failover_replays_conversation_then_audio_without_options(
    paused_realtime_main: None,
) -> None:
    model = InferenceRealtimeModel("openai/gpt-realtime", api_key="key", api_secret="secret")
    session = model.session()
    session._msg_ch.recv_nowait()  # initial session.update
    session._remote_chat_ctx.insert(
        None, llm.ChatMessage(id="user-1", role="user", content=["hello"])
    )
    session._uncommitted_audio = [b"first", b"second"]
    session._uncommitted_audio_bytes = len(b"firstsecond")
    ws = _FakeWebSocket()

    handled = await session._handle_extra_server_event(  # type: ignore[arg-type]
        _failover_event(), ws
    )

    assert handled is True
    assert [event["type"] for event in ws.sent] == [
        "conversation.item.create",
        "input_audio_buffer.append",
        "input_audio_buffer.append",
        "livekit.session.replay_completed",
    ]
    assert all(
        event["event_id"].startswith("livekit_replay_audio_")
        for event in ws.sent
        if event["type"] == "input_audio_buffer.append"
    )
    assert all(event["type"] != "session.update" for event in ws.sent)
    await session.aclose()


async def test_unsupported_gateway_failover_protocol_is_terminal(
    paused_realtime_main: None,
) -> None:
    model = InferenceRealtimeModel("openai/gpt-realtime", api_key="key", api_secret="secret")
    session = model.session()

    with pytest.raises(APIConnectionError, match="unsupported.*version") as exc_info:
        await session._handle_extra_server_event(  # type: ignore[arg-type]
            _failover_event(protocol_version=2), _FakeWebSocket()
        )

    assert exc_info.value.retryable is False
    await session.aclose()


@pytest.mark.parametrize("replay_timeout_ms", [None, 0, -1, "15000", 30_001])
async def test_gateway_failover_rejects_invalid_replay_timeout_budget(
    replay_timeout_ms: object,
    paused_realtime_main: None,
) -> None:
    model = InferenceRealtimeModel("openai/gpt-realtime", api_key="key", api_secret="secret")
    session = model.session()

    with pytest.raises(APIConnectionError, match="invalid.*replay_timeout_ms") as exc_info:
        await session._handle_extra_server_event(  # type: ignore[arg-type]
            _failover_event(replay_timeout_ms=replay_timeout_ms), _FakeWebSocket()
        )

    assert exc_info.value.retryable is False
    await session.aclose()


async def test_context_preserving_failover_only_acknowledges_the_handoff(
    paused_realtime_main: None,
) -> None:
    model = InferenceRealtimeModel("openai/gpt-realtime", api_key="key", api_secret="secret")
    session = model.session()
    session._record_sent_input_audio(b"active-user-turn")
    ws = _FakeWebSocket()

    handled = await session._handle_extra_server_event(  # type: ignore[arg-type]
        _failover_event(context_lost=False), ws
    )

    assert handled is True
    assert ws.sent == [{"type": "livekit.session.replay_completed"}]
    assert session._uncommitted_audio == [b"active-user-turn"]
    await session.aclose()


async def test_context_preserving_handoff_does_not_drop_interrupt_or_truncate(
    paused_realtime_main: None,
) -> None:
    model = InferenceRealtimeModel("openai/gpt-realtime", api_key="key", api_secret="secret")
    session = model.session()
    session._msg_ch.recv_nowait()  # initial session.update
    pending: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.Future()
    session._response_created_futures["response"] = pending
    ws = _BlockingFirstSendWebSocket()

    handoff_task = asyncio.create_task(
        session._handle_extra_server_event(  # type: ignore[arg-type]
            _failover_event(context_lost=False), ws
        )
    )
    await ws.first_send_started.wait()
    session.interrupt()
    session.truncate(message_id="assistant-item", modalities=["audio"], audio_end_ms=250)
    ws.release_first_send.set()
    assert await handoff_task is True

    queued = [session._msg_ch.recv_nowait(), session._msg_ch.recv_nowait()]
    event_types = [
        event.model_dump(exclude_unset=True)["type"]
        if hasattr(event, "model_dump")
        else event["type"]
        for event in queued
    ]
    assert event_types == ["response.cancel", "conversation.item.truncate"]
    session._response_created_futures.clear()
    pending.cancel()
    await session.aclose()


async def test_gateway_failover_requires_boolean_context_lost(
    paused_realtime_main: None,
) -> None:
    model = InferenceRealtimeModel("openai/gpt-realtime", api_key="key", api_secret="secret")
    session = model.session()

    with pytest.raises(APIConnectionError, match="invalid.*context_lost"):
        await session._handle_extra_server_event(  # type: ignore[arg-type]
            _failover_event(context_lost="yes"), _FakeWebSocket()
        )
    await session.aclose()


async def test_gateway_failover_interrupt_timeout_does_not_cancel_agent_interrupt(
    paused_realtime_main: None,
) -> None:
    class _AgentSession:
        agent_state = "listening"

        def __init__(self) -> None:
            self.current_agent = SimpleNamespace(chat_ctx=llm.ChatContext.empty())
            self.interrupt_started = asyncio.Event()
            self.release_interrupt = asyncio.Event()
            self.interrupt_completed = asyncio.Event()
            self.interrupt_cancelled = False

        async def interrupt(self, *, force: bool) -> None:
            assert force is True
            self.interrupt_started.set()
            try:
                await self.release_interrupt.wait()
            except asyncio.CancelledError:
                self.interrupt_cancelled = True
                raise
            self.interrupt_completed.set()

    model = InferenceRealtimeModel("openai/gpt-realtime", api_key="key", api_secret="secret")
    session = model.session()
    agent_session = _AgentSession()
    session._agent_session = agent_session  # type: ignore[assignment]
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)

    handled = await session._handle_extra_server_event(  # type: ignore[arg-type]
        _failover_event(replay_timeout_ms=250), _FakeWebSocket()
    )

    assert handled is True
    assert agent_session.interrupt_started.is_set()
    assert not agent_session.interrupt_completed.is_set()
    assert agent_session.interrupt_cancelled is False
    assert session._live_forwarding_allowed.is_set()
    assert session._gateway_failover_in_progress is False
    assert len(errors) == 1
    assert errors[0].recoverable is True
    assert "timed out interrupting the agent" in str(errors[0].error)

    agent_session.release_interrupt.set()
    await asyncio.wait_for(agent_session.interrupt_completed.wait(), timeout=1)
    assert agent_session.interrupt_cancelled is False
    await session.aclose()


async def test_gateway_failover_total_replay_timeout_restores_live_forwarding(
    paused_realtime_main: None,
) -> None:
    class _NeverCompletingWebSocket(_FakeWebSocket):
        def __init__(self) -> None:
            super().__init__()
            self.send_started = asyncio.Event()
            self.send_cancelled = asyncio.Event()

        async def send_str(self, data: str) -> None:
            self.send_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.send_cancelled.set()

    model = InferenceRealtimeModel("openai/gpt-realtime", api_key="key", api_secret="secret")
    session = model.session()
    session._msg_ch.recv_nowait()  # initial session.update
    desired_ctx = llm.ChatContext.empty()
    desired_ctx.add_message(role="user", content="must survive replay failure")
    session._deferred_chat_ctx = desired_ctx
    ws = _NeverCompletingWebSocket()

    with pytest.raises(APIConnectionError, match="timed out replaying"):
        await session._handle_extra_server_event(  # type: ignore[arg-type]
            _failover_event(replay_timeout_ms=100), ws
        )

    assert ws.send_started.is_set()
    assert ws.send_cancelled.is_set()
    assert session._live_forwarding_allowed.is_set()
    assert session._gateway_failover_in_progress is False
    assert not session._ws_send_lock.locked()
    await asyncio.sleep(0)
    deferred_event = session._msg_ch.recv_nowait()
    dumped = (
        deferred_event.model_dump(exclude_unset=True)
        if hasattr(deferred_event, "model_dump")
        else deferred_event
    )
    assert dumped["type"] == "conversation.item.create"
    await session.aclose()


async def test_close_cancels_a_still_pending_timed_out_interrupt(
    paused_realtime_main: None,
) -> None:
    class _AgentSession:
        agent_state = "listening"

        def __init__(self) -> None:
            self.current_agent = SimpleNamespace(chat_ctx=llm.ChatContext.empty())
            self.interrupt_cancelled = asyncio.Event()

        async def interrupt(self, *, force: bool) -> None:
            assert force is True
            try:
                await asyncio.Event().wait()
            finally:
                self.interrupt_cancelled.set()

    model = InferenceRealtimeModel("openai/gpt-realtime", api_key="key", api_secret="secret")
    session = model.session()
    agent_session = _AgentSession()
    session._agent_session = agent_session  # type: ignore[assignment]

    await session._handle_extra_server_event(  # type: ignore[arg-type]
        _failover_event(replay_timeout_ms=250), _FakeWebSocket()
    )
    assert session._pending_interrupt_futures

    await session.aclose()
    assert agent_session.interrupt_cancelled.is_set()
    assert not session._pending_interrupt_futures


async def test_gateway_failover_pauses_then_resumes_queued_live_audio(
    paused_realtime_main: None,
) -> None:
    class _BlockingWebSocket(_FakeWebSocket):
        def __init__(self) -> None:
            super().__init__()
            self.send_started = asyncio.Event()
            self.release_send = asyncio.Event()

        async def send_str(self, data: str) -> None:
            self.send_started.set()
            await self.release_send.wait()
            await super().send_str(data)

    model = InferenceRealtimeModel("openai/gpt-realtime", api_key="key", api_secret="secret")
    session = model.session()
    session._msg_ch.recv_nowait()
    session._uncommitted_audio = [b"replay"]
    session._uncommitted_audio_bytes = len(b"replay")
    ws = _BlockingWebSocket()

    failover = asyncio.create_task(
        session._handle_gateway_failover(  # type: ignore[arg-type]
            ws, replay_timeout_ms=_REPLAY_TIMEOUT_MS
        )
    )
    await ws.send_started.wait()
    session._queue_input_audio(data=b"live", duration=0.1)
    live_send = asyncio.create_task(session._wait_before_live_send())
    await asyncio.sleep(0)
    assert not live_send.done()

    ws.release_send.set()
    await failover
    await live_send

    assert ws.sent[-1]["type"] == "livekit.session.replay_completed"
    assert session._msg_ch.qsize() == 1
    await session.aclose()


async def test_audio_arriving_during_failover_setup_is_not_replayed_twice(
    paused_realtime_main: None,
) -> None:
    class _AgentSession:
        agent_state = "listening"

        def __init__(self) -> None:
            self.current_agent = SimpleNamespace(chat_ctx=llm.ChatContext.empty())
            self.interrupt_started = asyncio.Event()
            self.release_interrupt = asyncio.Event()

        async def interrupt(self, *, force: bool) -> None:
            assert force is True
            self.interrupt_started.set()
            await self.release_interrupt.wait()

    model = InferenceRealtimeModel("openai/gpt-realtime", api_key="key", api_secret="secret")
    session = model.session()
    session._msg_ch.recv_nowait()
    session._uncommitted_audio = [b"failed-leg"]
    session._uncommitted_audio_bytes = len(b"failed-leg")
    agent_session = _AgentSession()
    session._agent_session = agent_session  # type: ignore[assignment]
    ws = _FakeWebSocket()

    failover = asyncio.create_task(
        session._handle_gateway_failover(  # type: ignore[arg-type]
            ws, replay_timeout_ms=_REPLAY_TIMEOUT_MS
        )
    )
    await agent_session.interrupt_started.wait()
    session._queue_input_audio(data=b"resumed-live", duration=0.1)
    agent_session.release_interrupt.set()
    await failover

    replayed_audio = [
        event["audio"] for event in ws.sent if event["type"] == "input_audio_buffer.append"
    ]
    assert replayed_audio == [base64.b64encode(b"failed-leg").decode("utf-8")]
    assert session._msg_ch.qsize() == 1
    await session.aclose()


async def test_audio_buffer_is_bounded_and_never_replays_a_truncated_turn(
    paused_realtime_main: None,
) -> None:
    model = InferenceRealtimeModel(
        "openai/gpt-realtime",
        api_key="key",
        api_secret="secret",
        max_uncommitted_audio_bytes=4,
    )
    session = model.session()
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)

    session._record_sent_input_audio(b"123")
    session._record_sent_input_audio(b"45")

    assert session._uncommitted_audio_bytes == 0
    assert session._uncommitted_audio == []
    assert session._audio_buffer_overflowed is True
    assert len(errors) == 1
    assert errors[0].recoverable is True
    assert "replay disabled" in str(errors[0].error)

    ws = _FakeWebSocket()
    await session._handle_gateway_failover(  # type: ignore[arg-type]
        ws, replay_timeout_ms=_REPLAY_TIMEOUT_MS
    )
    assert [event["type"] for event in ws.sent] == ["livekit.session.replay_completed"]
    await session.aclose()


async def test_long_substituted_silence_does_not_overflow_or_end_the_call(
    paused_realtime_main: None,
) -> None:
    model = InferenceRealtimeModel("openai/gpt-realtime", api_key="key", api_secret="secret")
    session = model.session()
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)

    # Sixty seconds of 24 kHz mono PCM16 in 100 ms frames. AgentActivity
    # substitutes exactly this zero audio while the caller is not forwarded.
    for _ in range(600):
        session._record_sent_input_audio(bytes(4_800))

    assert session._audio_buffer_overflowed is False
    assert session._uncommitted_audio_bytes == 48_000
    assert session._active_user_turn_uncommitted is False
    assert errors == []
    await session.aclose()


async def test_only_audio_successfully_sent_to_the_failed_leg_is_replayable(
    paused_realtime_main: None,
) -> None:
    model = InferenceRealtimeModel("openai/gpt-realtime", api_key="key", api_secret="secret")
    session = model.session()
    ws = _FakeWebSocket()

    await session._send_ws_event(
        ws,  # type: ignore[arg-type]
        {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(b"provider-observed").decode("utf-8"),
        },
    )
    await session._send_ws_event(
        ws,  # type: ignore[arg-type]
        {
            "type": "input_audio_buffer.append",
            "event_id": "livekit_replay_audio_existing",
            "audio": base64.b64encode(b"already-replayed").decode("utf-8"),
        },
    )

    assert session._uncommitted_audio == [b"provider-observed"]
    await session.aclose()


async def test_in_flight_audio_is_included_in_the_frozen_failover_turn(
    paused_realtime_main: None,
) -> None:
    model = InferenceRealtimeModel("openai/gpt-realtime", api_key="key", api_secret="secret")
    session = model.session()
    ws = _BlockingFirstSendWebSocket()
    audio = base64.b64encode(b"in-flight").decode("utf-8")

    async def send_in_flight() -> None:
        async with session._ws_send_lock:
            await session._send_ws_event(  # type: ignore[arg-type]
                ws,
                {"type": "input_audio_buffer.append", "audio": audio},
            )

    send_task = asyncio.create_task(send_in_flight())
    await ws.first_send_started.wait()
    failover_task = asyncio.create_task(
        session._handle_gateway_failover(  # type: ignore[arg-type]
            ws, replay_timeout_ms=_REPLAY_TIMEOUT_MS
        )
    )
    await asyncio.sleep(0)
    assert not failover_task.done()

    ws.release_first_send.set()
    await asyncio.gather(send_task, failover_task)

    appends = [event for event in ws.sent if event["type"] == "input_audio_buffer.append"]
    assert len(appends) == 2
    assert appends[0]["audio"] == appends[1]["audio"] == audio
    assert appends[1]["event_id"].startswith("livekit_replay_audio_")
    await session.aclose()


async def test_audio_buffer_clears_on_local_and_upstream_boundaries(
    paused_realtime_main: None,
) -> None:
    model = InferenceRealtimeModel("openai/gpt-realtime", api_key="key", api_secret="secret")
    session = model.session()
    session._record_sent_input_audio(b"audio")
    session.commit_audio()
    assert session._uncommitted_audio == []

    session._record_sent_input_audio(b"more")
    await session._handle_extra_server_event(  # type: ignore[arg-type]
        {"type": "input_audio_buffer.committed"}, _FakeWebSocket()
    )
    assert session._uncommitted_audio == []

    session._record_sent_input_audio(b"again")
    session.clear_audio()
    assert session._uncommitted_audio == []
    await session.aclose()


async def test_failover_interrupts_and_regenerates_active_agent_reply(
    paused_realtime_main: None,
) -> None:
    class _AgentSession:
        # A response.create can be pending before AgentSession reaches thinking
        # or speaking. The pending realtime generation still has to be retried.
        agent_state = "listening"

        def __init__(self, realtime_session: inference_realtime.InferenceRealtimeSession) -> None:
            self.realtime_session = realtime_session
            self.current_agent = SimpleNamespace(chat_ctx=llm.ChatContext.empty())
            self.interrupt_calls = 0
            self.generate_reply_calls = 0

        async def interrupt(self, *, force: bool) -> None:
            assert force is True
            self.interrupt_calls += 1
            # AgentActivity may perform this provider sync while finalizing the
            # interrupted speech. It must not wait on the paused sender.
            interrupted_ctx = llm.ChatContext.empty()
            interrupted_ctx.add_message(role="assistant", content="heard before failover")
            await self.realtime_session.update_chat_ctx(interrupted_ctx)

        def generate_reply(self) -> None:
            self.generate_reply_calls += 1

    model = InferenceRealtimeModel("openai/gpt-realtime", api_key="key", api_secret="secret")
    session = model.session()
    agent_session = _AgentSession(session)
    session._agent_session = agent_session  # type: ignore[assignment]
    # Ordinary nonzero microphone noise is replayable audio, but only the
    # provider's speech_started event establishes response ownership.
    session._record_sent_input_audio(b"\x01\x00" * 20)
    pending: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.Future()
    session._response_created_futures["response"] = pending

    ws = _FakeWebSocket()
    await session._handle_gateway_failover(  # type: ignore[arg-type]
        ws, replay_timeout_ms=_REPLAY_TIMEOUT_MS
    )

    assert agent_session.interrupt_calls == 1
    assert agent_session.generate_reply_calls == 1
    assert isinstance(pending.exception(), llm.RealtimeError)
    assert any(event["type"] == "conversation.item.create" for event in ws.sent)
    assert any(event["type"] == "input_audio_buffer.append" for event in ws.sent)
    await session.aclose()


@pytest.mark.parametrize("activity_source", ["server_vad", "manual"])
async def test_failover_does_not_regenerate_while_replaying_an_active_user_turn(
    paused_realtime_main: None, activity_source: str
) -> None:
    class _AgentSession:
        agent_state = "speaking"

        def __init__(self) -> None:
            self.current_agent = SimpleNamespace(chat_ctx=llm.ChatContext.empty())
            self.generate_reply_calls = 0

        async def interrupt(self, *, force: bool) -> None:
            assert force is True

        def generate_reply(self) -> None:
            self.generate_reply_calls += 1

    model = InferenceRealtimeModel("openai/gpt-realtime", api_key="key", api_secret="secret")
    session = model.session()
    agent_session = _AgentSession()
    session._agent_session = agent_session  # type: ignore[assignment]
    ws = _FakeWebSocket()
    if activity_source == "server_vad":
        handled = await session._handle_extra_server_event(  # type: ignore[arg-type]
            {"type": "input_audio_buffer.speech_started"}, ws
        )
        assert handled is False
    else:
        session.start_user_activity()
    session._record_sent_input_audio(b"caller speech")
    pending: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.Future()
    session._response_created_futures["response"] = pending

    await session._handle_gateway_failover(  # type: ignore[arg-type]
        ws, replay_timeout_ms=_REPLAY_TIMEOUT_MS
    )

    assert any(event["type"] == "input_audio_buffer.append" for event in ws.sent)
    assert agent_session.generate_reply_calls == 0
    assert isinstance(pending.exception(), llm.RealtimeError)
    await session.aclose()


@pytest.mark.parametrize(
    ("activity_outcome", "expected_regenerations"),
    [("active", 0), ("committed", 0), ("cleared", 1)],
)
async def test_user_activity_outcome_during_replay_controls_reply_regeneration(
    paused_realtime_main: None,
    activity_outcome: str,
    expected_regenerations: int,
) -> None:
    class _AgentSession:
        agent_state = "speaking"

        def __init__(self) -> None:
            self.current_agent = SimpleNamespace(chat_ctx=llm.ChatContext.empty())
            self.generate_reply_calls = 0

        async def interrupt(self, *, force: bool) -> None:
            assert force is True

        def generate_reply(self) -> None:
            self.generate_reply_calls += 1

    model = InferenceRealtimeModel("openai/gpt-realtime", api_key="key", api_secret="secret")
    session = model.session()
    agent_session = _AgentSession()
    session._agent_session = agent_session  # type: ignore[assignment]
    session._record_sent_input_audio(b"background noise")
    pending: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.Future()
    session._response_created_futures["response"] = pending
    ws = _BlockingFirstSendWebSocket()

    failover_task = asyncio.create_task(
        session._handle_gateway_failover(  # type: ignore[arg-type]
            ws, replay_timeout_ms=_REPLAY_TIMEOUT_MS
        )
    )
    await ws.first_send_started.wait()
    session.start_user_activity()
    if activity_outcome == "committed":
        session.commit_audio()
    elif activity_outcome == "cleared":
        session.clear_audio()
    ws.release_first_send.set()
    await failover_task

    assert agent_session.generate_reply_calls == expected_regenerations
    assert isinstance(pending.exception(), llm.RealtimeError)
    await session.aclose()


async def test_chat_context_update_arriving_during_replay_is_deferred_not_lost(
    paused_realtime_main: None,
) -> None:
    model = InferenceRealtimeModel("openai/gpt-realtime", api_key="key", api_secret="secret")
    session = model.session()
    session._msg_ch.recv_nowait()  # initial session.update
    ws = _BlockingFirstSendWebSocket()

    failover_task = asyncio.create_task(
        session._handle_gateway_failover(  # type: ignore[arg-type]
            ws, replay_timeout_ms=_REPLAY_TIMEOUT_MS
        )
    )
    await ws.first_send_started.wait()

    updated_ctx = llm.ChatContext.empty()
    updated_ctx.add_message(role="user", content="arrived during replay")
    await session.update_chat_ctx(updated_ctx)

    ws.release_first_send.set()
    await failover_task
    await asyncio.sleep(0)

    deferred_event = session._msg_ch.recv_nowait()
    dumped = (
        deferred_event.model_dump(exclude_unset=True)
        if hasattr(deferred_event, "model_dump")
        else deferred_event
    )
    assert dumped["type"] == "conversation.item.create"
    await session.aclose()


async def test_failed_deferred_chat_sync_surfaces_a_recoverable_session_error(
    monkeypatch: pytest.MonkeyPatch,
    paused_realtime_main: None,
) -> None:
    async def fail_sync(self: RealtimeSession, chat_ctx: llm.ChatContext) -> None:
        raise llm.RealtimeError("provider did not acknowledge")

    monkeypatch.setattr(RealtimeSession, "update_chat_ctx", fail_sync)
    model = InferenceRealtimeModel("openai/gpt-realtime", api_key="key", api_secret="secret")
    session = model.session()
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)

    session._sync_deferred_chat_ctx(llm.ChatContext.empty())
    tasks = tuple(session._deferred_chat_ctx_tasks)
    await asyncio.gather(*tasks)

    assert len(errors) == 1
    assert errors[0].recoverable is True
    assert "failed to synchronize chat context" in str(errors[0].error)
    await session.aclose()


async def test_direct_openai_reconnect_replay_still_includes_session_options(
    paused_realtime_main: None,
) -> None:
    model = RealtimeModel(api_key="fake")
    session = model.session()

    events, _ = session._prepare_connection_replay(include_session_state=True)
    dumped = [
        event.model_dump(exclude_unset=True) if hasattr(event, "model_dump") else event
        for event in events
    ]

    assert dumped[0]["type"] == "session.update"
    await session.aclose()
