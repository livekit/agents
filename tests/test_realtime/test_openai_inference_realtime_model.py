from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from livekit.agents import llm
from livekit.plugins.openai.realtime import (
    InferenceRealtimeModel,
    inference_realtime_model as inference_realtime,
)
from livekit.plugins.openai.realtime.realtime_model import (
    RealtimeModel,
    RealtimeSession,
)

pytestmark = pytest.mark.unit


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
        {"type": "livekit.session.failover", "context_lost": True}, ws
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

    failover = asyncio.create_task(session._handle_gateway_failover(ws))  # type: ignore[arg-type]
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

    failover = asyncio.create_task(session._handle_gateway_failover(ws))  # type: ignore[arg-type]
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
    assert errors[0].recoverable is False

    ws = _FakeWebSocket()
    await session._handle_gateway_failover(ws)  # type: ignore[arg-type]
    assert [event["type"] for event in ws.sent] == ["livekit.session.replay_completed"]
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
        session._handle_gateway_failover(ws)  # type: ignore[arg-type]
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
    pending: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.Future()
    session._response_created_futures["response"] = pending

    ws = _FakeWebSocket()
    await session._handle_gateway_failover(ws)  # type: ignore[arg-type]

    assert agent_session.interrupt_calls == 1
    assert agent_session.generate_reply_calls == 1
    assert isinstance(pending.exception(), llm.RealtimeError)
    assert any(event["type"] == "conversation.item.create" for event in ws.sent)
    await session.aclose()


async def test_chat_context_update_arriving_during_replay_is_deferred_not_lost(
    paused_realtime_main: None,
) -> None:
    model = InferenceRealtimeModel("openai/gpt-realtime", api_key="key", api_secret="secret")
    session = model.session()
    session._msg_ch.recv_nowait()  # initial session.update
    ws = _BlockingFirstSendWebSocket()

    failover_task = asyncio.create_task(
        session._handle_gateway_failover(ws)  # type: ignore[arg-type]
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
