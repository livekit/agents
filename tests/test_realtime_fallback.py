from __future__ import annotations

import asyncio
from typing import cast

import pytest

from livekit import rtc
from livekit.agents.llm import (
    ChatContext,
    GenerationCreatedEvent,
    RealtimeError,
    RealtimeModelFallbackAdapter,
)
from livekit.agents.llm.realtime import _UserMessageSyncResult, _UserMessageSyncStatus

from .fake_realtime import FakeRealtimeModel, FakeRealtimeSession, fake_capabilities

pytestmark = pytest.mark.unit


def _audio_frame() -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=b"\x00\x00", sample_rate=24000, num_channels=1, samples_per_channel=1
    )


class _StubAgent:
    def __init__(self, chat_ctx: ChatContext) -> None:
        self.chat_ctx = chat_ctx


class _StubAgentSession:
    """Minimal stand-in for AgentSession that records orchestrated interrupt/generate_reply."""

    def __init__(self, agent_state: str = "listening", chat_ctx: ChatContext | None = None) -> None:
        self.agent_state = agent_state
        self.interrupt_calls = 0
        self.interrupt_entered = asyncio.Event()
        self.generate_reply_calls = 0
        self.current_agent = _StubAgent(chat_ctx if chat_ctx is not None else ChatContext.empty())

    def interrupt(self, *, force: bool = False) -> asyncio.Future[None]:
        self.interrupt_calls += 1
        self.interrupt_entered.set()
        fut: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        fut.set_result(None)
        return fut

    def generate_reply(self, **kwargs: object) -> object:
        self.generate_reply_calls += 1
        return object()


class _DependentInterruptAgentSession(_StubAgentSession):
    """Models an interrupt whose speech handle cannot finish until text sync advances."""

    def __init__(self, *, sync_advanced: asyncio.Event, chat_ctx: ChatContext) -> None:
        super().__init__(agent_state="thinking", chat_ctx=chat_ctx)
        self._sync_advanced = sync_advanced
        self.interrupt_entered = asyncio.Event()

    async def interrupt(self, *, force: bool = False) -> None:
        del force
        self.interrupt_calls += 1
        self.interrupt_entered.set()
        await self._sync_advanced.wait()


class _NamedRealtimeModel(FakeRealtimeModel):
    """FakeRealtimeModel with a configurable model/provider so tests can tell instances apart."""

    def __init__(self, *, model: str, provider: str) -> None:
        super().__init__()
        self._model_name = model
        self._provider_name = provider

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return self._provider_name


class _FailFirstReplacementModel(FakeRealtimeModel):
    """Fail the first replacement session, then recover if retried."""

    def session(self, *, turn_detection_disabled: bool = False) -> FakeRealtimeSession:
        session = super().session(turn_detection_disabled=turn_detection_disabled)
        if len(self.created_sessions) == 2:
            session.update_error = RuntimeError("first replacement cannot start")
        return session


class _SyncRealtimeSession(FakeRealtimeSession):
    def __init__(self, model: FakeRealtimeModel, *, sync_status: _UserMessageSyncStatus) -> None:
        super().__init__(model)
        self._sync_status = sync_status

    async def _sync_user_message(
        self, chat_ctx: ChatContext, message_id: str
    ) -> _UserMessageSyncResult:
        del chat_ctx, message_id
        error = RuntimeError("child synchronization uncertain")
        return _UserMessageSyncResult(self._sync_status, error)


class _SyncRealtimeModel(FakeRealtimeModel):
    def __init__(self, *, sync_status: _UserMessageSyncStatus) -> None:
        super().__init__()
        self._sync_status = sync_status

    def session(self, *, turn_detection_disabled: bool = False) -> FakeRealtimeSession:
        session = _SyncRealtimeSession(self, sync_status=self._sync_status)
        session.turn_detection_disabled = turn_detection_disabled
        self.created_sessions.append(session)
        return session


class _ControlledSyncRealtimeSession(FakeRealtimeSession):
    def __init__(self, model: FakeRealtimeModel, *, sync_status: _UserMessageSyncStatus) -> None:
        super().__init__(model)
        self._sync_status = sync_status
        self.sync_entered = asyncio.Event()
        self.release_sync = asyncio.Event()

    async def _sync_user_message(
        self, chat_ctx: ChatContext, message_id: str
    ) -> _UserMessageSyncResult:
        del chat_ctx, message_id
        self.sync_entered.set()
        await self.release_sync.wait()
        return _UserMessageSyncResult(
            self._sync_status,
            RuntimeError("retiring child synchronization result"),
        )


class _ControlledSyncRealtimeModel(FakeRealtimeModel):
    def __init__(self, *, sync_status: _UserMessageSyncStatus) -> None:
        super().__init__()
        self._sync_status = sync_status

    def session(self, *, turn_detection_disabled: bool = False) -> FakeRealtimeSession:
        session = _ControlledSyncRealtimeSession(self, sync_status=self._sync_status)
        session.turn_detection_disabled = turn_detection_disabled
        self.created_sessions.append(session)
        return session


async def test_reports_active_model_and_provider() -> None:
    primary = _NamedRealtimeModel(model="primary-model", provider="primary")
    backup = _NamedRealtimeModel(model="backup-model", provider="backup")
    adapter = RealtimeModelFallbackAdapter([primary, backup])

    # before any swap, the primary is reported
    assert adapter.metrics_metadata == {
        "model_name": "primary-model",
        "model_provider": "primary",
    }

    session = adapter.session()
    primary.active_session.emit_error(recoverable=False)
    await session._swap_task

    # the backup now serves the session, so metrics must be labeled with it
    assert adapter.metrics_metadata == {
        "model_name": "backup-model",
        "model_provider": "backup",
    }
    # the adapter keeps its own stable identity for spans, logs, and error events
    assert adapter.model == "RealtimeModelFallbackAdapter"


async def test_new_session_resets_active_model_to_primary() -> None:
    primary = _NamedRealtimeModel(model="primary-model", provider="primary")
    backup = _NamedRealtimeModel(model="backup-model", provider="backup")
    adapter = RealtimeModelFallbackAdapter([primary, backup])

    session = adapter.session()
    primary.active_session.emit_error(recoverable=False)
    await session._swap_task
    assert adapter.metrics_metadata["model_name"] == "backup-model"

    # a fresh session (e.g. a new agent activity) always starts on the primary,
    # so the label must follow it instead of sticking to the old failover target
    adapter.session()

    assert adapter.metrics_metadata == {
        "model_name": "primary-model",
        "model_provider": "primary",
    }


def test_requires_at_least_one_model() -> None:
    with pytest.raises(ValueError):
        RealtimeModelFallbackAdapter([])


def test_ands_soft_capabilities_across_models() -> None:
    primary = FakeRealtimeModel(capabilities=fake_capabilities(mutable_chat_context=True))
    backup = FakeRealtimeModel(capabilities=fake_capabilities(mutable_chat_context=False))

    adapter = RealtimeModelFallbackAdapter([primary, backup])

    # soft flags are conservatively ANDed: if any model can't mutate chat ctx, the adapter can't
    assert adapter.capabilities.mutable_chat_context is False
    # a flag both models support stays True
    assert adapter.capabilities.message_truncation is True


def test_raises_on_mismatched_hard_capabilities() -> None:
    primary = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=True))
    backup = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))

    with pytest.raises(ValueError):
        RealtimeModelFallbackAdapter([primary, backup])


def test_proxies_calls_to_active_child() -> None:
    primary = FakeRealtimeModel()
    backup = FakeRealtimeModel()
    session = RealtimeModelFallbackAdapter([primary, backup]).session()

    session.interrupt()
    session.commit_audio()
    session.generate_reply()

    child = primary.active_session
    assert child.interrupted is True
    assert child.committed is True
    assert child.generate_reply_calls == 1
    # the backup model has not been started
    assert backup.created_sessions == []


def test_forwards_child_events_to_wrapper_subscribers() -> None:
    primary = FakeRealtimeModel()
    session = RealtimeModelFallbackAdapter([primary]).session()

    received: list[object] = []
    session.on("generation_created", lambda ev: received.append(ev))

    primary.active_session.emit("generation_created", "sentinel")

    assert received == ["sentinel"]


def test_proxies_say_to_active_child() -> None:
    primary = FakeRealtimeModel()
    session = RealtimeModelFallbackAdapter([primary]).session()

    session.say("hello there")

    assert primary.active_session.say_calls == ["hello there"]


def test_proxies_start_user_activity_to_active_child() -> None:
    primary = FakeRealtimeModel()
    session = RealtimeModelFallbackAdapter([primary]).session()

    session.start_user_activity()

    assert primary.active_session.user_activity_started is True


async def test_restart_session_creates_fresh_child_same_model() -> None:
    primary = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary])
    session = adapter.session()
    old_child = primary.active_session

    await adapter.restart_session()

    assert old_child.closed is True
    assert len(primary.created_sessions) == 2
    assert session._active is primary.created_sessions[1]
    assert session._active.closed is False


async def test_switch_session_moves_to_next_model() -> None:
    primary = FakeRealtimeModel()
    backup = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary, backup])
    session = adapter.session()
    old = primary.active_session

    await adapter.restart_session(switch_model=True)

    assert old.closed is True
    assert session._active_index == 1
    assert session._active is backup.active_session


@pytest.mark.parametrize(
    "sync_status",
    [
        _UserMessageSyncStatus.ACCEPTED,
        _UserMessageSyncStatus.REJECTED,
        _UserMessageSyncStatus.UNKNOWN,
    ],
)
async def test_sync_user_message_waits_for_swap_and_preserves_child_result(
    sync_status: _UserMessageSyncStatus,
) -> None:
    primary = FakeRealtimeModel()
    backup = _SyncRealtimeModel(sync_status=sync_status)
    adapter = RealtimeModelFallbackAdapter([primary, backup])
    session = adapter.session()
    old = primary.active_session
    release_old = asyncio.Event()
    old.block_aclose = release_old

    old.emit_error(recoverable=False)
    await old.aclose_entered.wait()
    chat_ctx = ChatContext.empty()
    message = chat_ctx.add_message(role="user", content="synchronize exactly once")
    sync_task = asyncio.create_task(session._sync_user_message(chat_ctx, message.id))

    release_old.set()
    await session._swap_task
    result = await sync_task

    assert result.status is sync_status
    assert isinstance(result.error, RuntimeError)


async def test_sync_during_interrupting_swap_replays_on_replacement_without_watchdog() -> None:
    primary = FakeRealtimeModel()
    backup = _SyncRealtimeModel(sync_status=_UserMessageSyncStatus.ACCEPTED)
    adapter = RealtimeModelFallbackAdapter([primary, backup])
    session = adapter.session()
    chat_ctx = ChatContext.empty()
    message = chat_ctx.add_message(role="user", content="survive the provider swap")
    sync_advanced = asyncio.Event()
    agent_session = _DependentInterruptAgentSession(
        sync_advanced=sync_advanced,
        chat_ctx=chat_ctx,
    )
    session._agent_session = cast("object", agent_session)  # type: ignore[assignment]

    swap_task = asyncio.create_task(session.restart(switch_model=True))
    await agent_session.interrupt_entered.wait()

    async def _sync_then_generate() -> tuple[
        _UserMessageSyncResult, asyncio.Future[GenerationCreatedEvent]
    ]:
        result = await session._sync_user_message(chat_ctx, message.id)
        generation_fut = session.generate_reply()
        sync_advanced.set()
        return result, generation_fut

    turn_task = asyncio.create_task(_sync_then_generate())
    try:
        result, generation_fut = await asyncio.wait_for(turn_task, timeout=0.25)
        await asyncio.wait_for(swap_task, timeout=0.25)
    finally:
        sync_advanced.set()
        if not turn_task.done():
            turn_task.cancel()
        await asyncio.gather(turn_task, swap_task, return_exceptions=True)

    assert result.status is _UserMessageSyncStatus.UNKNOWN
    assert primary.active_session.generate_reply_calls == 0

    await asyncio.wait_for(backup.active_session.generate_reply_entered.wait(), timeout=0.25)
    assert backup.active_session.generate_reply_calls == 1
    assert [item.id for item in backup.active_session.chat_ctx.items].count(message.id) == 1

    backup.active_session._reply_futs[0].set_result(cast(GenerationCreatedEvent, object()))
    assert await generation_fut is not None


async def test_retiring_child_rejection_becomes_unknown_after_swap_starts() -> None:
    primary = _ControlledSyncRealtimeModel(sync_status=_UserMessageSyncStatus.REJECTED)
    backup = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary, backup])
    session = adapter.session()
    agent_session = _StubAgentSession()
    session._agent_session = cast("object", agent_session)  # type: ignore[assignment]
    chat_ctx = ChatContext.empty()
    message = chat_ctx.add_message(role="user", content="retiring child result")
    old = cast(_ControlledSyncRealtimeSession, primary.active_session)

    sync_task = asyncio.create_task(session._sync_user_message(chat_ctx, message.id))
    await old.sync_entered.wait()
    swap_task = asyncio.create_task(session.restart(switch_model=True))
    try:
        await asyncio.wait_for(agent_session.interrupt_entered.wait(), timeout=0.25)
        old.release_sync.set()
        result = await asyncio.wait_for(sync_task, timeout=0.25)
        await asyncio.wait_for(swap_task, timeout=0.25)
    finally:
        old.release_sync.set()
        await asyncio.gather(sync_task, swap_task, return_exceptions=True)

    assert result.status is _UserMessageSyncStatus.UNKNOWN
    assert isinstance(result.error, RealtimeError)


async def test_cancelling_cross_swap_generation_does_not_cancel_shared_swap() -> None:
    primary = FakeRealtimeModel()
    backup = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary, backup])
    session = adapter.session()
    old = primary.active_session
    release_old = asyncio.Event()
    old.block_aclose = release_old
    chat_ctx = ChatContext.empty()
    message = chat_ctx.add_message(role="user", content="cancel only this generation")
    sync_advanced = asyncio.Event()
    agent_session = _DependentInterruptAgentSession(
        sync_advanced=sync_advanced,
        chat_ctx=chat_ctx,
    )
    session._agent_session = cast("object", agent_session)  # type: ignore[assignment]

    swap_task = asyncio.create_task(session.restart(switch_model=True))
    await agent_session.interrupt_entered.wait()

    try:
        result = await asyncio.wait_for(
            session._sync_user_message(chat_ctx, message.id), timeout=0.25
        )
        generation_fut = session.generate_reply()
        sync_advanced.set()
        await old.aclose_entered.wait()
        generation_fut.cancel()
        release_old.set()
        await asyncio.wait_for(swap_task, timeout=0.25)
    finally:
        sync_advanced.set()
        release_old.set()
        await asyncio.gather(swap_task, return_exceptions=True)

    assert result.status is _UserMessageSyncStatus.UNKNOWN
    assert generation_fut.cancelled()
    assert backup.active_session.generate_reply_calls == 0


async def test_cross_swap_generation_fails_if_no_replacement_can_start() -> None:
    primary = FakeRealtimeModel()
    backup = FakeRealtimeModel()
    backup.bring_up_error = RuntimeError("cannot start replacement")
    adapter = RealtimeModelFallbackAdapter([primary, backup])
    session = adapter.session()
    chat_ctx = ChatContext.empty()
    message = chat_ctx.add_message(role="user", content="replacement must exist")
    sync_advanced = asyncio.Event()
    primary.bring_up_error = RuntimeError("primary also unavailable")
    agent_session = _DependentInterruptAgentSession(
        sync_advanced=sync_advanced,
        chat_ctx=chat_ctx,
    )
    session._agent_session = cast("object", agent_session)  # type: ignore[assignment]

    swap_task = asyncio.create_task(session.restart(switch_model=True))
    await agent_session.interrupt_entered.wait()

    async def _sync_then_generate() -> tuple[
        _UserMessageSyncResult, asyncio.Future[GenerationCreatedEvent]
    ]:
        result = await session._sync_user_message(chat_ctx, message.id)
        generation_fut = session.generate_reply()
        sync_advanced.set()
        return result, generation_fut

    turn_task = asyncio.create_task(_sync_then_generate())
    try:
        result, generation_fut = await asyncio.wait_for(turn_task, timeout=0.25)
        await asyncio.wait_for(swap_task, timeout=0.25)
    finally:
        sync_advanced.set()
        if not turn_task.done():
            turn_task.cancel()
        await asyncio.gather(turn_task, swap_task, return_exceptions=True)

    assert result.status is _UserMessageSyncStatus.UNKNOWN
    with pytest.raises(RealtimeError, match="failed to replace realtime session"):
        await generation_fut


async def test_sync_waits_through_consecutive_swaps_for_latest_child_result() -> None:
    primary = FakeRealtimeModel()
    backup = FakeRealtimeModel()
    final = _SyncRealtimeModel(sync_status=_UserMessageSyncStatus.REJECTED)
    adapter = RealtimeModelFallbackAdapter([primary, backup, final])
    session = adapter.session()

    primary.active_session.emit_error(recoverable=False)
    await session._swap_task
    assert session._active is backup.active_session

    old = backup.active_session
    release_old = asyncio.Event()
    old.block_aclose = release_old
    old.emit_error(recoverable=False)
    await old.aclose_entered.wait()
    chat_ctx = ChatContext.empty()
    message = chat_ctx.add_message(role="user", content="second replacement")
    sync_task = asyncio.create_task(session._sync_user_message(chat_ctx, message.id))

    release_old.set()
    await session._swap_task
    result = await sync_task

    assert session._active is final.active_session
    assert result.status is _UserMessageSyncStatus.REJECTED
    assert isinstance(result.error, RuntimeError)


async def test_switch_session_wraps_around() -> None:
    primary = FakeRealtimeModel()
    backup = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary, backup])
    session = adapter.session()

    await adapter.restart_session(switch_model=True)  # primary -> backup
    assert session._active_index == 1

    await adapter.restart_session(switch_model=True)  # backup -> wraps to primary
    assert session._active_index == 0
    assert session._active is primary.created_sessions[-1]


async def test_switch_session_single_model_restarts_same() -> None:
    primary = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary])
    session = adapter.session()
    old = primary.active_session

    await adapter.restart_session(switch_model=True)

    # only one model: degrades to a fresh session on the same model
    assert old.closed is True
    assert session._active_index == 0
    assert len(primary.created_sessions) == 2


async def test_restart_replays_chat_ctx_onto_new_child() -> None:
    primary = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary])
    session = adapter.session()
    old_child = primary.active_session

    ctx = ChatContext.empty()
    ctx.add_message(role="user", content="remember me")
    await old_child.update_chat_ctx(ctx)

    await adapter.restart_session()

    # the freshest chat context from the dying child is replayed onto the new child
    assert session._active.chat_ctx is ctx


async def test_restart_preserves_wrapper_subscribers() -> None:
    primary = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary])
    session = adapter.session()
    received: list[object] = []
    session.on("generation_created", lambda ev: received.append(ev))

    await adapter.restart_session()

    # events from the NEW child still reach the original wrapper subscriber, no rebinding
    session._active.emit("generation_created", "after-restart")
    assert received == ["after-restart"]


async def test_restart_preserves_provider_event_subscribers() -> None:
    primary = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary])
    session = adapter.session()
    received: list[object] = []
    session.on("provider_event", lambda ev: received.append(ev))

    primary.active_session.emit("provider_event", "before-restart")
    await adapter.restart_session()
    primary.active_session.emit("provider_event", "after-restart")

    assert received == ["before-restart", "after-restart"]


async def test_provider_event_subscribed_during_restart_skips_old_child() -> None:
    primary = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary])
    session = adapter.session()
    old_child = primary.active_session
    close_gate = asyncio.Event()
    old_child.block_aclose = close_gate

    restart_task = asyncio.create_task(adapter.restart_session())
    await old_child.aclose_entered.wait()

    received: list[object] = []
    session.on("provider_event", lambda ev: received.append(ev))
    old_child.emit("provider_event", "old-child")

    close_gate.set()
    await restart_task
    primary.active_session.emit("provider_event", "new-child")

    assert received == ["new-child"]


async def test_restart_emits_no_error() -> None:
    primary = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary])
    session = adapter.session()
    errors: list[object] = []
    session.on("error", lambda ev: errors.append(ev))

    await adapter.restart_session()

    assert errors == []


async def test_auto_swaps_to_next_model_on_non_recoverable_error() -> None:
    primary = FakeRealtimeModel()
    backup = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary, backup])
    session = adapter.session()
    old = primary.active_session

    old.emit_error(recoverable=False)
    await session._swap_task

    assert old.closed is True
    assert len(backup.created_sessions) == 1
    assert session._active is backup.active_session


async def test_non_recoverable_error_forwarded_as_recoverable_while_fallback_remains() -> None:
    primary = FakeRealtimeModel()
    backup = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary, backup])
    session = adapter.session()
    errors: list = []
    session.on("error", lambda e: errors.append(e))

    primary.active_session.emit_error(recoverable=False)
    await session._swap_task

    # the user is informed, but the error is re-stamped recoverable so the session is not closed
    assert len(errors) == 1
    assert errors[0].recoverable is True


async def test_background_swap_failure_emits_terminal_error() -> None:
    primary = FakeRealtimeModel()
    backup = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary, backup])
    session = adapter.session()
    errors: list = []
    session.on("error", lambda e: errors.append(e))

    class _BrokenAgent:
        @property
        def chat_ctx(self) -> ChatContext:
            raise RuntimeError("failed to snapshot chat context")

    agent_session = _StubAgentSession(agent_state="listening")
    agent_session.current_agent = cast(_StubAgent, _BrokenAgent())
    session._agent_session = agent_session

    primary.active_session.emit_error(recoverable=False)
    assert session._swap_task is not None
    with pytest.raises(RuntimeError, match="failed to snapshot chat context"):
        await session._swap_task
    await asyncio.sleep(0)

    assert [error.recoverable for error in errors] == [True, False]
    assert isinstance(errors[-1].error, RuntimeError)


async def test_recoverable_error_is_forwarded_unchanged_without_swap() -> None:
    primary = FakeRealtimeModel()
    backup = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary, backup])
    session = adapter.session()
    errors: list = []
    session.on("error", lambda e: errors.append(e))

    primary.active_session.emit_error(recoverable=True)

    assert len(errors) == 1
    assert errors[0].recoverable is True
    assert session._swap_task is None
    assert backup.created_sessions == []  # no swap on a recoverable error


async def test_exhausted_models_escalate_as_non_recoverable() -> None:
    primary = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary])  # no backup
    session = adapter.session()
    errors: list = []
    session.on("error", lambda e: errors.append(e))

    primary.active_session.emit_error(recoverable=False)

    # nothing left to fall back to: escalate so AgentSession can close
    assert session._swap_task is None
    assert len(errors) == 1
    assert errors[0].recoverable is False
    assert len(primary.created_sessions) == 1  # not auto-restarted


async def test_escalates_when_all_models_have_failed() -> None:
    primary = FakeRealtimeModel()
    backup = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary, backup])
    session = adapter.session()
    errors: list = []
    session.on("error", lambda e: errors.append(e))

    primary.active_session.emit_error(recoverable=False)
    await session._swap_task
    # now on backup; kill it too
    backup.active_session.emit_error(recoverable=False)

    assert session._swap_task.done()
    assert errors[-1].recoverable is False  # exhausted -> escalate


@pytest.mark.virtual_time
async def test_prefers_primary_again_after_cooldown_expires() -> None:
    primary = FakeRealtimeModel()
    backup = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary, backup], cooldown=5.0)
    session = adapter.session()

    # primary dies -> fall over to backup
    primary.active_session.emit_error(recoverable=False)
    await session._swap_task
    assert session._active_index == 1

    # let primary's cooldown expire
    await asyncio.sleep(6.0)

    # backup dies -> primary is available again and preferred by list order
    backup.active_session.emit_error(recoverable=False)
    await session._swap_task
    assert session._active_index == 0
    assert session._active is primary.created_sessions[-1]


async def test_regenerates_via_agent_session_when_speaking() -> None:
    primary = FakeRealtimeModel()
    backup = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary, backup])  # regenerate_on_swap default True
    session = adapter.session()
    session._agent_session = _StubAgentSession(agent_state="speaking")

    primary.active_session.emit_error(recoverable=False)
    await session._swap_task

    # regeneration goes through the AgentSession (orchestrated), not the raw child session
    assert session._agent_session.interrupt_calls == 1
    assert session._agent_session.generate_reply_calls == 1
    assert backup.active_session.generate_reply_calls == 0


async def test_regenerates_when_agent_thinking() -> None:
    primary = FakeRealtimeModel()
    backup = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary, backup])
    session = adapter.session()
    session._agent_session = _StubAgentSession(agent_state="thinking")

    # "thinking" is a reply in progress (generating, pre-audio), so it regenerates too
    primary.active_session.emit_error(recoverable=False)
    await session._swap_task

    assert session._agent_session.generate_reply_calls == 1


async def test_no_regenerate_when_agent_not_speaking() -> None:
    primary = FakeRealtimeModel()
    backup = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary, backup])
    session = adapter.session()
    session._agent_session = _StubAgentSession(agent_state="listening")

    primary.active_session.emit_error(recoverable=False)
    await session._swap_task

    assert session._agent_session.generate_reply_calls == 0


async def test_no_regenerate_when_disabled() -> None:
    primary = FakeRealtimeModel()
    backup = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary, backup], regenerate_on_swap=False)
    session = adapter.session()
    session._agent_session = _StubAgentSession(agent_state="speaking")

    primary.active_session.emit_error(recoverable=False)
    await session._swap_task

    # stale playout is still interrupted; only the re-issue of generate_reply is suppressed
    assert session._agent_session.interrupt_calls == 1
    assert session._agent_session.generate_reply_calls == 0


async def test_drops_audio_during_swap() -> None:
    primary = FakeRealtimeModel()
    backup = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary, backup])
    session = adapter.session()
    old = primary.active_session

    gate = asyncio.Event()
    old.block_aclose = gate

    old.emit_error(recoverable=False)
    # wait until the swap is mid-flight (inside the dying child's aclose)
    await old.aclose_entered.wait()

    frame = _audio_frame()
    session.push_audio(frame)
    # audio arriving mid-swap is dropped, not sent to the dying child
    assert frame not in old.pushed_audio

    gate.set()
    await session._swap_task

    # ...and not replayed into the new child (replaying would add permanent input latency)
    assert frame not in backup.active_session.pushed_audio


async def test_swap_cascades_past_a_model_that_cannot_start() -> None:
    primary = FakeRealtimeModel()
    backup1 = FakeRealtimeModel()
    backup2 = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary, backup1, backup2])
    session = adapter.session()
    errors: list = []
    session.on("error", lambda e: errors.append(e))

    # the first fallback fails to bring up; the swap should skip it and land on the next
    backup1.bring_up_error = RuntimeError("cannot start")
    primary.active_session.emit_error(recoverable=False)
    await session._swap_task

    assert session._active_index == 2
    assert session._active is backup2.active_session
    # only the recoverable hand-off was surfaced; nothing session-ending
    assert all(e.recoverable for e in errors)


async def test_zero_cooldown_attempts_each_model_only_once_per_swap() -> None:
    primary = _FailFirstReplacementModel()
    failing_backup = FakeRealtimeModel()
    final_backup = FakeRealtimeModel()
    failing_backup.bring_up_error = RuntimeError("backup cannot start")
    adapter = RealtimeModelFallbackAdapter(
        [primary, failing_backup, final_backup],
        cooldown=0,
    )
    session = adapter.session()

    await session.restart(switch_model=True)

    assert session._active_index == 2
    assert session._active is final_backup.active_session
    assert len(primary.created_sessions) == 2
    assert len(failing_backup.created_sessions) == 1
    assert len(final_backup.created_sessions) == 1


async def test_swap_escalates_when_no_model_can_start() -> None:
    primary = FakeRealtimeModel()
    backup = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary, backup])
    session = adapter.session()
    errors: list = []
    session.on("error", lambda e: errors.append(e))

    # the only fallback also fails to bring up: escalate a non-recoverable error
    backup.bring_up_error = RuntimeError("cannot start")
    primary.active_session.emit_error(recoverable=False)
    await session._swap_task

    assert any(not e.recoverable for e in errors)


async def test_emits_session_reconnected_on_swap() -> None:
    primary = FakeRealtimeModel()
    backup = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary, backup])
    session = adapter.session()

    events: list = []
    session.on("session_reconnected", lambda ev: events.append(ev))

    await adapter.restart_session(switch_model=True)

    assert len(events) == 1


async def test_swap_replays_child_current_ctx_not_stale_pushed_ctx() -> None:
    primary = FakeRealtimeModel()
    backup = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary, backup])
    session = adapter.session()
    old = primary.active_session

    # an update pushed earlier (not during a swap)
    ctx1 = ChatContext.empty()
    ctx1.add_message(role="user", content="old turn")
    await session.update_chat_ctx(ctx1)

    # the child then accumulates newer state (e.g. a server-side transcript)
    ctx2 = ChatContext.empty()
    ctx2.add_message(role="user", content="old turn")
    ctx2.add_message(role="assistant", content="newer transcript")
    await old.update_chat_ctx(ctx2)

    await adapter.restart_session()

    # replay must use the child's current context, not the stale earlier push
    assert session._active.chat_ctx is ctx2


async def test_swap_replays_agent_chat_ctx() -> None:
    primary = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary])
    session = adapter.session()

    agent_ctx = ChatContext.empty()
    agent_ctx.add_message(role="user", content="what the user heard")
    session._agent_session = _StubAgentSession(agent_state="listening", chat_ctx=agent_ctx)

    await adapter.restart_session()

    # the agent chat context (user-heard version) is replayed, not the child's own context
    assert session._active.chat_ctx is agent_ctx


async def test_emits_availability_changed_on_failure() -> None:
    primary = FakeRealtimeModel()
    backup = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary, backup])
    session = adapter.session()

    events: list = []
    adapter.on("realtime_availability_changed", lambda e: events.append(e))

    primary.active_session.emit_error(recoverable=False)
    await session._swap_task

    assert any(e.realtime_model is primary and e.available is False for e in events)


@pytest.mark.virtual_time
def test_session_exposes_model_capabilities() -> None:
    model = FakeRealtimeModel()
    session = model.session()

    assert session.capabilities is model.capabilities


def test_allows_mismatched_auto_tool_reply_generation() -> None:
    primary = FakeRealtimeModel(capabilities=fake_capabilities(auto_tool_reply_generation=True))
    backup = FakeRealtimeModel(capabilities=fake_capabilities(auto_tool_reply_generation=False))

    # no longer a hard capability: the active child's value is read per-turn from the session
    adapter = RealtimeModelFallbackAdapter([primary, backup])
    assert adapter.capabilities.auto_tool_reply_generation is False  # conservative AND on the model


def test_wrapper_capabilities_track_active_child() -> None:
    primary = FakeRealtimeModel(capabilities=fake_capabilities(auto_tool_reply_generation=True))
    backup = FakeRealtimeModel(capabilities=fake_capabilities(auto_tool_reply_generation=False))
    adapter = RealtimeModelFallbackAdapter([primary, backup])
    session = adapter.session()

    assert session.capabilities.auto_tool_reply_generation is True  # primary active


async def test_wrapper_capabilities_follow_swap() -> None:
    primary = FakeRealtimeModel(capabilities=fake_capabilities(auto_tool_reply_generation=True))
    backup = FakeRealtimeModel(capabilities=fake_capabilities(auto_tool_reply_generation=False))
    adapter = RealtimeModelFallbackAdapter([primary, backup])
    session = adapter.session()

    primary.active_session.emit_error(recoverable=False)
    await session._swap_task

    assert session.capabilities.auto_tool_reply_generation is False  # backup now active


async def test_emits_availability_changed_on_recovery() -> None:
    primary = FakeRealtimeModel()
    backup = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([primary, backup], cooldown=5.0)
    session = adapter.session()

    events: list = []
    adapter.on("realtime_availability_changed", lambda e: events.append(e))

    primary.active_session.emit_error(recoverable=False)
    await session._swap_task
    await asyncio.sleep(6.0)
    backup.active_session.emit_error(recoverable=False)
    await session._swap_task

    assert any(e.realtime_model is primary and e.available is True for e in events)
