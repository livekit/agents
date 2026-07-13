from __future__ import annotations

import asyncio

import pytest

from livekit import rtc
from livekit.agents.llm import ChatContext, RealtimeModelFallbackAdapter

from .fake_realtime import FakeRealtimeModel, fake_capabilities

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
        self.generate_reply_calls = 0
        self.current_agent = _StubAgent(chat_ctx if chat_ctx is not None else ChatContext.empty())

    def interrupt(self, *, force: bool = False) -> asyncio.Future[None]:
        self.interrupt_calls += 1
        fut: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        fut.set_result(None)
        return fut

    def generate_reply(self, **kwargs: object) -> object:
        self.generate_reply_calls += 1
        return object()


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
