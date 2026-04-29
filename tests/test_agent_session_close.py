from __future__ import annotations

import asyncio
from typing import Any

import pytest

from livekit.agents import Agent, AgentSession, AgentTask
from livekit.agents.voice import agent_session as agent_session_mod
from livekit.agents.voice.events import CloseEvent, CloseReason

TEST_CLOSE_TIMEOUT = 0.05


class _FakeActivity:
    def __init__(
        self,
        *,
        agent: object | None = None,
        drain_never_returns: bool = False,
        drain_raises: BaseException | None = None,
        current_speech: asyncio.Future[Any] | None = None,
    ) -> None:
        self.agent = agent or object()
        self._audio_recognition = None
        self.current_speech = current_speech
        self.drain_never_returns = drain_never_returns
        self.drain_raises = drain_raises
        self.drain_cancelled = asyncio.Event()
        self.aclose_called = asyncio.Event()

    def interrupt(self, *, force: bool = False) -> asyncio.Future[None]:
        fut: asyncio.Future[None] = asyncio.Future()
        fut.set_result(None)
        return fut

    async def drain(self) -> None:
        if self.drain_raises:
            raise self.drain_raises

        if not self.drain_never_returns:
            return

        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.drain_cancelled.set()
            raise

    async def aclose(self) -> None:
        self.aclose_called.set()


class _HangingCloser:
    def __init__(self) -> None:
        self.cancelled = asyncio.Event()

    async def aclose(self, *args: object, **kwargs: object) -> None:
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class _RaisingCloser:
    async def aclose(self) -> None:
        raise RuntimeError("close failed")


class _FailingAgentTask(AgentTask[None]):
    def __init__(self, old_agent: Agent) -> None:
        super().__init__(instructions="task")
        self._old_agent = old_agent

    def cancel(self) -> None:
        return None

    async def _wait_for_inactive(self) -> None:
        raise RuntimeError("inactive failed")


@pytest.fixture(autouse=True)
def _short_close_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_session_mod, "_AGENT_SESSION_CLOSE_TIMEOUT", TEST_CLOSE_TIMEOUT)


def _started_session() -> AgentSession:
    session = AgentSession(aec_warmup_duration=None)
    session._started = True
    return session


async def _close_session(session: AgentSession) -> list[CloseEvent]:
    close_events: list[CloseEvent] = []
    session.on("close", close_events.append)

    await asyncio.wait_for(session.aclose(), timeout=1.0)

    assert len(close_events) == 1
    assert close_events[0].reason == CloseReason.USER_INITIATED
    return close_events


async def test_aclose_emits_close_when_activity_drain_hangs() -> None:
    session = _started_session()
    activity = _FakeActivity(drain_never_returns=True)
    session._activity = activity

    await _close_session(session)

    await asyncio.wait_for(activity.drain_cancelled.wait(), timeout=1.0)
    assert session._activity is None


async def test_aclose_attempts_late_cleanup_after_drain_uses_deadline() -> None:
    session = _started_session()
    activity = _FakeActivity(drain_never_returns=True)
    session._activity = activity

    forward_cancelled = asyncio.Event()

    async def forward_audio() -> None:
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            forward_cancelled.set()
            raise

    session._forward_audio_atask = asyncio.create_task(forward_audio())

    await _close_session(session)

    await asyncio.wait_for(forward_cancelled.wait(), timeout=1.0)
    assert session._forward_audio_atask is None


async def test_aclose_emits_close_when_current_speech_hangs() -> None:
    session = _started_session()
    current_speech: asyncio.Future[Any] = asyncio.Future()
    activity = _FakeActivity(current_speech=current_speech)
    session._activity = activity

    await _close_session(session)

    assert not current_speech.cancelled()


async def test_aclose_emits_close_when_cleanup_step_raises() -> None:
    session = _started_session()
    session._recorder_io = _RaisingCloser()  # type: ignore[assignment]

    await _close_session(session)

    assert session._recorder_io is None


async def test_aclose_closes_agent_task_activity_and_parent_when_unwind_fails() -> None:
    session = _started_session()
    parent_activity = _FakeActivity()
    old_agent = Agent(instructions="old agent")
    old_agent._activity = parent_activity
    task_agent = _FailingAgentTask(old_agent)
    child_activity = _FakeActivity(agent=task_agent)
    session._activity = child_activity

    await _close_session(session)

    assert child_activity.aclose_called.is_set()
    assert parent_activity.aclose_called.is_set()


async def test_aclose_is_bounded_when_room_io_close_hangs() -> None:
    session = _started_session()
    room_io = _HangingCloser()
    session._room_io = room_io  # type: ignore[assignment]

    close_seen = asyncio.Event()
    session.on("close", lambda _: close_seen.set())

    close_task = asyncio.create_task(session.aclose())
    await asyncio.wait_for(close_seen.wait(), timeout=1.0)
    await asyncio.wait_for(close_task, timeout=1.0)

    assert room_io.cancelled.is_set()
    assert session._room_io is None


async def test_timed_out_close_step_is_cancelled() -> None:
    session = _started_session()
    closer = _HangingCloser()
    session._session_host = closer  # type: ignore[assignment]

    await _close_session(session)

    await asyncio.wait_for(closer.cancelled.wait(), timeout=1.0)
    assert session._session_host is None
