from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from livekit.agents.llm import FunctionCall
from livekit.agents.voice.events import AgentStateChangedEvent, RunContext, UserStateChangedEvent
from livekit.agents.voice.filler_scheduler import _FillerScheduler

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


class _FakeSpeechHandle:
    def __init__(self) -> None:
        self.num_steps = 1
        self.allow_interruptions = True
        self._interrupt_fut: asyncio.Future[None] = asyncio.Future()
        self._done_callbacks: list[Any] = []
        self._done = False

    @property
    def interrupted(self) -> bool:
        return self._interrupt_fut.done()

    def mark_interrupted(self) -> None:
        if not self._interrupt_fut.done():
            self._interrupt_fut.set_result(None)
        self.mark_done()

    def mark_done(self) -> None:
        if self._done:
            return
        self._done = True
        for cb in list(self._done_callbacks):
            cb(self)

    def add_done_callback(self, callback: Any) -> None:
        if self._done:
            callback(self)
        else:
            self._done_callbacks.append(callback)

    def remove_done_callback(self, callback: Any) -> None:
        if callback in self._done_callbacks:
            self._done_callbacks.remove(callback)

    async def wait_if_not_interrupted(self, aw: list[asyncio.Future[Any]]) -> None:
        gather_fut = asyncio.gather(*[asyncio.shield(f) for f in aw], return_exceptions=True)
        done, pending = await asyncio.wait(
            {gather_fut, self._interrupt_fut}, return_when=asyncio.FIRST_COMPLETED
        )
        if gather_fut in pending:
            gather_fut.cancel()
            try:
                await gather_fut
            except (asyncio.CancelledError, Exception):
                pass


class _FakeSession:
    """An AgentSession stand-in, idle until a test emits a state change."""

    def __init__(self) -> None:
        self.say_calls: list[dict[str, Any]] = []
        self._listeners: dict[str, list[Any]] = {}
        self._idle_event = asyncio.Event()
        self._idle_event.set()  # idle by default
        self._global_run_state = None  # read by RunContext

    def on(self, event: str, callback: Any) -> Any:
        self._listeners.setdefault(event, []).append(callback)
        return callback

    def emit(self, event: str, ev: Any) -> None:
        for cb in list(self._listeners.get(event, [])):
            cb(ev)

    def off(self, event: str, callback: Any) -> None:
        if callback in self._listeners.get(event, []):
            self._listeners[event].remove(callback)

    async def wait_for_idle(self) -> Any:
        await self._idle_event.wait()
        return None

    def set_idle(self, idle: bool) -> None:
        if idle:
            self._idle_event.set()
        else:
            self._idle_event.clear()

    def emit_agent_state(self, new_state: str, old_state: str = "idle") -> None:
        ev = AgentStateChangedEvent(old_state=old_state, new_state=new_state)  # type: ignore[arg-type]
        for cb in list(self._listeners.get("agent_state_changed", [])):
            cb(ev)

    def emit_user_state(self, new_state: str, old_state: str = "listening") -> None:
        ev = UserStateChangedEvent(old_state=old_state, new_state=new_state)  # type: ignore[arg-type]
        for cb in list(self._listeners.get("user_state_changed", [])):
            cb(ev)

    def say(self, text: Any, **kwargs: Any) -> Any:
        self.say_calls.append({"text": text, **kwargs})
        handle = MagicMock()

        async def _wait_for_playout() -> None:
            return None

        handle.wait_for_playout = _wait_for_playout
        return handle


def _scheduler(
    session: _FakeSession,
    handle: _FakeSpeechHandle,
    source: Any,
    *,
    delay: float,
    interval: float | None = None,
    max_steps: int | None = None,
) -> _FillerScheduler:
    return _FillerScheduler(
        session=session,  # type: ignore[arg-type]
        speech_handle=handle,  # type: ignore[arg-type]
        source=source,
        delay=delay,
        interval=interval,
        max_steps=max_steps,
    )


def _run_context(session: _FakeSession, handle: _FakeSpeechHandle) -> RunContext:
    return RunContext(
        session=session,  # type: ignore[arg-type]
        speech_handle=handle,  # type: ignore[arg-type]
        function_call=FunctionCall(call_id="c", name="t", arguments="{}"),
    )


def _texts(session: _FakeSession) -> list[Any]:
    return [c["text"] for c in session.say_calls]


async def test_scheduler_does_not_fire_before_dwell_elapses() -> None:
    session = _FakeSession()
    scheduler = _scheduler(session, _FakeSpeechHandle(), "let me check", delay=1.0)
    await asyncio.sleep(0.05)
    await scheduler.aclose()
    assert session.say_calls == []


async def test_scheduler_close_and_reset_dwell_are_safe_any_time() -> None:
    scheduler = _scheduler(_FakeSession(), _FakeSpeechHandle(), "x", delay=10.0)
    scheduler.reset_dwell()
    await scheduler.aclose()
    await scheduler.aclose()
    scheduler.reset_dwell()


async def test_scheduler_fires_once_after_idle_dwell_without_interval() -> None:
    session = _FakeSession()
    scheduler = _scheduler(session, _FakeSpeechHandle(), "let me check", delay=0.05)
    await asyncio.sleep(0.3)
    await scheduler.aclose()
    assert _texts(session) == ["let me check"]


@pytest.mark.parametrize("trigger", ["agent_state", "user_state", "reset_dwell"])
async def test_scheduler_dwell_restarts(trigger: str) -> None:
    session = _FakeSession()
    scheduler = _scheduler(session, _FakeSpeechHandle(), "ping", delay=0.1)

    await asyncio.sleep(0.05)
    if trigger == "agent_state":
        session.emit_agent_state("speaking")
    elif trigger == "user_state":
        session.emit_user_state("speaking")
    else:
        scheduler.reset_dwell()
    await asyncio.sleep(0.08)
    assert session.say_calls == []  # without the restart it would have fired at 0.1
    await asyncio.sleep(0.05)
    await scheduler.aclose()
    assert _texts(session) == ["ping"]


async def test_scheduler_invokes_callable_source_lazily() -> None:
    session = _FakeSession()
    invocations: list[int] = []

    def factory(step: int) -> Any:
        invocations.append(step)
        return session.say("from callable")

    scheduler = _scheduler(session, _FakeSpeechHandle(), factory, delay=0.05)
    assert invocations == []
    await asyncio.sleep(0.15)
    await scheduler.aclose()
    assert invocations == [0]
    assert _texts(session) == ["from callable"]


async def test_scheduler_fires_repeatedly_with_interval() -> None:
    session = _FakeSession()
    scheduler = _scheduler(session, _FakeSpeechHandle(), "tick", delay=0.05, interval=0.05)
    await asyncio.sleep(0.35)
    await scheduler.aclose()
    assert len(session.say_calls) >= 2
    assert set(_texts(session)) == {"tick"}


async def test_scheduler_does_not_fire_after_speech_interrupted() -> None:
    session = _FakeSession()
    handle = _FakeSpeechHandle()
    scheduler = _scheduler(session, handle, "x", delay=0.02, interval=0.02)

    await asyncio.sleep(0.05)
    fires_when_interrupted = len(session.say_calls)
    handle.mark_interrupted()
    await asyncio.sleep(0.2)
    await scheduler.aclose()
    # at most the one fire already in flight at interrupt time
    assert len(session.say_calls) <= fires_when_interrupted + 1


async def test_run_context_with_filler_fires_inside_the_block() -> None:
    session = _FakeSession()
    ctx = _run_context(session, _FakeSpeechHandle())
    async with ctx.with_filler("hello", delay=0.02, interval=None):
        await asyncio.sleep(0.1)
    assert _texts(session) == ["hello"]


async def test_run_context_with_filler_cancels_on_exit() -> None:
    session = _FakeSession()
    ctx = _run_context(session, _FakeSpeechHandle())
    async with ctx.with_filler("nope", delay=1.0, interval=None):
        await asyncio.sleep(0.01)
    assert session.say_calls == []


async def test_ctx_update_resets_pending_filler_dwell() -> None:
    session = _FakeSession()
    ctx = _run_context(session, _FakeSpeechHandle())
    async with ctx.with_filler("ping", delay=0.1, interval=None):
        await asyncio.sleep(0.05)
        await ctx.update("first")
        await ctx.update("second")
        await asyncio.sleep(0.08)
        fires_mid = len(session.say_calls)
        await asyncio.sleep(0.05)
        fires_late = len(session.say_calls)
    assert fires_mid == 0
    assert fires_late == 1


async def test_scheduler_reports_a_say_that_raises(caplog: pytest.LogCaptureFixture) -> None:
    """A failing say() is logged, since the scheduler's wait collects exceptions silently."""
    session = _FakeSession()

    def _failing_say(text: Any, **kwargs: Any) -> Any:
        raise RuntimeError("no TTS model and the RealtimeSession does not support say()")

    session.say = _failing_say  # type: ignore[assignment]

    with caplog.at_level(logging.ERROR, logger="livekit.agents"):
        scheduler = _scheduler(
            session, _FakeSpeechHandle(), "let me check", delay=0.05, interval=10, max_steps=3
        )
        await asyncio.sleep(0.15)
        await scheduler.aclose()

    assert [r.message for r in caplog.records if r.levelno >= logging.ERROR] == [
        "filler stopped on an error, no further filler will play for this tool call"
    ]
    assert any("does not support say()" in str(r.exc_info[1]) for r in caplog.records if r.exc_info)
