from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from livekit.agents.voice.events import AgentStateChangedEvent, UserStateChangedEvent
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
    """Minimal AgentSession stand-in. Idle by default; tests emit state events
    via ``emit_agent_state`` / ``emit_user_state``."""

    def __init__(self) -> None:
        self.say_calls: list[dict[str, Any]] = []
        self._listeners: dict[str, list[Any]] = {}
        self._idle_event = asyncio.Event()
        self._idle_event.set()  # idle by default

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


@pytest.mark.asyncio
async def test_scheduler_can_start_and_stop_without_firing() -> None:
    """A scheduler that is started and immediately stopped before delay elapses
    should fire nothing and leave no leaked tasks."""
    session = _FakeSession()
    handle = _FakeSpeechHandle()
    scheduler = _FillerScheduler(
        session=session,  # type: ignore[arg-type]
        speech_handle=handle,  # type: ignore[arg-type]
        source="should not fire",
        delay=10.0,
        interval=None,
        max_steps=None,
    )

    await asyncio.sleep(0.01)
    await scheduler.aclose()
    assert session.say_calls == []


@pytest.mark.asyncio
async def test_scheduler_stop_is_idempotent() -> None:
    """Calling stop() twice is safe."""
    session = _FakeSession()
    handle = _FakeSpeechHandle()
    scheduler = _FillerScheduler(
        session=session,  # type: ignore[arg-type]
        speech_handle=handle,  # type: ignore[arg-type]
        source="x",
        delay=10.0,
        interval=None,
        max_steps=None,
    )

    await scheduler.aclose()
    await scheduler.aclose()  # no-op


@pytest.mark.asyncio
async def test_scheduler_fires_after_idle_dwell() -> None:
    """When session is idle, scheduler fires after delay seconds and calls session.say."""
    session = _FakeSession()
    handle = _FakeSpeechHandle()
    scheduler = _FillerScheduler(
        session=session,  # type: ignore[arg-type]
        speech_handle=handle,  # type: ignore[arg-type]
        source="let me check",
        delay=0.05,
        interval=None,
        max_steps=None,
    )

    await asyncio.sleep(0.15)
    await scheduler.aclose()
    assert [c["text"] for c in session.say_calls] == ["let me check"]


@pytest.mark.asyncio
async def test_scheduler_does_not_fire_before_dwell_elapses() -> None:
    """If we stop before delay elapses, no fire happens."""
    session = _FakeSession()
    handle = _FakeSpeechHandle()
    scheduler = _FillerScheduler(
        session=session,  # type: ignore[arg-type]
        speech_handle=handle,  # type: ignore[arg-type]
        source="let me check",
        delay=1.0,
        interval=None,
        max_steps=None,
    )

    await asyncio.sleep(0.05)
    await scheduler.aclose()
    assert session.say_calls == []


@pytest.mark.asyncio
async def test_scheduler_dwell_resets_on_agent_state_change() -> None:
    """If agent becomes non-idle mid-dwell, the dwell restarts from zero."""
    session = _FakeSession()
    handle = _FakeSpeechHandle()
    scheduler = _FillerScheduler(
        session=session,  # type: ignore[arg-type]
        speech_handle=handle,  # type: ignore[arg-type]
        source="ping",
        delay=0.1,
        interval=None,
        max_steps=None,
    )

    # halfway through the dwell window, emit "non-idle" → dwell must restart
    await asyncio.sleep(0.05)
    session.emit_agent_state("speaking")
    # at this point we've used ~0.05s; if dwell didn't reset, fire would land
    # around t=0.10. With reset, fire shouldn't land until at least t=0.05+0.10=0.15
    await asyncio.sleep(0.08)
    fire_count_before = len(session.say_calls)
    await asyncio.sleep(0.05)  # cross past 0.15
    fire_count_after = len(session.say_calls)
    await scheduler.aclose()
    assert fire_count_before == 0  # didn't fire before reset+delay
    assert fire_count_after == 1


@pytest.mark.asyncio
async def test_scheduler_dwell_resets_on_user_speaking() -> None:
    """User speaking also resets the dwell."""
    session = _FakeSession()
    handle = _FakeSpeechHandle()
    scheduler = _FillerScheduler(
        session=session,  # type: ignore[arg-type]
        speech_handle=handle,  # type: ignore[arg-type]
        source="ping",
        delay=0.1,
        interval=None,
        max_steps=None,
    )

    await asyncio.sleep(0.05)
    session.emit_user_state("speaking")
    await asyncio.sleep(0.08)
    assert session.say_calls == []
    await asyncio.sleep(0.05)
    await scheduler.aclose()
    assert [c["text"] for c in session.say_calls] == ["ping"]


@pytest.mark.asyncio
async def test_scheduler_invokes_callable_source_lazily() -> None:
    """A Callable source is invoked at fire time and the returned handle awaited."""
    session = _FakeSession()
    handle = _FakeSpeechHandle()

    invocations: list[int] = []

    def factory(step: int) -> Any:
        invocations.append(step)
        return session.say("from callable")

    scheduler = _FillerScheduler(
        session=session,  # type: ignore[arg-type]
        speech_handle=handle,  # type: ignore[arg-type]
        source=factory,
        delay=0.05,
        interval=None,
        max_steps=None,
    )

    # at start, factory should not have run yet
    assert invocations == []
    await asyncio.sleep(0.15)
    await scheduler.aclose()
    assert invocations == [0]  # first fire, step counter starts at 0
    assert [c["text"] for c in session.say_calls] == ["from callable"]


@pytest.mark.asyncio
async def test_scheduler_fires_repeatedly_with_interval() -> None:
    """interval=X means fire, sleep X wall-clock, then restart dwell."""
    session = _FakeSession()
    handle = _FakeSpeechHandle()
    scheduler = _FillerScheduler(
        session=session,  # type: ignore[arg-type]
        speech_handle=handle,  # type: ignore[arg-type]
        source="tick",
        delay=0.05,
        interval=0.05,
        max_steps=None,
    )

    await asyncio.sleep(0.35)  # plenty of time for several fires
    await scheduler.aclose()
    # at least 2 fires; exact count depends on timing but cap at a sane upper bound
    assert 2 <= len(session.say_calls) <= 6
    assert all(c["text"] == "tick" for c in session.say_calls)


@pytest.mark.asyncio
async def test_scheduler_interval_none_fires_once_only() -> None:
    """interval=None means at-most-once."""
    session = _FakeSession()
    handle = _FakeSpeechHandle()
    scheduler = _FillerScheduler(
        session=session,  # type: ignore[arg-type]
        speech_handle=handle,  # type: ignore[arg-type]
        source="once",
        delay=0.02,
        interval=None,
        max_steps=None,
    )

    await asyncio.sleep(0.3)  # would fire ~15 times if it looped
    await scheduler.aclose()
    assert len(session.say_calls) == 1


@pytest.mark.asyncio
async def test_reset_dwell_aborts_current_dwell() -> None:
    """reset_dwell() called mid-dwell aborts the current dwell and restarts."""
    session = _FakeSession()
    handle = _FakeSpeechHandle()
    scheduler = _FillerScheduler(
        session=session,  # type: ignore[arg-type]
        speech_handle=handle,  # type: ignore[arg-type]
        source="x",
        delay=0.1,
        interval=None,
        max_steps=None,
    )

    await asyncio.sleep(0.05)
    scheduler.reset_dwell()
    await asyncio.sleep(0.08)
    fire_before = len(session.say_calls)
    await asyncio.sleep(0.05)
    await scheduler.aclose()
    fire_after = len(session.say_calls)
    assert fire_before == 0  # didn't fire before reset+delay completed
    assert fire_after == 1


@pytest.mark.asyncio
async def test_reset_dwell_is_safe_to_call_anytime() -> None:
    """Calling reset_dwell at any point is safe (it's just an event.set)."""
    session = _FakeSession()
    handle = _FakeSpeechHandle()
    scheduler = _FillerScheduler(
        session=session,  # type: ignore[arg-type]
        speech_handle=handle,  # type: ignore[arg-type]
        source="x",
        delay=10.0,
        interval=None,
        max_steps=None,
    )
    scheduler.reset_dwell()  # must not raise
    await scheduler.aclose()
    scheduler.reset_dwell()  # also safe after stop


@pytest.mark.asyncio
async def test_scheduler_does_not_fire_after_speech_interrupted() -> None:
    """Once speech_handle.interrupted goes True, subsequent ticks must not fire."""
    session = _FakeSession()
    handle = _FakeSpeechHandle()
    scheduler = _FillerScheduler(
        session=session,  # type: ignore[arg-type]
        speech_handle=handle,  # type: ignore[arg-type]
        source="x",
        delay=0.02,
        interval=0.02,
        max_steps=None,
    )

    await asyncio.sleep(0.05)  # at least one fire happens
    fires_when_interrupted = len(session.say_calls)
    handle.mark_interrupted()
    await asyncio.sleep(0.2)  # would have fired many more times if uninterrupted
    await scheduler.aclose()
    # at most one additional fire that was already in flight at interrupt time
    assert len(session.say_calls) <= fires_when_interrupted + 1


@pytest.mark.asyncio
async def test_run_context_with_filler_yields_and_fires() -> None:
    """async with ctx.with_filler(...) starts a scheduler and tears it down on exit."""
    from livekit.agents.llm import FunctionCall
    from livekit.agents.voice.events import RunContext

    session = _FakeSession()
    handle = _FakeSpeechHandle()
    ctx: RunContext = RunContext(
        session=session,  # type: ignore[arg-type]
        speech_handle=handle,  # type: ignore[arg-type]
        function_call=FunctionCall(call_id="c", name="t", arguments="{}"),
    )

    async with ctx.with_filler("hello", delay=0.02, interval=None):
        await asyncio.sleep(0.1)
    assert [c["text"] for c in session.say_calls] == ["hello"]


@pytest.mark.asyncio
async def test_run_context_with_filler_cancels_on_exit() -> None:
    """Exiting the cm before delay elapses cancels the pending fire."""
    from livekit.agents.llm import FunctionCall
    from livekit.agents.voice.events import RunContext

    session = _FakeSession()
    handle = _FakeSpeechHandle()
    ctx: RunContext = RunContext(
        session=session,  # type: ignore[arg-type]
        speech_handle=handle,  # type: ignore[arg-type]
        function_call=FunctionCall(call_id="c", name="t", arguments="{}"),
    )

    async with ctx.with_filler("nope", delay=1.0, interval=None):
        await asyncio.sleep(0.01)
    assert session.say_calls == []


@pytest.mark.asyncio
async def test_ctx_update_resets_pending_filler_dwell() -> None:
    """ctx.update() called mid-dwell aborts the current dwell on every active filler."""
    from livekit.agents.llm import FunctionCall
    from livekit.agents.voice.events import RunContext

    session = _FakeSession()
    handle = _FakeSpeechHandle()
    ctx: RunContext = RunContext(
        session=session,  # type: ignore[arg-type]
        speech_handle=handle,  # type: ignore[arg-type]
        function_call=FunctionCall(call_id="c", name="t", arguments="{}"),
    )

    async with ctx.with_filler("ping", delay=0.1, interval=None):
        await asyncio.sleep(0.05)
        # update() resets the dwell on every active filler before anything else;
        # fire two back to back to confirm each call resets it
        await ctx.update("first")
        await ctx.update("second")
        await asyncio.sleep(0.08)
        # not yet fired — reset_dwell pushed the fire past 0.05+0.1=0.15
        fires_mid = len(session.say_calls)
        await asyncio.sleep(0.05)  # cross past 0.15
        fires_late = len(session.say_calls)
    assert fires_mid == 0
    assert fires_late == 1
