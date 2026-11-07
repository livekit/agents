from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from livekit.agents import Agent, AgentSession
from livekit.agents.llm._realtime.gpt_live import GPTLiveModel, GPTLiveSession

from .test_gpt_live_model import _connect_hook, _response_event

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


@asynccontextmanager
async def _session(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[AgentSession, GPTLiveSession]]:
    _connect_hook(monkeypatch)
    model = GPTLiveModel(api_key="local-placeholder")
    session: AgentSession = AgentSession(llm=model, user_away_timeout=3.0)
    try:
        await session.start(Agent(instructions="Local test."))
        live = session.current_agent.duplex_session
        assert isinstance(live, GPTLiveSession)
        await live._session_started_fut
        yield session, live
    finally:
        await session.aclose()
        await model.aclose()


def _event(live: GPTLiveSession, event: str, delegation: str = "d1") -> None:
    live._handle_event(_response_event(delegation, {"type": event, "response": {"id": delegation}}))


@pytest.mark.parametrize(
    "terminal", ["response.completed", "response.failed", "response.incomplete"]
)
async def test_backend_holds_away_window_until_terminal(
    monkeypatch: pytest.MonkeyPatch, terminal: str
) -> None:
    async with _session(monkeypatch) as (session, live):
        _event(live, "response.created")
        await asyncio.sleep(4)
        assert str(session.user_state) == "listening"
        # Filler playout ending must not arm a timer while backend work continues.
        session._update_agent_state("speaking")
        session._update_agent_state("listening")
        await asyncio.sleep(4)
        assert str(session.user_state) == "listening"
        _event(live, terminal)
        await asyncio.sleep(2)
        assert str(session.user_state) == "listening"
        await asyncio.sleep(2)
        assert str(session.user_state) == "away"


async def test_overlapping_backends_hold_window_until_last_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _session(monkeypatch) as (session, live):
        _event(live, "response.created", "d1")
        _event(live, "response.created", "d2")
        _event(live, "response.completed", "d1")
        await asyncio.sleep(4)
        assert str(session.user_state) == "listening"
        _event(live, "response.completed", "d2")
        await asyncio.sleep(2)
        assert str(session.user_state) == "listening"
        await asyncio.sleep(2)
        assert str(session.user_state) == "away"


async def test_idle_session_still_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _session(monkeypatch) as (session, _live):
        await asyncio.sleep(4)
        assert str(session.user_state) == "away"


async def test_backend_event_does_not_revive_away_user(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _session(monkeypatch) as (session, live):
        await asyncio.sleep(4)
        assert str(session.user_state) == "away"
        _event(live, "response.created")
        _event(live, "response.completed")
        assert str(session.user_state) == "away"


async def test_reconnect_reset_releases_backend_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _session(monkeypatch) as (session, live):
        _event(live, "response.created")
        await asyncio.sleep(4)
        assert str(session.user_state) == "listening"
        live._reset_for_reconnect()
        # Restore startup ack for fake transport teardown after resetting the futures.
        live._session_started_fut.set_result(None)
        await asyncio.sleep(2)
        assert str(session.user_state) == "listening"
        await asyncio.sleep(2)
        assert str(session.user_state) == "away"


async def test_handoff_reuses_backend_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _session(monkeypatch) as (session, live):
        _event(live, "response.created")
        session.update_agent(Agent(instructions="Local test."))
        assert session._update_activity_atask is not None
        await session._update_activity_atask
        assert session.current_agent.duplex_session is live
        await asyncio.sleep(4)
        assert str(session.user_state) == "listening"
        _event(live, "response.completed")
        await asyncio.sleep(2)
        assert str(session.user_state) == "listening"
        await asyncio.sleep(2)
        assert str(session.user_state) == "away"


async def test_fallback_forwards_backend_hold_and_releases_it_on_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.agents.llm import RealtimeModelFallbackAdapter

    from .fake_realtime import FakeRealtimeModel, FakeRealtimeSession

    # The fallback delegates any active child's backend state, including plugins using it.
    monkeypatch.setattr(
        FakeRealtimeSession, "_backend_active", property(lambda self: not self.closed)
    )
    primary = FakeRealtimeModel()
    backup = FakeRealtimeModel()
    fallback = RealtimeModelFallbackAdapter([primary, backup])
    rt = fallback.session()
    events: list[bool] = []
    rt.on("_backend_activity_changed", lambda: events.append(rt._backend_active))
    try:
        assert rt._backend_active
        primary.active_session.emit("_backend_activity_changed")
        assert events == [True]
        # Make the replacement child idle; the old child is still active until closed.
        monkeypatch.setattr(
            FakeRealtimeSession,
            "_backend_active",
            property(lambda self: self is primary.active_session and not self.closed),
        )
        await rt._swap(1, was_speaking=False)
        assert not rt._backend_active
        assert events == [True, False]
    finally:
        await rt.aclose()
