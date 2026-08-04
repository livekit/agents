from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any, cast

import apsw
import pytest

from examples.hotel_receptionist.agent import (
    HotelReceptionistAgent,
    _close_session_resources,
    _expressive_options,
)
from examples.hotel_receptionist.capabilities import CAPABILITIES
from examples.hotel_receptionist.ui_view import UiView
from livekit import rtc

pytestmark = pytest.mark.unit


class _FakeDatabase:
    def __init__(self, events: list[str]) -> None:
        self.on_change: object | None = object()
        self._events = events

    async def aclose(self) -> None:
        self._events.append("database closed")


class _FakeUiView:
    def __init__(self, events: list[str], *, fail: bool = False) -> None:
        self._events = events
        self._fail = fail

    async def aclose(self) -> None:
        self._events.append("UI closed")
        if self._fail:
            raise RuntimeError("UI cleanup failed")


class _FakeLocalParticipant:
    def __init__(self) -> None:
        self.rpc_methods: dict[str, Callable[[Any], Any]] = {}
        self.unregister_calls: list[str] = []

    def register_rpc_method(
        self, method_name: str, handler: Callable[[Any], Any]
    ) -> Callable[[Any], Any]:
        self.rpc_methods[method_name] = handler
        return handler

    def unregister_rpc_method(self, method_name: str) -> None:
        self.unregister_calls.append(method_name)
        self.rpc_methods.pop(method_name, None)


class _FakeRoom:
    def __init__(self) -> None:
        self.local_participant = _FakeLocalParticipant()
        self.listeners: dict[str, list[Callable[[Any], None]]] = {}

    def on(self, event: str, callback: Callable[[Any], None]) -> Callable[[Any], None]:
        self.listeners.setdefault(event, []).append(callback)
        return callback

    def off(self, event: str, callback: Callable[[Any], None]) -> None:
        self.listeners[event].remove(callback)


def test_expressive_options_resolve_against_the_sdk() -> None:
    """The demo flips AgentSession's private ``_expressive`` switch (there is no
    public API for the pipeline yet); this pins the config shape to what the
    SDK's resolver actually accepts, so an upstream change fails here instead
    of crashing every simulation at startup."""
    from livekit.agents.voice.agent_session import (
        DEFAULT_EXPRESSIVE_OPTIONS,
        resolve_expressive_options,
    )

    assert _expressive_options(False) is False

    opts = _expressive_options(True)
    assert isinstance(opts, dict)
    resolved = resolve_expressive_options(
        opts,  # type: ignore[arg-type]
        provider_key="xai",
        default=DEFAULT_EXPRESSIVE_OPTIONS,
    )
    assert resolved["tts_instructions_template"]
    assert resolved["speech_steering"]


def test_agent_collects_tools_from_every_tool_module() -> None:
    agent = HotelReceptionistAgent(today=date(2026, 6, 8))

    # Capability gating: only the resident pair is visible until load_capability.
    assert {t.info.name for t in agent.tools} == {
        "load_capability",
        "say_goodbye_and_close_call",
    }
    # ...but the registry holds every mixin tool, ready to be switched on, and
    # every tool a capability advertises actually exists.
    assert len(agent._registry) == 36
    missing = {
        name for area in CAPABILITIES.values() for name in area.tools if name not in agent._registry
    }
    assert not missing


@pytest.mark.asyncio
async def test_session_resources_close_in_dependency_order() -> None:
    events: list[str] = []
    db = _FakeDatabase(events)
    ui = _FakeUiView(events)

    await _close_session_resources(cast(Any, db), cast(Any, ui))

    assert db.on_change is None
    assert events == ["UI closed", "database closed"]


@pytest.mark.asyncio
async def test_database_closes_when_ui_cleanup_fails() -> None:
    events: list[str] = []
    db = _FakeDatabase(events)
    ui = _FakeUiView(events, fail=True)

    with pytest.raises(RuntimeError, match="UI cleanup failed"):
        await _close_session_resources(cast(Any, db), cast(Any, ui))

    assert db.on_change is None
    assert events == ["UI closed", "database closed"]


@pytest.mark.asyncio
async def test_ui_view_unregisters_every_callback_once() -> None:
    room = _FakeRoom()
    connection = apsw.Connection(":memory:")
    ui = UiView(cast(rtc.Room, room), connection)

    try:
        await ui.start()
        assert set(room.local_participant.rpc_methods) == {
            "sqlite_diff:subscribe",
            "sqlite_diff:rebase",
        }
        assert len(room.listeners["participant_disconnected"]) == 1

        await ui.aclose()
        await ui.aclose()

        assert room.local_participant.rpc_methods == {}
        assert set(room.local_participant.unregister_calls) == {
            "sqlite_diff:subscribe",
            "sqlite_diff:rebase",
        }
        assert len(room.local_participant.unregister_calls) == 2
        assert room.listeners["participant_disconnected"] == []
    finally:
        await ui.aclose()
        connection.close()
