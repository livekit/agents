from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from livekit import rtc
from livekit.agents.ipc.job_proc_lazy_main import _JobProc
from livekit.agents.voice import AgentSession
from livekit.agents.voice.remote_session import SessionHost

pytestmark = pytest.mark.unit


class _FakeRoom(rtc.EventEmitter[str]):
    def __init__(self) -> None:
        super().__init__()
        self.connected = True
        self.remote_participants: dict[str, Any] = {"simulator": object()}

    def isconnected(self) -> bool:
        return self.connected


def _job_proc(room: _FakeRoom, timeout: float) -> Any:
    return SimpleNamespace(_room=room, _session_end_timeout=timeout)


def _host() -> SessionHost:
    return SessionHost(MagicMock())


class TestJobOwnedSessionHost:
    def test_job_owns_the_host_and_sessions_share_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        job_ctx = SimpleNamespace(_session_host=None)
        monkeypatch.setattr(
            "livekit.agents.voice.agent_session.get_job_context", lambda required=True: job_ctx
        )
        first, second = AgentSession(), AgentSession()
        host = _host()

        first._attach_session_host(lambda: host)
        second._attach_session_host(_host)

        assert job_ctx._session_host is host
        assert second._session_host is host
        assert not first._owns_session_host
        assert host.session is second

    def test_session_owns_the_host_outside_a_job(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "livekit.agents.voice.agent_session.get_job_context", lambda required=True: None
        )
        session = AgentSession()
        session._attach_session_host(_host)
        assert session._owns_session_host


class TestWaitForSimulationFinalize:
    @pytest.mark.asyncio
    async def test_ends_once_finalized(self) -> None:
        host = _host()
        wait = asyncio.ensure_future(
            _JobProc._wait_for_simulation_finalize(_job_proc(_FakeRoom(), 10.0), host)
        )
        await asyncio.sleep(0.05)
        assert not wait.done()

        host._simulation_finalized.set()
        await asyncio.wait_for(wait, timeout=0.5)

    @pytest.mark.asyncio
    async def test_ends_when_the_simulator_leaves(self) -> None:
        room = _FakeRoom()
        wait = asyncio.ensure_future(
            _JobProc._wait_for_simulation_finalize(_job_proc(room, 10.0), _host())
        )
        await asyncio.sleep(0.05)
        assert not wait.done()

        room.remote_participants.clear()
        room.emit("participant_disconnected", object())
        await asyncio.wait_for(wait, timeout=0.5)

    @pytest.mark.asyncio
    async def test_does_not_wait_once_the_room_is_gone(self) -> None:
        room = _FakeRoom()
        room.connected = False
        await asyncio.wait_for(
            _JobProc._wait_for_simulation_finalize(_job_proc(room, 10.0), _host()), timeout=0.5
        )

    @pytest.mark.asyncio
    async def test_gives_up_after_the_timeout(self) -> None:
        await asyncio.wait_for(
            _JobProc._wait_for_simulation_finalize(_job_proc(_FakeRoom(), 0.1), _host()),
            timeout=1.0,
        )
