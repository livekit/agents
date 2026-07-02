"""Unit tests for livekit.agents.inference._utils.get_inference_headers.

Regression coverage for the crash where accessing ``ctx.agent`` before the room
is connected raised ``Exception("cannot access local participant before
connecting")`` and propagated out of ``get_inference_headers`` (introduced by
#5937, surfaced during STT websocket connection).

Note the two distinct rooms on a JobContext:
- ``ctx.room`` is the live ``rtc.Room`` (exposes ``isconnected()``); the fix
  guards on it before touching the local participant.
- ``ctx.job.room`` is room metadata (the source of the room sid).
The fakes below keep them separate so the tests exercise the real code path.
"""

from __future__ import annotations

import pytest

from livekit.agents.inference._utils import (
    HEADER_AGENT_ID,
    HEADER_JOB_ID,
    HEADER_ROOM_ID,
    HEADER_USER_AGENT,
    get_inference_headers,
)

pytestmark = pytest.mark.unit


class _FakeLiveRoom:
    """Stands in for ``JobContext.room`` (the live rtc.Room with isconnected())."""

    def __init__(self, *, connected: bool) -> None:
        self._connected = connected

    def isconnected(self) -> bool:
        return self._connected


class _FakeJobRoom:
    """Stands in for ``ctx.job.room`` (metadata; source of the room sid)."""

    sid = "RM_test_room"


class _FakeJob:
    id = "AJ_test_job"
    room = _FakeJobRoom()


class _FakeAgent:
    def __init__(self, sid: object) -> None:
        self.sid = sid


class _CtxConnected:
    """Connected room: ``isconnected()`` is True and ``agent.sid`` is reachable."""

    job = _FakeJob()

    def __init__(self, agent_sid: object) -> None:
        self.room = _FakeLiveRoom(connected=True)
        self.agent = _FakeAgent(agent_sid)


class _CtxDisconnected:
    """Room not connected: ``isconnected()`` is False and ``agent`` access raises.

    Mirrors ``rtc.Room.local_participant`` raising a bare ``Exception`` when
    ``_local_participant`` is still ``None`` before the first connect.
    """

    job = _FakeJob()
    room = _FakeLiveRoom(connected=False)

    @property
    def agent(self):  # noqa: ANN201 - matches the raising property under test
        raise Exception("cannot access local participant before connecting")


def _patch_ctx(monkeypatch: pytest.MonkeyPatch, ctx: object) -> None:
    monkeypatch.setattr("livekit.agents.job.get_job_context", lambda: ctx)


def test_omits_agent_header_when_room_not_connected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before the room connects, the agent header is omitted without crashing.

    This is the regression test for the reported STT crash. The fake ``agent``
    raises exactly like ``rtc.Room.local_participant`` does pre-connect: on the
    pre-fix code (no ``isconnected()`` guard) the access raises and the test
    fails; with the guard the access is skipped and room/job headers remain.
    """
    _patch_ctx(monkeypatch, _CtxDisconnected())

    headers = get_inference_headers()

    assert headers[HEADER_ROOM_ID] == "RM_test_room"
    assert headers[HEADER_JOB_ID] == "AJ_test_job"
    assert HEADER_AGENT_ID not in headers
    assert HEADER_USER_AGENT in headers


def test_includes_agent_header_when_connected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A connected room with a string sid populates the agent header."""
    _patch_ctx(monkeypatch, _CtxConnected("PA_agent_sid"))

    headers = get_inference_headers()

    assert headers[HEADER_AGENT_ID] == "PA_agent_sid"
    assert headers[HEADER_ROOM_ID] == "RM_test_room"
    assert headers[HEADER_JOB_ID] == "AJ_test_job"


def test_omits_agent_header_when_sid_not_str(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-string sids (e.g. a Mock participant in tests) are not leaked.

    Guards the ``isinstance(..., str)`` check added by #5943. The room is
    connected so the test reaches the isinstance check rather than the
    connection guard.
    """
    _patch_ctx(monkeypatch, _CtxConnected(object()))

    headers = get_inference_headers()

    assert HEADER_AGENT_ID not in headers
    assert headers[HEADER_ROOM_ID] == "RM_test_room"
    assert headers[HEADER_JOB_ID] == "AJ_test_job"


def test_omits_agent_header_when_sid_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty-string sid is falsy and must be omitted (connected room)."""
    _patch_ctx(monkeypatch, _CtxConnected(""))

    headers = get_inference_headers()

    assert HEADER_AGENT_ID not in headers


def test_no_job_context_returns_only_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Outside a job context (console mode / tests) only User-Agent is set."""

    def _raise() -> object:
        raise RuntimeError("no job context found")

    monkeypatch.setattr("livekit.agents.job.get_job_context", _raise)

    headers = get_inference_headers()

    assert HEADER_USER_AGENT in headers
    assert HEADER_ROOM_ID not in headers
    assert HEADER_JOB_ID not in headers
    assert HEADER_AGENT_ID not in headers
