"""Tests for room-host (SessionHost over RoomSessionTransport) registration on start().

Covers the `_register_as_room_host` parameter introduced to decouple room-host
registration from the primary-session designation, so a session can opt out of
being the room host even without a JobContext.

Resolution rules (see AgentSession.start):
- not given      -> register as host
- explicit True  -> register as host (overrides non-primary auto-opt-out)
- explicit False -> do not register as host
- with a JobContext, a non-primary session auto-resolves to False when not given.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit.agents import Agent, AgentSession

from .fake_io import FakeAudioInput, FakeAudioOutput, FakeTextOutput
from .fake_llm import FakeLLM
from .fake_stt import FakeSTT
from .fake_tts import FakeTTS
from .fake_vad import FakeVAD

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

_MOD = "livekit.agents.voice.agent_session"


class SimpleAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a test agent.")


def _create_simple_session() -> AgentSession:
    """Minimal AgentSession with audio I/O set (so RoomIO leaves them alone)."""
    session = AgentSession[None](
        vad=FakeVAD(fake_user_speeches=[], min_silence_duration=0.5, min_speech_duration=0.05),
        stt=FakeSTT(fake_user_speeches=[]),
        llm=FakeLLM(fake_responses=[]),
        tts=FakeTTS(fake_responses=[]),
    )
    session.input.audio = FakeAudioInput()
    session.output.audio = FakeAudioOutput()
    session.output.transcription = FakeTextOutput()
    return session


async def _cleanup(session: AgentSession) -> None:
    with contextlib.suppress(RuntimeError):
        await session.drain()
    await session.aclose()


def _make_mock_job_ctx(primary: object | None = None) -> MagicMock:
    """A JobContext stub exposing only the fields start() touches in this path."""
    mock_ctx = MagicMock()
    mock_ctx.job.enable_recording = False  # keep recording off; this suite is host-only
    mock_ctx.job.id = "test-job-id"
    mock_ctx.job.agent_name = "test-agent"
    mock_ctx.room.name = "test-room"
    mock_ctx._primary_agent_session = primary
    mock_ctx.session_directory = Path("/tmp/test-session")
    mock_ctx.connect = AsyncMock()
    return mock_ctx


@contextlib.contextmanager
def _patch_host(job_ctx: MagicMock | None = None) -> Iterator[tuple[MagicMock, MagicMock]]:
    """Patch RoomIO and the host primitives; yield (RoomSessionTransport, SessionHost) mocks."""
    room_io_inst = MagicMock()
    room_io_inst.start = AsyncMock()
    room_io_inst.aclose = AsyncMock()
    with (
        patch(f"{_MOD}.get_job_context", return_value=job_ctx),
        patch(f"{_MOD}.room_io.RoomIO", return_value=room_io_inst),
        patch(f"{_MOD}.RoomSessionTransport") as mock_transport,
        patch(f"{_MOD}.SessionHost") as mock_host,
    ):
        # start()/aclose() await these on the registered host
        mock_host.return_value.start = AsyncMock()
        mock_host.return_value.aclose = AsyncMock()
        yield mock_transport, mock_host


def _assert_registered(
    session: AgentSession, transport: MagicMock, host: MagicMock, room: MagicMock
) -> None:
    transport.assert_called_once_with(room)
    host.assert_called_once_with(transport.return_value)
    host.return_value.register_session.assert_called_once_with(session)
    assert session._session_host is host.return_value


def _assert_not_registered(session: AgentSession, transport: MagicMock, host: MagicMock) -> None:
    transport.assert_not_called()
    host.assert_not_called()
    assert session._session_host is None


# ---------------------------------------------------------------------------
# Without a JobContext
# ---------------------------------------------------------------------------


async def test_registers_as_host_by_default_without_job_ctx() -> None:
    """No JobContext and the flag omitted: the session becomes the room host."""
    session = _create_simple_session()
    room = MagicMock()
    with _patch_host(job_ctx=None) as (transport, host):
        await session.start(SimpleAgent(), room=room)
        _assert_registered(session, transport, host, room)
        await _cleanup(session)


async def test_opt_out_of_host_without_job_ctx() -> None:
    """The new capability: opt out of being the room host even without a JobContext."""
    session = _create_simple_session()
    room = MagicMock()
    with _patch_host(job_ctx=None) as (transport, host):
        await session.start(SimpleAgent(), room=room, _register_as_room_host=False)
        _assert_not_registered(session, transport, host)
        await _cleanup(session)


async def test_explicit_opt_in_of_host_without_job_ctx() -> None:
    """Explicit True without a JobContext registers as host."""
    session = _create_simple_session()
    room = MagicMock()
    with _patch_host(job_ctx=None) as (transport, host):
        await session.start(SimpleAgent(), room=room, _register_as_room_host=True)
        _assert_registered(session, transport, host, room)
        await _cleanup(session)


# ---------------------------------------------------------------------------
# With a JobContext (primary-session interaction)
# ---------------------------------------------------------------------------


async def test_primary_session_registers_as_host() -> None:
    """The first session under a JobContext is primary and registers as host by default."""
    session = _create_simple_session()
    room = MagicMock()
    mock_ctx = _make_mock_job_ctx(primary=None)
    with _patch_host(job_ctx=mock_ctx) as (transport, host):
        await session.start(SimpleAgent(), room=room)
        assert mock_ctx._primary_agent_session is session
        _assert_registered(session, transport, host, room)
        await _cleanup(session)


async def test_non_primary_session_does_not_register_by_default() -> None:
    """A second session under the same JobContext is non-primary and is not the host."""
    session = _create_simple_session()
    room = MagicMock()
    mock_ctx = _make_mock_job_ctx(primary=object())  # another session already primary
    with _patch_host(job_ctx=mock_ctx) as (transport, host):
        await session.start(SimpleAgent(), room=room)
        _assert_not_registered(session, transport, host)
        await _cleanup(session)


async def test_non_primary_session_can_force_host() -> None:
    """Explicit True overrides the non-primary auto opt-out."""
    session = _create_simple_session()
    room = MagicMock()
    mock_ctx = _make_mock_job_ctx(primary=object())
    with _patch_host(job_ctx=mock_ctx) as (transport, host):
        await session.start(SimpleAgent(), room=room, _register_as_room_host=True)
        _assert_registered(session, transport, host, room)
        await _cleanup(session)


async def test_primary_session_can_opt_out_of_host() -> None:
    """Explicit False keeps a primary session from becoming the room host."""
    session = _create_simple_session()
    room = MagicMock()
    mock_ctx = _make_mock_job_ctx(primary=None)
    with _patch_host(job_ctx=mock_ctx) as (transport, host):
        await session.start(SimpleAgent(), room=room, _register_as_room_host=False)
        # still claims primary, just not the room host
        assert mock_ctx._primary_agent_session is session
        _assert_not_registered(session, transport, host)
        await _cleanup(session)
