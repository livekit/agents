"""Room-host (SessionHost over RoomSessionTransport) registration on start().

`_register_as_room_host` decouples host registration from primary designation:
omitted -> register; True -> register (overriding non-primary); False -> don't.
With a JobContext a non-primary session auto-resolves to False.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit.agents import Agent, AgentSession
from livekit.agents.types import NOT_GIVEN, NotGiven

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


def _make_session() -> AgentSession:
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


_PRIMARY, _OTHER = "self", "other"  # this session claims primary / another already holds it


def _job_ctx(marker: str | None) -> MagicMock | None:
    """JobContext stub for the given primary marker; None means no job context at all.

    Exposes only the fields start() touches here; recording stays off.
    """
    if marker is None:
        return None
    ctx = MagicMock()
    ctx.job.enable_recording = False
    ctx.job.id = "test-job-id"
    ctx.job.agent_name = "test-agent"
    ctx.room.name = "test-room"
    ctx._primary_agent_session = None if marker == _PRIMARY else object()
    ctx.session_directory = Path("/tmp/test-session")
    ctx.connect = AsyncMock()
    return ctx


@contextlib.contextmanager
def _patched(job_ctx: MagicMock | None) -> Iterator[tuple[MagicMock, MagicMock]]:
    room_io = MagicMock(start=AsyncMock(), aclose=AsyncMock())
    with (
        patch(f"{_MOD}.get_job_context", return_value=job_ctx),
        patch(f"{_MOD}.room_io.RoomIO", return_value=room_io),
        patch(f"{_MOD}.RoomSessionTransport") as transport,
        patch(f"{_MOD}.SessionHost") as host,
    ):
        host.return_value.start = AsyncMock()
        host.return_value.aclose = AsyncMock()
        yield transport, host


class Outcome(Enum):
    REGISTERS = True
    NOT_REGISTERS = False


# fmt: off
# @formatter:off
_CASES = [
    # "_register_as_room_host", "job_ctx primary marker" and  "expected outcome")
    pytest.param(True,      None,     Outcome.REGISTERS,     id="no_ctx/opt-in"),
    pytest.param(True,      _OTHER,   Outcome.REGISTERS,     id="non_primary/force"),
    pytest.param(NOT_GIVEN, None,     Outcome.REGISTERS,     id="no_ctx/default"),
    pytest.param(NOT_GIVEN, _PRIMARY, Outcome.REGISTERS,     id="primary/default"),
    pytest.param(NOT_GIVEN, _OTHER,   Outcome.NOT_REGISTERS, id="non_primary/default"),
    pytest.param(False,     _PRIMARY, Outcome.NOT_REGISTERS, id="primary/opt-out"),
    pytest.param(False,     None,     Outcome.NOT_REGISTERS, id="no_ctx/opt-out"),
]
# @formatter:on
# fmt: on


@pytest.mark.parametrize("flag, ctx_marker, expected_outcome", _CASES)
async def test_room_host_registration(
    flag: bool | NotGiven, ctx_marker: str | None, expected_outcome: Outcome
) -> None:
    session = _make_session()
    job_ctx = _job_ctx(ctx_marker)
    room = MagicMock()
    kwargs = {} if flag is NOT_GIVEN else {"_register_as_room_host": flag}

    with _patched(job_ctx) as (transport, host):
        await session.start(SimpleAgent(), room=room, **kwargs)  # type: ignore[arg-type]

        match expected_outcome:
            case Outcome.REGISTERS:
                transport.assert_called_once_with(room)
                host.assert_called_once_with(transport.return_value)
                host.return_value.register_session.assert_called_once_with(session)
                assert session._session_host is host.return_value
            case Outcome.NOT_REGISTERS:
                transport.assert_not_called()
                host.assert_not_called()
                assert session._session_host is None

        # primary designation is claimed regardless of host opt-out
        if ctx_marker == _PRIMARY:
            assert job_ctx._primary_agent_session is session

        with contextlib.suppress(RuntimeError):
            await session.drain()
        await session.aclose()
