import asyncio

import pytest

from livekit.agents import Agent, VoiceSession
from livekit.agents.voice import UserTurn
from livekit.agents.voice.events import UserTurnEndedEvent, UserTurnStartedEvent

from .fake_stt import FakeSTT

pytestmark = pytest.mark.unit


def _session(**turn_handling: object) -> VoiceSession:
    return VoiceSession(vad=None, stt=FakeSTT(), turn_handling=turn_handling)  # type: ignore[arg-type]


async def test_a_user_turn_needs_a_started_session() -> None:
    async with _session() as session:
        with pytest.raises(RuntimeError):
            session.start_user_turn()


async def test_an_empty_tap_adds_nothing_and_is_told() -> None:
    async with _session(push_to_talk=True) as session:
        started: list[UserTurnStartedEvent] = []
        ended: list[UserTurnEndedEvent] = []
        session.on("user_turn_started", started.append)
        session.on("user_turn_ended", ended.append)
        await session.start(Agent(instructions="Take the order.", name="orders"))
        turn = session.start_user_turn()
        assert isinstance(turn, UserTurn)
        result = await asyncio.wait_for(turn.end(), timeout=5)
        assert result.empty and result.message is None
        await asyncio.sleep(0)
        assert [e.turn_id for e in started] == [turn.id]
        assert [(e.turn_id, e.cancelled, e.empty) for e in ended] == [(turn.id, False, True)]
        assert not [m for m in session.history.items if m.type == "message" and m.role == "user"]


async def test_a_second_start_is_refused_while_a_turn_is_open() -> None:
    async with _session(push_to_talk=True) as session:
        await session.start(Agent(instructions="Take the order.", name="orders"))
        turn = session.start_user_turn()
        with pytest.raises(RuntimeError, match="already open"):
            session.start_user_turn()
        turn.cancel()
        session.start_user_turn().cancel()


async def test_ending_a_cancelled_turn_fails() -> None:
    async with _session() as session:
        await session.start(Agent(instructions="Take the order.", name="orders"))
        turn = session.start_user_turn()
        turn.cancel()
        with pytest.raises(RuntimeError, match="already ended"):
            await asyncio.wait_for(turn.end(), timeout=5)


async def test_push_to_talk_is_a_live_turn_handling_option() -> None:
    async with _session() as session:
        assert session.options["push_to_talk"] is False
        session.update_options(turn_handling={"push_to_talk": True})
        assert session.options["push_to_talk"] is True
        async with session.override_options(turn_handling={"push_to_talk": False}):
            assert session.options["push_to_talk"] is False
        assert session.options["push_to_talk"] is True
