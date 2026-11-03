from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

from livekit.agents import Agent, AgentSession
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.audio_recognition import _EndOfTurnInfo, _EndOfTurnMetrics

pytestmark = pytest.mark.unit


def _end_of_turn(turn_number: int) -> _EndOfTurnInfo:
    return _EndOfTurnInfo(
        skip_reply=False,
        new_transcript=f"turn {turn_number}",
        transcript_confidence=1.0,
        metrics=_EndOfTurnMetrics(
            started_speaking_at=None,
            stopped_speaking_at=None,
            transcription_delay=None,
            end_of_turn_delay=None,
        ),
    )


@pytest.mark.asyncio
async def test_cancelling_wait_for_idle_does_not_cancel_end_of_turn_task() -> None:
    session = AgentSession()
    activity = AgentActivity(Agent(instructions="test"), session)

    end_of_turn_task = asyncio.create_task(asyncio.Event().wait())
    activity._audio_recognition = cast(Any, SimpleNamespace(_end_of_turn_task=end_of_turn_task))

    idle_observer = asyncio.create_task(activity.wait_for_idle(wait_for_user=False))
    await asyncio.sleep(0)
    idle_observer.cancel()

    with pytest.raises(asyncio.CancelledError):
        await idle_observer

    assert not end_of_turn_task.cancelled()

    end_of_turn_task.cancel()
    await asyncio.gather(end_of_turn_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelling_wait_for_idle_does_not_poison_user_turns() -> None:
    session = AgentSession()
    activity = AgentActivity(Agent(instructions="test"), session)
    activity._scheduling_paused = False
    activity._new_turns_blocked = False

    active_turn = asyncio.create_task(asyncio.Event().wait())
    activity._user_turn_completed_atask = active_turn

    idle_observer = asyncio.create_task(activity.wait_for_idle())
    await asyncio.sleep(0)
    idle_observer.cancel()

    with pytest.raises(asyncio.CancelledError):
        await idle_observer

    assert not active_turn.cancelled()

    active_turn.cancel()
    await asyncio.gather(active_turn, return_exceptions=True)

    assert activity.on_end_of_turn(_end_of_turn(2))
    successor = activity._user_turn_completed_atask
    assert successor is not None
    await successor

    assert not successor.cancelled()


@pytest.mark.asyncio
async def test_cancelling_latest_user_turn_does_not_cancel_predecessor() -> None:
    session = AgentSession()
    activity = AgentActivity(Agent(instructions="test"), session)

    predecessor = asyncio.create_task(asyncio.Event().wait())
    latest_turn = asyncio.create_task(
        activity._user_turn_completed_task(predecessor, _end_of_turn(2))
    )
    await asyncio.sleep(0)
    latest_turn.cancel()

    with pytest.raises(asyncio.CancelledError):
        await latest_turn

    assert not predecessor.cancelled()

    predecessor.cancel()
    await asyncio.gather(predecessor, return_exceptions=True)


@pytest.mark.asyncio
async def test_wait_for_idle_does_not_spin_through_speech_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_sleep = asyncio.sleep
    yields = 0

    async def _counting_sleep(delay: float, *args: Any, **kwargs: Any) -> Any:
        nonlocal yields
        if delay == 0 and asyncio.current_task() is idle_task:
            yields += 1
        return await real_sleep(delay, *args, **kwargs)

    async with AgentSession(min_consecutive_speech_delay=0.3) as session:
        await session.start(Agent(instructions="test"))
        session.say("first")
        session.say("second")
        await real_sleep(0.02)

        monkeypatch.setattr(asyncio, "sleep", _counting_sleep)
        idle_task = asyncio.create_task(session.wait_for_idle())
        await asyncio.wait_for(idle_task, timeout=5.0)

    # a few passes per speech; a spinning loop yields thousands of times through the delay
    assert yields < 20, f"wait_for_idle yielded {yields} times through the speech delay"
