from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from livekit.agents import Agent, AgentSession
from livekit.agents.voice.agent_activity import AgentActivity

pytestmark = pytest.mark.unit


class _AudioRecognitionStub:
    def __init__(self, transcript_fut: asyncio.Future[str]) -> None:
        self._transcript_fut = transcript_fut
        self._end_of_turn_task: asyncio.Task[None] | None = None

    def _commit_user_turn(self, **_: Any) -> asyncio.Future[str]:
        return self._transcript_fut


def _create_activity() -> AgentActivity:
    session = AgentSession()
    return AgentActivity(Agent(instructions="test"), session)


@pytest.mark.asyncio
async def test_commit_user_turn_waits_for_current_turn_processing() -> None:
    loop = asyncio.get_running_loop()
    transcript_fut = loop.create_future()
    recognition = _AudioRecognitionStub(transcript_fut)
    activity = _create_activity()
    activity._audio_recognition = cast(Any, recognition)

    commit_fut = activity.commit_user_turn(transcript_timeout=2.0, stt_flush_duration=2.0)

    eou_gate = asyncio.Event()
    turn_gate = asyncio.Event()
    recognition._end_of_turn_task = asyncio.create_task(eou_gate.wait())
    activity._user_turn_completed_atask = asyncio.create_task(turn_gate.wait())
    transcript_fut.set_result("hello")

    await asyncio.sleep(0)
    assert not commit_fut.done()

    eou_gate.set()
    await asyncio.sleep(0)
    assert not commit_fut.done()

    turn_gate.set()
    assert await commit_fut == "hello"


@pytest.mark.asyncio
async def test_commit_user_turn_ignores_previous_turn_tasks() -> None:
    loop = asyncio.get_running_loop()
    transcript_fut = loop.create_future()
    transcript_fut.set_result("")
    recognition = _AudioRecognitionStub(transcript_fut)
    activity = _create_activity()

    previous_eou_task = asyncio.create_task(asyncio.Event().wait())
    previous_turn_task = asyncio.create_task(asyncio.Event().wait())
    recognition._end_of_turn_task = previous_eou_task
    activity._user_turn_completed_atask = previous_turn_task
    activity._audio_recognition = cast(Any, recognition)

    try:
        commit_fut = activity.commit_user_turn(transcript_timeout=2.0, stt_flush_duration=2.0)
        assert await asyncio.wait_for(commit_fut, timeout=1.0) == ""
    finally:
        previous_eou_task.cancel()
        previous_turn_task.cancel()
        await asyncio.gather(previous_eou_task, previous_turn_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelling_commit_wait_does_not_cancel_turn_processing() -> None:
    loop = asyncio.get_running_loop()
    transcript_fut = loop.create_future()
    recognition = _AudioRecognitionStub(transcript_fut)
    activity = _create_activity()
    activity._audio_recognition = cast(Any, recognition)

    commit_fut = activity.commit_user_turn(transcript_timeout=2.0, stt_flush_duration=2.0)
    eou_task = asyncio.create_task(asyncio.Event().wait())
    recognition._end_of_turn_task = eou_task
    transcript_fut.set_result("hello")
    await asyncio.sleep(0)

    commit_fut.cancel()
    with pytest.raises(asyncio.CancelledError):
        await commit_fut

    assert not eou_task.cancelled()
    eou_task.cancel()
    await asyncio.gather(eou_task, return_exceptions=True)
