from __future__ import annotations

import asyncio
import contextlib
from typing import Any, cast

import pytest

from livekit.agents import Agent, AgentSession, utils
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.audio_recognition import _EndOfTurnInfo, _EndOfTurnMetrics

from .fake_realtime import FakeRealtimeModel, FakeRealtimeSession, fake_capabilities

pytestmark = pytest.mark.unit


class _BlockingCloseRealtimeSession(FakeRealtimeSession):
    def __init__(self, model: FakeRealtimeModel) -> None:
        super().__init__(model)
        self.close_entered = asyncio.Event()
        self.close_release = asyncio.Event()
        self.commit_audio_calls = 0

    def commit_audio(self) -> None:
        self.commit_audio_calls += 1

    async def aclose(self) -> None:
        self.close_entered.set()
        await self.close_release.wait()
        await super().aclose()


class _ClosingRecognition:
    def __init__(self, bounded_turn_task: asyncio.Task[None]) -> None:
        self._bounded_turn_task = bounded_turn_task
        self.closed = False
        self.cancelled_bounded_turn = False

    async def _aclose(self) -> None:
        await utils.aio.cancel_and_wait(self._bounded_turn_task)
        self.closed = True
        self.cancelled_bounded_turn = self._bounded_turn_task.cancelled()


def _bounded_turn() -> _EndOfTurnInfo:
    return _EndOfTurnInfo(
        skip_reply=False,
        new_transcript="already bounded before close",
        transcript_confidence=0.9,
        metrics=_EndOfTurnMetrics(
            started_speaking_at=1.0,
            stopped_speaking_at=2.0,
            transcription_delay=0.1,
            end_of_turn_delay=0.2,
        ),
        backchannel_over_agent=False,
    )


async def test_close_retains_bounded_turn_without_starting_provider_work() -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=False,
            can_disable_turn_detection=False,
        )
    )
    session = AgentSession(llm=model)
    activity = AgentActivity(Agent(instructions="test"), session)
    rt_session = _BlockingCloseRealtimeSession(model)
    activity._rt_session = rt_session
    activity._scheduling_paused = True
    session._closing = True
    bounded_turn_finished = asyncio.Event()

    async def _finish_bounded_turn() -> None:
        try:
            await rt_session.close_entered.wait()
            await activity._user_turn_completed_task(None, _bounded_turn())
        finally:
            bounded_turn_finished.set()

    bounded_turn_task = asyncio.create_task(_finish_bounded_turn())
    activity._user_turn_completed_atask = bounded_turn_task
    recognition = _ClosingRecognition(bounded_turn_task)
    activity._audio_recognition = cast(Any, recognition)

    async def _close() -> None:
        async with activity._lock:
            await activity._close_session()

    close_task = asyncio.create_task(_close())
    await rt_session.close_entered.wait()
    await bounded_turn_finished.wait()
    rt_session.close_release.set()
    await close_task

    with contextlib.suppress(asyncio.CancelledError):
        await bounded_turn_task
    assert recognition.closed
    assert not recognition.cancelled_bounded_turn
    assert [message.raw_text_content for message in activity.agent.chat_ctx.messages()] == [
        "already bounded before close"
    ]
    assert rt_session.commit_audio_calls == 0
