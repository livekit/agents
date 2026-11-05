"""A parked preemptive generation must never block an agent handoff."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

import pytest

from livekit.agents import Agent, AgentSession
from livekit.agents.llm import LLM, ChatChunk, ChatContext, ChoiceDelta, LLMStream
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.voice.audio_recognition import (
    _EndOfTurnInfo,
    _EndOfTurnMetrics,
    _PreemptiveGenerationInfo,
)

pytestmark = pytest.mark.unit

TIMEOUT = 2.0


class _GatedStream(LLMStream):
    """Emits one token, then holds the generation open until released."""

    def __init__(self, *args: Any, gate: asyncio.Event, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._gate = gate

    async def _run(self) -> None:
        self._event_ch.send_nowait(
            ChatChunk(id="chunk", delta=ChoiceDelta(role="assistant", content="one moment"))
        )
        await self._gate.wait()


class _GatedLLM(LLM):
    """Text-only LLM whose replies keep playing until ``release()``."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()

    @property
    def model(self) -> str:
        return "gated"

    @property
    def provider(self) -> str:
        return "test"

    def release(self) -> None:
        self.gate.set()

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Any] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[Any] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        return _GatedStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            gate=self.gate,
        )


async def _wait_until(predicate: Callable[[], bool], *, timeout: float = TIMEOUT) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


def _end_of_turn(transcript: str) -> _EndOfTurnInfo:
    return _EndOfTurnInfo(
        skip_reply=False,
        new_transcript=transcript,
        transcript_confidence=1.0,
        metrics=_EndOfTurnMetrics(
            started_speaking_at=None,
            stopped_speaking_at=None,
            transcription_delay=None,
            end_of_turn_delay=None,
        ),
    )


async def _start_session() -> tuple[_GatedLLM, AgentSession[None]]:
    llm = _GatedLLM()
    session: AgentSession[None] = AgentSession(llm=llm)
    await session.start(agent=Agent(instructions="qualify the caller"))
    return llm, session


async def test_handoff_after_a_turn_discarded_for_uninterruptible_speech() -> None:
    """The reported case: a turn discarded mid-speech must not hang the next handoff."""
    llm, session = await _start_session()
    activity = session._activity
    assert activity is not None

    try:
        # the caller starts talking, so a speculative reply is started
        assert session.current_speech is None
        activity.on_preemptive_generation(
            _PreemptiveGenerationInfo(
                new_transcript="what is the rate",
                transcript_confidence=1.0,
                started_speaking_at=time.time(),
            )
        )
        assert activity._preemptive_generation is not None

        # the agent starts an uninterruptible message (e.g. a transfer hold)
        session.generate_reply(instructions="ask them to hold", allow_interruptions=False)
        await _wait_until(lambda: session.current_speech is not None)

        # the caller's turn ends while that message is still playing
        activity.on_end_of_turn(_end_of_turn("can you tell me the rate"))
        await _wait_until(
            lambda: (
                activity._user_turn_completed_atask is not None
                and activity._user_turn_completed_atask.done()
            )
        )

        # the uninterruptible message finishes; the discarded turn's reply is still parked
        llm.release()
        await _wait_until(lambda: session.current_speech is None)

        # awaiting an AgentTask pauses the activity and must not hang
        await asyncio.wait_for(
            session._update_activity(
                Agent(instructions="hand off"),
                previous_activity="pause",
                blocked_tasks=[],
                wait_on_enter=False,
            ),
            timeout=TIMEOUT,
        )
        assert activity._preemptive_generation is None
    finally:
        await asyncio.gather(session.aclose(), return_exceptions=True)


async def test_pause_clears_a_parked_preemptive_reply() -> None:
    """pause() drops any parked generation before waiting for the speech tasks."""
    _, session = await _start_session()
    activity = session._activity
    assert activity is not None

    try:
        # park a speculative reply without any turn discard
        activity.on_preemptive_generation(
            _PreemptiveGenerationInfo(
                new_transcript="what is the rate",
                transcript_confidence=1.0,
                started_speaking_at=time.time(),
            )
        )
        assert activity._preemptive_generation is not None

        await asyncio.wait_for(
            session._update_activity(
                Agent(instructions="hand off"),
                previous_activity="pause",
                blocked_tasks=[],
                wait_on_enter=False,
            ),
            timeout=TIMEOUT,
        )

        # the parked reply is gone, so a later resume cannot resurrect it
        assert activity._preemptive_generation is None
    finally:
        await asyncio.gather(session.aclose(), return_exceptions=True)


async def test_reply_parked_while_pausing_does_not_block_the_handoff() -> None:
    """A reply started after the handoff begins is still dropped before the drain wait.

    The cancel has to sit where scheduling is marked paused: anywhere earlier leaves an
    await in between, and a transcript arriving in that window parks a fresh reply that
    nothing will schedule. ``on_exit`` is the widest such window.
    """
    llm = _GatedLLM()
    session: AgentSession[None] = AgentSession(llm=llm)

    class _TalkingExit(Agent):
        async def on_exit(self) -> None:
            activity = session._activity
            assert activity is not None
            activity.on_preemptive_generation(
                _PreemptiveGenerationInfo(
                    new_transcript="what is the rate",
                    transcript_confidence=1.0,
                    started_speaking_at=time.time(),
                )
            )
            assert activity._preemptive_generation is not None

    await session.start(agent=_TalkingExit(instructions="qualify the caller"))
    activity = session._activity
    assert activity is not None

    try:
        await asyncio.wait_for(
            session._update_activity(Agent(instructions="hand off")),
            timeout=TIMEOUT,
        )
        assert activity._preemptive_generation is None
    finally:
        llm.release()
        await asyncio.gather(session.aclose(), return_exceptions=True)
