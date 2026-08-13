from __future__ import annotations

import asyncio

import pytest

from livekit.agents import (
    Agent,
    AgentSession,
    AgentTask,
    EndpointingOptions,
    TurnHandlingOptions,
    llm,
)

from .fake_io import FakeAudioInput
from .fake_llm import FakeLLM, FakeLLMResponse
from .fake_stt import FakeSTT, FakeUserSpeech
from .fake_vad import FakeVAD

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


class _HandoffTask(AgentTask[None]):
    def __init__(self) -> None:
        super().__init__(instructions="handoff task")
        self.received_turn = asyncio.Event()
        self.user_turns: list[str | None] = []

    async def on_user_turn_completed(
        self, _turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        self.user_turns.append(new_message.raw_text_content)
        self.received_turn.set()


class _HandoffSource(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="handoff source")
        self.task = _HandoffTask()
        self.speech_created = asyncio.Event()
        self.begin_handoff = asyncio.Event()

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="hold the handoff open")
        self.speech_created.set()
        await self.begin_handoff.wait()
        await self.task


class _TurnRecorder(Agent):
    def __init__(self, instructions: str) -> None:
        super().__init__(instructions=instructions)
        self.received_turn = asyncio.Event()
        self.user_turns: list[str | None] = []

    async def on_user_turn_completed(
        self, _turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        self.user_turns.append(new_message.raw_text_content)
        self.received_turn.set()


class _BlockingExitRecorder(_TurnRecorder):
    def __init__(self, instructions: str) -> None:
        super().__init__(instructions)
        self.exit_started = asyncio.Event()
        self.release_exit = asyncio.Event()

    async def on_exit(self) -> None:
        self.exit_started.set()
        await self.release_exit.wait()


class _FailingPrewarmLLM(FakeLLM):
    def prewarm(self, *, loop: asyncio.AbstractEventLoop | None = None) -> None:
        raise RuntimeError("successor prewarm failed")


async def _wait_for(predicate, *, message: str) -> None:
    for _ in range(300):
        if predicate():
            return
        await asyncio.sleep(0.01)
    pytest.fail(message)


async def test_agent_task_handoff_forwards_a_turn_arriving_while_scheduling_pauses() -> None:
    """The replacement activity owns one pipeline turn committed during its pause."""
    transcript = "the turn that crosses the handoff"
    speech = FakeUserSpeech(
        start_time=0.05,
        end_time=0.10,
        transcript=transcript,
        stt_delay=0.0,
    )
    session = AgentSession(
        stt=FakeSTT(fake_user_speeches=[speech]),
        vad=FakeVAD(
            fake_user_speeches=[speech], min_speech_duration=0.05, min_silence_duration=0.05
        ),
        llm=FakeLLM(
            fake_responses=[
                # Keep the old scheduler in its real pause/drain window long enough
                # for the VAD/STT end-of-turn task to commit the crossing turn.
                FakeLLMResponse(input="hold the handoff open", content="", ttft=2.0, duration=0.0),
                FakeLLMResponse(input=transcript, content="received", ttft=0.0, duration=0.0),
            ]
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            endpointing=EndpointingOptions(min_delay=0.4, max_delay=0.4),
        ),
        aec_warmup_duration=None,
    )
    audio_input = FakeAudioInput()
    session.input.audio = audio_input
    source = _HandoffSource()

    try:
        await session.start(source)
        await asyncio.wait_for(source.speech_created.wait(), timeout=2.0)
        await _wait_for(
            lambda: source._activity is not None and source._activity.current_speech is not None,
            message="the source reply never reached the real scheduler",
        )

        # Let the source recognition pipeline start a normal VAD/STT/EOU turn,
        # but leave its EOU timer pending while the AgentTask starts.
        audio_input.push(0.1)
        await _wait_for(
            lambda: (
                source._activity is not None
                and source._activity._audio_recognition is not None
                and source._activity._audio_recognition._end_of_turn_task is not None
                and not source._activity._audio_recognition._end_of_turn_task.done()
            ),
            message="the source activity never opened the end-of-turn decision",
        )

        source.begin_handoff.set()
        await _wait_for(
            lambda: source._activity is not None and source._activity.scheduling_paused,
            message="AgentTask handoff never paused the source scheduler",
        )

        await asyncio.wait_for(source.task.received_turn.wait(), timeout=4.0)
        await asyncio.wait_for(session.wait_for_idle(), timeout=4.0)

        assert source.task.user_turns == [transcript]
        assert [
            item.raw_text_content
            for item in source.task.chat_ctx.items
            if item.type == "message" and item.role == "user"
        ] == [transcript]
        assert [
            item.raw_text_content
            for item in session.history.items
            if item.type == "message" and item.role == "user"
        ] == [transcript]
    finally:
        if not source.task.done():
            source.task.complete(None)
        await asyncio.wait_for(session.aclose(), timeout=5.0)


async def test_consecutive_update_agent_handoffs_keep_a_crossing_turn_with_a_successor() -> None:
    """A turn queued before two serialized updates is never delivered to the old activity."""
    transcript = "the turn between two updates"
    speech = FakeUserSpeech(
        start_time=0.05,
        end_time=0.10,
        transcript=transcript,
        stt_delay=0.0,
    )
    session = AgentSession(
        stt=FakeSTT(fake_user_speeches=[speech]),
        vad=FakeVAD(
            fake_user_speeches=[speech], min_speech_duration=0.05, min_silence_duration=0.05
        ),
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(input=transcript, content="received", ttft=0.0, duration=0.0),
            ]
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            endpointing=EndpointingOptions(min_delay=0.4, max_delay=0.4),
        ),
        aec_warmup_duration=None,
    )
    audio_input = FakeAudioInput()
    session.input.audio = audio_input
    source = _TurnRecorder("source")
    first_successor = _TurnRecorder("first successor")
    second_successor = _TurnRecorder("second successor")

    try:
        await session.start(source)
        audio_input.push(0.1)
        await _wait_for(
            lambda: (
                source._activity is not None
                and source._activity._audio_recognition is not None
                and source._activity._audio_recognition._end_of_turn_task is not None
                and not source._activity._audio_recognition._end_of_turn_task.done()
            ),
            message="the source activity never opened the end-of-turn decision",
        )

        # update_agent() blocks the source synchronously. Holding the activity
        # lock only makes the otherwise tiny pre-replacement window deterministic;
        # the turn itself still arrives through normal VAD/STT/EOU processing.
        async with session._activity_lock:
            session.update_agent(first_successor)
            session.update_agent(second_successor)
            await _wait_for(
                lambda: source._activity is not None and bool(source._activity._handoff_user_turns),
                message="the blocked source did not retain its crossing turn",
            )

        await _wait_for(
            lambda: (
                first_successor.received_turn.is_set() or second_successor.received_turn.is_set()
            ),
            message="no successor received the crossing turn",
        )
        await asyncio.wait_for(session.wait_for_idle(), timeout=4.0)

        assert source.user_turns == []
        assert first_successor.user_turns + second_successor.user_turns == [transcript]
        assert [
            item.raw_text_content
            for item in session.history.items
            if item.type == "message" and item.role == "user"
        ] == [transcript]
    finally:
        await asyncio.wait_for(session.aclose(), timeout=5.0)


async def test_handoff_turn_is_persisted_when_close_rejects_the_successor() -> None:
    """A queued turn becomes terminal history if close prevents its successor from starting."""
    transcript = "the turn closed during the handoff"
    speech = FakeUserSpeech(
        start_time=0.05,
        end_time=0.10,
        transcript=transcript,
        stt_delay=0.0,
    )
    session = AgentSession(
        stt=FakeSTT(fake_user_speeches=[speech]),
        vad=FakeVAD(
            fake_user_speeches=[speech], min_speech_duration=0.05, min_silence_duration=0.05
        ),
        llm=FakeLLM(),
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            endpointing=EndpointingOptions(min_delay=0.4, max_delay=0.4),
        ),
        aec_warmup_duration=None,
        session_close_transcript_timeout=0.01,
    )
    audio_input = FakeAudioInput()
    session.input.audio = audio_input
    source = _BlockingExitRecorder("source")
    successor = _TurnRecorder("successor")
    update_task: asyncio.Task[None] | None = None
    close_task: asyncio.Task[None] | None = None

    try:
        await session.start(source)
        audio_input.push(0.1)
        await _wait_for(
            lambda: (
                source._activity is not None
                and source._activity._audio_recognition is not None
                and source._activity._audio_recognition._end_of_turn_task is not None
                and not source._activity._audio_recognition._end_of_turn_task.done()
            ),
            message="the source activity never opened the end-of-turn decision",
        )

        session.update_agent(successor)
        update_task = session._update_activity_atask
        assert update_task is not None
        await asyncio.wait_for(source.exit_started.wait(), timeout=2.0)
        await _wait_for(
            lambda: (
                session._next_activity is not None
                and bool(session._next_activity._handoff_user_turns)
            ),
            message="the handoff never queued the crossing turn on its successor",
        )

        close_task = asyncio.create_task(session.aclose())
        await _wait_for(lambda: session._closing, message="session close never started")
        source.release_exit.set()
        await asyncio.wait_for(close_task, timeout=5.0)
        await asyncio.wait_for(update_task, timeout=2.0)

        assert source.user_turns == []
        assert successor.user_turns == []
        assert [
            item.raw_text_content
            for item in successor.chat_ctx.items
            if item.type == "message"
            and item.role == "user"
            and item.raw_text_content == transcript
        ] == [transcript]
        assert [
            item.raw_text_content
            for item in session.history.items
            if item.type == "message"
            and item.role == "user"
            and item.raw_text_content == transcript
        ] == [transcript]
    finally:
        source.release_exit.set()
        if close_task is None:
            await asyncio.wait_for(session.aclose(), timeout=5.0)


async def test_handoff_turn_is_persisted_when_successor_start_fails() -> None:
    """A start failure writes a queued turn once instead of silently clearing it."""
    transcript = "the turn lost by a failing successor"
    speech = FakeUserSpeech(
        start_time=0.05,
        end_time=0.10,
        transcript=transcript,
        stt_delay=0.0,
    )
    session = AgentSession(
        stt=FakeSTT(fake_user_speeches=[speech]),
        vad=FakeVAD(
            fake_user_speeches=[speech], min_speech_duration=0.05, min_silence_duration=0.05
        ),
        llm=FakeLLM(),
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            endpointing=EndpointingOptions(min_delay=0.4, max_delay=0.4),
        ),
        aec_warmup_duration=None,
    )
    audio_input = FakeAudioInput()
    session.input.audio = audio_input
    source = _TurnRecorder("source")
    successor = Agent(instructions="failing successor", llm=_FailingPrewarmLLM())
    update_task: asyncio.Task[None] | None = None

    try:
        await session.start(source)
        audio_input.push(0.1)
        await _wait_for(
            lambda: (
                source._activity is not None
                and source._activity._audio_recognition is not None
                and source._activity._audio_recognition._end_of_turn_task is not None
                and not source._activity._audio_recognition._end_of_turn_task.done()
            ),
            message="the source activity never opened the end-of-turn decision",
        )

        async with session._activity_lock:
            session.update_agent(successor)
            update_task = session._update_activity_atask
            assert update_task is not None
            await _wait_for(
                lambda: source._activity is not None and bool(source._activity._handoff_user_turns),
                message="the blocked source did not retain its crossing turn",
            )

        with pytest.raises(RuntimeError, match="successor prewarm failed"):
            await asyncio.wait_for(update_task, timeout=2.0)

        assert source.user_turns == []
        assert [
            item.raw_text_content
            for item in successor.chat_ctx.items
            if item.type == "message"
            and item.role == "user"
            and item.raw_text_content == transcript
        ] == [transcript]
        assert [
            item.raw_text_content
            for item in session.history.items
            if item.type == "message"
            and item.role == "user"
            and item.raw_text_content == transcript
        ] == [transcript]
    finally:
        await asyncio.wait_for(session.aclose(), timeout=5.0)


async def test_source_close_persists_a_turn_queued_before_update_starts() -> None:
    """Closing the source consumes its early-update queue exactly once."""
    transcript = "the turn queued before the update starts"
    speech = FakeUserSpeech(
        start_time=0.05,
        end_time=0.10,
        transcript=transcript,
        stt_delay=0.0,
    )
    session = AgentSession(
        stt=FakeSTT(fake_user_speeches=[speech]),
        vad=FakeVAD(
            fake_user_speeches=[speech], min_speech_duration=0.05, min_silence_duration=0.05
        ),
        llm=FakeLLM(),
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            endpointing=EndpointingOptions(min_delay=0.4, max_delay=0.4),
        ),
        aec_warmup_duration=None,
        session_close_transcript_timeout=0.01,
    )
    audio_input = FakeAudioInput()
    session.input.audio = audio_input
    source = _TurnRecorder("source")
    successor = _TurnRecorder("successor")
    update_task: asyncio.Task[None] | None = None
    close_task: asyncio.Task[None] | None = None

    try:
        await session.start(source)
        audio_input.push(0.1)
        await _wait_for(
            lambda: (
                source._activity is not None
                and source._activity._audio_recognition is not None
                and source._activity._audio_recognition._end_of_turn_task is not None
                and not source._activity._audio_recognition._end_of_turn_task.done()
            ),
            message="the source activity never opened the end-of-turn decision",
        )

        async with session._activity_lock:
            session.update_agent(successor)
            update_task = session._update_activity_atask
            assert update_task is not None
            await _wait_for(
                lambda: source._activity is not None and bool(source._activity._handoff_user_turns),
                message="the blocked source did not retain its crossing turn",
            )
            close_task = asyncio.create_task(session.aclose())
            await _wait_for(lambda: session._closing, message="session close never started")
            await asyncio.wait_for(close_task, timeout=5.0)

        await asyncio.wait_for(update_task, timeout=2.0)

        assert [
            item.raw_text_content
            for item in source.chat_ctx.items
            if item.type == "message"
            and item.role == "user"
            and item.raw_text_content == transcript
        ] == [transcript]
        assert [
            item.raw_text_content
            for item in session.history.items
            if item.type == "message"
            and item.role == "user"
            and item.raw_text_content == transcript
        ] == [transcript]
    finally:
        if close_task is None:
            await asyncio.wait_for(session.aclose(), timeout=5.0)
