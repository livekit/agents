from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterable, AsyncIterator
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from livekit import rtc
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    AgentStateChangedEvent,
    AgentTask,
    APIConnectionError,
    ConversationItemAddedEvent,
    FlushSentinel,
    LanguageCode,
    MetricsCollectedEvent,
    ModelSettings,
    NotGivenOr,
    RunContext,
    StopResponse,
    TurnHandlingOptions,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
    function_tool,
    inference,
    vad,
)
from livekit.agents.llm import (
    FunctionTool,
    FunctionToolCall,
    InputTranscriptionCompleted,
    RawFunctionTool,
    ToolFlag,
    Toolset,
)
from livekit.agents.llm.chat_context import ChatContext, ChatMessage
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType, STTError
from livekit.agents.utils import aio
from livekit.agents.voice import audio_recognition
from livekit.agents.voice.agent_activity import AgentActivity, _transcripts_equivalent
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.agents.voice.audio_recognition import AudioRecognition, _EndOfTurnInfo, _STTPipeline
from livekit.agents.voice.endpointing import BaseEndpointing
from livekit.agents.voice.events import AgentState, FunctionToolsExecutedEvent, UserState
from livekit.agents.voice.io import AudioOutput, AudioOutputCapabilities, PlaybackFinishedEvent
from livekit.agents.voice.tool_executor import UPDATE_TEMPLATE
from livekit.agents.voice.turn import _StreamingTurnDetector

from .fake_llm import FakeLLM
from .fake_session import FakeActions, create_session, run_session
from .fake_vad import FakeVAD

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


class MyAgent(Agent):
    def __init__(
        self,
        *,
        generate_reply_on_enter: bool = False,
        say_on_user_turn_completed: bool = False,
        on_user_turn_completed_delay: float = 0.0,
        turn_handling: NotGivenOr[TurnHandlingOptions] = NOT_GIVEN,
    ) -> None:
        super().__init__(
            instructions=("You are a helpful assistant."),
            turn_handling=turn_handling,
        )
        self.generate_reply_on_enter = generate_reply_on_enter
        self.say_on_user_turn_completed = say_on_user_turn_completed
        self.on_user_turn_completed_delay = on_user_turn_completed_delay

        self._close_session_task: asyncio.Task[None] | None = None

    async def on_enter(self) -> None:
        if self.generate_reply_on_enter:
            self.session.generate_reply(instructions="instructions:say hello to the user")

    @function_tool
    async def get_weather(self, location: str) -> str:
        """
        Called when the user asks about the weather.

        Args:
            location: The location to get the weather for
        """
        return f"The weather in {location} is sunny today."

    @function_tool
    async def goodbye(self) -> None:
        await self.session.generate_reply(instructions="instructions:say goodbye to the user")
        self._close_session_task = asyncio.create_task(self.session.aclose())

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        if self.say_on_user_turn_completed:
            self.session.say("session.say from on_user_turn_completed")

        if self.on_user_turn_completed_delay > 0.0:
            await asyncio.sleep(self.on_user_turn_completed_delay)


SESSION_TIMEOUT = 60.0


def test_realtime_user_input_transcription_preserves_item_id() -> None:
    captured_events: list[UserInputTranscribedEvent] = []

    class DummySession:
        def _user_input_transcribed(self, ev: UserInputTranscribedEvent) -> None:
            captured_events.append(ev)

    activity = object.__new__(AgentActivity)
    activity._session = DummySession()

    AgentActivity._on_input_audio_transcription_completed(
        activity,
        InputTranscriptionCompleted(
            item_id="item_123",
            transcript="hello",
            is_final=False,
        ),
    )

    assert len(captured_events) == 1
    assert captured_events[0].transcript == "hello"
    assert captured_events[0].is_final is False
    assert captured_events[0].item_id == "item_123"


@pytest.mark.parametrize(
    ("has_local_vad", "should_interrupt"),
    [
        (False, True),
        (True, False),
    ],
)
def test_interim_transcript_interrupts_only_without_local_vad(
    has_local_vad: bool, should_interrupt: bool
) -> None:
    captured_events: list[UserInputTranscribedEvent] = []
    activity = object.__new__(AgentActivity)
    activity._agent = SimpleNamespace(llm=NOT_GIVEN, vad=NOT_GIVEN)
    activity._session = SimpleNamespace(
        _text_only=False,
        llm=None,
        vad=Mock(spec=vad.VAD) if has_local_vad else None,
        _user_input_transcribed=captured_events.append,
    )
    activity._turn_detection = None
    activity._paused_speech = None
    activity._interrupt_by_audio_activity = Mock()

    activity.on_interim_transcript(
        SpeechEvent(
            type=SpeechEventType.INTERIM_TRANSCRIPT,
            alternatives=[SpeechData(text="hello", language=LanguageCode("en"))],
        ),
        speaking=None,
    )

    assert len(captured_events) == 1
    assert captured_events[0].transcript == "hello"
    if should_interrupt:
        activity._interrupt_by_audio_activity.assert_called_once_with()
    else:
        activity._interrupt_by_audio_activity.assert_not_called()


async def test_events_and_metrics() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Hello, how are you?", stt_delay=0.2)  # EOU at 2.5+0.5=3.0s
    actions.add_llm("I'm doing well, thank you!", ttft=0.1, duration=0.3)
    actions.add_tts(
        2.0, ttfb=0.2, duration=0.3
    )  # audio playout starts at 3.0+0.3+0.2=3.5s, ends at 5.5s

    session = create_session(actions)
    agent = MyAgent()

    user_state_events: list[UserStateChangedEvent] = []
    agent_state_events: list[AgentStateChangedEvent] = []
    metrics_events: list[MetricsCollectedEvent] = []
    conversation_events: list[ConversationItemAddedEvent] = []
    user_transcription_events: list[UserInputTranscribedEvent] = []

    session.on("user_state_changed", user_state_events.append)
    session.on("agent_state_changed", agent_state_events.append)
    session.on("metrics_collected", metrics_events.append)
    session.on("conversation_item_added", conversation_events.append)
    session.on("user_input_transcribed", user_transcription_events.append)

    t_origin = await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    # conversation_item_added
    assert len(conversation_events) == 3
    assert conversation_events[0].item.type == "agent_handoff"
    assert conversation_events[1].item.type == "message"
    assert conversation_events[1].item.role == "user"
    assert conversation_events[1].item.text_content == "Hello, how are you?"
    check_timestamp(conversation_events[1].created_at - t_origin, 3.0)
    assert conversation_events[2].item.type == "message"
    assert conversation_events[2].item.role == "assistant"
    assert conversation_events[2].item.text_content == "I'm doing well, thank you!"
    check_timestamp(conversation_events[2].created_at - t_origin, 5.5)

    # user_input_transcribed
    assert len(user_transcription_events) >= 1
    assert user_transcription_events[-1].transcript == "Hello, how are you?"
    assert user_transcription_events[-1].is_final is True
    check_timestamp(user_transcription_events[-1].created_at - t_origin, 2.7)

    # user_state_changed
    assert len(user_state_events) == 2
    check_timestamp(user_state_events[0].created_at - t_origin, 0.5)
    assert user_state_events[0].new_state == "speaking"
    check_timestamp(user_state_events[1].created_at - t_origin, 3.0)
    assert user_state_events[1].new_state == "listening"

    # agent_state_changed
    assert len(agent_state_events) == 4
    assert agent_state_events[0].old_state == "initializing"
    assert agent_state_events[0].new_state == "listening"
    assert agent_state_events[1].new_state == "thinking"
    check_timestamp(agent_state_events[1].created_at - t_origin, 3.0)
    assert agent_state_events[2].new_state == "speaking"
    check_timestamp(agent_state_events[2].created_at - t_origin, 3.5)
    assert agent_state_events[3].new_state == "listening"
    check_timestamp(agent_state_events[3].created_at - t_origin, 5.5)

    # metrics
    metrics_events = [ev for ev in metrics_events if ev.metrics.type != "vad_metrics"]
    assert len(metrics_events) == 3
    assert metrics_events[0].metrics.type == "eou_metrics"
    check_timestamp(metrics_events[0].metrics.end_of_utterance_delay, 0.5)
    check_timestamp(metrics_events[0].metrics.transcription_delay, 0.2)
    assert metrics_events[1].metrics.type == "llm_metrics"
    check_timestamp(metrics_events[1].metrics.ttft, 0.1)
    check_timestamp(metrics_events[1].metrics.duration, 0.3)
    assert metrics_events[2].metrics.type == "tts_metrics"
    check_timestamp(metrics_events[2].metrics.ttfb, 0.2)
    check_timestamp(metrics_events[2].metrics.audio_duration, 2.0)


async def test_llm_node_ttfs_anchors_on_synthesis_start() -> None:
    # the fake TTS only synthesizes once the LLM stream ends (~2.0s in), so the LLM streaming
    # time must land in ttfs and not be misattributed to the TTS ttfb
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Hello, how are you?", stt_delay=0.2)
    actions.add_llm("I'm doing well, thank you!", ttft=0.1, duration=2.0)
    actions.add_tts(1.0, ttfb=0.2, duration=0.3)

    session = create_session(actions)
    agent = MyAgent()

    conversation_events: list[ConversationItemAddedEvent] = []
    session.on("conversation_item_added", conversation_events.append)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assistant_messages = [
        ev.item
        for ev in conversation_events
        if ev.item.type == "message" and ev.item.role == "assistant"
    ]
    assert len(assistant_messages) == 1
    metrics = assistant_messages[0].metrics
    check_timestamp(metrics["llm_node_ttfs"], 2.0)
    check_timestamp(metrics["tts_node_ttfb"], 0.2)


async def test_tool_call() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "What's the weather in Tokyo?")
    actions.add_llm(
        content="Let me check the weather for you.",
        tool_calls=[
            FunctionToolCall(name="get_weather", arguments='{"location": "Tokyo"}', call_id="1")
        ],
    )
    actions.add_tts(2.0)  # audio for the content alongside the tool call
    actions.add_llm(
        content="The weather in Tokyo is sunny today.",
        input="The weather in Tokyo is sunny today.",
    )
    actions.add_tts(3.0)  # audio for the tool response

    session = create_session(actions)
    agent = MyAgent()

    agent_state_events: list[AgentStateChangedEvent] = []
    tool_executed_events: list[FunctionToolsExecutedEvent] = []
    playback_finished_events: list[PlaybackFinishedEvent] = []

    session.on("agent_state_changed", agent_state_events.append)
    session.on("function_tools_executed", tool_executed_events.append)
    session.output.audio.on("playback_finished", playback_finished_events.append)

    agent_speech_end_states: list[str] = []
    on_end_of_agent_speech = AudioRecognition._on_end_of_agent_speech

    def _record_agent_speech_end(
        recognition: AudioRecognition,
        *,
        ended_at: float,
    ) -> None:
        agent_speech_end_states.append(session.agent_state)
        on_end_of_agent_speech(
            recognition,
            ended_at=ended_at,
        )

    with patch.object(
        AudioRecognition,
        "_on_end_of_agent_speech",
        _record_agent_speech_end,
    ):
        t_origin = await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert len(playback_finished_events) == 2
    check_timestamp(playback_finished_events[0].playback_position, 2.0)
    check_timestamp(playback_finished_events[1].playback_position, 3.0)
    assert agent_speech_end_states[0] == "thinking"
    assert all(state == "listening" for state in agent_speech_end_states[1:])

    assert len(agent_state_events) == 6
    assert agent_state_events[0].old_state == "initializing"
    assert agent_state_events[0].new_state == "listening"
    assert agent_state_events[1].new_state == "thinking"
    assert agent_state_events[2].new_state == "speaking"
    assert (
        agent_state_events[3].new_state == "thinking"
    )  # from speaking to thinking when tool call is executed
    check_timestamp(agent_state_events[3].created_at - t_origin, 5.5)
    assert agent_state_events[4].new_state == "speaking"
    assert agent_state_events[5].new_state == "listening"

    assert len(tool_executed_events) == 1
    assert tool_executed_events[0].function_calls[0].name == "get_weather"
    assert tool_executed_events[0].function_calls[0].arguments == '{"location": "Tokyo"}'
    assert tool_executed_events[0].function_calls[0].call_id == "1"

    # chat context
    chat_ctx_items = agent.chat_ctx.items
    assert len(chat_ctx_items) == 7
    assert chat_ctx_items[0].type == "message"
    assert chat_ctx_items[0].role == "system"
    assert chat_ctx_items[1].type == "agent_config_update"
    assert chat_ctx_items[2].type == "message"
    assert chat_ctx_items[2].role == "user"
    assert chat_ctx_items[2].text_content == "What's the weather in Tokyo?"
    assert chat_ctx_items[3].type == "message"
    assert chat_ctx_items[3].role == "assistant"
    assert chat_ctx_items[3].text_content == "Let me check the weather for you."
    assert chat_ctx_items[4].type == "function_call"
    assert chat_ctx_items[4].name == "get_weather"
    assert chat_ctx_items[5].type == "function_call_output"
    assert chat_ctx_items[5].output == "The weather in Tokyo is sunny today."
    assert chat_ctx_items[6].type == "message"
    assert chat_ctx_items[6].role == "assistant"
    assert chat_ctx_items[6].text_content == "The weather in Tokyo is sunny today."


async def test_slow_tool_keeps_agent_thinking_after_filler() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Look up my order")
    actions.add_llm(
        content="Let me check.",
        tool_calls=[FunctionToolCall(name="lookup_order", arguments="{}", call_id="1")],
    )
    actions.add_tts(2.0)
    actions.add_tts(1.0, input="Just a moment.")
    actions.add_llm(content="Order 42 is on the way.", input="order 42 shipped")
    actions.add_tts(1.0)

    class SlowToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You are a helpful assistant.")

        @function_tool
        async def lookup_order(self, context: RunContext) -> str:
            async with context.with_filler("Just a moment.", delay=0.5):
                await asyncio.sleep(8.0)
            return "order 42 shipped"

    session = create_session(actions, extra_kwargs={"user_away_timeout": 3.0})

    agent_state_events: list[AgentStateChangedEvent] = []
    user_state_events: list[UserStateChangedEvent] = []
    session.on("agent_state_changed", agent_state_events.append)
    session.on("user_state_changed", user_state_events.append)

    await asyncio.wait_for(
        run_session(session, SlowToolAgent(), drain_delay=10), timeout=SESSION_TIMEOUT
    )

    # both the reply and the filler end while the tool is still running, so each
    # must hand back "thinking", not "listening" (which arms the user-away timer)
    speaking_ends = [ev.new_state for ev in agent_state_events if ev.old_state == "speaking"]
    assert speaking_ends == ["thinking", "thinking", "listening"]
    assert all(ev.new_state != "away" for ev in user_state_events)


async def test_tool_without_a_reply_returns_the_agent_to_listening() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Look up my order")
    actions.add_llm(
        content="Let me check.",
        tool_calls=[FunctionToolCall(name="silent_lookup", arguments="{}", call_id="1")],
    )
    actions.add_tts(2.0)

    class SilentToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You are a helpful assistant.")

        @function_tool
        async def silent_lookup(self, context: RunContext) -> None:
            """A slow tool that returns nothing, so no reply follows it."""
            await asyncio.sleep(8.0)

    session = create_session(actions, extra_kwargs={"user_away_timeout": 3.0})

    agent_state_events: list[AgentStateChangedEvent] = []
    session.on("agent_state_changed", agent_state_events.append)

    await asyncio.wait_for(
        run_session(session, SilentToolAgent(), drain_delay=20), timeout=SESSION_TIMEOUT
    )

    # the tool holds the agent in "thinking", and nothing follows it to hand the turn back
    transitions = [(ev.old_state, ev.new_state) for ev in agent_state_events]
    assert transitions[-2:] == [("speaking", "thinking"), ("thinking", "listening")]


@pytest.mark.parametrize("reset_away_timer", [False, True])
async def test_async_tool_defers_user_away_until_its_reply_lands(reset_away_timer: bool) -> None:
    """A background tool speaks for itself, so "away" must wait for that reply."""
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Look up my order")
    actions.add_llm(
        content="Let me check.",
        tool_calls=[FunctionToolCall(name="async_lookup", arguments="{}", call_id="1")],
    )
    actions.add_tts(2.0)
    actions.add_llm(
        content="I'm looking it up.",
        input=UPDATE_TEMPLATE.format(
            function_name="async_lookup", call_id="1", message="looking it up"
        ),
    )
    actions.add_tts(1.0)

    tool_returned_at = 0.0

    class AsyncToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You are a helpful assistant.")

        @function_tool
        async def async_lookup(self, context: RunContext) -> str:
            """Look an order up in the background."""
            nonlocal tool_returned_at
            await context.update("looking it up")
            await asyncio.sleep(10.0)
            if reset_away_timer:
                assert context.session.agent_state == "listening"
                context.session.reset_away_timer()
            await asyncio.sleep(10.0)
            tool_returned_at = time.time()
            return "order 42 shipped"

    session = create_session(actions, extra_kwargs={"user_away_timeout": 3.0})

    away_times: list[float] = []
    session.on(
        "user_state_changed",
        lambda ev: away_times.append(time.time()) if ev.new_state == "away" else None,
    )

    await asyncio.wait_for(
        run_session(session, AsyncToolAgent(), drain_delay=40), timeout=SESSION_TIMEOUT
    )

    # the agent is genuinely listening while the tool runs, so nothing but the tool holds
    # the window off; landing restarts it in full rather than firing on what was left
    assert tool_returned_at > 0
    assert len(away_times) == 1
    assert away_times[0] >= tool_returned_at + 3.0


async def test_async_tool_without_a_reply_still_restarts_the_away_window() -> None:
    """A tool returning None after an update files no reply, yet still restarts the window."""
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Look up my order")
    actions.add_llm(
        content="Let me check.",
        tool_calls=[FunctionToolCall(name="async_lookup", arguments="{}", call_id="1")],
    )
    actions.add_tts(2.0)
    actions.add_llm(
        content="I'm looking it up.",
        input=UPDATE_TEMPLATE.format(
            function_name="async_lookup", call_id="1", message="looking it up"
        ),
    )
    actions.add_tts(1.0)

    tool_returned_at = 0.0

    class AsyncToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You are a helpful assistant.")

        @function_tool
        async def async_lookup(self, context: RunContext) -> None:
            """Look an order up in the background and say nothing more."""
            nonlocal tool_returned_at
            await context.update("looking it up")
            await asyncio.sleep(20.0)
            tool_returned_at = time.time()

    session = create_session(actions, extra_kwargs={"user_away_timeout": 3.0})

    away_times: list[float] = []
    session.on(
        "user_state_changed",
        lambda ev: away_times.append(time.time()) if ev.new_state == "away" else None,
    )

    await asyncio.wait_for(
        run_session(session, AsyncToolAgent(), drain_delay=40), timeout=SESSION_TIMEOUT
    )

    assert len(away_times) == 1
    assert away_times[0] >= tool_returned_at + 3.0


@pytest.mark.parametrize(
    "resume_false_interruption, expected_interruption_time",
    [
        (False, 5.5),  # when vad event, 5 + 0.5
        (True, 5.5),  # pause/resume is disabled for fake audio output
    ],
)
async def test_interruption(
    resume_false_interruption: bool, expected_interruption_time: float
) -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Here is a long story for you ... the end.")
    actions.add_tts(10.0)  # playout starts at 3.5s
    actions.add_user_speech(5.0, 6.0, "Stop!", stt_delay=0.2)
    # interrupted at 5.5s, min_interruption_duration=0.5

    session = create_session(
        actions,
        turn_handling={"interruption": {"resume_false_interruption": resume_false_interruption}},
    )
    agent = MyAgent()

    agent_state_events: list[AgentStateChangedEvent] = []
    playback_finished_events: list[PlaybackFinishedEvent] = []
    session.on("agent_state_changed", agent_state_events.append)
    session.output.audio.on("playback_finished", playback_finished_events.append)

    t_origin = await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    chat_ctx_items = agent.chat_ctx.items
    assert len(chat_ctx_items) == 5
    assert chat_ctx_items[1].type == "agent_config_update"
    assert chat_ctx_items[3].type == "message"
    assert chat_ctx_items[3].role == "assistant"
    assert chat_ctx_items[3].interrupted is True

    assert len(agent_state_events) == 6
    assert agent_state_events[0].old_state == "initializing"
    assert agent_state_events[0].new_state == "listening"
    assert agent_state_events[1].new_state == "thinking"
    assert agent_state_events[2].new_state == "speaking"
    assert agent_state_events[3].new_state == "listening"
    check_timestamp(agent_state_events[3].created_at - t_origin, expected_interruption_time)
    assert agent_state_events[4].new_state == "thinking"
    check_timestamp(agent_state_events[4].created_at - t_origin, 6.5)
    assert agent_state_events[5].new_state == "listening"
    check_timestamp(agent_state_events[5].created_at - t_origin, 6.5)

    assert len(playback_finished_events) == 1
    assert playback_finished_events[0].interrupted is True
    if not resume_false_interruption:
        # fake audio output doesn't support pause/resume
        check_timestamp(playback_finished_events[0].playback_position, 2.0)


async def test_interrupted_say_uses_speech_start_timestamp() -> None:
    class SayOnEnterAgent(MyAgent):
        async def on_enter(self) -> None:
            self.session.say("I saw your application for a Honda")

    actions = FakeActions()
    actions.add_tts(10.0, input="I saw your application for a Honda")
    actions.add_user_speech(1.0, 1.2, "Who is this?")
    actions.add_llm("This is John.")
    actions.add_tts(1.0)

    session = create_session(actions)
    agent = SayOnEnterAgent()

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    messages = [item for item in agent.chat_ctx.items if item.type == "message"]
    assert [message.role for message in messages] == ["system", "assistant", "user", "assistant"]
    assert messages[1].interrupted is True
    assert messages[1].created_at == messages[1].metrics["started_speaking_at"]
    assert messages[2].text_content == "Who is this?"
    assert messages[3].text_content == "This is John."


async def test_interruption_options() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Here is a long story for you ... the end.")
    actions.add_tts(5.0)  # playout starts at 3.5s
    actions.add_user_speech(5.0, 6.0, "Stop!")
    actions.add_user_speech(6.5, 7.5, "ok, stop!", stt_delay=0.0)
    # it should interrupt at 7.5s after stt, playback position is 4.0s

    # test min_interruption_words
    session = create_session(
        actions,
        turn_handling={"interruption": {"min_words": 3}},
    )
    playback_finished_events: list[PlaybackFinishedEvent] = []
    session.output.audio.on("playback_finished", playback_finished_events.append)

    await asyncio.wait_for(run_session(session, MyAgent()), timeout=SESSION_TIMEOUT)

    assert len(playback_finished_events) == 1
    assert playback_finished_events[0].interrupted is True
    check_timestamp(playback_finished_events[0].playback_position, 4.0)

    # test allow_interruptions=False
    session = create_session(
        actions,
        turn_handling={"interruption": {"enabled": False}},
    )
    playback_finished_events.clear()
    session.output.audio.on("playback_finished", playback_finished_events.append)

    await asyncio.wait_for(run_session(session, MyAgent()), timeout=SESSION_TIMEOUT)

    assert len(playback_finished_events) == 1
    assert playback_finished_events[0].interrupted is False
    check_timestamp(playback_finished_events[0].playback_position, 5.0)


async def test_interruption_by_text_input() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Here is a long story for you ... the end.")
    actions.add_tts(15.0)
    actions.add_llm("Ok, I'll stop now.", input="stop from text input")
    actions.add_tts(2.0)

    session = create_session(actions)
    agent = MyAgent()

    agent_state_events: list[AgentStateChangedEvent] = []
    playback_finished_events: list[PlaybackFinishedEvent] = []
    session.on("agent_state_changed", agent_state_events.append)
    session.output.audio.on("playback_finished", playback_finished_events.append)

    def fake_text_input() -> None:
        session.interrupt()
        session.generate_reply(user_input="stop from text input")

    asyncio.get_event_loop().call_later(5, fake_text_input)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert len(playback_finished_events) == 2
    assert playback_finished_events[0].interrupted is True

    assert len(agent_state_events) == 6
    assert agent_state_events[0].old_state == "initializing"
    assert agent_state_events[0].new_state == "listening"
    assert agent_state_events[1].new_state == "thinking"
    assert agent_state_events[2].new_state == "speaking"
    # interrupted by text while speaking -> straight to thinking for the new reply
    assert agent_state_events[3].new_state == "thinking"
    assert agent_state_events[4].new_state == "speaking"
    assert agent_state_events[5].new_state == "listening"

    chat_ctx_items = agent.chat_ctx.items
    assert len(chat_ctx_items) == 6
    assert chat_ctx_items[0].type == "message"
    assert chat_ctx_items[0].role == "system"
    assert chat_ctx_items[1].type == "agent_config_update"
    assert chat_ctx_items[2].type == "message"
    assert chat_ctx_items[2].role == "user"
    assert chat_ctx_items[2].text_content == "Tell me a story."
    assert chat_ctx_items[3].type == "message"
    assert chat_ctx_items[3].role == "assistant"
    assert chat_ctx_items[3].interrupted is True  # assistant message should be before text input
    assert chat_ctx_items[4].type == "message"
    assert chat_ctx_items[4].role == "user"
    assert chat_ctx_items[4].text_content == "stop from text input"
    assert chat_ctx_items[5].type == "message"
    assert chat_ctx_items[5].role == "assistant"
    assert chat_ctx_items[5].text_content == "Ok, I'll stop now."


@pytest.mark.parametrize(
    "resume_false_interruption, expected_interruption_time",
    [
        (False, 3.5),  # 3 + 0.5
        (True, 3.5),  # pause/resume is disabled for fake audio output
    ],
)
async def test_interruption_before_speaking(
    resume_false_interruption: bool, expected_interruption_time: float
) -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Here is a long story for you ... the end.", duration=1.0)
    actions.add_tts(10.0)
    actions.add_user_speech(3.0, 4.0, "Stop!", stt_delay=0.2)

    session = create_session(
        actions,
        turn_handling={"interruption": {"resume_false_interruption": resume_false_interruption}},
    )
    agent = MyAgent()

    agent_state_events: list[AgentStateChangedEvent] = []
    playback_finished_events: list[PlaybackFinishedEvent] = []
    session.on("agent_state_changed", agent_state_events.append)
    session.output.audio.on("playback_finished", playback_finished_events.append)

    t_origin = await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert len(agent_state_events) == 5
    assert agent_state_events[0].old_state == "initializing"
    assert agent_state_events[0].new_state == "listening"
    assert agent_state_events[1].new_state == "thinking"  # without speaking state
    assert agent_state_events[2].new_state == "listening"
    check_timestamp(
        agent_state_events[2].created_at - t_origin, expected_interruption_time
    )  # interrupted at 3.5s
    assert agent_state_events[3].new_state == "thinking"
    assert agent_state_events[4].new_state == "listening"

    assert len(playback_finished_events) == 0

    assert len(agent.chat_ctx.items) == 4
    assert agent.chat_ctx.items[0].type == "message"
    assert agent.chat_ctx.items[0].role == "system"
    assert agent.chat_ctx.items[1].type == "agent_config_update"
    assert agent.chat_ctx.items[2].type == "message"
    assert agent.chat_ctx.items[2].role == "user"
    assert agent.chat_ctx.items[2].text_content == "Tell me a story."
    # an interrupted reply with no text is not recorded
    assert agent.chat_ctx.items[3].type == "message"
    assert agent.chat_ctx.items[3].role == "user"
    assert agent.chat_ctx.items[3].text_content == "Stop!"


async def test_interrupt_before_speaking_with_pausable_audio() -> None:
    """A user turn starting while thinking pauses the output so the stale reply never speaks."""
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Here is a long story for you ... the end.", duration=1.0)
    actions.add_tts(10.0)
    actions.add_user_speech(3.0, 4.0, "Stop!", stt_delay=0.2)

    session = create_session(actions, can_pause_audio=True)
    agent = MyAgent()

    agent_state_events: list[AgentStateChangedEvent] = []
    playback_finished_events: list[PlaybackFinishedEvent] = []
    session.on("agent_state_changed", agent_state_events.append)
    session.output.audio.on("playback_finished", playback_finished_events.append)

    t_origin = await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    # core assertion: the stale reply never promotes to "speaking"
    assert not any(ev.new_state == "speaking" for ev in agent_state_events), (
        "stale reply should have been paused before the first frame reached the transport"
    )

    # state sequence mirrors test_interruption_before_speaking (can_pause=False variant),
    # proving the pause path is observationally equivalent to the interrupt path
    assert len(agent_state_events) == 5
    assert agent_state_events[0].old_state == "initializing"
    assert agent_state_events[0].new_state == "listening"
    assert agent_state_events[1].new_state == "thinking"
    assert agent_state_events[2].new_state == "listening"
    check_timestamp(agent_state_events[2].created_at - t_origin, 3.5)

    # nothing audible reached the transport — the pause cleanup emits a single
    # playback_finished with interrupted=True and playback_position=0
    assert len(playback_finished_events) == 1
    assert playback_finished_events[0].interrupted is True
    assert playback_finished_events[0].playback_position == 0.0

    # stale assistant reply is dropped; chat_ctx holds user turn 1 and (after
    # the on_final_transcript commit) user turn 2
    user_messages = [
        item for item in agent.chat_ctx.items if item.type == "message" and item.role == "user"
    ]
    assert [m.text_content for m in user_messages] == ["Tell me a story.", "Stop!"]
    assert not any(
        item.type == "message" and item.role == "assistant" for item in agent.chat_ctx.items
    )


async def test_forced_interrupt_of_unplayed_speeches_returns_to_listening() -> None:
    """A reply cancelled while still queued must not leave the agent in "thinking"."""

    class StalledAudioOutput(AudioOutput):
        def __init__(self) -> None:
            super().__init__(
                label="Stalled",
                capabilities=AudioOutputCapabilities(pause=False),
                sample_rate=24000,
            )

        async def capture_frame(self, frame: rtc.AudioFrame) -> None:
            await super().capture_frame(frame)
            await asyncio.Event().wait()

        def flush(self) -> None:
            super().flush()

        def clear_buffer(self) -> None:
            pass

    async def frames() -> AsyncIterable[rtc.AudioFrame]:
        yield rtc.AudioFrame(b"\x00\x00" * 480, 24000, 1, 480)

    session = AgentSession[None](llm=FakeLLM(), aec_warmup_duration=None)
    session.output.audio = audio = StalledAudioOutput()
    await session.start(Agent(instructions="test"))

    said = session.say("One moment.", audio=frames(), add_to_chat_ctx=False)
    reply = session.generate_reply(user_input="A question.")
    await asyncio.sleep(0.5)
    session.interrupt(force=True)
    await asyncio.sleep(10.0)

    assert said.done() and reply.done()
    assert session.agent_state == "listening"

    # release the playout waiters the stalled output never finishes
    for _ in range(audio._pending_playback_count):
        audio.on_playback_finished(playback_position=0.0, interrupted=True)
    await session.aclose()


async def test_false_interruption_before_speaking_resumes() -> None:
    """Brief VAD-only noise while thinking pauses, then resumes the reply on VAD EOS."""
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Here is a short reply.", ttft=0.05, duration=0.05)
    actions.add_tts(5.0, ttfb=0.05, duration=0.05)
    # brief VAD-only noise — same shape as the can_pause=False test, different capability
    actions.add_user_speech(3.0, 3.3, "")

    session = create_session(actions, can_pause_audio=True)
    agent = MyAgent()

    agent_state_events: list[AgentStateChangedEvent] = []
    playback_finished_events: list[PlaybackFinishedEvent] = []
    session.on("agent_state_changed", agent_state_events.append)
    session.output.audio.on("playback_finished", playback_finished_events.append)

    t_origin = await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    # the agent resumes and speaks the reply after the false interruption clears
    speaking_events = [ev for ev in agent_state_events if ev.new_state == "speaking"]
    assert len(speaking_events) == 1

    # playout was postponed: the noise ran 3.0–3.3s, so "speaking" should fire at
    # ~3.8s (resume on VAD EOS=3.3s + 0.5s min_silence_duration)
    check_timestamp(speaking_events[0].created_at - t_origin, 3.8)

    # the reply plays to completion (not interrupted); playback_position covers the
    # full audio duration
    assert len(playback_finished_events) == 1
    assert playback_finished_events[0].interrupted is False
    check_timestamp(playback_finished_events[0].playback_position, 5.0)


async def test_generate_reply() -> None:
    """generate_reply in on_enter and a tool, say in on_user_turn_completed."""

    actions = FakeActions()
    # llm and tts response for generate_reply() and say()
    actions.add_llm("What can I do for you!", input="instructions:say hello to the user")
    actions.add_tts(2.0)
    actions.add_llm("Goodbye! have a nice day!", input="instructions:say goodbye to the user")
    actions.add_tts(3.0)
    actions.add_tts(1.0, ttfb=0, input="session.say from on_user_turn_completed")
    # user speech
    actions.add_user_speech(3.0, 4.0, "bye")
    actions.add_llm(
        content="",
        tool_calls=[FunctionToolCall(name="goodbye", arguments="", call_id="1")],
    )
    # tool started at 4.5(EOU) + 1.0(session.say)
    # tool finished at 5.5 + 0.5 (second LLM + TTS) + 3 (audio)

    session = create_session(actions)
    agent = MyAgent(generate_reply_on_enter=True, say_on_user_turn_completed=True)

    conversation_events: list[ConversationItemAddedEvent] = []
    playback_finished_events: list[PlaybackFinishedEvent] = []
    tool_executed_events: list[FunctionToolsExecutedEvent] = []
    session.on("conversation_item_added", conversation_events.append)
    session.on("function_tools_executed", tool_executed_events.append)
    session.output.audio.on("playback_finished", playback_finished_events.append)

    t_origin = await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    # playback_finished
    assert len(playback_finished_events) == 3
    assert playback_finished_events[0].interrupted is False
    check_timestamp(playback_finished_events[0].playback_position, 2.0)
    assert playback_finished_events[1].interrupted is False
    check_timestamp(playback_finished_events[1].playback_position, 1.0)
    assert playback_finished_events[2].interrupted is False
    check_timestamp(playback_finished_events[2].playback_position, 3.0)

    # function_tools_executed
    assert len(tool_executed_events) == 1
    assert tool_executed_events[0].function_calls[0].name == "goodbye"

    # conversation_item_added
    assert len(conversation_events) == 5
    assert conversation_events[0].item.type == "agent_handoff"
    assert conversation_events[1].item.type == "message"
    assert conversation_events[1].item.role == "assistant"
    assert conversation_events[1].item.text_content == "What can I do for you!"
    check_timestamp(conversation_events[1].created_at - t_origin, 2.5)
    assert conversation_events[2].item.type == "message"
    assert conversation_events[2].item.role == "user"
    assert conversation_events[2].item.text_content == "bye"
    check_timestamp(conversation_events[2].created_at - t_origin, 4.5)
    assert conversation_events[3].item.type == "message"
    assert conversation_events[3].item.role == "assistant"
    assert conversation_events[3].item.text_content == "session.say from on_user_turn_completed"
    check_timestamp(conversation_events[3].created_at - t_origin, 5.5, max_abs_diff=1.0)
    assert conversation_events[4].item.type == "message"
    assert conversation_events[4].item.role == "assistant"
    assert conversation_events[4].item.text_content == "Goodbye! have a nice day!"
    check_timestamp(conversation_events[4].created_at - t_origin, 9.0, max_abs_diff=1.0)

    # chat context
    assert len(agent.chat_ctx.items) == 8
    assert agent.chat_ctx.items[0].type == "message"
    assert agent.chat_ctx.items[0].role == "system"
    assert agent.chat_ctx.items[1].type == "agent_config_update"
    assert agent.chat_ctx.items[2].type == "message"
    assert agent.chat_ctx.items[2].role == "assistant"
    assert agent.chat_ctx.items[2].text_content == "What can I do for you!"
    assert agent.chat_ctx.items[3].type == "message"
    assert agent.chat_ctx.items[3].role == "user"
    assert agent.chat_ctx.items[3].text_content == "bye"
    assert agent.chat_ctx.items[4].type == "message"
    assert agent.chat_ctx.items[4].role == "assistant"
    assert agent.chat_ctx.items[4].text_content == "session.say from on_user_turn_completed"
    assert agent.chat_ctx.items[5].type == "function_call"
    assert agent.chat_ctx.items[6].type == "message"
    assert agent.chat_ctx.items[6].role == "assistant"
    assert agent.chat_ctx.items[6].text_content == "Goodbye! have a nice day!"
    assert agent.chat_ctx.items[7].type == "function_call_output"


class _OnEnterToolset(Toolset):
    def __init__(self) -> None:
        end_call = function_tool(
            self._end_call, name="end_call", description="ends", flags=ToolFlag.IGNORE_ON_ENTER
        )
        keep = function_tool(self._keep, name="keep", description="a normal tool")
        super().__init__(id="ts", tools=[end_call, keep])

    async def _end_call(self) -> None: ...

    async def _keep(self) -> str:
        return "kept"


@function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
async def _bare_ignored() -> None:
    """A bare flagged tool."""


def _record_offered_tools(session: AgentSession) -> list[set[str]]:
    offered: list[set[str]] = []
    orig_chat = session.llm.chat

    def _recording_chat(*, chat_ctx, tools=None, **kwargs):  # type: ignore[no-untyped-def]
        offered.append(
            {t.info.name for t in (tools or []) if isinstance(t, (FunctionTool, RawFunctionTool))}
        )
        return orig_chat(chat_ctx=chat_ctx, tools=tools, **kwargs)

    session.llm.chat = _recording_chat  # type: ignore[method-assign]
    return offered


@pytest.mark.parametrize("with_task", [False, True])
async def test_on_enter_hides_ignore_on_enter_tools(with_task: bool) -> None:
    """Flagged tools are hidden inside on_enter, also after on_enter awaits an AgentTask."""

    class _Task(AgentTask[None]):
        def __init__(self) -> None:
            super().__init__(instructions="complete the setup")

        async def on_enter(self) -> None:
            self.complete(None)

    class _Agent(Agent):
        def __init__(self) -> None:
            super().__init__(
                instructions="you are helpful", tools=[_OnEnterToolset(), _bare_ignored]
            )

        async def on_enter(self) -> None:
            if with_task:
                await _Task()
                # the fake input sends only one frame; prime it again after resume
                session.input.audio.push(0.1)
            self.session.generate_reply(instructions="instructions:say hello to the user")

    actions = FakeActions()
    actions.add_llm("Hello!", input="instructions:say hello to the user")
    actions.add_tts(1.0)
    actions.add_user_speech(2.0, 3.0, "hi there")
    actions.add_llm("How can I help?")
    actions.add_tts(1.0)

    session = create_session(actions)

    captured = _record_offered_tools(session)

    await asyncio.wait_for(run_session(session, _Agent()), timeout=SESSION_TIMEOUT)

    assert len(captured) == 2
    # on_enter reply: flagged tools hidden, normal tool still offered
    assert captured[0] == {"keep"}
    # a later (non-on_enter) turn sees every tool again
    assert captured[1] == {"keep", "end_call", "_bare_ignored"}


async def test_on_enter_marker_does_not_follow_a_handoff_back() -> None:
    """Handing back to an agent that handed off from on_enter must not keep its tools hidden."""

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def end_call() -> None:
        """Ends the call."""

    @function_tool
    async def keep() -> None:
        """A normal tool."""

    class _Router(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="you route the caller", tools=[end_call, keep])
            self.entered = 0

        async def on_enter(self) -> None:
            self.entered += 1
            if self.entered == 1:
                await self.session.generate_reply(instructions="instructions:greet the user")
                self.session.update_agent(specialist)

    class _Specialist(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="you are a specialist")

        async def on_user_turn_completed(
            self, turn_ctx: ChatContext, new_message: ChatMessage
        ) -> None:
            self.session.update_agent(router)
            raise StopResponse()

    router = _Router()
    specialist = _Specialist()

    actions = FakeActions()
    actions.add_llm("Hello!", input="instructions:greet the user")
    actions.add_tts(1.0)
    actions.add_user_speech(3.0, 4.0, "take me back")
    actions.add_user_speech(6.0, 7.0, "goodbye")
    actions.add_llm("Bye!")
    actions.add_tts(1.0)

    # no preemptive reply from the specialist, so the only LLM calls are the router's
    session = create_session(actions, turn_handling={"preemptive_generation": {"enabled": False}})

    captured = _record_offered_tools(session)

    await asyncio.wait_for(run_session(session, router), timeout=SESSION_TIMEOUT)

    assert router.entered == 2
    assert len(captured) == 2
    greeting, user_turn = captured
    assert greeting == {"keep"}
    # the router's turn after the handoff back sees every tool again
    assert user_turn == {"keep", "end_call"}


async def test_on_enter_hides_tools_in_nested_tool_reply() -> None:
    """The tool reply of an on_enter reply also hides the flagged tools."""

    class _Agent(Agent):
        def __init__(self) -> None:
            super().__init__(
                instructions="you are helpful", tools=[_OnEnterToolset(), _bare_ignored]
            )

        async def on_enter(self) -> None:
            self.session.generate_reply(instructions="instructions:say hello to the user")

    actions = FakeActions()
    # on_enter reply calls the visible `keep` tool instead of speaking, spawning a follow-up
    actions.add_llm(
        "",
        tool_calls=[FunctionToolCall(name="keep", arguments="{}", call_id="1")],
        input="instructions:say hello to the user",
    )
    # tool-response follow-up (keyed on the `keep` return value)
    actions.add_llm("Hello there!", input="kept")
    actions.add_tts(1.0)
    # a user turn ends the run and confirms tools are restored afterwards
    actions.add_user_speech(4.0, 5.0, "hi there")
    actions.add_llm("How can I help?")
    actions.add_tts(1.0)

    session = create_session(actions)

    captured = _record_offered_tools(session)

    await asyncio.wait_for(run_session(session, _Agent()), timeout=SESSION_TIMEOUT)

    assert len(captured) == 3
    greeting, tool_reply, user_turn = captured
    # both the greeting reply and its nested tool-response follow-up hide the flagged tools
    assert greeting == {"keep"}
    assert tool_reply == {"keep"}
    # the later (non-on_enter) user turn sees every tool again
    assert user_turn == {"keep", "end_call", "_bare_ignored"}


_SIP = rtc.ParticipantKind.PARTICIPANT_KIND_SIP
_STANDARD = rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD


@pytest.mark.parametrize(
    ("explicit", "kind", "attributes", "simulation", "expected_duration"),
    [
        pytest.param(NOT_GIVEN, _SIP, {}, False, None, id="outbound-sip"),
        pytest.param(NOT_GIVEN, _SIP, {"sip.ruleID": "SDR_inbound"}, False, 3.0, id="inbound-sip"),
        pytest.param(NOT_GIVEN, _STANDARD, {}, False, 3.0, id="standard"),
        pytest.param(None, _SIP, {}, False, None, id="explicit-none"),
        pytest.param(0.0, _SIP, {}, False, 0.0, id="explicit-zero"),
        pytest.param(1.5, _SIP, {}, False, 1.5, id="explicit-outbound-sip"),
        pytest.param(NOT_GIVEN, _STANDARD, {}, True, None, id="simulation"),
        pytest.param(1.5, _STANDARD, {}, True, 1.5, id="explicit-simulation"),
    ],
)
def test_aec_warmup_duration_resolves_from_call_type(
    explicit: NotGivenOr[float | None],
    kind: rtc.ParticipantKind.ValueType,
    attributes: dict[str, str],
    simulation: bool,
    expected_duration: float | None,
) -> None:
    session = AgentSession(vad=None, aec_warmup_duration=explicit)
    participant = MagicMock(spec=rtc.RemoteParticipant)
    participant.kind = kind
    participant.attributes = attributes

    with patch(
        "livekit.agents.job.current_simulation", return_value=object() if simulation else None
    ):
        session._on_room_io_participant_linked(participant)

    assert session.options.aec_warmup_duration == expected_duration
    assert session._aec_warmup_remaining == (expected_duration or 0.0)


def test_outbound_sip_cancels_aec_warmup_that_already_started() -> None:
    session = AgentSession(vad=None)
    timer = MagicMock(spec=asyncio.TimerHandle)
    session._aec_warmup_timer = timer
    participant = MagicMock(spec=rtc.RemoteParticipant)
    participant.kind = rtc.ParticipantKind.PARTICIPANT_KIND_SIP
    participant.attributes = {}

    session._on_room_io_participant_linked(participant)

    timer.cancel.assert_called_once()
    assert session._aec_warmup_timer is None


@pytest.mark.parametrize(
    ("aec_warmup_duration", "interrupted_at"),
    [
        # warmup (3.5s + 3.0s) blocks the VAD path, so the EOU at 5.0 + 0.5 interrupts
        (3.0, 5.5),
        # without warmup VAD interrupts at 4.0 + 0.5
        (None, 4.5),
    ],
)
async def test_aec_warmup_blocks_vad_interruption(
    aec_warmup_duration: float | None, interrupted_at: float
) -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Here is a long story for you ... the end.")
    actions.add_tts(15.0)  # playout starts at 3.5s
    actions.add_user_speech(4.0, 5.0, "Stop!", stt_delay=0.2)

    session = create_session(actions, extra_kwargs={"aec_warmup_duration": aec_warmup_duration})
    agent_state_events: list[AgentStateChangedEvent] = []
    playback_finished_events: list[PlaybackFinishedEvent] = []
    session.on("agent_state_changed", agent_state_events.append)
    session.output.audio.on("playback_finished", playback_finished_events.append)

    t_origin = await asyncio.wait_for(run_session(session, MyAgent()), timeout=SESSION_TIMEOUT)

    assert len(playback_finished_events) == 1
    assert playback_finished_events[0].interrupted is True
    assert [ev.new_state for ev in agent_state_events[:3]] == ["listening", "thinking", "speaking"]
    speaking_to_listening = next(e for e in agent_state_events[3:] if e.new_state == "listening")
    check_timestamp(speaking_to_listening.created_at - t_origin, interrupted_at)


async def test_backchannel_boundary_suppresses_start_boundary_backchannel() -> None:
    actions = FakeActions()
    session = create_session(
        actions,
        turn_handling={"interruption": {"backchannel_boundary": (0.05, 0.0)}},
    )
    hooks = _TestRecognitionHooks()
    recognition = _recognition(session, turn_detection="vad", hooks=hooks)

    try:
        recognition._on_start_of_agent_speech(started_at=time.time())
        # backchannels during the cooldown are dropped
        event = _backchannel_event()
        hooks.on_overlap_speech(event)
        recognition._on_overlap_speech_event(event)
        assert hooks.interruptions == []

        # a real interruption during the cooldown must still fire
        event = _interruption_event()
        hooks.on_overlap_speech(event)
        recognition._on_overlap_speech_event(event)
        assert len(hooks.interruptions) == 1

        # after cooldown, both event types behave normally
        await asyncio.sleep(0.06)
        backchannel_event = _backchannel_event()
        hooks.on_overlap_speech(backchannel_event)
        recognition._on_overlap_speech_event(backchannel_event)
        interruption_event = _interruption_event()
        hooks.on_overlap_speech(interruption_event)
        recognition._on_overlap_speech_event(interruption_event)
        assert len(hooks.interruptions) == 2
    finally:
        await _close_test_session(session)


@pytest.mark.parametrize("has_vad_stream", [True, False])
async def test_stt_eos_resets_vad(has_vad_stream: bool) -> None:
    """STT EOS flushes an active VAD stream in place, or recreates VAD when none is running."""
    recognition = _recognition(
        create_session(FakeActions()), turn_detection="stt", min_delay=0.0, max_delay=0.0
    )
    recognition._speaking = True
    recognition._vad_speech_started = True
    recognition._vad = MagicMock()
    vad_stream = MagicMock() if has_vad_stream else None
    recognition._vad_stream = vad_stream

    try:
        with patch.object(recognition, "_update_vad") as update_vad:
            await recognition._on_stt_event(SpeechEvent(type=SpeechEventType.END_OF_SPEECH))

        if vad_stream is not None:
            vad_stream.flush.assert_called_once_with()
            update_vad.assert_not_called()
            assert recognition._vad_stream is vad_stream
        else:
            update_vad.assert_called_once_with(recognition._vad)
    finally:
        if recognition._end_of_turn_task is not None:
            await aio.cancel_and_wait(recognition._end_of_turn_task)
        await _close_test_session(recognition._session)


@contextlib.asynccontextmanager
async def _stt_turn_recognition(
    *, min_delay: float
) -> AsyncIterator[tuple[AudioRecognition, list[str]]]:
    committed: list[str] = []

    class _CommitRecordingHooks(_TestRecognitionHooks):
        def on_end_of_turn(self, info: _EndOfTurnInfo) -> bool:
            committed.append(info.new_transcript)
            return True

    recognition = _recognition(
        create_session(FakeActions()),
        turn_detection="stt",
        hooks=_CommitRecordingHooks(),
        min_delay=min_delay,
        max_delay=min_delay,
        stt=MagicMock(),
    )
    try:
        yield recognition, committed
    finally:
        if recognition._end_of_turn_task is not None:
            await aio.cancel_and_wait(recognition._end_of_turn_task)
        await _close_test_session(recognition._session)


async def _send_stt_event(
    recognition: AudioRecognition, type: SpeechEventType, text: str | None = None
) -> None:
    alternatives = [SpeechData(text=text, language=LanguageCode(""))] if text else []
    await recognition._on_stt_event(SpeechEvent(type=type, alternatives=alternatives))
    await asyncio.sleep(0.05)


async def _send_vad_end(recognition: AudioRecognition) -> None:
    await recognition._on_vad_event(
        vad.VADEvent(
            type=vad.VADEventType.END_OF_SPEECH,
            samples_index=0,
            timestamp=time.time(),
            speech_duration=0.5,
            silence_duration=0.3,
        )
    )
    await asyncio.sleep(0.05)


@pytest.mark.parametrize(
    ("min_delay", "first_turns"),
    [
        pytest.param(0.0, ["I need the OTP.", "For the guard."], id="committed-before-final"),
        pytest.param(0.5, ["I need the OTP. For the guard."], id="final-within-delay"),
    ],
)
async def test_stt_final_after_a_later_end_of_speech_still_ends_a_turn(
    min_delay: float, first_turns: list[str]
) -> None:
    async with _stt_turn_recognition(min_delay=min_delay) as (recognition, committed):
        await _send_stt_event(recognition, SpeechEventType.START_OF_SPEECH)
        await _send_stt_event(recognition, SpeechEventType.END_OF_SPEECH)
        await _send_stt_event(recognition, SpeechEventType.START_OF_SPEECH)
        await _send_stt_event(recognition, SpeechEventType.FINAL_TRANSCRIPT, "I need the OTP.")
        await _send_stt_event(recognition, SpeechEventType.END_OF_SPEECH)
        await _send_stt_event(recognition, SpeechEventType.FINAL_TRANSCRIPT, "For the guard.")
        await asyncio.sleep(min_delay + 0.1)

        assert committed == first_turns
        assert recognition._audio_transcript == ""

        await _send_stt_event(recognition, SpeechEventType.START_OF_SPEECH)
        await _send_stt_event(recognition, SpeechEventType.FINAL_TRANSCRIPT, "Thanks.")
        await _send_stt_event(recognition, SpeechEventType.END_OF_SPEECH)
        await asyncio.sleep(min_delay + 0.1)

        assert committed == [*first_turns, "Thanks."]


async def test_stt_resume_supersedes_a_pending_end_of_speech() -> None:
    """A provider START_OF_SPEECH reopens the turn; only its next END_OF_SPEECH ends it."""
    async with _stt_turn_recognition(min_delay=0.0) as (recognition, committed):
        await _send_stt_event(recognition, SpeechEventType.START_OF_SPEECH)
        await _send_stt_event(recognition, SpeechEventType.END_OF_SPEECH)
        await _send_stt_event(recognition, SpeechEventType.START_OF_SPEECH)
        await _send_stt_event(recognition, SpeechEventType.FINAL_TRANSCRIPT, "I need the OTP.")
        await _send_vad_end(recognition)
        assert committed == []

        await _send_vad_end(recognition)
        await _send_stt_event(recognition, SpeechEventType.FINAL_TRANSCRIPT, "For the")
        assert committed == []

        await _send_stt_event(recognition, SpeechEventType.END_OF_SPEECH)
        assert committed == ["I need the OTP. For the"]


async def test_stt_resume_after_final_is_not_ended_by_a_vad_pause() -> None:
    """Deepgram Flux order: EndOfTurn sends FINAL_TRANSCRIPT then END_OF_SPEECH."""
    async with _stt_turn_recognition(min_delay=0.3) as (recognition, committed):
        await _send_stt_event(recognition, SpeechEventType.START_OF_SPEECH)
        await _send_stt_event(recognition, SpeechEventType.FINAL_TRANSCRIPT, "I live in Cantu.")
        await _send_stt_event(recognition, SpeechEventType.END_OF_SPEECH)
        # the user resumes within min_delay
        await _send_stt_event(recognition, SpeechEventType.START_OF_SPEECH)
        await _send_stt_event(recognition, SpeechEventType.INTERIM_TRANSCRIPT, "in the province")
        # a short pause: VAD ends before the provider does
        await _send_vad_end(recognition)
        await asyncio.sleep(0.4)
        assert committed == []

        await _send_stt_event(
            recognition, SpeechEventType.FINAL_TRANSCRIPT, "in the province of Como."
        )
        await _send_stt_event(recognition, SpeechEventType.END_OF_SPEECH)
        await asyncio.sleep(0.4)
        assert committed == ["I live in Cantu. in the province of Como."]


async def test_stt_resume_without_words_is_ended_by_the_next_end_of_speech() -> None:
    """A resume that transcribes nothing still commits the pending text on the provider's end."""
    async with _stt_turn_recognition(min_delay=0.3) as (recognition, committed):
        await _send_stt_event(recognition, SpeechEventType.START_OF_SPEECH)
        await _send_stt_event(recognition, SpeechEventType.FINAL_TRANSCRIPT, "I live in Cantu.")
        await _send_stt_event(recognition, SpeechEventType.END_OF_SPEECH)
        await _send_stt_event(recognition, SpeechEventType.START_OF_SPEECH)
        await asyncio.sleep(0.4)
        assert committed == []

        # e.g. a Flux EndOfTurn without words: no final, only END_OF_SPEECH
        await _send_stt_event(recognition, SpeechEventType.END_OF_SPEECH)
        await asyncio.sleep(0.4)
        assert committed == ["I live in Cantu."]


async def test_stt_final_after_resume_waits_for_the_provider_end_of_speech() -> None:
    """A final for the resumed speech doesn't let a VAD pause end the turn early."""
    async with _stt_turn_recognition(min_delay=0.3) as (recognition, committed):
        await _send_stt_event(recognition, SpeechEventType.START_OF_SPEECH)
        await _send_stt_event(recognition, SpeechEventType.FINAL_TRANSCRIPT, "I need")
        await _send_stt_event(recognition, SpeechEventType.END_OF_SPEECH)
        # the user resumes within min_delay
        await _send_stt_event(recognition, SpeechEventType.START_OF_SPEECH)
        # a mid-turn final for the resumed speech
        await _send_stt_event(recognition, SpeechEventType.FINAL_TRANSCRIPT, "the OTP")
        await _send_vad_end(recognition)
        await asyncio.sleep(0.4)
        assert committed == []

        await _send_stt_event(recognition, SpeechEventType.FINAL_TRANSCRIPT, "for the guard")
        await _send_stt_event(recognition, SpeechEventType.END_OF_SPEECH)
        await asyncio.sleep(0.4)
        assert committed == ["I need the OTP for the guard"]


async def test_stt_pending_end_of_turn_survives_vad_only_speech() -> None:
    """VAD speech the provider ignores cancels the pending end of turn; its VAD end resumes it."""
    async with _stt_turn_recognition(min_delay=0.3) as (recognition, committed):
        await _send_stt_event(recognition, SpeechEventType.START_OF_SPEECH)
        await _send_stt_event(recognition, SpeechEventType.FINAL_TRANSCRIPT, "I live in Cantu.")
        await _send_stt_event(recognition, SpeechEventType.END_OF_SPEECH)
        await recognition._on_vad_event(
            vad.VADEvent(
                type=vad.VADEventType.START_OF_SPEECH,
                samples_index=0,
                timestamp=time.time(),
                speech_duration=0.1,
                silence_duration=0.0,
            )
        )
        await asyncio.sleep(0.4)
        assert committed == []

        await _send_vad_end(recognition)
        await asyncio.sleep(0.4)
        assert committed == ["I live in Cantu."]


async def test_stt_turn_end_does_not_carry_into_manual_turn_detection() -> None:
    async with _stt_turn_recognition(min_delay=0.0) as (recognition, committed):
        await _send_stt_event(recognition, SpeechEventType.START_OF_SPEECH)
        await _send_stt_event(recognition, SpeechEventType.FINAL_TRANSCRIPT, "I need the OTP.")
        await _send_stt_event(recognition, SpeechEventType.END_OF_SPEECH)
        assert committed == ["I need the OTP."]

        recognition._update_options(turn_detection="manual")
        await _send_stt_event(recognition, SpeechEventType.FINAL_TRANSCRIPT, "For the guard.")
        await asyncio.sleep(0.1)

        assert committed == ["I need the OTP."]
        assert recognition._audio_transcript == "For the guard."


async def test_stt_turn_is_not_ended_by_vad_after_the_user_resumes() -> None:
    async with _stt_turn_recognition(min_delay=0.0) as (recognition, committed):
        await _send_stt_event(recognition, SpeechEventType.START_OF_SPEECH)
        await _send_stt_event(recognition, SpeechEventType.FINAL_TRANSCRIPT, "I need the OTP.")
        await _send_stt_event(recognition, SpeechEventType.END_OF_SPEECH)
        assert committed == ["I need the OTP."]

        await _send_stt_event(recognition, SpeechEventType.START_OF_SPEECH)
        await _send_vad_end(recognition)
        await _send_stt_event(recognition, SpeechEventType.FINAL_TRANSCRIPT, "For the")
        assert committed == ["I need the OTP."]

        await _send_stt_event(recognition, SpeechEventType.END_OF_SPEECH)
        assert committed == ["I need the OTP.", "For the"]


async def test_stt_cleared_turn_waits_for_its_own_end_of_speech() -> None:
    async with _stt_turn_recognition(min_delay=0.0) as (recognition, committed):
        await _send_stt_event(recognition, SpeechEventType.START_OF_SPEECH)
        await _send_stt_event(recognition, SpeechEventType.FINAL_TRANSCRIPT, "I need the OTP.")
        await _send_stt_event(recognition, SpeechEventType.END_OF_SPEECH)
        assert committed == ["I need the OTP."]

        recognition._clear_user_turn()
        await _send_stt_event(recognition, SpeechEventType.FINAL_TRANSCRIPT, "For the guard.")
        assert committed == ["I need the OTP."]

        await _send_stt_event(recognition, SpeechEventType.END_OF_SPEECH)
        assert committed == ["I need the OTP.", "For the guard."]


async def test_backchannel_boundary_releases_end_boundary_transcript() -> None:
    actions = FakeActions()
    session = create_session(
        actions,
        turn_handling={"interruption": {"backchannel_boundary": (0.0, 0.5)}},
    )
    recognition = _recognition(session)
    recognition._interruption_enabled = True
    recognition._interruption_ch = aio.Chan[inference.InterruptionDataFrameType]()

    try:
        recognition._on_start_of_agent_speech(started_at=8.0)
        recognition._on_start_of_speech(started_at=8.1)
        await recognition._on_stt_event(
            _final_transcript_event(
                text="before the boundary", start_time=0.0, end_time=0.0, created_at=9.0
            )
        )
        await recognition._on_stt_event(
            _final_transcript_event(
                text="near the boundary",
                start_time=10_000.0,
                end_time=20_000.0,
                created_at=9.75,
            )
        )
        recognition._on_end_of_agent_speech(ended_at=10.0)
        await recognition._on_stt_event(
            _final_transcript_event(
                text="after agent", start_time=0.0, end_time=0.0, created_at=10.1
            )
        )

        assert recognition._hooks.final_transcripts == [  # type: ignore[attr-defined]
            "near the boundary",
            "after agent",
        ]
    finally:
        recognition._interruption_ch.close()
        await _close_test_session(session)


async def test_interruption_detection_error_is_not_session_error() -> None:
    actions = FakeActions()
    session = create_session(actions)
    activity = AgentActivity(MyAgent(), session)
    fallback = Mock()
    activity._fallback_to_vad_interruption = fallback
    error_events: list[object] = []
    session.on("error", error_events.append)

    try:
        recoverable = inference.InterruptionDetectionError(
            label="test",
            error=RuntimeError("temporary failure"),
            recoverable=True,
        )
        activity._on_error(recoverable)

        unrecoverable = inference.InterruptionDetectionError(
            label="test",
            error=RuntimeError("adaptive unavailable"),
            recoverable=False,
        )
        activity._on_error(unrecoverable)

        assert error_events == []
        fallback.assert_called_once_with(unrecoverable)
    finally:
        await _close_test_session(session)


async def test_reset_away_timer_restarts_the_full_timeout() -> None:
    async with AgentSession(
        vad=None, turn_handling={"turn_detection": None}, user_away_timeout=3.0
    ) as session:
        user_states: list[str] = []
        session.on("user_state_changed", lambda ev: user_states.append(ev.new_state))
        await session.start(MyAgent())

        for _ in range(2):
            await asyncio.sleep(2.0)
            session.reset_away_timer()

        await asyncio.sleep(2.0)
        assert session.user_state == "listening"
        assert user_states == []

        await asyncio.sleep(2.0)
        assert session.user_state == "away"
        assert user_states == ["away"]


@pytest.mark.parametrize("agent_state", ["listening", "speaking", "thinking"])
async def test_reset_away_timer_returns_away_user_to_listening(agent_state: AgentState) -> None:
    async with AgentSession(
        vad=None, turn_handling={"turn_detection": None}, user_away_timeout=3.0
    ) as session:
        transitions: list[tuple[str, str]] = []
        session.on(
            "user_state_changed", lambda ev: transitions.append((ev.old_state, ev.new_state))
        )
        await session.start(MyAgent())
        await asyncio.sleep(4.0)
        assert session.user_state == "away"

        session._update_agent_state(agent_state)
        session.reset_away_timer()
        assert session.user_state == "listening"
        assert transitions == [("listening", "away"), ("away", "listening")]

        await asyncio.sleep(2.0)
        assert session.user_state == "listening"
        await asyncio.sleep(2.0)
        assert session.user_state == ("away" if agent_state == "listening" else "listening")


@pytest.mark.parametrize(
    ("user_state", "agent_state"),
    [("speaking", "listening"), ("listening", "speaking"), ("listening", "thinking")],
)
async def test_reset_away_timer_preserves_active_turns(
    user_state: UserState, agent_state: AgentState
) -> None:
    async with AgentSession(
        vad=None, turn_handling={"turn_detection": None}, user_away_timeout=3.0
    ) as session:
        await session.start(MyAgent())
        session._update_user_state(user_state)
        session._update_agent_state(agent_state)

        session.reset_away_timer()
        await asyncio.sleep(4.0)
        assert session.user_state == user_state
        assert session.agent_state == agent_state

        session._update_user_state("listening")
        session._update_agent_state("listening")
        await asyncio.sleep(4.0)
        assert session.user_state == "away"


async def test_reset_away_timer_is_noop_when_disabled() -> None:
    async with AgentSession(
        vad=None, turn_handling={"turn_detection": None}, user_away_timeout=None
    ) as session:
        await session.start(MyAgent())
        session.reset_away_timer()
        await asyncio.sleep(30.0)
        assert session.user_state == "listening"
        assert session._user_away_timer is None


async def test_reset_away_timer_is_noop_outside_session_lifetime() -> None:
    async with AgentSession(
        vad=None, turn_handling={"turn_detection": None}, user_away_timeout=3.0
    ) as session:
        session.reset_away_timer()
        assert session._user_away_timer is None

        await session.start(MyAgent())
        timer = session._user_away_timer
        session.shutdown()
        session.reset_away_timer()
        assert session._user_away_timer is timer

        await session.aclose()
        session.reset_away_timer()
        await asyncio.sleep(4.0)
        assert session.user_state == "listening"
        assert session._user_away_timer is None


async def test_final_transcript_resets_away_timer_when_not_speaking() -> None:
    session = create_session(FakeActions(), extra_kwargs={"user_away_timeout": 15.0})
    try:
        session._agent_state = "listening"
        session._user_state = "listening"

        with patch.object(session, "_set_user_away_timer") as set_timer:
            session._user_input_transcribed(
                UserInputTranscribedEvent(transcript="hello", is_final=True)
            )
            set_timer.assert_called_once()
            assert session.user_state == "listening"

        with patch.object(session, "_set_user_away_timer") as set_timer:
            session._user_input_transcribed(
                UserInputTranscribedEvent(transcript="hello", is_final=False)
            )
            set_timer.assert_not_called()

        session._user_state = "speaking"
        with patch.object(session, "_set_user_away_timer") as set_timer:
            session._user_input_transcribed(
                UserInputTranscribedEvent(transcript="hello", is_final=True)
            )
            set_timer.assert_not_called()
    finally:
        await _close_test_session(session)


async def test_stt_error_count_resets_on_user_transcript() -> None:
    session = create_session(
        FakeActions(),
        extra_kwargs={"conn_options": SessionConnectOptions(max_unrecoverable_errors=1)},
    )

    def _stt_error() -> STTError:
        return STTError(
            timestamp=time.time(),
            label="test",
            error=RuntimeError("stt unavailable"),
            recoverable=False,
        )

    try:
        # the first unrecoverable error is tolerated, like llm/tts
        session._on_error(_stt_error())
        assert session._closing_task is None
        assert session._stt_error_counts == 1

        # a real transcript means the stt recovered -> tolerance resets
        session._user_input_transcribed(
            UserInputTranscribedEvent(transcript="hello", is_final=True)
        )
        assert session._stt_error_counts == 0

        # an empty placeholder transcript is not a recovery -> no reset
        session._on_error(_stt_error())
        session._user_input_transcribed(UserInputTranscribedEvent(transcript="", is_final=False))
        assert session._stt_error_counts == 1

        # tolerance still trips once the (reset) budget is exhausted again
        session._on_error(_stt_error())
        assert session._closing_task is not None
        await session._closing_task
    finally:
        await _close_test_session(session)


@pytest.fixture
def _no_stt_reconnect_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_recognition, "_STT_RECONNECT_INTERVAL", 0.0)


@pytest.mark.usefixtures("_no_stt_reconnect_delay")
async def test_stt_pipeline_recreates_stream_after_unrecoverable_error() -> None:
    attempts = 0

    async def stt_node(audio, model_settings):  # type: ignore[no-untyped-def]
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise APIConnectionError("stt unavailable")

        yield SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[SpeechData(text="recovered", language="en")],
        )
        # stay open like a live stream until the pipeline is closed
        async for _ in audio:
            pass

    pipeline = _STTPipeline(stt_node)
    try:
        # the first stream dies on a connection error; the pump must recreate it
        # and forward the transcript produced by the second stream
        ev = await asyncio.wait_for(pipeline.event_ch.recv(), timeout=5)
        assert ev.type == SpeechEventType.FINAL_TRANSCRIPT
        assert ev.alternatives[0].text == "recovered"
        assert ev.created_at > 0
        assert attempts == 2
    finally:
        await pipeline.aclose()


@pytest.mark.usefixtures("_no_stt_reconnect_delay")
@pytest.mark.parametrize(
    ("error", "closing"),
    [
        pytest.param(ValueError("bug in stt_node"), False, id="non-connection-error"),
        pytest.param(APIConnectionError("stt unavailable"), True, id="session-closing"),
    ],
)
async def test_stt_pipeline_does_not_recreate_stream(error: Exception, closing: bool) -> None:
    attempts = 0

    async def stt_node(audio, model_settings):  # type: ignore[no-untyped-def]
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise error
        yield SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[SpeechData(text="recovered", language="en")],
        )

    pipeline = _STTPipeline(stt_node, is_closing=lambda: closing)
    try:
        events = [ev async for ev in pipeline.event_ch]
        assert events == []
        assert attempts == 1
    finally:
        await pipeline.aclose()


@pytest.mark.usefixtures("_no_stt_reconnect_delay")
async def test_stt_pipeline_recreation_uses_rebound_node() -> None:
    started_old = asyncio.Event()
    fail_old = asyncio.Event()

    async def old_node(audio, model_settings):  # type: ignore[no-untyped-def]
        # the stream captured from the previous agent, still running at handoff
        started_old.set()
        await fail_old.wait()
        raise APIConnectionError("stt unavailable")
        yield  # pragma: no cover - makes this an async generator

    async def new_node(audio, model_settings):  # type: ignore[no-untyped-def]
        yield SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[SpeechData(text="recovered", language="en")],
        )
        async for _ in audio:
            pass

    # a handoff rebinds the pipeline; recreation must use the new agent's node
    pipeline = _STTPipeline(old_node)
    try:
        await asyncio.wait_for(started_old.wait(), timeout=5)
        pipeline._rebind_node(new_node)
        fail_old.set()
        ev = await asyncio.wait_for(pipeline.event_ch.recv(), timeout=5)
        assert ev.type == SpeechEventType.FINAL_TRANSCRIPT
        assert ev.alternatives[0].text == "recovered"
    finally:
        await pipeline.aclose()


async def test_vad_fallback_uses_next_vad_inference_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    actions = FakeActions()
    session = create_session(actions)
    activity = AgentActivity(MyAgent(), session)
    error = inference.InterruptionDetectionError(
        label="test",
        error=RuntimeError("adaptive unavailable"),
        recoverable=False,
    )

    audio_recognition = MagicMock()
    current_speech = MagicMock()
    current_speech.interrupted = False
    current_speech.allow_interruptions = True

    activity._audio_recognition = audio_recognition
    activity._current_speech = current_speech
    activity._interruption_detection_enabled = True
    activity._interruption_by_audio_activity_enabled = False
    activity._default_interruption_by_audio_activity_enabled = True

    caplog.set_level(logging.INFO, logger="livekit.agents")

    try:
        activity._fallback_to_vad_interruption(error)

        audio_recognition._update_interruption_detection.assert_called_once_with(None)
        current_speech.interrupt.assert_not_called()
        assert activity._interruption_detection_enabled is False
        assert activity._interruption_by_audio_activity_enabled is True

        activity.on_vad_inference_done(
            vad.VADEvent(
                type=vad.VADEventType.INFERENCE_DONE,
                samples_index=0,
                timestamp=time.time(),
                speech_duration=session.options.interruption["min_duration"] - 0.01,
                silence_duration=0.0,
                speaking=True,
            )
        )
        current_speech.interrupt.assert_not_called()

        activity.on_vad_inference_done(
            vad.VADEvent(
                type=vad.VADEventType.INFERENCE_DONE,
                samples_index=0,
                timestamp=time.time(),
                speech_duration=session.options.interruption["min_duration"],
                silence_duration=0.0,
                speaking=True,
            )
        )
        current_speech.interrupt.assert_called_once_with(source="audio_activity")
        assert any(
            record.levelno == logging.INFO
            and "falling back to VAD-based interruption" in record.message
            for record in caplog.records
        )
        assert not [record for record in caplog.records if record.levelno >= logging.WARNING]
    finally:
        await _close_test_session(session)


async def test_flush_held_transcripts_emits_buffered_events() -> None:
    actions = FakeActions()
    session = create_session(actions)
    hooks = _TestRecognitionHooks()
    recognition = _recognition(session, hooks=hooks)
    recognition._transcript_buffer.append(
        _final_transcript_event(text="held transcript", start_time=0.0, end_time=1.0)
    )

    try:
        recognition._flush_held_transcripts()

        assert hooks.final_transcripts == ["held transcript"]
        assert not recognition._transcript_buffer
    finally:
        await _close_test_session(session)


async def test_held_final_transcript_cancels_timeout_on_arrival() -> None:
    session = create_session(FakeActions())
    hooks = _TestRecognitionHooks()
    recognition = _recognition(session, hooks=hooks)
    recognition._transcript_gate_active = True
    timeout_handle = asyncio.get_running_loop().call_later(60.0, lambda: None)
    recognition._transcription_timeout_handle = timeout_handle
    event = _final_transcript_event(text="held transcript", start_time=0.0, end_time=1.0)

    try:
        await recognition._on_stt_event(event)

        assert timeout_handle.cancelled()
        assert recognition._turn_transcript_received
        assert list(recognition._transcript_buffer) == [event]
        assert hooks.final_transcripts == []

        recognition._flush_held_transcripts()

        assert timeout_handle.cancelled()
        assert recognition._turn_transcript_received
        assert hooks.final_transcripts == ["held transcript"]
        assert recognition._current_transcript == "held transcript"
    finally:
        timeout_handle.cancel()
        await _close_test_session(session)


async def test_true_verdict_releases_late_transcripts() -> None:
    session = create_session(FakeActions())
    hooks = _TestRecognitionHooks()
    recognition = _recognition(session, hooks=hooks)
    recognition._transcript_gate_active = True
    recognition._agent_speaking = True
    recognition._agent_speech_started_at = time.time() - 1.0
    overlap_started_at = time.time() - 0.5
    recognition._transcript_buffer.append(
        _final_transcript_event(
            text="already held",
            start_time=0.0,
            end_time=0.0,
            created_at=overlap_started_at,
        )
    )

    try:
        event = inference.OverlappingSpeechEvent(
            is_interruption=True,
            overlap_started_at=overlap_started_at,
        )
        hooks.on_overlap_speech(event)
        recognition._on_overlap_speech_event(event)
        await recognition._on_stt_event(
            _final_transcript_event(text="arrived later", start_time=0.0, end_time=0.0)
        )

        assert hooks.final_transcripts == ["already held", "arrived later"]
        assert len(hooks.interruptions) == 1
        assert not recognition._transcript_buffer
    finally:
        await _close_test_session(session)


async def test_preflight_transcript_does_not_cancel_transcription_timeout() -> None:
    session = create_session(FakeActions())
    hooks = _TestRecognitionHooks()
    recognition = _recognition(session, hooks=hooks)
    timeout_handle = asyncio.get_running_loop().call_later(60.0, lambda: None)
    recognition._transcription_timeout_handle = timeout_handle
    event = SpeechEvent(
        type=SpeechEventType.PREFLIGHT_TRANSCRIPT,
        alternatives=[SpeechData(text="preflight transcript", language=LanguageCode(""))],
    )

    try:
        await recognition._on_stt_event(event)

        assert not timeout_handle.cancelled()
        assert recognition._turn_transcript_received is False
    finally:
        timeout_handle.cancel()
        await _close_test_session(session)


async def test_transcription_timeout_accounts_for_vad_endpointing_delay() -> None:
    session = create_session(FakeActions(), extra_kwargs={"transcription_timeout": 2.0})
    recognition = _recognition(session, turn_detection="vad")
    recognition._stt_pipeline = MagicMock()
    recognition._vad_speech_started = True
    event_loop = asyncio.get_running_loop()

    try:
        await recognition._on_vad_event(
            vad.VADEvent(
                type=vad.VADEventType.END_OF_SPEECH,
                samples_index=0,
                timestamp=time.time(),
                speech_duration=1.0,
                silence_duration=0.5,
                inference_duration=0.25,
            )
        )

        timeout_handle = recognition._transcription_timeout_handle
        assert timeout_handle is not None
        assert timeout_handle.when() - event_loop.time() == pytest.approx(1.25)
    finally:
        recognition._stt_pipeline = None
        await recognition._aclose()
        await _close_test_session(session)


async def test_late_vad_eos_after_committed_turn_does_not_arm_transcription_timeout() -> None:
    session = create_session(FakeActions(), extra_kwargs={"transcription_timeout": 2.0})
    recognition = _recognition(session, turn_detection="stt")
    recognition._stt_pipeline = MagicMock()
    recognition._vad_speech_started = False

    try:
        await recognition._on_vad_event(
            vad.VADEvent(
                type=vad.VADEventType.END_OF_SPEECH,
                samples_index=0,
                timestamp=time.time(),
                speech_duration=1.0,
                silence_duration=0.5,
            )
        )

        assert recognition._transcription_timeout_handle is None
    finally:
        recognition._stt_pipeline = None
        await recognition._aclose()
        await _close_test_session(session)


async def test_clear_user_turn_resets_transcription_timeout() -> None:
    session = create_session(FakeActions(), extra_kwargs={"transcription_timeout": 1.0})
    recognition = _recognition(session, turn_detection="vad")
    timeout_handle = asyncio.get_running_loop().call_later(60.0, lambda: None)
    recognition._transcription_timeout_handle = timeout_handle
    recognition._turn_speech_duration = 2.0
    recognition._turn_transcript_received = True
    recognition._user_turn_start = time.time()

    try:
        recognition._clear_user_turn()

        assert timeout_handle.cancelled()
        assert recognition._transcription_timeout_handle is None
        assert recognition._turn_speech_duration == 0.0
        assert recognition._turn_transcript_received is False
        assert recognition._user_turn_start is None

        recognition._arm_transcription_timeout(1.0, delay=0.0)
        assert recognition._transcription_timeout_handle is not None
    finally:
        await recognition._aclose()
        await _close_test_session(session)


@pytest.mark.parametrize(
    ("session_preemptive", "agent_preemptive", "expected_latency"),
    [
        # with TTS: 0.1 + 0.3 + 0.3
        ({"preemptive_tts": True}, NOT_GIVEN, 0.7),
        # without TTS: max(0.1 + 0.3, 0.5) + 0.3
        ({"preemptive_tts": False}, NOT_GIVEN, 0.8),
        # disabled: 0.5 + 0.3 + 0.3
        ({"enabled": False}, NOT_GIVEN, 1.1),
        # the agent's setting overrides the session's
        ({"preemptive_tts": True}, {"enabled": False}, 1.1),
        ({"enabled": False}, {"enabled": True, "preemptive_tts": True}, 0.7),
    ],
)
async def test_preemptive_generation(
    session_preemptive: dict, agent_preemptive: NotGivenOr[dict], expected_latency: float
) -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.0, "Hello, how are you?", stt_delay=0.1)
    actions.add_llm("I'm doing great, thank you!", ttft=0.1, duration=0.3)
    actions.add_tts(3.0, ttfb=0.3)

    session = create_session(actions, turn_handling={"preemptive_generation": session_preemptive})
    agent = MyAgent(
        turn_handling=NOT_GIVEN
        if agent_preemptive is NOT_GIVEN
        else {"preemptive_generation": agent_preemptive}
    )
    agent_state_events: list[AgentStateChangedEvent] = []
    user_state_events: list[UserStateChangedEvent] = []
    session.on("agent_state_changed", agent_state_events.append)
    session.on("user_state_changed", user_state_events.append)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert [ev.new_state for ev in user_state_events] == ["speaking", "listening"]
    assert [ev.new_state for ev in agent_state_events] == [
        "listening",
        "thinking",
        "speaking",
        "listening",
    ]
    t_user_stop_speaking = user_state_events[1].speech_timestamp
    assert t_user_stop_speaking is not None
    check_timestamp(
        agent_state_events[2].created_at - t_user_stop_speaking,
        expected_latency,
        max_abs_diff=0.2,
    )


@pytest.mark.parametrize(
    "preflight, final, expected",
    [
        ("hello how are you", "Hello, how are you?", True),
        ("مرحبا كيف حالك", "مرحبا، كيف حالك؟", True),
        ("نعم أريد الحجز", "نعم، أريد الحجز؛", True),
        ("آپ کیسے ہیں", "آپ کیسے ہیں۔", True),
        ("आप कैसे हैं", "आप कैसे हैं।", True),
        ("你好 我很好", "你好，我很好。", True),
        # different words must still invalidate the preemptive generation
        ("مرحبا كيف حالك", "مرحبا كيف حالكم؟", False),
    ],
)
def test_preemptive_transcripts_equivalent(preflight: str, final: str, expected: bool) -> None:
    assert _transcripts_equivalent(preflight, final) is expected


@pytest.mark.parametrize(
    "preemptive_generation, expected",
    [
        # bool form: the legacy on/off flag
        (True, {"enabled": True, "preemptive_tts": False}),
        (False, {"enabled": False, "preemptive_tts": False}),
        # dict form: an options mapping, documented on the same kwarg (#7343)
        ({"enabled": False}, {"enabled": False, "preemptive_tts": False}),
        ({"enabled": True, "preemptive_tts": True}, {"enabled": True, "preemptive_tts": True}),
    ],
)
async def test_deprecated_preemptive_generation_kwarg(
    preemptive_generation: bool | dict, expected: dict
) -> None:
    session = AgentSession(preemptive_generation=preemptive_generation)
    opts = session.options.preemptive_generation
    assert {k: opts[k] for k in expected} == expected
    assert isinstance(opts["enabled"], bool)


@pytest.mark.parametrize(
    "preemptive_generation, on_user_turn_completed_delay",
    [
        (False, 0.0),
        (False, 2.0),
        (True, 0.0),
        (True, 2.0),
    ],
)
async def test_interrupt_during_on_user_turn_completed(
    preemptive_generation: bool, on_user_turn_completed_delay: float
) -> None:

    actions = FakeActions()
    actions.add_user_speech(0.5, 2.0, "Tell me a story", stt_delay=0.2)
    actions.add_llm("Here is a story for you...", ttft=0.1, duration=0.3)
    actions.add_tts(10.0, ttfb=1.0)  # latency after end of turn: 1.3s
    actions.add_user_speech(2.6, 3.2, "about a firefighter.")  # interrupt before speaking
    actions.add_llm("Here is a story about a firefighter...", ttft=0.1, duration=0.3)
    actions.add_tts(10.0, ttfb=0.3)

    session = create_session(
        actions,
        turn_handling={"preemptive_generation": {"enabled": preemptive_generation}},
    )
    agent = MyAgent(on_user_turn_completed_delay=on_user_turn_completed_delay)

    agent_state_events: list[AgentStateChangedEvent] = []
    conversation_events: list[ConversationItemAddedEvent] = []
    session.on("agent_state_changed", agent_state_events.append)
    session.on("conversation_item_added", conversation_events.append)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert agent_state_events[0].old_state == "initializing"
    assert agent_state_events[0].new_state == "listening"
    if on_user_turn_completed_delay == 0.0:
        # on_user_turn_completed already committed before interrupting
        assert len(agent_state_events) == 6
        assert agent_state_events[1].new_state == "thinking"
        assert agent_state_events[2].new_state == "listening"
        assert agent_state_events[3].new_state == "thinking"
        assert agent_state_events[4].new_state == "speaking"
        assert agent_state_events[5].new_state == "listening"
    else:
        assert len(agent_state_events) == 4
        assert agent_state_events[1].new_state == "thinking"
        assert agent_state_events[2].new_state == "speaking"
        assert agent_state_events[3].new_state == "listening"

    assert len(conversation_events) == 4
    assert conversation_events[0].item.type == "agent_handoff"
    assert conversation_events[1].item.type == "message"
    assert conversation_events[1].item.role == "user"
    assert conversation_events[1].item.text_content == "Tell me a story"
    assert conversation_events[2].item.type == "message"
    assert conversation_events[2].item.role == "user"
    assert conversation_events[2].item.text_content == "about a firefighter."
    assert conversation_events[3].item.type == "message"
    assert conversation_events[3].item.role == "assistant"
    assert conversation_events[3].item.text_content == "Here is a story about a firefighter..."


async def test_unknown_function_call() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Check the weather")
    actions.add_llm(
        content="",
        tool_calls=[
            FunctionToolCall(
                name="nonexistent_tool", arguments='{"location": "Tokyo"}', call_id="1"
            )
        ],
    )
    actions.add_llm(
        content="I don't have access to that function.",
        input="Unknown function: nonexistent_tool",
    )
    actions.add_tts(2.0)

    session = create_session(actions)
    agent = MyAgent()

    tool_executed_events: list[FunctionToolsExecutedEvent] = []
    session.on("function_tools_executed", tool_executed_events.append)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert len(tool_executed_events) == 1
    assert tool_executed_events[0].function_calls[0].name == "nonexistent_tool"
    assert tool_executed_events[0].function_call_outputs[0].is_error is True
    assert "Unknown function" in tool_executed_events[0].function_call_outputs[0].output

    chat_ctx_items = agent.chat_ctx.items
    error_outputs = [
        item
        for item in chat_ctx_items
        if item.type == "function_call_output" and item.is_error is True
    ]
    assert len(error_outputs) == 1
    assert "Unknown function: nonexistent_tool" in error_outputs[0].output


async def test_invalid_tool_arguments_surface_as_tool_error() -> None:
    """Invalid tool arguments stay in the history with a descriptive error the model can fix."""
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "What's the weather?")
    # get_weather requires `location`
    actions.add_llm(
        content="",
        tool_calls=[
            FunctionToolCall(name="get_weather", arguments="{}", call_id="1"),
        ],
    )

    session = create_session(actions)
    agent = MyAgent()

    tool_executed_events: list[FunctionToolsExecutedEvent] = []
    session.on("function_tools_executed", tool_executed_events.append)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert len(tool_executed_events) == 1
    ev = tool_executed_events[0]
    assert len(ev.function_calls) == 1
    assert ev.function_calls[0].name == "get_weather"
    assert ev.function_call_outputs[0] is not None
    output = ev.function_call_outputs[0]
    assert output.is_error is True

    assert "An internal error occurred" not in output.output
    assert "get_weather" in output.output
    assert "location" in output.output

    items = agent.chat_ctx.items
    function_calls = [i for i in items if i.type == "function_call"]
    function_call_outputs = [i for i in items if i.type == "function_call_output"]
    assert len(function_calls) == 1
    assert function_calls[0].name == "get_weather"
    assert function_calls[0].call_id == "1"
    assert len(function_call_outputs) == 1
    assert function_call_outputs[0].call_id == "1"
    assert function_call_outputs[0].is_error is True


async def test_tool_internal_exception_returns_generic_error() -> None:
    """A tool's unexpected exception reaches the model as a generic message, never its details."""

    class _BrokenToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You are a helpful assistant.")

        @function_tool
        async def get_weather(self, location: str) -> str:
            """Always blows up."""
            raise RuntimeError("kaboom: secret database password leaked")

    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "What's the weather in Tokyo?")
    actions.add_llm(
        content="",
        tool_calls=[
            FunctionToolCall(name="get_weather", arguments='{"location": "Tokyo"}', call_id="1"),
        ],
    )

    session = create_session(actions)
    agent = _BrokenToolAgent()

    tool_executed_events: list[FunctionToolsExecutedEvent] = []
    session.on("function_tools_executed", tool_executed_events.append)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert len(tool_executed_events) == 1
    output = tool_executed_events[0].function_call_outputs[0]
    assert output is not None
    assert output.is_error is True
    assert output.output == "An internal error occurred"
    assert "kaboom" not in output.output
    assert "secret" not in output.output


class _TestRecognitionHooks:
    def __init__(self) -> None:
        self.interruptions: list[inference.OverlappingSpeechEvent] = []
        self.final_transcripts: list[str] = []
        self.interruption_by_audio_activity_enabled = False

    def on_overlap_speech(self, ev: inference.OverlappingSpeechEvent) -> None:
        if ev.is_interruption:
            self.interruptions.append(ev)

    def on_backchannel_confirmed(self) -> None:
        pass

    def on_start_of_speech(self, ev: object, speech_start_time: float) -> None:
        pass

    def on_vad_inference_done(self, ev: object) -> None:
        pass

    def on_end_of_speech(self, ev: object, *, speech_end_time: float) -> None:
        pass

    def on_interim_transcript(self, ev: SpeechEvent, *, speaking: bool | None) -> None:
        pass

    def on_final_transcript(self, ev: SpeechEvent, *, speaking: bool | None = None) -> None:
        self.final_transcripts.append(ev.alternatives[0].text)

    def on_end_of_turn(self, info: _EndOfTurnInfo) -> bool:
        return True

    def on_preemptive_generation(self, info: object) -> None:
        pass

    def on_preflight_stale(self, info: object) -> None:
        pass

    def retrieve_chat_ctx(self) -> ChatContext:
        return ChatContext.empty()


def _recognition(
    session: AgentSession,
    *,
    turn_detection: str = "manual",
    hooks: _TestRecognitionHooks | None = None,
    min_delay: float = 0.1,
    max_delay: float = 1.0,
    stt: object = None,
) -> AudioRecognition:
    return AudioRecognition(
        session,
        hooks=hooks or _TestRecognitionHooks(),
        endpointing=BaseEndpointing(min_delay=min_delay, max_delay=max_delay),
        stt=stt,
        vad=None,
        interruption_detection=None,
        turn_detection=turn_detection,
    )


def _interruption_event() -> inference.OverlappingSpeechEvent:
    return inference.OverlappingSpeechEvent(
        type="overlapping_speech",
        is_interruption=True,
        overlap_started_at=time.time(),
        detected_at=time.time(),
    )


def _backchannel_event() -> inference.OverlappingSpeechEvent:
    return inference.OverlappingSpeechEvent(
        type="overlapping_speech",
        is_interruption=False,
        overlap_started_at=time.time(),
        detected_at=time.time(),
    )


def _final_transcript_event(
    *, text: str, start_time: float, end_time: float, created_at: float | None = None
) -> SpeechEvent:
    ev = SpeechEvent(
        type=SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[
            SpeechData(
                text=text,
                language=LanguageCode(""),
                start_time=start_time,
                end_time=end_time,
            )
        ],
    )
    if created_at is not None:
        ev.created_at = created_at
    return ev


async def _close_test_session(session: object) -> None:
    await session.aclose()
    audio_output = session.output.audio
    synchronizer = getattr(audio_output, "_synchronizer", None)
    if synchronizer is not None:
        await synchronizer.aclose()


def check_timestamp(t_event: float, t_target: float, *, max_abs_diff: float = 0.75) -> None:
    assert abs(t_event - t_target) <= max_abs_diff, (
        f"{t_event} is not within {max_abs_diff} of {t_target}"
    )


async def test_silent_tool_call_pause_state_does_not_leak_into_tool_reply() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.1, 0.2, "What's the weather in Tokyo?", stt_delay=0.05)

    # silent tool-call step: no spoken preamble before the function call
    actions.add_llm(
        content="",
        tool_calls=[
            FunctionToolCall(
                name="get_weather",
                arguments='{"location": "Tokyo"}',
                call_id="1",
            )
        ],
        ttft=0.05,
        duration=1.0,
    )

    # VAD-only speech starts during the silent step and lasts past the tool reply start
    actions.add_user_speech(0.85, 2.0, "", stt_delay=0.05)

    actions.add_llm(
        content="The weather in Tokyo is sunny today.",
        input="The weather in Tokyo is sunny today.",
        ttft=0.0,
        duration=0.0,
    )
    actions.add_tts(0.5, ttfb=0.0, duration=0.0)

    session = create_session(
        actions,
        can_pause_audio=True,
        turn_handling={"interruption": {"false_interruption_timeout": 0.2}},
    )
    agent = MyAgent()

    agent_state_events: list[AgentStateChangedEvent] = []
    false_interruption_events: list[AgentFalseInterruptionEvent] = []

    session.on("agent_state_changed", agent_state_events.append)
    session.on("agent_false_interruption", false_interruption_events.append)

    await asyncio.wait_for(
        run_session(session, agent, drain_delay=0.8),
        timeout=SESSION_TIMEOUT,
    )

    transitions = [(ev.old_state, ev.new_state) for ev in agent_state_events]
    silent_step_finished = transitions.index(("speaking", "listening"))

    # the pause state of the silent tool-call step must not leak into the tool reply
    assert transitions[silent_step_finished + 1] == ("listening", "speaking")
    assert false_interruption_events
    assert false_interruption_events[-1].resumed is True


@pytest.mark.parametrize("vad_arg", [NOT_GIVEN, None, "user"])
async def test_session_vad_defaults(vad_arg: object) -> None:
    user_vad = FakeVAD(fake_user_speeches=[])
    session = AgentSession(vad=user_vad if vad_arg == "user" else vad_arg)
    try:
        if vad_arg is NOT_GIVEN:
            assert session.vad is not None
        else:
            assert session.vad is (user_vad if vad_arg == "user" else None)
        assert session._using_default_vad is (vad_arg is NOT_GIVEN)
    finally:
        await session.aclose()


@pytest.mark.parametrize("turn_detection", [NOT_GIVEN, None, "user"])
async def test_session_turn_detection_defaults(turn_detection: object) -> None:
    user_detector = inference.TurnDetector(version="v1-mini")
    turn_handling = (
        {}
        if turn_detection is NOT_GIVEN
        else {"turn_detection": user_detector if turn_detection == "user" else None}
    )
    session = AgentSession(turn_handling=turn_handling)
    try:
        if turn_detection is NOT_GIVEN:
            assert isinstance(session.turn_detection, _StreamingTurnDetector)
        else:
            assert session.turn_detection is (user_detector if turn_detection == "user" else None)
    finally:
        await session.aclose()


async def test_stt_context_options_defaults_forward_chat_context_on() -> None:
    """forward_chat_context is on by default in the resolved stt_context_options."""
    session = AgentSession(vad=None)
    try:
        assert session._opts.stt_context_options["forward_chat_context"] is True
    finally:
        await session.aclose()


async def test_stt_context_options_passthrough() -> None:
    session = AgentSession(
        vad=None,
        stt_context_options={"keyterms": ["LiveKit"], "forward_chat_context": False},
    )
    try:
        opts = session._opts.stt_context_options
        assert opts["keyterms"] == ["LiveKit"]
        assert opts["forward_chat_context"] is False
    finally:
        await session.aclose()


async def test_deprecated_keyterms_options_maps_to_stt_context(caplog) -> None:
    """keyterms_options still works, maps onto stt_context_options, and warns."""
    with caplog.at_level(logging.WARNING):
        session = AgentSession(vad=None, keyterms_options={"keyterms": ["Acme"]})
    try:
        opts = session._opts.stt_context_options
        assert opts["keyterms"] == ["Acme"]
        assert opts["forward_chat_context"] is True  # default still applies
        assert "keyterms_options is deprecated" in caplog.text
    finally:
        await session.aclose()


async def test_stt_context_options_wins_over_keyterms_options() -> None:
    """When both are passed, stt_context_options takes precedence over the deprecated option."""
    session = AgentSession(
        vad=None,
        stt_context_options={"keyterms": ["new"]},
        keyterms_options={"keyterms": ["old"]},
    )
    try:
        assert session._opts.stt_context_options["keyterms"] == ["new"]
    finally:
        await session.aclose()


@pytest.mark.parametrize(
    ("kwargs", "min_delay", "max_delay"),
    [
        # streaming detectors get the tighter defaults
        pytest.param({}, 0.3, 2.5, id="default-streaming"),
        pytest.param(
            {"turn_handling": {"turn_detection": inference.TurnDetector(version="v1-mini")}},
            0.3,
            2.5,
            id="user-streaming",
        ),
        # non-streaming modes keep the legacy defaults
        pytest.param({"turn_handling": {"turn_detection": "vad"}}, 0.5, 3.0, id="vad"),
        pytest.param({"turn_detection": "vad"}, 0.5, 3.0, id="deprecated-vad"),
        # an explicit delay wins per key
        pytest.param(
            {"turn_handling": {"endpointing": {"min_delay": 0.4}}}, 0.4, 2.5, id="explicit"
        ),
    ],
)
async def test_endpointing_defaults_follow_turn_detector(
    kwargs: dict, min_delay: float, max_delay: float
) -> None:
    session = AgentSession(**kwargs)
    try:
        assert session._opts.endpointing["min_delay"] == min_delay
        assert session._opts.endpointing["max_delay"] == max_delay
    finally:
        await session.aclose()


async def test_agent_turn_detection_override_resolves_endpointing_per_activity() -> None:
    """endpointing_opts uses the activity's resolved detector, not just the session's."""
    # session default → streaming detector; provide VAD so it validates
    session = AgentSession(vad=FakeVAD(fake_user_speeches=[]))
    try:
        streaming_activity = AgentActivity(Agent(instructions="test"), session)
        assert streaming_activity.endpointing_opts["min_delay"] == 0.3
        assert streaming_activity.endpointing_opts["max_delay"] == 2.5

        # an agent overriding to VAD falls back to legacy defaults for this activity
        vad_activity = AgentActivity(Agent(instructions="test", turn_detection="vad"), session)
        assert vad_activity.endpointing_opts["min_delay"] == 0.5
        assert vad_activity.endpointing_opts["max_delay"] == 3.0
    finally:
        await session.aclose()


@pytest.mark.parametrize(
    ("session_min_delay", "agent_min_delay", "reply_held"),
    [
        pytest.param(0.2, 1.0, True, id="agent-holds-longer"),
        pytest.param(1.0, 0.2, False, id="agent-releases-earlier"),
    ],
)
async def test_reply_holdoff_uses_agent_endpointing_delay(
    session_min_delay: float, agent_min_delay: float, reply_held: bool
) -> None:
    """VAD silence uses the active agent's delay for pending reply hold-off."""
    session = AgentSession(
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling={
            "turn_detection": "vad",
            "endpointing": {"min_delay": session_min_delay},
        },
    )
    try:
        activity = AgentActivity(
            Agent(
                instructions="test",
                turn_handling={"endpointing": {"min_delay": agent_min_delay}},
            ),
            session,
        )
        assert activity.endpointing_opts["min_delay"] == agent_min_delay

        activity.on_vad_inference_done(
            vad.VADEvent(
                type=vad.VADEventType.INFERENCE_DONE,
                samples_index=0,
                timestamp=0.0,
                speech_duration=0.0,
                silence_duration=0.25,
                speaking=True,
                raw_accumulated_silence=0.25,
            )
        )

        assert activity._user_silence_event.is_set() == (not reply_held)
    finally:
        await session.aclose()


async def test_runtime_endpointing_opts_survive_handoff() -> None:
    """update_options changes are recorded as overrides, so a new activity keeps them."""
    session = AgentSession(vad=FakeVAD(fake_user_speeches=[]))
    try:
        session.update_options(endpointing_opts={"mode": "dynamic", "alpha": 0.5, "min_delay": 0.4})

        # a fresh activity (as built on agent handoff) re-resolves from overrides
        activity = AgentActivity(Agent(instructions="test"), session)
        assert activity.endpointing_opts["mode"] == "dynamic"
        assert activity.endpointing_opts["alpha"] == 0.5
        assert activity.endpointing_opts["min_delay"] == 0.4
        # untouched key still gets the streaming default
        assert activity.endpointing_opts["max_delay"] == 2.5
    finally:
        await session.aclose()


class FlushMultiSegmentAgent(Agent):
    """Agent whose llm_node flushes the reply into two segments via FlushSentinel."""

    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful assistant.")

    async def llm_node(
        self,
        chat_ctx: ChatContext,
        tools: list,
        model_settings: ModelSettings,
    ) -> AsyncIterable[str | FlushSentinel]:
        yield "Hello there. "
        yield FlushSentinel()
        yield "How are you?"


async def test_pipeline_multi_segment_flush() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Hello, how are you?", stt_delay=0.2)
    # the agent's llm_node injects a FlushSentinel, splitting the reply into two
    # segments; register a TTS response keyed by each segment's text
    actions.add_tts(1.0, input="Hello there. ", ttfb=0.1, duration=0.1)
    actions.add_tts(1.0, input="How are you?", ttfb=0.1, duration=0.1)

    session = create_session(actions)
    agent = FlushMultiSegmentAgent()

    playback_finished_events: list[PlaybackFinishedEvent] = []
    session.output.audio.on("playback_finished", playback_finished_events.append)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    # each FlushSentinel-delimited segment plays out independently
    assert len(playback_finished_events) == 2
    assert all(not ev.interrupted for ev in playback_finished_events)

    # but both segments join into a single assistant message
    assistant_msgs = [
        it for it in agent.chat_ctx.items if it.type == "message" and it.role == "assistant"
    ]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0].text_content == "Hello there. How are you?"


async def test_pipeline_multi_segment_interrupted() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Hello, how are you?", stt_delay=0.2)
    # long first segment so the interrupt lands while it is still playing
    actions.add_tts(15.0, input="Hello there. ", ttfb=0.1, duration=0.1)
    actions.add_tts(1.0, input="How are you?", ttfb=0.1, duration=0.1)

    session = create_session(actions)
    agent = FlushMultiSegmentAgent()

    playback_finished_events: list[PlaybackFinishedEvent] = []
    session.output.audio.on("playback_finished", playback_finished_events.append)

    asyncio.get_event_loop().call_later(5, session.interrupt)

    await asyncio.wait_for(run_session(session, agent, drain_delay=0.5), timeout=SESSION_TIMEOUT)

    # interrupted during the first segment: only that segment plays, the second
    # segment is never forwarded
    assert len(playback_finished_events) == 1
    assert playback_finished_events[0].interrupted is True

    assistant_msgs = [
        it for it in agent.chat_ctx.items if it.type == "message" and it.role == "assistant"
    ]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0].interrupted is True
    assert "How are you?" not in (assistant_msgs[0].text_content or "")
