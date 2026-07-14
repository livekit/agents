from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterable
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentFalseInterruptionEvent,
    AgentStateChangedEvent,
    ConversationItemAddedEvent,
    FlushSentinel,
    LanguageCode,
    MetricsCollectedEvent,
    ModelSettings,
    NotGivenOr,
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
    ToolContext,
    ToolFlag,
    Toolset,
)
from livekit.agents.llm.chat_context import ChatContext, ChatMessage
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.utils import aio
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.audio_recognition import AudioRecognition, _EndOfTurnInfo
from livekit.agents.voice.endpointing import BaseEndpointing
from livekit.agents.voice.events import FunctionToolsExecutedEvent
from livekit.agents.voice.io import PlaybackFinishedEvent

from .fake_session import FakeActions, create_session, run_session

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


async def test_events_and_metrics() -> None:
    speed = 1
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Hello, how are you?", stt_delay=0.2)  # EOU at 2.5+0.5=3.0s
    actions.add_llm("I'm doing well, thank you!", ttft=0.1, duration=0.3)
    actions.add_tts(
        2.0, ttfb=0.2, duration=0.3
    )  # audio playout starts at 3.0+0.3+0.2=3.5s, ends at 5.5s

    session = create_session(actions, speed_factor=speed)
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
    check_timestamp(conversation_events[1].created_at - t_origin, 3.0, speed_factor=speed)
    assert conversation_events[2].item.type == "message"
    assert conversation_events[2].item.role == "assistant"
    assert conversation_events[2].item.text_content == "I'm doing well, thank you!"
    check_timestamp(conversation_events[2].created_at - t_origin, 5.5, speed_factor=speed)

    # user_input_transcribed
    assert len(user_transcription_events) >= 1
    assert user_transcription_events[-1].transcript == "Hello, how are you?"
    assert user_transcription_events[-1].is_final is True
    check_timestamp(user_transcription_events[-1].created_at - t_origin, 2.7, speed_factor=speed)

    # user_state_changed
    assert len(user_state_events) == 2
    check_timestamp(user_state_events[0].created_at - t_origin, 0.5, speed_factor=speed)
    assert user_state_events[0].new_state == "speaking"
    check_timestamp(user_state_events[1].created_at - t_origin, 3.0, speed_factor=speed)
    assert user_state_events[1].new_state == "listening"

    # agent_state_changed
    assert len(agent_state_events) == 4
    assert agent_state_events[0].old_state == "initializing"
    assert agent_state_events[0].new_state == "listening"
    assert agent_state_events[1].new_state == "thinking"
    check_timestamp(agent_state_events[1].created_at - t_origin, 3.0, speed_factor=speed)
    assert agent_state_events[2].new_state == "speaking"
    check_timestamp(agent_state_events[2].created_at - t_origin, 3.5, speed_factor=speed)
    assert agent_state_events[3].new_state == "listening"
    check_timestamp(agent_state_events[3].created_at - t_origin, 5.5, speed_factor=speed)

    # metrics
    metrics_events = [ev for ev in metrics_events if ev.metrics.type != "vad_metrics"]
    assert len(metrics_events) == 3
    assert metrics_events[0].metrics.type == "eou_metrics"
    check_timestamp(metrics_events[0].metrics.end_of_utterance_delay, 0.5, speed_factor=speed)
    check_timestamp(metrics_events[0].metrics.transcription_delay, 0.2, speed_factor=speed)
    assert metrics_events[1].metrics.type == "llm_metrics"
    check_timestamp(metrics_events[1].metrics.ttft, 0.1, speed_factor=speed)
    check_timestamp(metrics_events[1].metrics.duration, 0.3, speed_factor=speed)
    assert metrics_events[2].metrics.type == "tts_metrics"
    check_timestamp(metrics_events[2].metrics.ttfb, 0.2, speed_factor=speed)
    check_timestamp(metrics_events[2].metrics.audio_duration, 2.0, speed_factor=speed)


async def test_tts_node_ttfb_excludes_upstream_latency() -> None:
    # the LLM stream stays open for its full duration and the fake TTS only starts
    # synthesizing once its input is flushed. tts_node_ttfb must anchor on the text
    # being handed to the TTS provider (~2.0s in), not on the first LLM token (~0.1s in),
    # otherwise the LLM streaming time is misattributed to the TTS
    speed = 1
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Hello, how are you?", stt_delay=0.2)
    actions.add_llm("I'm doing well, thank you!", ttft=0.1, duration=2.0)
    actions.add_tts(1.0, ttfb=0.2, duration=0.3)

    session = create_session(actions, speed_factor=speed)
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
    assert "tts_node_ttfb" in metrics
    check_timestamp(metrics["tts_node_ttfb"], 0.2, speed_factor=speed)


async def test_tool_call() -> None:
    speed = 1
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

    session = create_session(actions, speed_factor=speed)
    agent = MyAgent()

    agent_state_events: list[AgentStateChangedEvent] = []
    tool_executed_events: list[FunctionToolsExecutedEvent] = []
    playback_finished_events: list[PlaybackFinishedEvent] = []

    session.on("agent_state_changed", agent_state_events.append)
    session.on("function_tools_executed", tool_executed_events.append)
    session.output.audio.on("playback_finished", playback_finished_events.append)

    t_origin = await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert len(playback_finished_events) == 2
    check_timestamp(playback_finished_events[0].playback_position, 2.0, speed_factor=speed)
    check_timestamp(playback_finished_events[1].playback_position, 3.0, speed_factor=speed)

    assert len(agent_state_events) == 6
    assert agent_state_events[0].old_state == "initializing"
    assert agent_state_events[0].new_state == "listening"
    assert agent_state_events[1].new_state == "thinking"
    assert agent_state_events[2].new_state == "speaking"
    assert (
        agent_state_events[3].new_state == "thinking"
    )  # from speaking to thinking when tool call is executed
    check_timestamp(agent_state_events[3].created_at - t_origin, 5.5, speed_factor=speed)
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
    speed = 1
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Here is a long story for you ... the end.")
    actions.add_tts(10.0)  # playout starts at 3.5s
    actions.add_user_speech(5.0, 6.0, "Stop!", stt_delay=0.2)
    # interrupted at 5.5s, min_interruption_duration=0.5

    session = create_session(
        actions,
        speed_factor=speed,
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
    check_timestamp(
        agent_state_events[3].created_at - t_origin, expected_interruption_time, speed_factor=speed
    )
    assert agent_state_events[4].new_state == "thinking"
    check_timestamp(agent_state_events[4].created_at - t_origin, 6.5, speed_factor=speed)
    assert agent_state_events[5].new_state == "listening"
    check_timestamp(agent_state_events[5].created_at - t_origin, 6.5, speed_factor=speed)

    assert len(playback_finished_events) == 1
    assert playback_finished_events[0].interrupted is True
    if not resume_false_interruption:
        # fake audio output doesn't support pause/resume
        check_timestamp(playback_finished_events[0].playback_position, 2.0, speed_factor=speed)


async def test_interruption_options() -> None:
    speed = 1
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
        speed_factor=speed,
        turn_handling={"interruption": {"min_words": 3}},
    )
    playback_finished_events: list[PlaybackFinishedEvent] = []
    session.output.audio.on("playback_finished", playback_finished_events.append)

    await asyncio.wait_for(run_session(session, MyAgent()), timeout=SESSION_TIMEOUT)

    assert len(playback_finished_events) == 1
    assert playback_finished_events[0].interrupted is True
    check_timestamp(playback_finished_events[0].playback_position, 4.0, speed_factor=speed)

    # test allow_interruptions=False
    session = create_session(
        actions,
        speed_factor=speed,
        turn_handling={"interruption": {"enabled": False}},
    )
    playback_finished_events.clear()
    session.output.audio.on("playback_finished", playback_finished_events.append)

    await asyncio.wait_for(run_session(session, MyAgent()), timeout=SESSION_TIMEOUT)

    assert len(playback_finished_events) == 1
    assert playback_finished_events[0].interrupted is False
    check_timestamp(playback_finished_events[0].playback_position, 5.0, speed_factor=speed)


async def test_min_interruption_duration_applies_to_stt_transcripts() -> None:
    """Regression test for https://github.com/livekit/agents/issues/3515.

    STTs that stream continuous interim results (e.g. Amazon Transcribe) used to
    trigger an interruption on the first non-empty interim transcript, bypassing
    ``interruption.min_duration`` entirely (it was only enforced on the VAD path).
    The interruption must not fire before the user has spoken for min_duration.
    """
    speed = 1
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Here is a long story for you ... the end.")
    actions.add_tts(10.0)  # playout starts at 3.5s
    # user speaks for 4s while interim transcripts stream every 0.3s;
    # the first interim lands at ~5.3s, min_duration is reached at 7.0s
    actions.add_user_speech(
        5.0, 9.0, "please stop talking right now", stt_delay=0.2, interim_interval=0.3
    )

    session = create_session(
        actions,
        speed_factor=speed,
        turn_handling={"interruption": {"min_duration": 2.0}},
    )
    agent_state_events: list[AgentStateChangedEvent] = []
    playback_finished_events: list[PlaybackFinishedEvent] = []
    session.on("agent_state_changed", agent_state_events.append)
    session.output.audio.on("playback_finished", playback_finished_events.append)

    t_origin = await asyncio.wait_for(run_session(session, MyAgent()), timeout=SESSION_TIMEOUT)

    assert len(playback_finished_events) == 1
    assert playback_finished_events[0].interrupted is True
    # interrupted at 7.0s (5.0 + min_duration), i.e. 3.5s into the playout —
    # not at ~5.3s when the first interim transcript arrived
    check_timestamp(playback_finished_events[0].playback_position, 3.5, speed_factor=speed)

    interrupted_at = next(
        ev.created_at
        for ev in agent_state_events
        if ev.new_state == "listening" and ev.old_state == "speaking"
    )
    check_timestamp(interrupted_at - t_origin, 7.0, speed_factor=speed)


async def test_interruption_by_text_input() -> None:
    speed = 1
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Here is a long story for you ... the end.")
    actions.add_tts(15.0)
    actions.add_llm("Ok, I'll stop now.", input="stop from text input")
    actions.add_tts(2.0)

    session = create_session(actions, speed_factor=speed)
    agent = MyAgent()

    agent_state_events: list[AgentStateChangedEvent] = []
    playback_finished_events: list[PlaybackFinishedEvent] = []
    session.on("agent_state_changed", agent_state_events.append)
    session.output.audio.on("playback_finished", playback_finished_events.append)

    def fake_text_input() -> None:
        session.interrupt()
        session.generate_reply(user_input="stop from text input")

    asyncio.get_event_loop().call_later(5 / speed, fake_text_input)

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
    speed = 1
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Here is a long story for you ... the end.", duration=1.0)
    actions.add_tts(10.0)
    actions.add_user_speech(3.0, 4.0, "Stop!", stt_delay=0.2)

    session = create_session(
        actions,
        speed_factor=speed,
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
        agent_state_events[2].created_at - t_origin, expected_interruption_time, speed_factor=speed
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
    # before we insert an empty assistant message with interrupted=True
    # now we ignore it when the text is empty
    assert agent.chat_ctx.items[3].type == "message"
    assert agent.chat_ctx.items[3].role == "user"
    assert agent.chat_ctx.items[3].text_content == "Stop!"


async def test_interrupt_before_speaking_with_pausable_audio() -> None:
    """
    Regression test for https://github.com/livekit/agents/issues/5509
    User turn starting while the agent is ``thinking`` must pause the
    pausable output so the stale reply never promotes to ``speaking``.
    """
    speed = 1
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Here is a long story for you ... the end.", duration=1.0)
    actions.add_tts(10.0)
    actions.add_user_speech(3.0, 4.0, "Stop!", stt_delay=0.2)

    session = create_session(actions, speed_factor=speed, can_pause_audio=True)
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
    check_timestamp(agent_state_events[2].created_at - t_origin, 3.5, speed_factor=speed)

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


async def test_false_interruption_before_speaking_resumes() -> None:
    """
    Brief VAD-only noise during ``thinking`` must pause then resume on VAD EOS,
    letting the stale reply play through normally.
    """
    speed = 1
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Here is a short reply.", ttft=0.05, duration=0.05)
    actions.add_tts(5.0, ttfb=0.05, duration=0.05)
    # brief VAD-only noise — same shape as the can_pause=False test, different capability
    actions.add_user_speech(3.0, 3.3, "")

    session = create_session(actions, speed_factor=speed, can_pause_audio=True)
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
    check_timestamp(speaking_events[0].created_at - t_origin, 3.8, speed_factor=speed)

    # the reply plays to completion (not interrupted); playback_position covers the
    # full audio duration
    assert len(playback_finished_events) == 1
    assert playback_finished_events[0].interrupted is False
    check_timestamp(playback_finished_events[0].playback_position, 5.0, speed_factor=speed)


async def test_generate_reply() -> None:
    """
    Test `generate_reply` in `on_enter` and tool call, `say` in `on_user_turn_completed`
    """
    speed = 1

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

    session = create_session(actions, speed_factor=speed)
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
    check_timestamp(playback_finished_events[0].playback_position, 2.0, speed_factor=speed)
    assert playback_finished_events[1].interrupted is False
    check_timestamp(playback_finished_events[1].playback_position, 1.0, speed_factor=speed)
    assert playback_finished_events[2].interrupted is False
    check_timestamp(playback_finished_events[2].playback_position, 3.0, speed_factor=speed)

    # function_tools_executed
    assert len(tool_executed_events) == 1
    assert tool_executed_events[0].function_calls[0].name == "goodbye"

    # conversation_item_added
    assert len(conversation_events) == 5
    assert conversation_events[0].item.type == "agent_handoff"
    assert conversation_events[1].item.type == "message"
    assert conversation_events[1].item.role == "assistant"
    assert conversation_events[1].item.text_content == "What can I do for you!"
    check_timestamp(conversation_events[1].created_at - t_origin, 2.5, speed_factor=speed)
    assert conversation_events[2].item.type == "message"
    assert conversation_events[2].item.role == "user"
    assert conversation_events[2].item.text_content == "bye"
    check_timestamp(conversation_events[2].created_at - t_origin, 4.5, speed_factor=speed)
    assert conversation_events[3].item.type == "message"
    assert conversation_events[3].item.role == "assistant"
    assert conversation_events[3].item.text_content == "session.say from on_user_turn_completed"
    check_timestamp(
        conversation_events[3].created_at - t_origin, 5.5, speed_factor=speed, max_abs_diff=1.0
    )
    assert conversation_events[4].item.type == "message"
    assert conversation_events[4].item.role == "assistant"
    assert conversation_events[4].item.text_content == "Goodbye! have a nice day!"
    check_timestamp(
        conversation_events[4].created_at - t_origin, 9.0, speed_factor=speed, max_abs_diff=1.0
    )

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


async def test_on_enter_hides_ignore_on_enter_tools() -> None:
    """IGNORE_ON_ENTER tools (bare + toolset-nested) are hidden inside on_enter, restored after."""

    class _Toolset(Toolset):
        def __init__(self) -> None:
            end_call = function_tool(
                self._end_call,
                name="end_call",
                description="ends the call",
                flags=ToolFlag.IGNORE_ON_ENTER,
            )
            keep = function_tool(self._keep, name="keep", description="a normal tool")
            super().__init__(id="ts", tools=[end_call, keep])

        async def _end_call(self) -> None: ...

        async def _keep(self) -> None: ...

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def bare_ignored() -> None:
        """A bare flagged tool."""

    class _Agent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="you are helpful", tools=[_Toolset(), bare_ignored])

        async def on_enter(self) -> None:
            self.session.generate_reply(instructions="instructions:say hello to the user")

    actions = FakeActions()
    actions.add_llm("Hello!", input="instructions:say hello to the user")
    actions.add_tts(1.0)
    actions.add_user_speech(2.0, 3.0, "hi there")
    actions.add_llm("How can I help?")
    actions.add_tts(1.0)

    session = create_session(actions)

    captured: list[set[str]] = []
    orig_chat = session.llm.chat

    def _recording_chat(*, chat_ctx, tools=None, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(
            {t.info.name for t in (tools or []) if isinstance(t, (FunctionTool, RawFunctionTool))}
        )
        return orig_chat(chat_ctx=chat_ctx, tools=tools, **kwargs)

    session.llm.chat = _recording_chat  # type: ignore[method-assign]

    await asyncio.wait_for(run_session(session, _Agent()), timeout=SESSION_TIMEOUT)

    assert len(captured) == 2
    # on_enter reply: flagged tools hidden, normal tool still offered
    assert captured[0] == {"keep"}
    # a later (non-on_enter) turn sees every tool again
    assert captured[1] == {"keep", "end_call", "bare_ignored"}


async def test_on_enter_hides_tools_in_nested_tool_reply() -> None:
    """When an on_enter reply calls a tool, the tool-response follow-up (a nested speech task)
    also hides the flagged tools — proving the on_enter contextvar reaches nested tasks."""

    class _Toolset(Toolset):
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
    async def bare_ignored() -> None:
        """A bare flagged tool."""

    class _Agent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="you are helpful", tools=[_Toolset(), bare_ignored])

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

    captured: list[set[str]] = []
    orig_chat = session.llm.chat

    def _recording_chat(*, chat_ctx, tools=None, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(
            {t.info.name for t in (tools or []) if isinstance(t, (FunctionTool, RawFunctionTool))}
        )
        return orig_chat(chat_ctx=chat_ctx, tools=tools, **kwargs)

    session.llm.chat = _recording_chat  # type: ignore[method-assign]

    await asyncio.wait_for(run_session(session, _Agent()), timeout=SESSION_TIMEOUT)

    assert len(captured) == 3
    greeting, tool_reply, user_turn = captured
    # both the greeting reply and its nested tool-response follow-up hide the flagged tools
    assert greeting == {"keep"}
    assert tool_reply == {"keep"}
    # the later (non-on_enter) user turn sees every tool again
    assert user_turn == {"keep", "end_call", "bare_ignored"}


def test_on_enter_ignored_tools() -> None:
    """_on_enter_ignored_tools returns flagged tools only inside this agent/session's on_enter."""
    from livekit.agents.voice.agent_activity import _OnEnterContextVar, _OnEnterData

    class _Toolset(Toolset):
        def __init__(self) -> None:
            end_call = function_tool(
                self._end_call, name="end_call", description="ends", flags=ToolFlag.IGNORE_ON_ENTER
            )
            ts_keep = function_tool(self._keep, name="ts_keep", description="normal")
            super().__init__(id="ts", tools=[end_call, ts_keep])

        async def _end_call(self) -> None: ...

        async def _keep(self) -> None: ...

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def bare_ignored() -> None:
        """A flagged bare tool."""

    @function_tool
    async def bare_keep() -> None:
        """A normal bare tool."""

    activity = object.__new__(AgentActivity)
    activity._agent = object()  # type: ignore[assignment]
    activity._session = object()  # type: ignore[assignment]
    tool_ctx = ToolContext([_Toolset(), bare_ignored, bare_keep])

    # outside on_enter: nothing is ignored
    assert activity._on_enter_ignored_tools(tool_ctx) == []

    # inside this agent/session's on_enter: flagged tools (bare + nested) are returned
    tk = _OnEnterContextVar.set(_OnEnterData(session=activity._session, agent=activity._agent))
    try:
        ignored = {t.info.name for t in activity._on_enter_ignored_tools(tool_ctx)}
    finally:
        _OnEnterContextVar.reset(tk)
    assert ignored == {"end_call", "bare_ignored"}

    # a different agent's on_enter must not leak in
    tk = _OnEnterContextVar.set(_OnEnterData(session=activity._session, agent=object()))
    try:
        assert activity._on_enter_ignored_tools(tool_ctx) == []
    finally:
        _OnEnterContextVar.reset(tk)


async def test_aec_warmup() -> None:
    """AEC warmup should block audio-activity-based interruptions during the warmup window.

    Without warmup, VAD-based interruption fires at 4.0 + 0.5 = 4.5s.
    With warmup (3.0s from speaking at 3.5s, expires at 6.5s), the VAD path is blocked.
    The interruption is delayed to 5.5s (EOU: speech end 5.0 + 0.5 endpointing delay)
    because FakeSTT is timer-based and still produces transcripts during warmup.
    """
    speed = 1
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Here is a long story for you ... the end.")
    actions.add_tts(15.0)  # playout starts at 3.5s
    # user speaks at 4.0-5.0s — within warmup window (3.5 + 3.0 = 6.5s expiry)
    # without warmup: VAD interruption at 4.0 + 0.5 = 4.5s
    # with warmup: VAD blocked, falls through to EOU at 5.0 + 0.5 = 5.5s
    actions.add_user_speech(4.0, 5.0, "Stop!", stt_delay=0.2)

    session = create_session(
        actions,
        speed_factor=speed,
        extra_kwargs={"aec_warmup_duration": 3.0},
    )
    agent = MyAgent()

    agent_state_events: list[AgentStateChangedEvent] = []
    playback_finished_events: list[PlaybackFinishedEvent] = []
    session.on("agent_state_changed", agent_state_events.append)
    session.output.audio.on("playback_finished", playback_finished_events.append)

    t_origin = await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert len(playback_finished_events) == 1
    assert playback_finished_events[0].interrupted is True

    assert agent_state_events[0].new_state == "listening"
    assert agent_state_events[1].new_state == "thinking"
    assert agent_state_events[2].new_state == "speaking"
    # interruption delayed to 5.5s (EOU), not 4.5s (VAD was blocked by warmup)
    speaking_to_listening = next(e for e in agent_state_events[3:] if e.new_state == "listening")
    check_timestamp(speaking_to_listening.created_at - t_origin, 5.5, speed_factor=speed)


async def test_start_boundary_does_not_block_vad_interruption() -> None:
    """backchannel boundary should not interfere with VAD-based interruption when adaptive
    detection is not active. The cooldown timer runs but has no effect on the VAD path.

    This validates that the backchannel_boundary config is properly handled and doesn't
    regress normal interruption behavior.
    """
    speed = 1
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Here is a long story for you ... the end.")
    actions.add_tts(15.0)  # playout starts at ~3.5s
    # user speaks at 4.0-5.0s — within the 1s warmup window (3.5 + 1.0 = 4.5s expiry)
    # VAD interruption at 4.0 + 0.5 = 4.5s (warmup does NOT block VAD)
    actions.add_user_speech(4.0, 5.0, "Stop!", stt_delay=0.2)

    session = create_session(
        actions,
        speed_factor=speed,
        extra_kwargs={"aec_warmup_duration": None},
    )
    agent = MyAgent()

    agent_state_events: list[AgentStateChangedEvent] = []
    playback_finished_events: list[PlaybackFinishedEvent] = []
    session.on("agent_state_changed", agent_state_events.append)
    session.output.audio.on("playback_finished", playback_finished_events.append)

    t_origin = await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert len(playback_finished_events) == 1
    assert playback_finished_events[0].interrupted is True

    assert agent_state_events[0].new_state == "listening"
    assert agent_state_events[1].new_state == "thinking"
    assert agent_state_events[2].new_state == "speaking"
    # VAD interruption fires normally at ~4.5s (warmup doesn't block VAD path)
    speaking_to_listening = next(e for e in agent_state_events[3:] if e.new_state == "listening")
    check_timestamp(speaking_to_listening.created_at - t_origin, 4.5, speed_factor=speed)


async def test_backchannel_boundary_suppresses_start_boundary_backchannel() -> None:
    actions = FakeActions()
    session = create_session(
        actions,
        turn_handling={"interruption": {"backchannel_boundary": (0.05, 0.0)}},
    )
    hooks = _TestRecognitionHooks()
    recognition = AudioRecognition(
        session,
        hooks=hooks,
        endpointing=BaseEndpointing(min_delay=0.1, max_delay=1.0),
        stt=None,
        vad=None,
        using_default_vad=False,
        interruption_detection=None,
        turn_detection="vad",
    )

    try:
        recognition._on_start_of_agent_speech(started_at=time.time())
        # backchannels during the cooldown are dropped (they are a no-op anyway,
        # but this guards against the gate firing on `on_interruption`)
        await recognition._on_overlap_speech_event(_backchannel_event())
        assert hooks.interruptions == []

        # a real interruption during the cooldown must still fire
        await recognition._on_overlap_speech_event(_interruption_event())
        assert len(hooks.interruptions) == 1

        # after cooldown, both event types behave normally
        await asyncio.sleep(0.06)
        await recognition._on_overlap_speech_event(_backchannel_event())
        await recognition._on_overlap_speech_event(_interruption_event())
        assert len(hooks.interruptions) == 2
    finally:
        await _close_test_session(session)


async def _make_stt_eos_recognition() -> AudioRecognition:
    return AudioRecognition(
        create_session(FakeActions()),
        hooks=_TestRecognitionHooks(),
        endpointing=BaseEndpointing(min_delay=0.0, max_delay=0.0),
        stt=None,
        vad=None,
        using_default_vad=False,
        interruption_detection=None,
        turn_detection="stt",
    )


async def test_stt_eos_resets_active_vad_stream_without_restarting_vad() -> None:
    recognition = await _make_stt_eos_recognition()
    recognition._speaking = True
    recognition._vad_speech_started = True
    recognition._vad = MagicMock()
    resettable_stream = MagicMock()
    recognition._vad_stream = resettable_stream

    try:
        with patch.object(recognition, "_update_vad") as update_vad:
            await recognition._on_stt_event(SpeechEvent(type=SpeechEventType.END_OF_SPEECH))

        resettable_stream.flush.assert_called_once_with()
        update_vad.assert_not_called()
        assert recognition._vad_stream is resettable_stream
    finally:
        if recognition._end_of_turn_task is not None:
            await aio.cancel_and_wait(recognition._end_of_turn_task)
        await _close_test_session(recognition._session)


async def test_stt_eos_falls_back_to_update_vad_when_no_active_stream() -> None:
    recognition = await _make_stt_eos_recognition()
    recognition._speaking = True
    recognition._vad_speech_started = True
    recognition._vad = MagicMock()
    recognition._vad_stream = None

    try:
        with patch.object(recognition, "_update_vad") as update_vad:
            await recognition._on_stt_event(SpeechEvent(type=SpeechEventType.END_OF_SPEECH))

        update_vad.assert_called_once_with(recognition._vad)
    finally:
        if recognition._end_of_turn_task is not None:
            await aio.cancel_and_wait(recognition._end_of_turn_task)
        await _close_test_session(recognition._session)


async def test_backchannel_boundary_releases_end_boundary_transcript() -> None:
    actions = FakeActions()
    session = create_session(
        actions,
        turn_handling={"interruption": {"backchannel_boundary": (0.0, 0.5)}},
    )
    recognition = AudioRecognition(
        session,
        hooks=_TestRecognitionHooks(),
        endpointing=BaseEndpointing(min_delay=0.1, max_delay=1.0),
        stt=None,
        vad=None,
        using_default_vad=False,
        interruption_detection=None,
        turn_detection="vad",
    )
    recognition._interruption_enabled = True
    recognition._interruption_ch = aio.Chan[inference.InterruptionDataFrameType]()
    input_started_at = time.time() - 10.0
    # the input anchor lives on the STT pipeline (see _STTPipeline.input_started_at)
    recognition._stt_pipeline = SimpleNamespace(input_started_at=input_started_at)  # type: ignore[assignment]

    try:
        # the agent speaks for a couple of seconds so the held transcript still lands
        # after the agent-speech start (the lower bound of the ignore window)
        recognition._on_start_of_agent_speech(started_at=time.time() - 2.0)
        speech_ended_at = time.time()
        recognition._on_end_of_agent_speech(ignore_user_transcript_until=speech_ended_at)

        assert not recognition._should_hold_stt_event(
            _final_transcript_event(
                text="near the boundary",
                start_time=speech_ended_at - input_started_at - 0.25,
                end_time=speech_ended_at - input_started_at,
            )
        )
        assert recognition._should_hold_stt_event(
            _final_transcript_event(
                text="before the boundary",
                start_time=speech_ended_at - input_started_at - 0.75,
                end_time=speech_ended_at - input_started_at - 0.5,
            )
        )
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
    # unknown speech duration: the min_duration gate lets it through (the VAD
    # event below carries its own speech_duration check)
    audio_recognition.current_speech_duration = None
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
        current_speech.interrupt.assert_called_once_with()
        assert any(
            record.levelno == logging.INFO
            and "falling back to VAD-based interruption" in record.message
            for record in caplog.records
        )
        assert not [record for record in caplog.records if record.levelno >= logging.WARNING]
    finally:
        await _close_test_session(session)


async def test_force_flush_held_transcripts_emits_buffered_events() -> None:
    actions = FakeActions()
    session = create_session(actions)
    hooks = _TestRecognitionHooks()
    recognition = AudioRecognition(
        session,
        hooks=hooks,
        endpointing=BaseEndpointing(min_delay=0.1, max_delay=1.0),
        stt=None,
        vad=None,
        using_default_vad=False,
        interruption_detection=None,
        turn_detection="manual",
    )
    recognition._transcript_buffer.append(
        _final_transcript_event(text="held transcript", start_time=0.0, end_time=1.0)
    )

    try:
        await recognition._flush_held_transcripts(cooldown=0.0, force=True)

        assert hooks.final_transcripts == ["held transcript"]
        assert not recognition._transcript_buffer
    finally:
        await _close_test_session(session)


@pytest.mark.parametrize(
    "preemptive_generation, expected_latency",
    [
        ({"preemptive_tts": True}, 0.7),
        ({"preemptive_tts": False}, 0.8),
        ({"enabled": False}, 1.1),
    ],
)
async def test_preemptive_generation(preemptive_generation: dict, expected_latency: float) -> None:
    speed = 1
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.0, "Hello, how are you?", stt_delay=0.1)
    actions.add_llm("I'm doing great, thank you!", ttft=0.1, duration=0.3)
    actions.add_tts(3.0, ttfb=0.3)
    # preemptive_generation with TTS enabled: e2e latency is 0.1+0.3+0.3=0.7s
    # preemptive_generation without TTS enabled: e2e latency is max(0.1+0.3, 0.5)+0.3=0.8s
    # preemptive_generation disabled: e2e latency is 0.5+0.3+0.3=1.1s

    session = create_session(
        actions,
        speed_factor=speed,
        turn_handling={"preemptive_generation": preemptive_generation},
    )
    agent = MyAgent()

    agent_state_events: list[AgentStateChangedEvent] = []
    user_state_events: list[UserStateChangedEvent] = []
    session.on("agent_state_changed", agent_state_events.append)
    session.on("user_state_changed", user_state_events.append)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)
    assert len(user_state_events) == 2
    assert user_state_events[0].old_state == "listening"
    assert user_state_events[0].new_state == "speaking"
    assert user_state_events[1].new_state == "listening"
    t_user_stop_speaking = user_state_events[1].created_at

    assert len(agent_state_events) == 4
    assert agent_state_events[0].old_state == "initializing"
    assert agent_state_events[0].new_state == "listening"
    assert agent_state_events[1].new_state == "thinking"
    assert agent_state_events[2].new_state == "speaking"
    t_agent_start_speaking = agent_state_events[2].created_at
    check_timestamp(
        t_agent_start_speaking - t_user_stop_speaking,
        t_target=expected_latency,
        speed_factor=speed,
        max_abs_diff=0.2,
    )
    assert agent_state_events[3].new_state == "listening"


@pytest.mark.parametrize(
    "session_preemptive, agent_preemptive, expected_latency",
    [
        # agent disables what the session enabled -> no preemptive generation (1.1s)
        ({"preemptive_tts": True}, {"enabled": False}, 1.1),
        # agent enables (with TTS) what the session disabled -> fully preemptive (0.7s)
        ({"enabled": False}, {"enabled": True, "preemptive_tts": True}, 0.7),
    ],
)
async def test_preemptive_generation_on_agent(
    session_preemptive: dict, agent_preemptive: dict, expected_latency: float
) -> None:
    # preemptive generation set on the agent must override the session value
    speed = 1
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.0, "Hello, how are you?", stt_delay=0.1)
    actions.add_llm("I'm doing great, thank you!", ttft=0.1, duration=0.3)
    actions.add_tts(3.0, ttfb=0.3)
    # preemptive_generation with TTS enabled: e2e latency is 0.1+0.3+0.3=0.7s
    # preemptive_generation disabled: e2e latency is 0.5+0.3+0.3=1.1s

    session = create_session(
        actions,
        speed_factor=speed,
        turn_handling={"preemptive_generation": session_preemptive},
    )
    agent = MyAgent(turn_handling={"preemptive_generation": agent_preemptive})

    agent_state_events: list[AgentStateChangedEvent] = []
    user_state_events: list[UserStateChangedEvent] = []
    session.on("agent_state_changed", agent_state_events.append)
    session.on("user_state_changed", user_state_events.append)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)
    t_user_stop_speaking = user_state_events[1].created_at
    t_agent_start_speaking = agent_state_events[2].created_at
    check_timestamp(
        t_agent_start_speaking - t_user_stop_speaking,
        t_target=expected_latency,
        speed_factor=speed,
        max_abs_diff=0.2,
    )


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
    """
    Test interrupt during preemptive generation and on_user_turn_completed.
    """
    speed = 1
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.0, "Tell me a story", stt_delay=0.2)
    actions.add_llm("Here is a story for you...", ttft=0.1, duration=0.3)
    actions.add_tts(10.0, ttfb=1.0)  # latency after end of turn: 1.3s
    actions.add_user_speech(2.6, 3.2, "about a firefighter.")  # interrupt before speaking
    actions.add_llm("Here is a story about a firefighter...", ttft=0.1, duration=0.3)
    actions.add_tts(10.0, ttfb=0.3)

    session = create_session(
        actions,
        speed_factor=speed,
        turn_handling={"preemptive_generation": {"enabled": preemptive_generation}},
    )
    agent = MyAgent(on_user_turn_completed_delay=on_user_turn_completed_delay / speed)

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
    speed = 1
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

    session = create_session(actions, speed_factor=speed)
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
    """When the LLM emits a tool call with invalid arguments (missing required
    field, wrong type, malformed JSON, etc.), the faulty turn must NOT be
    stripped from the conversation. Instead the schema error is wrapped in a
    ToolError so the model receives a descriptive message and can self-correct
    on the next turn."""
    speed = 1
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "What's the weather?")
    # get_weather requires `location: str` — emit a call with no args so it
    # fails pydantic validation.
    actions.add_llm(
        content="",
        tool_calls=[
            FunctionToolCall(name="get_weather", arguments="{}", call_id="1"),
        ],
    )

    session = create_session(actions, speed_factor=speed)
    agent = MyAgent()

    tool_executed_events: list[FunctionToolsExecutedEvent] = []
    session.on("function_tools_executed", tool_executed_events.append)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    # Event was emitted with both the call AND a non-None output (i.e., not stripped).
    assert len(tool_executed_events) == 1
    ev = tool_executed_events[0]
    assert len(ev.function_calls) == 1
    assert ev.function_calls[0].name == "get_weather"
    assert ev.function_call_outputs[0] is not None
    output = ev.function_call_outputs[0]
    assert output.is_error is True

    # The model must see a descriptive, schema-specific error — NOT the generic
    # "An internal error occurred" string we reserve for unexpected exceptions.
    assert "An internal error occurred" not in output.output
    assert "get_weather" in output.output
    # Pydantic validation error references the missing field.
    assert "location" in output.output

    # The faulty call AND its error output must both end up in chat history so
    # the LLM can see what it did wrong on the next turn (not stripped).
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
    """When a tool body raises a non-ToolError exception, the model receives
    the generic "An internal error occurred" message so we don't leak internal
    details. Validation-error path is tested separately."""

    class _BrokenToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You are a helpful assistant.")

        @function_tool
        async def get_weather(self, location: str) -> str:
            """Always blows up."""
            raise RuntimeError("kaboom: secret database password leaked")

    speed = 1
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "What's the weather in Tokyo?")
    actions.add_llm(
        content="",
        tool_calls=[
            FunctionToolCall(name="get_weather", arguments='{"location": "Tokyo"}', call_id="1"),
        ],
    )

    session = create_session(actions, speed_factor=speed)
    agent = _BrokenToolAgent()

    tool_executed_events: list[FunctionToolsExecutedEvent] = []
    session.on("function_tools_executed", tool_executed_events.append)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert len(tool_executed_events) == 1
    output = tool_executed_events[0].function_call_outputs[0]
    assert output is not None
    assert output.is_error is True
    # Generic message — the RuntimeError details must NOT leak to the model.
    assert output.output == "An internal error occurred"
    assert "kaboom" not in output.output
    assert "secret" not in output.output


# helpers


class _TestRecognitionHooks:
    def __init__(self) -> None:
        self.interruptions: list[inference.OverlappingSpeechEvent] = []
        self.final_transcripts: list[str] = []

    def on_interruption(self, ev: inference.OverlappingSpeechEvent) -> None:
        self.interruptions.append(ev)

    def on_start_of_speech(self, ev: object, speech_start_time: float) -> None:
        pass

    def on_vad_inference_done(self, ev: object) -> None:
        pass

    def on_end_of_speech(self, ev: object) -> None:
        pass

    def on_interim_transcript(self, ev: SpeechEvent, *, speaking: bool | None) -> None:
        pass

    def on_final_transcript(self, ev: SpeechEvent, *, speaking: bool | None = None) -> None:
        self.final_transcripts.append(ev.alternatives[0].text)

    def on_end_of_turn(self, info: _EndOfTurnInfo) -> bool:
        return True

    def on_preemptive_generation(self, info: object) -> None:
        pass

    def retrieve_chat_ctx(self) -> ChatContext:
        return ChatContext.empty()


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


def _final_transcript_event(*, text: str, start_time: float, end_time: float) -> SpeechEvent:
    return SpeechEvent(
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


async def _close_test_session(session: object) -> None:
    await session.aclose()
    audio_output = session.output.audio
    synchronizer = getattr(audio_output, "_synchronizer", None)
    if synchronizer is not None:
        await synchronizer.aclose()


def check_timestamp(
    t_event: float,
    t_target: float,
    *,
    speed_factor: float = 1.0,
    max_abs_diff: float = 0.75,
    min_real_time_diff: float = 0.3,
) -> None:
    """
    Check if the event timestamp is within the target timestamp +/- max_abs_diff.
    The event timestamp is scaled by the speed factor.

    ``max_abs_diff`` is expressed in scaled time. A real-time floor of
    ``min_real_time_diff`` (wallclock seconds) is also applied so high
    ``speed_factor`` values don't compress the effective tolerance below the
    scheduling-jitter noise floor on CI runners — without this, the real-time
    tolerance is ``max_abs_diff / speed_factor``, which at speed=5 is only
    150 ms and routinely flakes.
    """
    t_event_scaled = t_event * speed_factor
    effective_diff = max(max_abs_diff, min_real_time_diff * speed_factor)
    print(
        f"check_timestamp: t_event={t_event_scaled} (real {t_event:.3f}s), "
        f"t_target={t_target}, effective_diff={effective_diff} "
        f"(max_abs_diff={max_abs_diff}, min_real_time_diff={min_real_time_diff})"
    )
    assert abs(t_event_scaled - t_target) <= effective_diff, (
        f"event timestamp {t_event_scaled} is not within {effective_diff} of target {t_target} "
        f"(real-time tolerance {effective_diff / speed_factor:.3f}s)"
    )


async def test_silent_tool_call_pause_state_does_not_leak_into_tool_reply() -> None:
    speed = 1
    actions = FakeActions()
    actions.add_user_speech(0.1, 0.2, "What's the weather in Tokyo?", stt_delay=0.05)

    # Silent tool-call step: no spoken preamble/audio before the function call.
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

    # VAD-only speech starts during the silent tool-call generation and remains
    # active after the tool reply starts.
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
        speed_factor=speed,
        can_pause_audio=True,
        turn_handling={"interruption": {"false_interruption_timeout": 0.2 / speed}},
    )
    agent = MyAgent()

    agent_state_events: list[AgentStateChangedEvent] = []
    false_interruption_events: list[AgentFalseInterruptionEvent] = []

    session.on("agent_state_changed", agent_state_events.append)
    session.on("agent_false_interruption", false_interruption_events.append)

    await asyncio.wait_for(
        run_session(session, agent, drain_delay=0.8 / speed),
        timeout=SESSION_TIMEOUT,
    )

    transitions = [(ev.old_state, ev.new_state) for ev in agent_state_events]
    silent_step_finished = transitions.index(("speaking", "listening"))

    # Before the fix this is ("listening", "thinking") because the pause state
    # captured during the silent tool-call step leaks into the tool reply.
    assert transitions[silent_step_finished + 1] == ("listening", "speaking")
    assert false_interruption_events
    assert false_interruption_events[-1].resumed is True


async def test_default_vad_is_auto_provisioned() -> None:
    from livekit.agents.voice.agent_session import AgentSession

    session = AgentSession()
    try:
        assert session.vad is not None
        assert session._using_default_vad is True
    finally:
        await session.aclose()


async def test_explicit_vad_none_opts_out() -> None:
    from livekit.agents.voice.agent_session import AgentSession

    session = AgentSession(vad=None)
    try:
        assert session.vad is None
        assert session._using_default_vad is False
    finally:
        await session.aclose()


async def test_user_supplied_vad_clears_default_flag() -> None:
    from livekit.agents.voice.agent_session import AgentSession

    from .fake_vad import FakeVAD

    user_vad = FakeVAD(fake_user_speeches=[])

    session = AgentSession(vad=user_vad)
    try:
        assert session.vad is user_vad
        assert session._using_default_vad is False
    finally:
        await session.aclose()


async def test_default_turn_detection_builds_default_eot() -> None:
    """No turn_detection given → session auto-provisions a default TurnDetector."""
    from livekit.agents.voice.agent_session import AgentSession
    from livekit.agents.voice.turn import _StreamingTurnDetector

    session = AgentSession()
    try:
        assert isinstance(session.turn_detection, _StreamingTurnDetector)
    finally:
        await session.aclose()


async def test_turn_detection_none_opts_out() -> None:
    """Explicit None opts out of turn detection (no default detector built)."""
    from livekit.agents.voice.agent_session import AgentSession

    session = AgentSession(turn_handling={"turn_detection": None})
    try:
        assert session.turn_detection is None
    finally:
        await session.aclose()


async def test_user_supplied_turn_detector_passes_through() -> None:
    from livekit.agents import inference
    from livekit.agents.voice.agent_session import AgentSession

    user_detector = inference.TurnDetector(version="v1-mini")
    session = AgentSession(turn_handling={"turn_detection": user_detector})
    try:
        assert session.turn_detection is user_detector
    finally:
        await session.aclose()


async def test_streaming_detector_uses_streaming_endpointing_defaults() -> None:
    """Default session → streaming detector → tighter 0.3/2.5 endpointing defaults."""
    from livekit.agents.voice.agent_session import AgentSession

    session = AgentSession()
    try:
        assert session._opts.endpointing["min_delay"] == 0.3
        assert session._opts.endpointing["max_delay"] == 2.5
        assert session._opts.endpointing_overrides == {}
    finally:
        await session.aclose()


async def test_non_streaming_detector_uses_legacy_endpointing_defaults() -> None:
    """A non-streaming mode keeps the legacy 0.5/3.0 defaults."""
    from livekit.agents.voice.agent_session import AgentSession

    session = AgentSession(turn_handling={"turn_detection": "vad"})
    try:
        assert session._opts.endpointing["min_delay"] == 0.5
        assert session._opts.endpointing["max_delay"] == 3.0
    finally:
        await session.aclose()


async def test_explicit_endpointing_overrides_streaming_default_per_key() -> None:
    """An explicit delay is honored; the unset one still gets the streaming default."""
    from livekit.agents.voice.agent_session import AgentSession

    session = AgentSession(turn_handling={"endpointing": {"min_delay": 0.4}})
    try:
        assert session._opts.endpointing["min_delay"] == 0.4
        assert session._opts.endpointing["max_delay"] == 2.5
        assert session._opts.endpointing_overrides == {"min_delay": 0.4}
    finally:
        await session.aclose()


async def test_user_streaming_detector_uses_streaming_defaults() -> None:
    """A user-constructed streaming detector also triggers the streaming defaults."""
    from livekit.agents import inference
    from livekit.agents.voice.agent_session import AgentSession

    session = AgentSession(
        turn_handling={"turn_detection": inference.TurnDetector(version="v1-mini")}
    )
    try:
        assert session._opts.endpointing["min_delay"] == 0.3
        assert session._opts.endpointing["max_delay"] == 2.5
    finally:
        await session.aclose()


async def test_deprecated_turn_detection_vad_uses_legacy_defaults() -> None:
    """Deprecated turn_detection arg + no delays → legacy defaults (non-streaming)."""
    from livekit.agents.voice.agent_session import AgentSession

    session = AgentSession(turn_detection="vad")
    try:
        assert session._opts.endpointing["min_delay"] == 0.5
        assert session._opts.endpointing["max_delay"] == 3.0
    finally:
        await session.aclose()


async def test_agent_turn_detection_override_resolves_endpointing_per_activity() -> None:
    """endpointing_opts uses the activity's resolved detector, not just the session's."""
    from livekit.agents.voice.agent_session import AgentSession

    from .fake_vad import FakeVAD

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


async def test_runtime_endpointing_opts_survive_handoff() -> None:
    """update_options changes are recorded as overrides, so a new activity keeps them."""
    from livekit.agents.voice.agent_session import AgentSession

    from .fake_vad import FakeVAD

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
    speed = 1
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Hello, how are you?", stt_delay=0.2)
    # the agent's llm_node injects a FlushSentinel, splitting the reply into two
    # segments; register a TTS response keyed by each segment's text
    actions.add_tts(1.0, input="Hello there. ", ttfb=0.1, duration=0.1)
    actions.add_tts(1.0, input="How are you?", ttfb=0.1, duration=0.1)

    session = create_session(actions, speed_factor=speed)
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
    speed = 1
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Hello, how are you?", stt_delay=0.2)
    # long first segment so the interrupt lands while it is still playing
    actions.add_tts(15.0, input="Hello there. ", ttfb=0.1, duration=0.1)
    actions.add_tts(1.0, input="How are you?", ttfb=0.1, duration=0.1)

    session = create_session(actions, speed_factor=speed)
    agent = FlushMultiSegmentAgent()

    playback_finished_events: list[PlaybackFinishedEvent] = []
    session.output.audio.on("playback_finished", playback_finished_events.append)

    asyncio.get_event_loop().call_later(5 / speed, session.interrupt)

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
