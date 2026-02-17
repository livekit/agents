from __future__ import annotations

import asyncio

import pytest

from livekit.agents import (
    Agent,
    AgentStateChangedEvent,
    ConversationItemAddedEvent,
    MetricsCollectedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
    function_tool,
)
from livekit.agents.llm import FunctionToolCall
from livekit.agents.llm.chat_context import ChatContext, ChatMessage
from livekit.agents.voice.events import FunctionToolsExecutedEvent
from livekit.agents.voice.io import PlaybackFinishedEvent

from .fake_session import FakeActions, create_session, run_session


class MyAgent(Agent):
    def __init__(
        self,
        *,
        generate_reply_on_enter: bool = False,
        say_on_user_turn_completed: bool = False,
        on_user_turn_completed_delay: float = 0.0,
    ) -> None:
        super().__init__(
            instructions=("You are a helpful assistant."),
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


async def test_events_and_metrics() -> None:
    speed = 5.0
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
    assert len(conversation_events) == 2
    assert conversation_events[0].item.type == "message"
    assert conversation_events[0].item.role == "user"
    assert conversation_events[0].item.text_content == "Hello, how are you?"
    check_timestamp(conversation_events[0].created_at - t_origin, 3.0, speed_factor=speed)
    assert conversation_events[1].item.type == "message"
    assert conversation_events[1].item.role == "assistant"
    assert conversation_events[1].item.text_content == "I'm doing well, thank you!"
    check_timestamp(conversation_events[1].created_at - t_origin, 5.5, speed_factor=speed)

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


async def test_tool_call() -> None:
    speed = 5.0
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
    speed = 5.0
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Here is a long story for you ... the end.")
    actions.add_tts(10.0)  # playout starts at 3.5s
    actions.add_user_speech(5.0, 6.0, "Stop!", stt_delay=0.2)
    # interrupted at 5.5s, min_interruption_duration=0.5

    session = create_session(
        actions,
        speed_factor=speed,
        extra_kwargs={"resume_false_interruption": resume_false_interruption},
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
    speed = 5.0
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Here is a long story for you ... the end.")
    actions.add_tts(5.0)  # playout starts at 3.5s
    actions.add_user_speech(5.0, 6.0, "Stop!")
    actions.add_user_speech(6.5, 7.5, "ok, stop!", stt_delay=0.0)
    # it should interrupt at 7.5s after stt, playback position is 4.0s

    # test min_interruption_words
    session = create_session(
        actions, speed_factor=speed, extra_kwargs={"min_interruption_words": 3}
    )
    playback_finished_events: list[PlaybackFinishedEvent] = []
    session.output.audio.on("playback_finished", playback_finished_events.append)

    await asyncio.wait_for(run_session(session, MyAgent()), timeout=SESSION_TIMEOUT)

    assert len(playback_finished_events) == 1
    assert playback_finished_events[0].interrupted is True
    check_timestamp(playback_finished_events[0].playback_position, 4.0, speed_factor=speed)

    # test allow_interruptions=False
    session = create_session(
        actions, speed_factor=speed, extra_kwargs={"allow_interruptions": False}
    )
    playback_finished_events.clear()
    session.output.audio.on("playback_finished", playback_finished_events.append)

    await asyncio.wait_for(run_session(session, MyAgent()), timeout=SESSION_TIMEOUT)

    assert len(playback_finished_events) == 1
    assert playback_finished_events[0].interrupted is False
    check_timestamp(playback_finished_events[0].playback_position, 5.0, speed_factor=speed)


async def test_interruption_by_text_input() -> None:
    speed = 5.0
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

    await asyncio.wait_for(run_session(session, agent, drain_delay=0.5), timeout=SESSION_TIMEOUT)

    assert len(playback_finished_events) == 2
    assert playback_finished_events[0].interrupted is True

    print(agent_state_events)
    assert len(agent_state_events) == 7
    assert agent_state_events[0].old_state == "initializing"
    assert agent_state_events[0].new_state == "listening"
    assert agent_state_events[1].new_state == "thinking"
    assert agent_state_events[2].new_state == "speaking"
    assert (
        agent_state_events[3].new_state == "listening"
    )  # not sure how we can avoid listening here?
    # speaking to thinking when interrupted by text
    assert agent_state_events[4].new_state == "thinking"
    assert agent_state_events[5].new_state == "speaking"
    assert agent_state_events[6].new_state == "listening"

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
    speed = 5.0
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Here is a long story for you ... the end.", duration=1.0)
    actions.add_tts(10.0)
    actions.add_user_speech(3.0, 4.0, "Stop!", stt_delay=0.2)

    session = create_session(
        actions,
        speed_factor=speed,
        extra_kwargs={"resume_false_interruption": resume_false_interruption},
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


async def test_generate_reply() -> None:
    """
    Test `generate_reply` in `on_enter` and tool call, `say` in `on_user_turn_completed`
    """
    speed = 5.0

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

    t_origin = await asyncio.wait_for(
        run_session(session, agent, drain_delay=0.5), timeout=SESSION_TIMEOUT
    )

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
    assert len(conversation_events) == 4
    assert conversation_events[0].item.type == "message"
    assert conversation_events[0].item.role == "assistant"
    assert conversation_events[0].item.text_content == "What can I do for you!"
    check_timestamp(conversation_events[0].created_at - t_origin, 2.5, speed_factor=speed)
    assert conversation_events[1].item.type == "message"
    assert conversation_events[1].item.role == "user"
    assert conversation_events[1].item.text_content == "bye"
    check_timestamp(conversation_events[1].created_at - t_origin, 4.5, speed_factor=speed)
    assert conversation_events[2].item.type == "message"
    assert conversation_events[2].item.role == "assistant"
    assert conversation_events[2].item.text_content == "session.say from on_user_turn_completed"
    check_timestamp(
        conversation_events[2].created_at - t_origin, 5.5, speed_factor=speed, max_abs_diff=1.0
    )
    assert conversation_events[3].item.type == "message"
    assert conversation_events[3].item.role == "assistant"
    assert conversation_events[3].item.text_content == "Goodbye! have a nice day!"
    check_timestamp(
        conversation_events[3].created_at - t_origin, 9.0, speed_factor=speed, max_abs_diff=1.0
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


@pytest.mark.parametrize(
    "preemptive_generation, expected_latency",
    [
        (True, 0.8),
        (False, 1.1),
    ],
)
async def test_preemptive_generation(preemptive_generation: bool, expected_latency: float) -> None:
    speed = 5.0
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.0, "Hello, how are you?", stt_delay=0.2)
    actions.add_llm("I'm doing great, thank you!", ttft=0.1, duration=0.3)
    actions.add_tts(3.0, ttfb=0.3)
    # preemptive_generation enabled: e2e latency is 0.2+0.3+0.3=0.8s
    # preemptive_generation disabled: e2e latency is 0.5+0.3+3.0=1.1s

    session = create_session(
        actions, speed_factor=speed, extra_kwargs={"preemptive_generation": preemptive_generation}
    )
    agent = MyAgent()

    agent_state_events: list[AgentStateChangedEvent] = []
    session.on("agent_state_changed", agent_state_events.append)

    t_origin = await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)
    assert len(agent_state_events) == 4
    assert agent_state_events[0].old_state == "initializing"
    assert agent_state_events[0].new_state == "listening"
    assert agent_state_events[1].new_state == "thinking"
    assert agent_state_events[2].new_state == "speaking"
    check_timestamp(
        agent_state_events[2].created_at - t_origin,
        t_target=2.0 + expected_latency,
        speed_factor=speed,
        max_abs_diff=0.2,
    )
    assert agent_state_events[3].new_state == "listening"


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
    speed = 5.0
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
        extra_kwargs={"preemptive_generation": preemptive_generation},
    )
    agent = MyAgent(on_user_turn_completed_delay=on_user_turn_completed_delay / speed)

    agent_state_events: list[AgentStateChangedEvent] = []
    conversation_events: list[ConversationItemAddedEvent] = []
    session.on("agent_state_changed", agent_state_events.append)
    session.on("conversation_item_added", conversation_events.append)

    await asyncio.wait_for(run_session(session, agent, drain_delay=1.0), timeout=SESSION_TIMEOUT)

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

    assert len(conversation_events) == 3
    assert conversation_events[0].item.type == "message"
    assert conversation_events[0].item.role == "user"
    assert conversation_events[0].item.text_content == "Tell me a story"
    assert conversation_events[1].item.type == "message"
    assert conversation_events[1].item.role == "user"
    assert conversation_events[1].item.text_content == "about a firefighter."
    assert conversation_events[2].item.type == "message"
    assert conversation_events[2].item.role == "assistant"
    assert conversation_events[2].item.text_content == "Here is a story about a firefighter..."


# helpers


def check_timestamp(
    t_event: float, t_target: float, *, speed_factor: float = 1.0, max_abs_diff: float = 0.5
) -> None:
    """
    Check if the event timestamp is within the target timestamp +/- max_abs_diff.
    The event timestamp is scaled by the speed factor.
    """
    t_event = t_event * speed_factor
    print(
        f"check_timestamp: t_event: {t_event}, t_target: {t_target}, max_abs_diff: {max_abs_diff}"
    )
    assert abs(t_event - t_target) <= max_abs_diff, (
        f"event timestamp {t_event} is not within {max_abs_diff} of target {t_target}"
    )
