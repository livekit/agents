from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentSession,
    AgentStateChangedEvent,
    ConversationItemAddedEvent,
    MetricsCollectedEvent,
    NotGivenOr,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
    function_tool,
    utils,
)
from livekit.agents.llm import FunctionToolCall
from livekit.agents.llm.chat_context import ChatContext, ChatMessage
from livekit.agents.voice.events import FunctionToolsExecutedEvent
from livekit.agents.voice.io import PlaybackFinishedEvent
from livekit.agents.voice.transcription.synchronizer import (
    TranscriptSynchronizer,
    _SyncedAudioOutput,
)

from .fake_io import FakeAudioInput, FakeAudioOutput, FakeTextOutput
from .fake_llm import FakeLLM, FakeLLMResponse
from .fake_stt import FakeSTT, FakeUserSpeech
from .fake_tts import FakeTTS, FakeTTSResponse
from .fake_vad import FakeVAD


class MyAgent(Agent):
    def __init__(
        self,
        *,
        generate_reply_on_enter: bool = False,
        say_on_user_turn_completed: bool = False,
    ) -> None:
        super().__init__(
            instructions=("You are a helpful assistant."),
        )
        self.generate_reply_on_enter = generate_reply_on_enter
        self.say_on_user_turn_completed = say_on_user_turn_completed

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
    assert len(chat_ctx_items) == 6
    assert chat_ctx_items[0].type == "message"
    assert chat_ctx_items[0].role == "system"
    assert chat_ctx_items[1].type == "message"
    assert chat_ctx_items[1].role == "user"
    assert chat_ctx_items[1].text_content == "What's the weather in Tokyo?"
    assert chat_ctx_items[2].type == "message"
    assert chat_ctx_items[2].role == "assistant"
    assert chat_ctx_items[2].text_content == "Let me check the weather for you."
    assert chat_ctx_items[3].type == "function_call"
    assert chat_ctx_items[3].name == "get_weather"
    assert chat_ctx_items[4].type == "function_call_output"
    assert chat_ctx_items[4].output == "The weather in Tokyo is sunny today."
    assert chat_ctx_items[5].type == "message"
    assert chat_ctx_items[5].role == "assistant"
    assert chat_ctx_items[5].text_content == "The weather in Tokyo is sunny today."


async def test_interruption() -> None:
    speed = 5.0
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Here is a long story for you ... the end.")
    actions.add_tts(10.0)  # playout starts at 3.5s
    actions.add_user_speech(5.0, 6.0, "Stop!")
    # interrupted at 5.5s, min_interruption_duration=0.5

    session = create_session(actions, speed_factor=speed)
    agent = MyAgent()

    agent_state_events: list[AgentStateChangedEvent] = []
    playback_finished_events: list[PlaybackFinishedEvent] = []
    session.on("agent_state_changed", agent_state_events.append)
    session.output.audio.on("playback_finished", playback_finished_events.append)

    t_origin = await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    chat_ctx_items = agent.chat_ctx.items
    assert len(chat_ctx_items) == 4
    assert chat_ctx_items[2].type == "message"
    assert chat_ctx_items[2].role == "assistant"
    assert chat_ctx_items[2].interrupted is True

    assert len(agent_state_events) == 6
    assert agent_state_events[0].old_state == "initializing"
    assert agent_state_events[0].new_state == "listening"
    assert agent_state_events[1].new_state == "thinking"
    assert agent_state_events[2].new_state == "speaking"
    assert agent_state_events[3].new_state == "listening"
    check_timestamp(agent_state_events[3].created_at - t_origin, 5.5, speed_factor=speed)
    assert agent_state_events[4].new_state == "thinking"
    check_timestamp(agent_state_events[4].created_at - t_origin, 6.5, speed_factor=speed)
    assert agent_state_events[5].new_state == "listening"
    check_timestamp(agent_state_events[5].created_at - t_origin, 6.5, speed_factor=speed)

    assert len(playback_finished_events) == 1
    assert playback_finished_events[0].interrupted is True
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

    await asyncio.wait_for(run_session(session, agent, drain_delay=2.0), timeout=SESSION_TIMEOUT)

    assert len(playback_finished_events) == 2
    assert playback_finished_events[0].interrupted is True

    assert len(agent_state_events) == 6
    assert agent_state_events[0].old_state == "initializing"
    assert agent_state_events[0].new_state == "listening"
    assert agent_state_events[1].new_state == "thinking"
    assert agent_state_events[2].new_state == "speaking"
    # speaking to thinking when interrupted by text
    assert agent_state_events[3].new_state == "thinking"
    assert agent_state_events[4].new_state == "speaking"
    assert agent_state_events[5].new_state == "listening"

    chat_ctx_items = agent.chat_ctx.items
    assert len(chat_ctx_items) == 5
    assert chat_ctx_items[0].type == "message"
    assert chat_ctx_items[0].role == "system"
    assert chat_ctx_items[1].type == "message"
    assert chat_ctx_items[1].role == "user"
    assert chat_ctx_items[1].text_content == "Tell me a story."
    assert chat_ctx_items[2].type == "message"
    assert chat_ctx_items[2].role == "assistant"
    assert chat_ctx_items[2].interrupted is True  # assistant message should be before text input
    assert chat_ctx_items[3].type == "message"
    assert chat_ctx_items[3].role == "user"
    assert chat_ctx_items[3].text_content == "stop from text input"
    assert chat_ctx_items[4].type == "message"
    assert chat_ctx_items[4].role == "assistant"
    assert chat_ctx_items[4].text_content == "Ok, I'll stop now."


async def test_interruption_before_speaking() -> None:
    speed = 5.0
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Here is a long story for you ... the end.", duration=1.0)
    actions.add_tts(10.0)
    actions.add_user_speech(3.0, 4.0, "Stop!")

    session = create_session(actions, speed_factor=speed)
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
        agent_state_events[2].created_at - t_origin, 3.5, speed_factor=speed
    )  # interrupted at 3.5s
    assert agent_state_events[3].new_state == "thinking"
    assert agent_state_events[4].new_state == "listening"

    assert len(playback_finished_events) == 0

    assert len(agent.chat_ctx.items) == 3
    assert agent.chat_ctx.items[0].type == "message"
    assert agent.chat_ctx.items[0].role == "system"
    assert agent.chat_ctx.items[1].type == "message"
    assert agent.chat_ctx.items[1].role == "user"
    assert agent.chat_ctx.items[1].text_content == "Tell me a story."
    # before we insert an empty assistant message with interrupted=True
    # now we ignore it when the text is empty
    assert agent.chat_ctx.items[2].type == "message"
    assert agent.chat_ctx.items[2].role == "user"
    assert agent.chat_ctx.items[2].text_content == "Stop!"


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
    assert len(agent.chat_ctx.items) == 7
    assert agent.chat_ctx.items[0].type == "message"
    assert agent.chat_ctx.items[0].role == "system"
    assert agent.chat_ctx.items[1].type == "message"
    assert agent.chat_ctx.items[1].role == "assistant"
    assert agent.chat_ctx.items[1].text_content == "What can I do for you!"
    assert agent.chat_ctx.items[2].type == "message"
    assert agent.chat_ctx.items[2].role == "user"
    assert agent.chat_ctx.items[2].text_content == "bye"
    assert agent.chat_ctx.items[3].type == "message"
    assert agent.chat_ctx.items[3].role == "assistant"
    assert agent.chat_ctx.items[3].text_content == "session.say from on_user_turn_completed"
    assert agent.chat_ctx.items[4].type == "function_call"
    assert agent.chat_ctx.items[5].type == "function_call_output"
    assert agent.chat_ctx.items[6].type == "message"
    assert agent.chat_ctx.items[6].role == "assistant"
    assert agent.chat_ctx.items[6].text_content == "Goodbye! have a nice day!"


# helpers


def create_session(
    actions: FakeActions,
    *,
    speed_factor: float = 1.0,
    extra_kwargs: dict[str, Any] | None = None,
) -> AgentSession:
    user_speeches = actions.get_user_speeches(speed_factor=speed_factor)
    llm_responses = actions.get_llm_responses(speed_factor=speed_factor)
    tts_responses = actions.get_tts_responses(speed_factor=speed_factor)

    stt = FakeSTT(fake_user_speeches=user_speeches)
    session = AgentSession(
        vad=FakeVAD(
            fake_user_speeches=user_speeches,
            min_silence_duration=0.5 / speed_factor,
            min_speech_duration=0.05 / speed_factor,
        ),
        stt=stt,
        llm=FakeLLM(fake_responses=llm_responses),
        tts=FakeTTS(fake_responses=tts_responses),
        min_interruption_duration=0.5 / speed_factor,
        min_endpointing_delay=0.5 / speed_factor,
        max_endpointing_delay=6.0 / speed_factor,
        **(extra_kwargs or {}),
    )

    # setup io with transcription sync
    audio_input = FakeAudioInput()
    audio_output = FakeAudioOutput()
    transcription_output = FakeTextOutput()

    transcript_sync = TranscriptSynchronizer(
        next_in_chain_audio=audio_output,
        next_in_chain_text=transcription_output,
        speed=speed_factor,
    )
    session.input.audio = audio_input
    session.output.audio = transcript_sync.audio_output
    session.output.transcription = transcript_sync.text_output
    return session


async def run_session(session: AgentSession, agent: Agent, *, drain_delay: float = 1.0) -> float:
    stt = session.stt
    audio_input = session.input.audio
    assert isinstance(stt, FakeSTT)
    assert isinstance(audio_input, FakeAudioInput)

    transcription_sync: TranscriptSynchronizer | None = None
    if isinstance(session.output.audio, _SyncedAudioOutput):
        transcription_sync = session.output.audio._synchronizer

    await session.start(agent)

    # start the fake vad and stt
    t_origin = time.time()
    audio_input.push(0.1)

    # wait for the user speeches to be processed
    await stt.fake_user_speeches_done

    await asyncio.sleep(drain_delay)
    with contextlib.suppress(RuntimeError):
        await session.drain()
    await session.aclose()

    if transcription_sync is not None:
        await transcription_sync.aclose()

    return t_origin


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


class FakeActions:
    def __init__(self) -> None:
        self._items: list[FakeUserSpeech | FakeLLMResponse | FakeTTSResponse] = []

    def add_user_speech(
        self, start_time: float, end_time: float, transcript: str, *, stt_delay: float = 0.2
    ) -> None:
        self._items.append(
            FakeUserSpeech(
                start_time=start_time,
                end_time=end_time,
                transcript=transcript,
                stt_delay=stt_delay,
            )
        )

    def add_llm(
        self,
        content: str,
        tool_calls: list[FunctionToolCall] | None = None,
        *,
        input: NotGivenOr[str] = NOT_GIVEN,
        ttft: float = 0.1,
        duration: float = 0.3,
    ) -> None:
        if (
            not utils.is_given(input)
            and self._items
            and isinstance(self._items[-1], FakeUserSpeech)
        ):
            # use the last user speech as input
            input = self._items[-1].transcript

        if not utils.is_given(input):
            raise ValueError("input is required or previous item needs to be a user speech")

        self._items.append(
            FakeLLMResponse(
                content=content,
                input=input,
                ttft=ttft,
                duration=duration,
                tool_calls=tool_calls or [],
            )
        )

    def add_tts(
        self,
        audio_duration: float,
        *,
        input: NotGivenOr[str] = NOT_GIVEN,
        ttfb: float = 0.2,
        duration: float = 0.3,
    ) -> None:
        if (
            not utils.is_given(input)
            and self._items
            and isinstance(self._items[-1], FakeLLMResponse)
        ):
            input = self._items[-1].content

        if not utils.is_given(input):
            raise ValueError("input is required or previous item needs to be a llm response")

        self._items.append(
            FakeTTSResponse(
                audio_duration=audio_duration,
                input=input,
                ttfb=ttfb,
                duration=duration,
            )
        )

    def get_user_speeches(self, *, speed_factor: float = 1.0) -> list[FakeUserSpeech]:
        return [item.speed_up(speed_factor) for item in self._items if item.type == "user_speech"]

    def get_llm_responses(self, *, speed_factor: float = 1.0) -> list[FakeLLMResponse]:
        return [item.speed_up(speed_factor) for item in self._items if item.type == "llm"]

    def get_tts_responses(self, *, speed_factor: float = 1.0) -> list[FakeTTSResponse]:
        return [item.speed_up(speed_factor) for item in self._items if item.type == "tts"]
