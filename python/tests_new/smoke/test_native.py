"""The public voice API on the native session: speech handles, tools, handoffs, realtime and
duplex models, and the text run."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator

import pytest

from livekit import rtc
from livekit.agents import Agent, RunContext, VoiceSession, llm
from livekit.agents.voice import SpeechHandle
from tests_new.fakes import (
    FakeAudioOutput,
    FakeDuplex,
    FakeLLM,
    FakeRealtime,
    FakeTTS,
    tool_call,
)

pytestmark = pytest.mark.usefixtures("no_logged_tracebacks")


def said(session: VoiceSession) -> list[tuple[str, str | None]]:
    return [(i.role, i.text_content) for i in session.history.items if i.type == "message"]


async def test_done_callbacks_run_in_order_once_the_speech_ends() -> None:
    """A speech's done callbacks run in order, a removed one never, a late one soon after.

    https://github.com/livekit/agents/issues/4849
    """
    calls: list[str] = []
    async with VoiceSession(tts=FakeTTS(streaming=False), vad=None) as session:
        await session.start(Agent(instructions="Help."))
        handle = session.say("Let me say something.")
        handle.add_done_callback(lambda h: calls.append("first"))
        handle.add_done_callback(dropped := lambda h: calls.append("dropped"))
        handle.add_done_callback(lambda h: calls.append("second"))
        handle.remove_done_callback(dropped)

        await handle
        handle.add_done_callback(lambda h: calls.append("late"))
        await asyncio.sleep(0)

    assert calls == ["first", "second", "late"]


async def test_allow_interruptions_set_before_the_speech_starts_is_kept() -> None:
    """Turning interruptions off on a fresh handle survives the speech's creation."""
    async with VoiceSession(tts=FakeTTS(streaming=False), vad=None) as session:
        await session.start(Agent(instructions="Help."))
        handle = session.say("Let me say something.")

        handle.allow_interruptions = False
        await handle

    assert not handle.allow_interruptions


async def test_an_interrupt_ends_the_wait_but_never_the_awaited_work() -> None:
    """``wait_if_not_interrupted`` returns on interrupt and leaves the work it watched running."""

    async def unfinished() -> AsyncIterator[str]:
        yield "A long sentence the agent will not finish saying."
        await asyncio.Event().wait()

    async with VoiceSession(tts=FakeTTS(streaming=False), vad=None) as session:
        await session.start(Agent(instructions="Help."))
        handle = session.say(unfinished())
        work: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        waiting = asyncio.ensure_future(handle.wait_if_not_interrupted([work]))

        handle.interrupt()
        await waiting

        assert not work.done()
        work.cancel()
    assert SpeechHandle.SPEECH_PRIORITY_LOW < SpeechHandle.SPEECH_PRIORITY_HIGH


async def test_say_speaks_a_str_a_stream_of_words_or_pre_synthesized_audio() -> None:
    """say() takes a str, an async stream of words, or text with its own audio."""

    async def words() -> AsyncIterable[str]:
        yield "Your booking "
        yield "is confirmed."

    async def frames() -> AsyncIterable[rtc.AudioFrame]:
        yield rtc.AudioFrame(b"\x00\x00" * 480, 24000, 1, 480)

    session = VoiceSession(tts=FakeTTS(streaming=False), vad=None)
    session.output.audio = FakeAudioOutput()
    async with session:
        await session.start(Agent(instructions="Help."))

        await session.say("One moment.")
        await session.say(words())
        await session.say("Goodbye.", audio=frames())

    assert said(session) == [
        ("assistant", "One moment."),
        ("assistant", "Your booking is confirmed."),
        ("assistant", "Goodbye."),
    ]


async def test_say_refuses_a_stream_of_words_with_its_own_audio() -> None:
    """Pre-synthesized audio needs the words it speaks as a str."""

    async def words() -> AsyncIterable[str]:
        yield "Hello."

    async def frames() -> AsyncIterable[rtc.AudioFrame]:
        yield rtc.AudioFrame(b"\x00\x00" * 480, 24000, 1, 480)

    async with VoiceSession(tts=FakeTTS(streaming=False), vad=None) as session:
        await session.start(Agent(instructions="Help."))

        with pytest.raises(ValueError, match="takes the words it speaks as a str"):
            session.say(words(), audio=frames())


async def test_a_tool_call_runs_and_its_output_is_answered() -> None:
    """A Python tool the model calls runs, and the model answers its output."""
    agent = Agent(instructions="You are the front desk.")

    @agent.tool
    async def get_weather(ctx: RunContext, city: str) -> str:
        """The weather in a city."""
        return f"sunny in {city}"

    model = FakeLLM(
        {
            "what's the weather like?": tool_call("get_weather", city="Paris"),
            "sunny in Paris": "It is sunny in Paris.",
        }
    )
    async with VoiceSession(llm=model, tts=FakeTTS(streaming=False), vad=None) as session:
        await session.start(agent)

        await session.generate_reply(user_input="what's the weather like?")

    assert said(session)[-1] == ("assistant", "It is sunny in Paris.")


async def test_a_tool_hands_the_call_to_another_agent() -> None:
    """A tool's ``update_agent`` makes the other agent active, which then greets the caller."""
    billing = Agent(instructions="You handle billing.", name="Billing")

    @billing.on_enter
    async def greet(session: VoiceSession) -> None:
        session.say("Billing here, how can I help?")

    desk = Agent(instructions="You are the front desk.")

    @desk.tool
    async def to_billing(ctx: RunContext) -> None:
        """Transfer the caller to billing."""
        assert ctx.live_session is not None
        ctx.live_session.update_agent(billing)

    model = FakeLLM({"put me through to billing": tool_call("to_billing")})
    async with VoiceSession(llm=model, tts=FakeTTS(streaming=False), vad=None) as session:
        await session.start(desk)

        await session.generate_reply(user_input="put me through to billing")
        await session.say("One moment.")

    assert session.current_agent.name == "Billing"
    assert ("assistant", "Billing here, how can I help?") in said(session)


async def test_an_agent_runs_as_text() -> None:
    """``agent.run`` answers a text message with the model's reply."""
    agent = Agent(instructions="You answer by text.", llm=FakeLLM(["You said: hello"]))

    result = await agent.run("hello")

    assert result.status == "completed"
    assert result.output == "You said: hello"


async def test_a_realtime_model_replies() -> None:
    """A Python realtime model drives the session's reply."""
    async with VoiceSession(llm=FakeRealtime("Hello from the realtime model."), vad=None) as s:
        await s.start(Agent(instructions="You are realtime."))

        await s.generate_reply(instructions="greet the user")

    assert ("assistant", "Hello from the realtime model.") in said(s)


async def test_a_duplex_model_receives_its_configuration_as_one_unit() -> None:
    """A duplex model is configured once with the agent's instructions, then replies."""
    model = FakeDuplex()
    async with VoiceSession(llm=llm.DuplexRealtimeAdapter(model), vad=None) as session:
        await session.start(Agent(instructions="You are duplex."))

        await session.generate_reply()

    assert model.sessions[0].configured_with == ["You are duplex."]
