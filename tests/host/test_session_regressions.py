"""Regression contracts for application callbacks driving a live voice session."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator

import pytest

from livekit import rtc
from livekit.agents import (
    Agent,
    APIError,
    Completed,
    Failed,
    ModelSettings,
    RunContext,
    VoiceSession,
    VoiceTask,
    VoiceTaskFailed,
)
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.metrics import VADMetrics
from tests.fakes import (
    FakeAudioInput,
    FakeAudioOutput,
    FakeLLM,
    FakeSTT,
    FakeTTS,
    FakeVAD,
    Hang,
    audio_frame,
    tool_call,
    until,
)

IN_PROGRESS = "The tool call is still in progress."


def _messages(session: VoiceSession) -> list[tuple[str, str | None]]:
    return [(i.role, i.text_content) for i in session.history.items if i.type == "message"]


async def test_two_sessions_never_mint_the_same_chat_item_id() -> None:
    """Two calls on one thread never mint the same chat item id, nor the same reply group."""
    ids: list[set[str]] = []
    for _ in range(2):
        async with VoiceSession(llm=FakeLLM(["Hi."])) as session:
            await session.start(Agent(instructions="Help."))
            await session.generate_reply(user_input="hello")
            ids.append({item.id for item in session.history.items})

    assert ids[0] and ids[1]
    assert not ids[0] & ids[1]


async def test_a_failed_reply_reports_its_error_on_the_awaited_handle() -> None:
    """A failed `generate_reply` settles with the provider's error on its handle.
    https://github.com/livekit/agents/issues/6224"""
    failure = APIError("scripted failure", retryable=False)
    async with VoiceSession(llm=FakeLLM([failure])) as session:
        await session.start(Agent(instructions="Help."))
        speech = session.generate_reply(user_input="hello")
        await speech

    assert speech.exception() is failure


@pytest.mark.parametrize("blocked_on", ["event", "task_tool"])
async def test_on_enter_waits_do_not_block_session_progress(blocked_on: str) -> None:
    """An `on_enter` still awaiting an event or a task tool does not hold up the session.
    https://github.com/livekit/agents/issues/6313"""
    collector = Agent(
        instructions="Collect.",
        name="collector",
        output_type=str,
        llm=FakeLLM({IN_PROGRESS: tool_call("collect", call_id="collect-1")}),
    )

    @collector.tool
    async def collect(ctx: RunContext) -> Completed[str]:
        """Collect the reference."""
        return Completed("ref-7")

    front = Agent(instructions="Help.", name="front")
    seen: list[str] = []
    released = asyncio.Event()

    @front.tool
    async def lookup(ctx: RunContext) -> str:
        """Find the reference."""
        seen.append(await VoiceTask(agent=collector).run())
        return seen[-1]

    @front.on_enter
    async def entered(session: VoiceSession) -> None:
        if blocked_on == "event":
            await released.wait()
        else:
            await session.generate_reply(instructions="greet")
        seen.append("entered")

    model = FakeLLM(
        {
            "greet": tool_call("lookup", call_id="lookup-1"),
            "ref-7": "Found it.",
            "hello": "Hi.",
        }
    )
    async with VoiceSession(
        llm=model, vad=None, turn_handling={"turn_detection": "manual"}
    ) as session:
        await session.start(front)
        if blocked_on == "event":
            await session.generate_reply(user_input="hello")
            released.set()
        await until(lambda: "entered" in seen, "on_enter never finished")
        expected = ["entered"] if blocked_on == "event" else ["ref-7", "entered"]
        assert seen == expected
        assert _messages(session)[-1] == (
            ("assistant", "Hi.") if blocked_on == "event" else ("assistant", "Found it.")
        )


async def test_turn_hook_edits_shape_the_reply_that_is_recorded() -> None:
    """What an awaited `on_user_turn_completed` adds or edits reaches the request whose reply is
    kept. https://github.com/livekit/agents/issues/1970"""
    # A request made before the hook finished answers "stale" and must not be recorded.
    model = FakeLLM({"hello": "stale", "note 1": "fresh"})
    session = VoiceSession(
        vad=FakeVAD(),
        stt=FakeSTT(["hello"]),
        llm=model,
        tts=FakeTTS(),
        turn_handling={"turn_detection": None, "endpointing": {"min_delay": 0.1}},
        aec_warmup_duration=None,
    )
    audio = FakeAudioInput()
    session.input.audio = audio
    session.output.audio = FakeAudioOutput()
    agent = Agent(instructions="Help.")
    turns = 0

    @agent.instructions
    def numbered(ctx: RunContext) -> str:
        return f"turn {turns}"

    @session.on_user_turn_completed
    async def hook(ctx: RunContext, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        nonlocal turns
        await asyncio.sleep(0.05)
        turns += 1
        turn_ctx.add_message(role="system", content=f"note {turns}")
        new_message.content = [f"{new_message.text_content} (edited)"]

    async with session:
        await session.start(agent)
        audio.speak(0.4)
        await until(lambda: ("assistant", "fresh") in _messages(session), "no reply was kept")
    assert _messages(session) == [("user", "hello (edited)"), ("assistant", "fresh")]
    answered = next(r for r in model.requests if r.messages()[-1].text_content == "note 1")
    assert "turn 1" in (answered.messages()[0].text_content or "")


async def test_cancelling_an_idle_waiter_leaves_the_reply_and_later_turns_intact() -> None:
    """Cancelling `wait_for_idle()` cancels only the waiter, not the reply or later turns.
    https://github.com/livekit/agents/issues/6911"""
    answering = Hang()
    model = FakeLLM({"one": [answering, "first answer"], "two": "second answer"})
    async with VoiceSession(llm=model) as session:
        await session.start(Agent(instructions="Help."))
        speech = session.generate_reply(user_input="one")
        waiter = asyncio.ensure_future(session.wait_for_idle())
        await answering.started.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        answering.release()
        await speech
        await session.generate_reply(user_input="two")
        await session.wait_for_idle()

        assert not speech.interrupted and speech.exception() is None
        assert _messages(session) == [
            ("user", "one"),
            ("assistant", "first answer"),
            ("user", "two"),
            ("assistant", "second answer"),
        ]


async def test_a_task_failure_handled_by_its_tool_does_not_fail_the_run() -> None:
    """A `VoiceTask` failure its tool catches leaves the reply and later turns healthy.
    https://github.com/livekit/agents/pull/8269"""
    collector = Agent(
        instructions="Collect.",
        name="collector",
        output_type=str,
        llm=FakeLLM({IN_PROGRESS: tool_call("give_up", call_id="give-1")}),
    )

    @collector.tool
    async def give_up(ctx: RunContext) -> Failed:
        """Give up collecting."""
        return Failed("caller refused")

    front = Agent(instructions="Help.", name="front")

    @front.tool
    async def lookup(ctx: RunContext) -> str:
        """Find the reference."""
        try:
            return await VoiceTask(agent=collector).run()
        except VoiceTaskFailed as failure:
            return f"handled: {failure}"

    model = FakeLLM(
        [tool_call("lookup", call_id="lookup-1"), "Sorry about that.", "Anything else?"]
    )
    async with VoiceSession(
        llm=model, vad=None, turn_handling={"turn_detection": "manual"}
    ) as session:
        await session.start(front)
        await session.generate_reply(user_input="find")
        await session.generate_reply(user_input="next")

        assert _messages(session)[-3:] == [
            ("assistant", "Sorry about that."),
            ("user", "next"),
            ("assistant", "Anything else?"),
        ]
        assert session.current_agent is front


async def test_a_tts_node_can_feed_an_async_text_generator_to_synthesis() -> None:
    """Text a `tts_node` rewrites through an async generator is what gets synthesized.
    https://github.com/livekit/agents/issues/728"""
    tts = FakeTTS()
    session = VoiceSession(vad=None, tts=tts, turn_handling={"turn_detection": None})
    session.output.audio = FakeAudioOutput()

    @session.tts_node
    def louder(
        ctx: RunContext, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        async def upper() -> AsyncIterator[str]:
            async for chunk in text:
                yield chunk.upper()

        return session.default.tts_node(ctx, upper(), model_settings)

    async with session:
        await session.start(Agent(instructions="Help."))
        speech = session.say("One moment, please.")
        await speech

    assert speech.exception() is None
    assert "".join(tts.texts) == "ONE MOMENT, PLEASE."


async def test_a_sessions_vad_meters_its_inferences_on_itself() -> None:
    """The VAD a session drives emits its own ``metrics_collected`` with its inference count."""
    vad = FakeVAD()
    metrics: list[VADMetrics] = []
    vad.on("metrics_collected", metrics.append)
    session = VoiceSession(vad=vad, aec_warmup_duration=None)
    microphone = FakeAudioInput()
    session.input.audio = microphone

    async with session:
        await session.start(Agent(instructions="Listen."))
        while not metrics:  # a person speaking in real time, until the VAD reports
            microphone.push(audio_frame(0.1))
            await asyncio.sleep(0.1)

    assert metrics[0].inference_count > 0


async def test_subscribing_to_the_removed_session_metrics_event_says_what_replaces_it() -> None:
    """``VoiceSession.on("metrics_collected")`` raises naming its 2.0 replacements."""
    session = VoiceSession(vad=None)

    with pytest.raises(ValueError, match="session.usage"):
        session.on("metrics_collected", print)  # type: ignore[call-overload]  # removed event
