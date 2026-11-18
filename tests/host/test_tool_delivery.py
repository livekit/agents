"""Exact tool output keeps its delivery boundary separate from business completion."""

import pytest

from livekit.agents import Agent, RunContext, ToolResult, VoiceSession
from livekit.agents.storage import ContextStore
from tests.fakes import FakeAudioOutput, FakeLLM, FakeTTS, tool_call, until


async def test_plain_agent_run_refuses_exact_output_without_claiming_the_person_heard_it() -> None:
    failures: list[str] = []
    agent = Agent(
        instructions="Look up the reservation.", llm=FakeLLM([tool_call("lookup"), "Done."])
    )

    @agent.tool
    async def lookup(ctx: RunContext) -> str:
        """Look up the reservation."""
        try:
            await ctx.say("Reference ABC.")
        except RuntimeError as error:
            failures.append(str(error))
        return "The reservation is confirmed."

    result = await agent.run("Find my reservation")

    assert result.status == "completed" and result.output == "Done."
    assert failures == ["exact tool output requires a live voice, text stream or A2A output"]
    outputs = [item for item in result.new_messages() if item.type == "function_call_output"]
    assert len(outputs) == 1 and not outputs[0].is_error
    assert outputs[0].output == "The reservation is confirmed."
    assert all(
        item.text_content != "Reference ABC."
        for item in result.new_messages()
        if item.type == "message"
    ), "an absent output boundary cannot become a perceived assistant message"


async def test_unawaited_exact_say_survives_the_tools_immediate_final_result() -> None:
    body_calls: list[str] = []
    model = FakeLLM([tool_call("lookup", call_id="booking")])
    agent = Agent(instructions="Find the reservation.")

    @agent.tool
    def lookup(ctx: RunContext) -> ToolResult:
        """Look up the reservation."""
        body_calls.append("lookup")
        ctx.say("Your booking reference is ABC.")
        return ToolResult("The reservation is confirmed.", reply_required=False)

    async with VoiceSession(llm=model, vad=None) as session:
        await session.start(agent)
        await session.generate_reply(user_input="Find my reservation")
        await until(
            lambda: any(
                message.text_content == "Your booking reference is ABC."
                for message in session.history.messages()
            ),
            "the exact output admitted before the tool immediately returned",
        )
        await session.wait_for_idle()

        spoken = [
            message.text_content
            for message in session.history.messages()
            if message.role == "assistant"
        ]
        finals = [
            item
            for item in session.history.items
            if item.type == "function_call_output"
            and item.output == "The reservation is confirmed."
        ]
        assert spoken == ["Your booking reference is ABC."]
        assert len(finals) == 1 and not finals[0].is_error
        assert body_calls == ["lookup"], "exact presentation cannot repeat business work"
        assert len(model.requests) == 1, "exact output does not need an acknowledgement model turn"


@pytest.mark.parametrize("bound", [False, True], ids=["direct", "bound"])
async def test_unawaited_progress_and_exact_output_keep_order_before_the_tools_final_result(
    bound: bool,
) -> None:
    agent = Agent(
        instructions="Look up the reservation.", llm=FakeLLM([tool_call("lookup"), "Done."])
    )

    @agent.tool
    def lookup(ctx: RunContext) -> str:
        """Look up the reservation."""
        for n in range(100):
            ctx.update(f"Checking record {n}.", silent=n % 2 == 0)
        ctx.say("Reference ABC.")
        return "The reservation is confirmed."

    context = ContextStore().bind() if bound else None
    async with agent.run_stream("Find my reservation", context=context) as stream:
        events = [event async for event in stream.stream_events()]
        result = await stream.result()

    assert result.status == "completed" and result.output == "Done."
    progress = [event for event in events if event.type == "progress"]
    assert [event.message for event in progress] == [f"Checking record {n}." for n in range(100)]
    assert [event.silent for event in progress] == [n % 2 == 0 for n in range(100)]
    exact = [
        event for event in events if event.type == "text_delta" and event.delta == "Reference ABC."
    ]
    assert len(exact) == 1
    final = next(event for event in events if event.type == "tool_result")
    assert progress[-1].sequence < exact[0].sequence < final.sequence
    messages = [
        message
        for message in result.new_messages()
        if message.type == "message" and message.text_content == "Reference ABC."
    ]
    assert len(messages) == 1 and messages[0].id == exact[0].message_id
    assert result.verbatim is None


async def test_a_failed_exact_voice_output_does_not_cancel_or_repeat_the_business_tool() -> None:
    effects: list[str] = []
    failures: list[Exception] = []
    agent = Agent(instructions="Find the reservation.")

    @agent.tool
    async def lookup(ctx: RunContext) -> ToolResult:
        """Look up the reservation."""
        effects.append("booked")
        try:
            await ctx.say("Your booking reference is ABC.")
        except RuntimeError as error:
            failures.append(error)
        return ToolResult("The reservation is confirmed.", reply_required=False)

    async with VoiceSession(
        llm=FakeLLM([tool_call("lookup")]),
        tts=FakeTTS(error=RuntimeError("the exact TTS output failed")),
        vad=None,
    ) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(agent)
        await session.generate_reply(user_input="Find my reservation")
        await until(lambda: len(failures) == 1, "the exact output reader reported its TTS failure")
        await session.wait_for_idle()
        assert str(failures[0]) == "the exact tool output was not delivered completely"
        assert effects == ["booked"]
        assert not any(
            message.text_content == "Your booking reference is ABC."
            for message in session.history.messages()
        )
        outputs = [item for item in session.history.items if item.type == "function_call_output"]
        assert outputs[-1].output == "The reservation is confirmed." and not outputs[-1].is_error
