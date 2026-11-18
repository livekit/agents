"""Exact tool output keeps its delivery boundary separate from business completion: awaiting
``ctx.say()`` reads its outcome at this hop's output boundary and never raises for it."""

import asyncio
from collections.abc import Awaitable

import pytest

from livekit.agents import Agent, RunContext, ToolResult, VoiceSession
from livekit.agents.a2a import RemoteAgent
from livekit.agents.llm import FunctionCallOutput
from livekit.agents.storage import ContextStore
from livekit.agents.voice.events import SayOutcome
from tests.fakes import FakeAudioOutput, FakeLLM, FakeTTS, tool_call, until

from .a2a_peer import URI, Peer


async def test_plain_agent_run_fails_exact_output_without_claiming_the_person_heard_it() -> None:
    outcomes: list[SayOutcome] = []
    agent = Agent(
        instructions="Look up the reservation.", llm=FakeLLM([tool_call("lookup"), "Done."])
    )

    @agent.tool
    async def lookup(ctx: RunContext) -> str:
        """Look up the reservation."""
        outcomes.append(await ctx.say("Reference ABC."))
        return "The reservation is confirmed."

    result = await agent.run("Find my reservation")

    assert result.status == "completed" and result.output == "Done."
    assert outcomes == ["failed"], "a result-only run has no live output to deliver it to"
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
    outcomes: list[SayOutcome] = []
    agent = Agent(instructions="Find the reservation.")

    @agent.tool
    async def lookup(ctx: RunContext) -> ToolResult:
        """Look up the reservation."""
        effects.append("booked")
        outcomes.append(await ctx.say("Your booking reference is ABC."))
        return ToolResult("The reservation is confirmed.", reply_required=False)

    async with VoiceSession(
        llm=FakeLLM([tool_call("lookup")]),
        tts=FakeTTS(error=RuntimeError("the exact TTS output failed")),
        vad=None,
    ) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(agent)
        await session.generate_reply(user_input="Find my reservation")
        await until(lambda: outcomes == ["failed"], "the exact output reader read its TTS failure")

        def finals() -> list[FunctionCallOutput]:
            return [
                item
                for item in session.history.items
                if item.type == "function_call_output"
                and item.output == "The reservation is confirmed."
            ]

        await until(finals, "the failed output kept the tool from returning its result")
        await session.wait_for_idle()
        assert effects == ["booked"]
        assert not any(
            message.text_content == "Your booking reference is ABC."
            for message in session.history.messages()
        )
        assert len(finals()) == 1 and not finals()[0].is_error


def _exact_speaker(
    outcomes: list[SayOutcome], effects: list[str], business: asyncio.Event
) -> Agent[None, str]:
    """An agent whose tool speaks a booking reference exactly, then keeps working on the
    business until ``business`` is set."""
    agent = Agent(instructions="Find the reservation.")

    @agent.tool
    async def lookup(ctx: RunContext) -> ToolResult:
        """Look up the reservation."""
        outcomes.append(await ctx.say("Your booking reference is ABC."))
        await business.wait()
        effects.append("confirmed")
        return ToolResult("The reservation is confirmed.", reply_required=False)

    return agent


async def test_an_interrupted_exact_voice_output_is_read_as_interrupted_and_work_continues() -> (
    None
):
    outcomes: list[SayOutcome] = []
    effects: list[str] = []
    business = asyncio.Event()
    output = FakeAudioOutput()
    async with VoiceSession(
        llm=FakeLLM([tool_call("lookup")]), tts=FakeTTS(seconds=30), vad=None
    ) as session:
        session.output.audio = output
        await session.start(_exact_speaker(outcomes, effects, business))
        await session.generate_reply(user_input="Find my reservation")
        await until(output.captured.is_set, "the exact words never reached the speaker")

        await session.interrupt()

        await until(lambda: outcomes == ["interrupted"], "the interruption left the reader parked")
        assert effects == [], "presentation cannot settle the business work"
        business.set()
        await until(
            lambda: effects == ["confirmed"], "the interruption cancelled the business work"
        )
        await session.wait_for_idle()
        assert not any(
            message.text_content == "Your booking reference is ABC." and not message.interrupted
            for message in session.history.messages()
        ), "interrupted words are never recorded as heard"


async def test_closing_the_session_mid_playout_reads_interrupted_and_leaks_no_host_task() -> None:
    receipts: list[Awaitable[SayOutcome]] = []
    agent = Agent(instructions="Find the reservation.")

    @agent.tool
    async def lookup(ctx: RunContext) -> ToolResult:
        """Look up the reservation."""
        receipts.append(ctx.say("Your booking reference is ABC."))
        return ToolResult("The reservation is confirmed.", reply_required=False)

    before = asyncio.all_tasks()
    output = FakeAudioOutput()
    async with VoiceSession(
        llm=FakeLLM([tool_call("lookup")]), tts=FakeTTS(seconds=30), vad=None
    ) as session:
        session.output.audio = output
        await session.start(agent)
        await session.generate_reply(user_input="Find my reservation")
        await until(output.captured.is_set, "the exact words never reached the speaker")

    assert await receipts[0] == "interrupted", "the close settled the reader without raising"
    leaked = asyncio.all_tasks() - before - {asyncio.current_task()}
    assert not leaked, f"the closed session left host tasks running: {leaked}"


async def test_a_busy_remote_never_delays_stopping_its_exact_words_on_the_call(peer: Peer) -> None:
    task = {"id": "booking-42", "contextId": "trip"}
    exact = "Your booking reference is ABC."
    peer.events = lambda _: [
        {"task": {**task, "status": {"state": "TASK_STATE_WORKING"}}},
        {
            "statusUpdate": {
                **{"taskId": task["id"], "contextId": task["contextId"]},
                "status": {
                    "state": "TASK_STATE_WORKING",
                    "message": {
                        "messageId": "exact-1",
                        "role": "ROLE_AGENT",
                        "parts": [{"text": exact}],
                        "metadata": {f"{URI}/progress": True, f"{URI}/verbatim": exact},
                    },
                },
            }
        },
        {
            "artifactUpdate": {
                **{"taskId": task["id"], "contextId": task["contextId"]},
                "artifact": {"artifactId": "answer", "parts": [{"text": "Booked"}]},
            }
        },
        {
            "statusUpdate": {
                "taskId": task["id"],
                "contextId": task["contextId"],
                "status": {"state": "TASK_STATE_COMPLETED"},
            }
        },
    ]
    peer.response_gate = asyncio.Event()
    peer.gated_after = 1
    output = FakeAudioOutput()
    agent = Agent(
        name="desk",
        instructions="Help.",
        llm=FakeLLM([tool_call("bookings", request="Book a table"), ""]),
        tools=[RemoteAgent(peer.url, name="bookings").as_tool()],
    )
    async with VoiceSession(tts=FakeTTS(seconds=30), vad=None) as session:
        session.output.audio = output
        await session.start(agent)
        session.generate_reply(user_input="Book a table")
        try:
            await until(
                output.captured.is_set, "the remote's exact words never reached the speaker"
            )

            await session.interrupt()

            assert not peer.response_gate.is_set() and not peer.cancelled, (
                "the playback stopped while the remote was still working on the business"
            )
        finally:
            peer.response_gate.set()
        await until(
            lambda: any(
                item.type == "function_call_output" and "Booked" in item.output
                for item in session.history.items
            ),
            "stopping the exact words lost the remote's later result",
        )
        assert not any(
            message.text_content == exact and not message.interrupted
            for message in session.history.messages()
        ), "the stopped words were recorded as heard"
