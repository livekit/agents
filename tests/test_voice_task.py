"""A VoiceTask fails with its tool's ``Failed(reason)``: the awaiting tool gets ``VoiceTaskFailed``
and the caller takes the floor back."""

import asyncio

import pytest

from livekit.agents import (
    Agent,
    Completed,
    Failed,
    RunContext,
    VoiceSession,
    VoiceTask,
    VoiceTaskFailed,
)
from livekit.agents.llm import FunctionToolCall
from livekit.agents.voice.events import ToolCallUpdated, ToolExecutionUpdatedEvent

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit


def _calls(input: str, tool: str, then: str | None = None) -> FakeLLM:
    responses = [
        FakeLLMResponse(
            input=input,
            content="",
            ttft=0,
            duration=0,
            tool_calls=[FunctionToolCall(name=tool, arguments="{}", call_id=f"{tool}-1")],
        )
    ]
    if then is not None:
        responses.append(FakeLLMResponse(input=then, content="Okay.", ttft=0, duration=0))
    return FakeLLM(fake_responses=responses)


async def test_a_failed_task_raises_its_reason_in_the_awaiting_tool() -> None:
    reason = "the passenger will not give an email"
    collector = Agent(
        instructions="Collect an email.",
        name="collector",
        output_type=str,
        # the task's model reads the awaiting call's placeholder output last
        llm=_calls("The tool call is still in progress.", "decline"),
    )

    @collector.tool
    async def decline(ctx: RunContext) -> Failed:
        """The passenger declines."""
        return Failed(reason)

    front = Agent(instructions="Help.", name="front")
    failures: list[str] = []

    @front.tool
    async def change_email(ctx: RunContext) -> str:
        """Change the receipt email."""
        try:
            await VoiceTask(agent=collector).run()
        except VoiceTaskFailed as failure:
            failures.append(failure.message)
            return "not changed"
        return "changed"

    model = _calls("change my email", "change_email", then="not changed")
    async with VoiceSession(llm=model, vad=None, turn_detection="manual") as session:
        await session.start(front)
        await asyncio.wait_for(session._run(user_input="change my email"), timeout=5)
        assert failures == [reason]
        assert session.current_agent is front


async def test_a_task_tool_can_report_progress_then_return_a_validated_completion() -> None:
    release = asyncio.Event()
    collector = Agent(
        instructions="Collect the reference.",
        name="collector",
        output_type=str,
        llm=_calls("The tool call is still in progress.", "collect"),
    )

    @collector.output_validator
    async def normalize(ctx: RunContext, value: str) -> str:
        return value.upper()

    @collector.tool
    async def collect(ctx: RunContext) -> Completed[str]:
        """Collect the reference."""
        await ctx.update("Collecting the reference.")
        await release.wait()
        return Completed("booking-1")

    front = Agent(instructions="Help.", name="front")
    results: list[str] = []

    @front.tool
    async def lookup(ctx: RunContext) -> str:
        """Find the booking."""
        results.append(await VoiceTask(agent=collector).run())
        return results[-1]

    def progress(event: ToolExecutionUpdatedEvent) -> None:
        if isinstance(event.update, ToolCallUpdated) and event.update.call_id == "collect-1":
            release.set()

    async with VoiceSession(
        llm=_calls("find my booking", "lookup", then="BOOKING-1"),
        vad=None,
        turn_detection="manual",
    ) as session:
        session.on("tool_execution_updated", progress)
        await session.start(front)
        await asyncio.wait_for(session._run(user_input="find my booking"), timeout=5)
        assert results == ["BOOKING-1"]
        assert session.current_agent is front


async def test_a_handoff_refused_inside_a_task_leaves_the_task_agent_speaking() -> None:
    collector = Agent(
        instructions="Collect an email.",
        name="collector",
        output_type=str,
        llm=_calls("The tool call is still in progress.", "decline"),
    )
    billing = Agent(instructions="Take the payment.", name="billing")
    speaking: list[Agent] = []

    @collector.tool
    async def decline(ctx: RunContext) -> Failed:
        """The passenger declines."""
        # a task holds the floor: the session refuses to hand it to another agent
        session = ctx.live_session
        assert session is not None
        session.update_agent(billing)
        await asyncio.sleep(0.1)
        speaking.append(session.current_agent)
        return Failed("declined")

    front = Agent(instructions="Help.", name="front")

    @front.tool
    async def change_email(ctx: RunContext) -> str:
        """Change the receipt email."""
        with pytest.raises(VoiceTaskFailed):
            await VoiceTask(agent=collector).run()
        return "not changed"

    model = _calls("change my email", "change_email", then="not changed")
    async with VoiceSession(llm=model, vad=None, turn_detection="manual") as session:
        await session.start(front)
        await asyncio.wait_for(session._run(user_input="change my email"), timeout=5)
        assert speaking == [collector]
        assert session.current_agent is front
