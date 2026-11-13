"""Regression contracts for which tools a reply may call and how long a tool's interruption
protection lasts."""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, RunContext, VoiceSession, function_tool
from livekit.agents.beta import EndCallTool
from livekit.agents.llm import ToolContext
from tests.fakes import FakeAudioOutput, FakeLLM, FakeTTS, tool_call, until


async def test_an_on_enter_reply_cannot_call_a_toolset_tool_flagged_ignore_on_enter() -> None:
    """A toolset's IGNORE_ON_ENTER tool is withheld from the on_enter reply: calling it there
    is refused, and later turns go on. https://github.com/livekit/agents/issues/6354"""
    model = FakeLLM(
        [tool_call("end_call", call_id="greet-end"), "Hello, how can I help?", "Anything else?"]
    )
    agent = Agent(instructions="Help.", llm=model, tools=[EndCallTool(ignore_on_enter=True)])

    @agent.on_enter
    async def greet(session: VoiceSession) -> None:
        await session.generate_reply(instructions="greet")

    async with VoiceSession(vad=None, turn_handling={"turn_detection": "manual"}) as session:
        await session.start(agent)
        await until(lambda: any(i.type == "message" for i in session.history.items), "no greeting")
        await session.generate_reply(user_input="hello")

    refused = next(i for i in session.history.items if i.type == "function_call_output")
    assert refused.call_id == "greet-end" and refused.is_error
    assert session.history.messages()[-1].text_content == "Anything else?"


async def test_disallowed_interruptions_hold_until_the_tool_body_completes() -> None:
    """A tool that disallowed interruptions keeps running through its intermediate speech and
    an interruption attempt, and later turns stay usable.
    https://github.com/livekit/agents/issues/4560"""
    model = FakeLLM([tool_call("transfer", call_id="t1"), "Transferred.", "Anything else?"])
    agent = Agent(instructions="Help.", llm=model)
    working, release = asyncio.Event(), asyncio.Event()
    outcome: list[str] = []

    @agent.tool
    async def transfer(ctx: RunContext) -> str:
        """Transfer the booking."""
        ctx.disallow_interruptions()
        session = ctx.live_session
        assert session is not None
        await session.say("Please wait.")
        working.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            outcome.append("cancelled")
            raise
        outcome.append("completed")
        return "done"

    session = VoiceSession(
        vad=None, tts=FakeTTS(seconds=0.05), turn_handling={"turn_detection": None}
    )
    session.output.audio = FakeAudioOutput()
    async with session:
        await session.start(agent)
        speech = session.generate_reply(user_input="go")
        await working.wait()
        interrupting = session.interrupt()  # refused, or pending until the protected reply ends
        release.set()
        await speech
        await asyncio.gather(interrupting, return_exceptions=True)
        await session.generate_reply(user_input="next")

    assert outcome == ["completed"]
    assert not speech.interrupted
    assert session.history.messages()[-1].text_content == "Anything else?"


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/pull/5884 reproduces: tool order follows insertion",
)
@pytest.mark.parametrize("fmt", ["openai", "anthropic", "aws"])
def test_serialized_tools_do_not_depend_on_registration_order(fmt: str) -> None:
    """The same tool set serializes identically whatever order it was given in, so provider
    prompt caches keep hitting. https://github.com/livekit/agents/pull/5884"""

    async def book(ctx: RunContext) -> str:
        """Book a room."""
        return "booked"

    async def cancel(ctx: RunContext) -> str:
        """Cancel a booking."""
        return "cancelled"

    async def lookup(ctx: RunContext) -> str:
        """Look a booking up."""
        return "found"

    a, b, c = function_tool(book), function_tool(cancel), function_tool(lookup)

    shuffled = ToolContext([c, a, b]).parse_function_tools(fmt)  # type: ignore[call-overload]
    ordered = ToolContext([a, b, c]).parse_function_tools(fmt)  # type: ignore[call-overload]
    assert shuffled == ordered
