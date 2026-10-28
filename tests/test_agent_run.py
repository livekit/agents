"""``agent.run()`` and ``agent.run_stream()`` through the core's ``AgentRunner``: the tool
outcome table in text mode, output contracts, the stream's events and lifecycle, and usage."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from livekit.agents import Agent, ModelRetry, RunContext, RunUsage, StopResponse
from livekit.agents.llm import FunctionToolCall

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit


def respond(input: str, content: str = "", *calls: FunctionToolCall) -> FakeLLMResponse:
    return FakeLLMResponse(input=input, content=content, ttft=0, duration=0, tool_calls=list(calls))


def call(name: str, arguments: str = "{}", call_id: str = "c1") -> FunctionToolCall:
    return FunctionToolCall(name=name, arguments=arguments, call_id=call_id)


async def test_model_retry_from_a_tool_is_feedback_the_model_corrects() -> None:
    llm = FakeLLM(
        fake_responses=[
            respond("book", "", call("book", '{"day": "someday"}')),
            respond("give a weekday", "", call("book", '{"day": "monday"}', "c2")),
            respond("booked monday", "Booked for Monday."),
        ]
    )
    desk = Agent(instructions="desk", llm=llm)

    @desk.tool
    async def book(ctx: RunContext, day: str) -> str:
        """Book a day."""
        if day == "someday":
            raise ModelRetry("give a weekday")
        return f"booked {day}"

    result = await desk.run("book")
    assert result.status == "completed" and result.output == "Booked for Monday."
    retried = [i for i in result.new_messages() if i.type == "function_call_output"][0]
    assert retried.output == "give a weekday" and retried.is_error


async def test_a_none_return_continues_and_stop_response_ends_without_output() -> None:
    llm = FakeLLM(
        fake_responses=[
            respond("note it", "", call("note")),
            respond("", "Noted."),
            respond("hang up", "", call("hang_up")),
        ]
    )
    desk = Agent(instructions="desk", llm=llm)

    @desk.tool
    async def note(ctx: RunContext) -> None:
        """Note."""

    @desk.tool
    def hang_up(ctx: RunContext) -> str:  # a sync body
        """Hang up."""
        raise StopResponse()

    noted = await desk.run("note it")
    assert noted.status == "completed" and noted.output == "Noted."
    assert noted.usage.requests == 2

    stopped = await desk.run("hang up")
    assert stopped.status == "completed" and stopped.output is None
    assert stopped.usage.requests == 1


@dataclass
class Booking:
    reference: str


async def test_output_types_wrap_scalars_and_output_functions_resolve_the_value() -> None:
    count = Agent(
        instructions="count",
        output_type=int,
        llm=FakeLLM(
            fake_responses=[respond("how many", "", call("final_result", '{"result": 3}'))]
        ),
    )
    counted = await count.run("how many")
    assert counted.status == "completed" and counted.output == 3

    async def resolve(ctx: RunContext[dict[str, str]], code: str) -> Booking:
        if code not in ctx.context:
            raise ModelRetry(f"no booking {code}")
        return Booking(reference=ctx.context[code])

    lookup = Agent[dict[str, str], Booking](
        instructions="bookings",
        output_type=resolve,
        llm=FakeLLM(
            fake_responses=[
                respond("my booking", "", call("final_result", '{"code": "X"}')),
                respond(
                    "invalid final_result: no booking X",
                    "",
                    call("final_result", '{"code": "A1"}', "c2"),
                ),
            ]
        ),
    )
    resolved = await lookup.run("my booking", context={"A1": "BK-1"})
    assert resolved.status == "completed" and resolved.output == Booking(reference="BK-1")


async def test_run_stream_events_are_ordered_and_text_excludes_tools() -> None:
    llm = FakeLLM(
        fake_responses=[
            respond("weather", "", call("lookup", '{"city": "Paris"}')),
            respond("sunny", "It is sunny in Paris."),
        ]
    )
    agent = Agent(instructions="weather", llm=llm, name="forecaster")

    @agent.tool
    async def lookup(ctx: RunContext, city: str) -> str:
        """Look the weather up."""
        return "sunny"

    async with agent.run_stream("weather") as stream:
        events = [e async for e in stream.stream_events()]
        with pytest.raises(RuntimeError):
            stream.stream_text()
        result = await stream.result()

    types = [e.type for e in events]
    assert types[:2] == ["tool_call", "tool_result"]
    assert types[-1] == "text_end" and set(types[2:-1]) == {"text_delta"}
    assert [e.sequence for e in events] == sorted({e.sequence for e in events})
    assert all(e.agent_name == "forecaster" and e.frame_id is None for e in events)
    tool_call, tool_result = events[0], events[1]
    assert tool_call.type == "tool_call" and tool_call.arguments == {"city": "Paris"}
    assert tool_result.type == "tool_result" and tool_result.outcome == "completed"
    assert "".join(e.delta for e in events if e.type == "text_delta") == "It is sunny in Paris."
    assert result.status == "completed" and result.output == "It is sunny in Paris."
    assert await stream.result() is result, "result() is idempotent after exit"


async def test_run_stream_cancel_settles_the_tool_and_raises_cancellation() -> None:
    entered, cancelled = asyncio.Event(), asyncio.Event()
    llm = FakeLLM(fake_responses=[respond("wait", "", call("slow"))])
    agent = Agent(instructions="slow", llm=llm)

    @agent.tool
    async def slow(ctx: RunContext) -> str:
        """Slow."""
        entered.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "done"

    async with agent.run_stream("wait") as stream:
        events = stream.stream_events()
        assert (await anext(events)).type == "tool_call"
        await entered.wait()
        stream.cancel()
        stream.cancel()  # idempotent
        with pytest.raises(asyncio.CancelledError):
            async for _ in events:
                pass
        with pytest.raises(asyncio.CancelledError):
            await stream.result()
    assert cancelled.is_set(), "the tool body was cancelled before the run settled"


async def test_usage_is_the_invocation_delta_and_accumulates_into_the_caller() -> None:
    llm = FakeLLM(fake_responses=[respond("hi", "hello")])
    agent = Agent(instructions="hi", llm=llm)
    total = RunUsage()
    first = await agent.run("hi", usage=total)
    second = await agent.run("hi", usage=total)
    assert first.usage.requests == 1 and second.usage.requests == 1
    assert total.requests == 2
