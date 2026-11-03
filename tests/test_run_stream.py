"""Public consumer and cancellation contracts over the shared native stream owner."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, is_dataclass
from typing import get_type_hints

import pytest
from pydantic import JsonValue

from livekit.agents import Agent, RunContext
from livekit.agents.llm import Toolset
from livekit.agents.voice.run_stream import TextDeltaEvent, ToolCallEvent, ToolResultEvent

from .fake_llm import FakeLLM
from .test_agent_run import call, respond

pytestmark = pytest.mark.unit


def blocked(tools=()):
    entered, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    agent = Agent(
        name="agent",
        instructions="Help",
        tools=tools,
        llm=FakeLLM(fake_responses=[respond("start", "", call("wait")), respond("ready", "Done")]),
    )

    @agent.tool
    async def wait(ctx: RunContext) -> str:
        """Wait for the application."""
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "ready"

    return agent, entered, release, cancelled


def test_generated_public_event_records_preserve_types_and_identity():
    event = ToolCallEvent(
        sequence=1,
        agent_name="agent",
        invocation_id="run",
        call_id="call",
        tool_name="tool",
        arguments={"value": 3},
    )
    assert is_dataclass(event)
    assert event.__class__.__module__ == "livekit.agents.voice.run_stream"
    assert get_type_hints(ToolCallEvent)["arguments"] == dict[str, JsonValue]
    assert get_type_hints(ToolResultEvent)["output"] == JsonValue
    assert asdict(event)["type"] == "tool_call"


async def test_concurrent_result_calls_are_refused_and_the_accepted_result_is_reused():
    agent, entered, release, _ = blocked()
    async with agent.run_stream("start") as stream:
        first = asyncio.create_task(stream.result())
        try:
            await asyncio.wait_for(entered.wait(), 3)
            with pytest.raises(RuntimeError, match="another consumer"):
                await stream.result()
            release.set()
            result = await asyncio.wait_for(first, 3)
            assert result.output == "Done"
            assert await stream.result() is result
        finally:
            release.set()
            await asyncio.gather(first, return_exceptions=True)


async def test_an_unstarted_iterator_can_close_without_cancelling_execution():
    agent = Agent(instructions="Help", llm=FakeLLM(fake_responses=[respond("start", "Done")]))
    async with agent.run_stream("start") as stream:
        events = stream.stream_events()
        await events.aclose()
        with pytest.raises(RuntimeError, match="already has an iterator"):
            stream.stream_text()
        assert (await stream.result()).output == "Done"


async def test_cancelling_a_result_waiter_does_not_cancel_execution():
    agent, entered, release, cancelled = blocked()
    async with agent.run_stream("start") as stream:
        waiting = asyncio.create_task(stream.result())
        try:
            await asyncio.wait_for(entered.wait(), 3)
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
            assert not cancelled.is_set()
            release.set()
            assert (await asyncio.wait_for(stream.result(), 3)).output == "Done"
        finally:
            release.set()
            await asyncio.gather(waiting, return_exceptions=True)


async def test_concurrent_iterator_reads_are_refused_and_a_cancelled_read_can_retry():
    agent, entered, release, cancelled = blocked()
    async with agent.run_stream("start") as stream:
        events = stream.stream_events()
        assert (await anext(events)).type == "tool_call"
        await asyncio.wait_for(entered.wait(), 3)
        reading = asyncio.create_task(anext(events))
        try:
            await asyncio.sleep(0)  # enter the native read admission before the second caller
            with pytest.raises(RuntimeError, match="another consumer"):
                await anext(events)
            reading.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reading
            assert not cancelled.is_set()
            release.set()
            remaining = [event async for event in events]
            assert any(isinstance(event, TextDeltaEvent) for event in remaining)
            assert (await stream.result()).output == "Done"
        finally:
            release.set()
            await events.aclose()
            await asyncio.gather(reading, return_exceptions=True)


async def test_context_exit_settles_an_abandoned_iterator_and_retains_cancellation():
    agent, entered, release, cancelled = blocked()
    events = None
    try:
        async with agent.run_stream("start") as stream:
            events = stream.stream_events()
            assert (await anext(events)).type == "tool_call"
            await asyncio.wait_for(entered.wait(), 3)
        assert cancelled.is_set()
        with pytest.raises(asyncio.CancelledError):
            await stream.result()
        with pytest.raises(asyncio.CancelledError):
            await anext(events)
    finally:
        release.set()
        if events is not None:
            await events.aclose()


async def test_body_exception_remains_primary_when_native_cleanup_also_fails():
    closed = asyncio.Event()

    class Resource(Toolset):
        async def aclose(self):
            await super().aclose()
            closed.set()
            raise ValueError("cleanup failed")

    agent, entered, release, cancelled = blocked([Resource(id="resource")])
    body = LookupError("body failed")
    events = None
    try:
        with pytest.raises(LookupError) as caught:
            async with agent.run_stream("start") as stream:
                events = stream.stream_events()
                await anext(events)
                await asyncio.wait_for(entered.wait(), 3)
                raise body
        assert caught.value is body
        assert closed.is_set() and cancelled.is_set()
        with pytest.raises(ValueError, match="cleanup failed"):
            await stream.result()
    finally:
        release.set()
        if events is not None:
            await events.aclose()


async def test_a_view_opened_after_result_is_empty_and_keeps_the_cached_result():
    agent = Agent(instructions="Help", llm=FakeLLM(fake_responses=[respond("start", "Done")]))
    async with agent.run_stream("start") as stream:
        result = await stream.result()
        assert [event async for event in stream.stream_events()] == []
        assert await stream.result() is result
    assert await stream.result() is result
