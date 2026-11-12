"""Public consumer and cancellation contracts over the shared native stream owner."""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, RunContext
from livekit.agents.llm import Toolset
from livekit.agents.voice.run_result import TextDeltaEvent

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
