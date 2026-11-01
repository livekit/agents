"""Direct invocations expose the ordinary thread API without configured storage."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Self

import pytest

from livekit.agents import Agent, RunContext, _core, persisted
from livekit.agents.llm import FunctionToolCall, Toolset
from livekit.agents.storage.context_store import ContextStore
from livekit.agents.storage.errors import StorageUnavailable
from livekit.agents.storage.resolve import Thread

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit


@dataclass
class Preferences:
    title: str = persisted(default="")


async def test_direct_runs_have_independent_in_memory_threads() -> None:
    threads: list[Thread] = []
    agent = Agent(
        name="desk",
        instructions="Help",
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="remember",
                    content="",
                    tool_calls=[FunctionToolCall(name="remember", arguments="{}", call_id="c1")],
                    ttft=0,
                    duration=0,
                ),
                FakeLLMResponse(input="memory ready", content="Remembered", ttft=0, duration=0),
            ]
        ),
    )

    @agent.tool
    async def remember(ctx: RunContext) -> str:
        """Keep a title for this conversation."""
        thread = ctx.thread
        threads.append(thread)
        assert (await thread.info()).attributes == {}
        await thread.update(attributes={"title": "A conversation"})
        assert (await thread.info()).attributes == {"title": "A conversation"}
        assert (await thread.chat_history()).items == []
        assert await thread.lane(agent.name).tasks() == []
        assert await thread.journal() == []
        preferences = Preferences()
        await ctx.store.load_context(preferences)
        preferences.title = "Saved in memory"
        await ctx.store.save_context(preferences)
        restored = Preferences()
        await ctx.store.load_context(restored)
        assert restored.title == "Saved in memory"
        return "memory ready"

    try:
        for _ in range(2):
            result = await agent.run("remember")
            assert result.status == "completed" and result.output == "Remembered"
        assert threads[0].id != threads[1].id
        for thread in threads:
            with pytest.raises(StorageUnavailable, match="closed"):
                await thread.info()
    finally:
        for thread in threads:
            await thread._storage.aclose()


async def test_cancelled_direct_run_closes_its_in_memory_thread() -> None:
    entered = asyncio.Event()
    threads: list[Thread] = []
    agent = Agent(
        name="desk",
        instructions="Help",
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="wait",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[FunctionToolCall(name="wait", arguments="{}", call_id="c1")],
                )
            ]
        ),
    )

    @agent.tool
    async def wait(ctx: RunContext) -> None:
        """Wait for cancellation."""
        threads.append(ctx.thread)
        entered.set()
        await asyncio.Future()

    try:
        async with agent.run_stream("wait") as stream:
            await entered.wait()
            stream.cancel()
            with pytest.raises(asyncio.CancelledError):
                await stream.result()
        with pytest.raises(StorageUnavailable, match="closed"):
            await threads[0].info()
    finally:
        for thread in threads:
            await thread._storage.aclose()


async def test_toolset_setup_failure_releases_each_owned_resource_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    from livekit.agents.voice import agent_run

    owners: list[_core._core_RunResources] = []
    original = agent_run.run_resources

    def record_owner(*args):
        owner = original(*args)
        owners.append(owner)
        return owner

    monkeypatch.setattr(agent_run, "run_resources", record_owner)

    class Startup(Toolset):
        async def setup(self) -> Self:
            events.append(f"setup:{self.id}")
            if self.id == "refused":
                raise ValueError("startup refused")
            return self

        async def aclose(self) -> None:
            events.append(f"close:{self.id}")

    agent = Agent(
        name="desk",
        instructions="Help",
        llm=FakeLLM(),
        tools=[Startup(id="ready"), Startup(id="refused")],
    )
    with pytest.raises(ValueError, match="startup refused"):
        await agent.run("start")
    assert events == ["setup:ready", "setup:refused", "close:ready"]
    assert len(owners) == 1
    assert owners[0].closed
    await owners[0].rollback()
    assert events == ["setup:ready", "setup:refused", "close:ready"]
    with pytest.raises(RuntimeError, match="resources are closed"):
        await owners[0].open()


async def test_failing_toolset_release_waits_for_siblings_before_closing_storage() -> None:
    events: list[str] = []
    threads: list[Thread] = []
    stores: list[ContextStore] = []

    class Release(Toolset):
        async def aclose(self) -> None:
            if self.id == "failed":
                raise ValueError("release failed")
            await asyncio.sleep(0.01)
            # Closing callbacks still have their thread and activation until every sibling ends.
            await threads[0].info()
            assert not stores[0]._closed
            events.append("sibling released")

    agent = Agent(
        instructions="Help",
        tools=[Release(id="failed"), Release(id="delayed")],
        llm=FakeLLM(
            fake_responses=[FakeLLMResponse(input="hi", content="Hello", ttft=0, duration=0)]
        ),
    )

    @agent.instructions
    def capture(ctx: RunContext) -> str:
        threads.append(ctx.thread)
        stores.append(ctx.store)
        return "Help"

    async with agent.run_stream("hi") as stream:
        with pytest.raises(ValueError, match="release failed") as first:
            await stream.result()
        # Retrieval and context exit observe the settled error without re-running cleanup or
        # consuming the native captured exception a second time.
        with pytest.raises(ValueError) as repeated:
            await stream.result()
        assert repeated.value is first.value
    assert events == ["sibling released"]
    assert stores[0]._closed
    with pytest.raises(StorageUnavailable, match="closed"):
        await threads[0].info()
