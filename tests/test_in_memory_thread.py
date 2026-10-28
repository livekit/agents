"""Direct invocations expose the ordinary thread API without configured storage."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Self

import pytest

from livekit.agents import Agent, RunContext, persisted
from livekit.agents.llm import FunctionToolCall, Toolset
from livekit.agents.storage import Storage, StorageUnavailable
from livekit.agents.storage.context_store import ContextStore
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
        assert await thread.tasks() == []
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
            await thread.storage.aclose()


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
            await thread.storage.aclose()


async def test_toolset_setup_failure_releases_each_owned_resource_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    stores: list[ContextStore] = []
    storages: list[Storage] = []
    close_store = ContextStore.close
    close_storage = Storage.aclose

    def record_store_close(store: ContextStore) -> None:
        stores.append(store)
        close_store(store)

    async def record_storage_close(storage: Storage) -> None:
        storages.append(storage)
        await close_storage(storage)

    monkeypatch.setattr(ContextStore, "close", record_store_close)
    monkeypatch.setattr(Storage, "aclose", record_storage_close)

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
    assert len(stores) == len(storages) == 1


async def test_failing_toolset_release_waits_for_siblings_before_closing_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    close_storage = Storage.aclose

    async def record_storage_close(storage: Storage) -> None:
        events.append("storage closed")
        await close_storage(storage)

    monkeypatch.setattr(Storage, "aclose", record_storage_close)

    class Release(Toolset):
        async def aclose(self) -> None:
            if self.id == "failed":
                raise ValueError("release failed")
            await asyncio.sleep(0.01)
            events.append("sibling released")

    agent = Agent(
        instructions="Help",
        tools=[Release(id="failed"), Release(id="delayed")],
        llm=FakeLLM(
            fake_responses=[FakeLLMResponse(input="hi", content="Hello", ttft=0, duration=0)]
        ),
    )
    with pytest.raises(ValueError, match="release failed"):
        await agent.run("hi")
    # Let delayed releases settle before asserting, keeping a failing regression hermetic.
    await asyncio.sleep(0.02)
    assert events == ["sibling released", "storage closed"]
