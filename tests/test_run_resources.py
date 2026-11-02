"""Public text invocations use native resources while retaining Python hydration only."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from livekit.agents import Agent, RunContext, TextSession, persisted
from livekit.agents.llm import Toolset
from livekit.agents.storage import SQLiteStorage
from livekit.agents.storage.context_store import ContextStore
from livekit.agents.storage.resolve import Thread
from livekit.agents.voice.agent_run import RunHooks, reset_run_hooks, set_run_hooks

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit


@dataclass
class State:
    count: int = persisted(default=0)


def agent(tools=()) -> Agent:
    return Agent(
        name="agent",
        instructions="Help",
        output_type=str,
        tools=tools,
        llm=FakeLLM(
            fake_responses=[FakeLLMResponse(input="Hello", content="Hi", ttft=0, duration=0)]
        ),
    )


@pytest.mark.parametrize("kind", ["agent", "session"])
async def test_owned_context_ends_and_releases_python_hydration(kind):
    configured = agent()
    stores = []

    @configured.instructions
    async def capture(ctx: RunContext) -> str:
        store = ctx.store
        await store.load_context(State())
        assert store._objects.loaded
        stores.append(store)
        return "Help"

    await (configured if kind == "agent" else TextSession(agent=configured)).run("Hello")
    assert stores
    for store in stores:
        assert store._closed
        assert not store._objects.loaded
        assert not store._objects.values
        with pytest.raises(RuntimeError, match="activation has ended"):
            await store.load_context(State())


@pytest.mark.parametrize("kind", ["agent", "session"])
@pytest.mark.parametrize("supply_store", [True, False])
async def test_managed_storage_and_supplied_store_remain_open(kind, supply_store, tmp_path):
    configured = agent()
    namespace = SQLiteStorage(directory=tmp_path)
    info = await namespace.threads.create_thread()
    thread = Thread(namespace, info.thread_id, agent_name="agent")
    activation = await namespace.threads._core_context_store(info.thread_id, "agent")
    supplied = ContextStore(activation=activation) if supply_store else None
    stores = []

    @configured.instructions
    async def capture(ctx: RunContext) -> str:
        assert ctx.thread is thread
        await ctx.store.load_context(State())
        stores.append(ctx.store)
        return "Help"

    token = set_run_hooks(RunHooks(thread=thread, store=supplied))
    try:
        await (configured if kind == "agent" else TextSession(agent=configured)).run("Hello")
        assert (await namespace.threads.get_thread(info.thread_id)).thread_id == info.thread_id
        assert stores
        if supply_store:
            assert all(store._activation is supplied._activation for store in stores)
            assert not supplied._closed
            await supplied.load_context(State())
        else:
            assert all(store._closed for store in stores)
    finally:
        reset_run_hooks(token)
        activation.close()
        await namespace.aclose()


async def test_cancelled_initialization_waits_for_authored_cleanup():
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    finished = []

    class Resource(Toolset):
        async def setup(self):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()
                finished.append(True)

    configured = agent([Resource(id="resource")])
    opening = asyncio.create_task(configured.run("Hello"))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        opening.cancel()
        await asyncio.wait_for(cleaning.wait(), 3)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(opening), 0.05)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await opening
        assert finished == [True]
    finally:
        release.set()
        await asyncio.gather(opening, return_exceptions=True)


@pytest.mark.parametrize("kind", ["agent", "session"])
async def test_host_initialization_error_survives_a_cleanup_error(kind):
    closed = []

    class Resource(Toolset):
        async def aclose(self):
            closed.append(True)
            raise ValueError("cleanup failure")

    # Model selection fails after native resources have opened.
    configured = Agent(name="agent", instructions="Help", tools=[Resource(id="resource")])
    with pytest.raises(RuntimeError, match="has no LLM"):
        await (configured if kind == "agent" else TextSession(agent=configured)).run("Hello")
    assert closed == [True]
