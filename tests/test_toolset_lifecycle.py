"""Text and voice use native toolset leases and settle rollback before failure escapes."""

import asyncio
from typing import Self

import pytest

from livekit.agents import Agent, VoiceSession, _core
from livekit.agents.llm import Toolset

pytestmark = pytest.mark.unit


class Counted(Toolset):
    def __init__(self, *, tools=None, fail=False):
        # A public ID is a label; separate objects with the same ID own separate resources.
        super().__init__(id="same-label", tools=tools)
        self.setups = self.closes = 0
        self.fail = fail

    async def setup(self) -> Self:
        self.setups += 1
        await super().setup()
        if self.fail:
            raise ValueError("toolset initialization failed")
        return self

    async def aclose(self) -> None:
        self.closes += 1
        await super().aclose()


async def test_nested_and_direct_users_share_the_same_child_lifetime():
    child = Counted()
    parent = Counted(tools=[child])
    a = _core._core_ToolsetGroup([parent, parent])
    b = _core._core_ToolsetGroup([child])
    try:
        async with asyncio.timeout(3):
            await asyncio.gather(a.start(), b.start())
            assert (child.setups, child.closes) == (1, 0)
            await a.aclose()
            assert (child.setups, child.closes) == (1, 0)
            await b.aclose()
        assert (parent.setups, parent.closes) == (1, 1)
        assert (child.setups, child.closes) == (1, 1)
    finally:
        await asyncio.gather(a.aclose(), b.aclose())


async def test_closed_group_releases_python_toolset_callbacks():
    import gc
    import weakref

    toolset = Counted()
    reference = weakref.ref(toolset)
    group = _core._core_ToolsetGroup([toolset])
    await group.start()
    await group.aclose()
    del toolset
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("name", [None, "same"])
async def test_same_name_agents_keep_their_own_hooks_and_toolset_scopes(name):
    first_tools, second_tools = Counted(), Counted()
    first = Agent(name=name, instructions="First", tools=[first_tools])
    second = Agent(name=name, instructions="Second", tools=[second_tools])
    events = []

    @first.on_enter
    async def first_enter(session):
        events.append("first entered")

    @first.on_exit
    async def first_exit(session):
        events.append("first exited")

    @second.on_enter
    async def second_enter(session):
        events.append("second entered")

    @second.on_exit
    async def second_exit(session):
        events.append("second exited")

    async with VoiceSession(vad=None, turn_handling={"turn_detection": None}) as session:
        await session.start(first)
        assert session.current_agent is first
        session.update_agent(second)
        async with asyncio.timeout(3):
            while first_tools.closes != 1 or events != [
                "first entered",
                "first exited",
                "second entered",
            ]:
                await asyncio.sleep(0)
        assert session.current_agent is second
        assert second_tools.closes == 0
    assert events[-1] == "second exited"
    assert second_tools.closes == 1


async def test_cancellation_waits_for_authored_setup_finally_to_settle():
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    finished = []

    class Resource(Counted):
        async def setup(self):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()
                finished.append(True)

    scope = _core._core_ToolsetGroup([Resource()])
    opening = asyncio.ensure_future(scope.start())
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), 3)
        opening.cancel()
        await asyncio.gather(opening, return_exceptions=True)
        await asyncio.wait_for(cleaning.wait(), 3)
        closing = asyncio.ensure_future(scope.aclose())
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(closing), 0.05)
        release.set()
        await asyncio.wait_for(closing, 3)
        assert finished == [True]
    finally:
        release.set()
        opening.cancel()
        await asyncio.gather(opening, return_exceptions=True)
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await scope.aclose()


async def test_reflection_failure_rolls_back_acquired_handoff_tools(monkeypatch):
    from livekit.agents.voice._session_callbacks import _AgentHooks

    first_tools, rejected_tools = Counted(), Counted()
    first = Agent(instructions="First", tools=[first_tools])
    rejected = Agent(instructions="Rejected", tools=[rejected_tools])
    original = _AgentHooks._core_tool_infos

    def reflect(self, tools):
        if rejected_tools in tools:
            raise ValueError("invalid dynamic tool schema")
        return original(self, tools)

    monkeypatch.setattr(_AgentHooks, "_core_tool_infos", reflect)
    async with VoiceSession(vad=None, turn_handling={"turn_detection": None}) as session:
        await session.start(first)
        session.update_agent(rejected)
        async with asyncio.timeout(3):
            while rejected_tools.closes != 1 or session.current_agent is not first:
                await asyncio.sleep(0)
        assert rejected_tools.setups == 1
        assert session.current_agent is first
        assert first_tools.closes == 0
    assert first_tools.closes == 1
