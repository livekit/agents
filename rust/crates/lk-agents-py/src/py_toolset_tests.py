"""Resource ownership behind search/proxy tool lists, through the Cargo-linked runtime."""

import asyncio
from collections import deque
from typing import Annotated, Any

import pytest
from pydantic import Field

from livekit.agents import Agent, RunContext, _core
from livekit.agents.beta import (
    KeywordSearchStrategy,
    SearchItem,
    ToolProxyToolset,
    ToolSearchToolset,
)
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
    FunctionCallOutput,
    FunctionToolCall,
    LLMStream,
    Tool,
    Toolset,
    function_tool,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from tests.fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit


class _Child(Toolset):
    def __init__(self, name: str, *, fail: bool = False) -> None:
        super().__init__(id=name)
        self.setups = 0
        self.closes = 0
        self.fail = fail

    @function_tool
    async def lookup(self) -> str:
        """Look up a reservation."""
        return "found"

    async def setup(self) -> "_Child":
        await super().setup()
        self.setups += 1
        if self.fail:
            raise RuntimeError("child setup failed")
        return self

    async def aclose(self) -> None:
        await super().aclose()
        self.closes += 1


@pytest.mark.parametrize("wrapper", [ToolSearchToolset, ToolProxyToolset])
async def test_hidden_children_close_after_last_shared_owner(
    wrapper: type[ToolSearchToolset],
) -> None:
    _core.init(_core.Role.Job)
    child = _Child("reservations")
    left = _core._core_ToolsetGroup([wrapper(id="left", tools=[child, child])])
    right = _core._core_ToolsetGroup([wrapper(id="right", tools=[child])])
    async with asyncio.timeout(5):
        try:
            await left.start()
            await right.start()
            assert (child.setups, child.closes) == (1, 0)
            await left.aclose()
            assert child.closes == 0
        finally:
            await left.aclose()
            await right.aclose()
    assert (child.setups, child.closes) == (1, 1)


@pytest.mark.parametrize("wrapper", [ToolSearchToolset, ToolProxyToolset])
async def test_hidden_children_roll_back_when_sibling_setup_fails(
    wrapper: type[ToolSearchToolset],
) -> None:
    _core.init(_core.Role.Job)
    ready, failed = _Child("ready"), _Child("failed", fail=True)
    group = _core._core_ToolsetGroup([wrapper(id="search", tools=[ready, failed])])
    async with asyncio.timeout(5):
        try:
            with pytest.raises(Exception, match="child setup failed"):
                await group.start()
            # Failed resources do not receive their authored close callback.
            assert (ready.setups, ready.closes) == (1, 1)
            assert (failed.setups, failed.closes) == (1, 0)
        finally:
            await group.aclose()


class _FailingIndex(KeywordSearchStrategy):
    def build_index(self, items: list[SearchItem]) -> None:
        raise RuntimeError("index build failed")


@pytest.mark.parametrize("wrapper", [ToolSearchToolset, ToolProxyToolset])
async def test_hidden_children_roll_back_when_index_build_fails(
    wrapper: type[ToolSearchToolset],
) -> None:
    _core.init(_core.Role.Job)
    child = _Child("reservations")
    group = _core._core_ToolsetGroup(
        [wrapper(id="search", tools=[child], search_strategy=_FailingIndex())]
    )
    async with asyncio.timeout(5):
        try:
            with pytest.raises(Exception, match="index build failed"):
                await group.start()
            assert (child.setups, child.closes) == (1, 1)
        finally:
            await group.aclose()


class _WaitingIndex(KeywordSearchStrategy):
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def build_index(self, items: list[SearchItem]) -> None:  # type: ignore[override]
        self.entered.set()
        try:
            await asyncio.Future[None]()
        finally:
            self.cancelled.set()


@pytest.mark.parametrize("wrapper", [ToolSearchToolset, ToolProxyToolset])
async def test_close_during_index_build_settles_hidden_children(
    wrapper: type[ToolSearchToolset],
) -> None:
    _core.init(_core.Role.Job)
    child, strategy = _Child("reservations"), _WaitingIndex()
    group = _core._core_ToolsetGroup(
        [wrapper(id="search", tools=[child], search_strategy=strategy)]
    )
    async with asyncio.timeout(5):
        opening = asyncio.ensure_future(group.start())
        try:
            await strategy.entered.wait()
            await group.aclose()
            assert strategy.cancelled.is_set()
            assert (child.setups, child.closes) == (1, 1)
            with pytest.raises(asyncio.CancelledError):
                await opening
        finally:
            await group.aclose()
            await asyncio.gather(opening, return_exceptions=True)


class ScriptedLLM(FakeLLM):
    def __init__(self, *steps: FunctionToolCall | str) -> None:
        super().__init__()
        self.steps = deque(steps)
        self.offered: list[list[str]] = []

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> LLMStream:
        from livekit.agents.llm import ToolContext

        self.offered.append(list(ToolContext(tools or []).function_tools))
        incoming = chat_ctx.items[-1]
        assert isinstance(incoming, (ChatMessage, FunctionCallOutput))
        text = (
            (incoming.text_content or "") if isinstance(incoming, ChatMessage) else incoming.output
        )
        step = self.steps.popleft()
        self.fake_response_map[text] = FakeLLMResponse(
            input=text,
            content=step if isinstance(step, str) else "",
            tool_calls=[step] if isinstance(step, FunctionToolCall) else [],
            ttft=0,
            duration=0,
        )
        return super().chat(chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)


async def test_discovered_tool_executes_on_the_next_model_step() -> None:
    booked: list[int] = []

    @function_tool
    async def reserve_seats(ctx: RunContext, count: int) -> str:
        """Reserve seats for the customer."""
        booked.append(count)
        return f"reserved {count} seats"

    agent = Agent(
        instructions="Find and use the reservations tool.",
        tools=[ToolSearchToolset(id="reservations", tools=[reserve_seats])],
        llm=ScriptedLLM(
            FunctionToolCall(
                name="tool_search", arguments='{"query":"reserve seats"}', call_id="s"
            ),
            FunctionToolCall(name="reserve_seats", arguments='{"count":2}', call_id="r"),
            "Your two seats are reserved.",
        ),
    )
    result = await agent.run("Reserve two seats.")
    assert isinstance(agent.llm, ScriptedLLM)
    assert "reserve_seats" not in agent.llm.offered[0]
    assert "reserve_seats" in agent.llm.offered[1]
    assert booked == [2], [
        i.output for i in result.new_messages() if i.type == "function_call_output"
    ]
    assert result.status == "completed" and result.output == "Your two seats are reserved."
    output = next(
        i for i in result.new_messages() if i.type == "function_call_output" and i.call_id == "r"
    )
    assert output.output == "reserved 2 seats" and not output.is_error


@pytest.mark.parametrize(
    "count", [2, 0], ids=["executes-valid-arguments", "refuses-invalid-arguments"]
)
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_proxy_invokes_a_discovered_tool_only_with_valid_arguments(
    count: int, asynchronous: bool
) -> None:
    booked: list[int] = []

    def reserve(ctx: RunContext, count: Annotated[int, Field(ge=1)]) -> str:
        """Reserve seats for the customer."""
        booked.append(count)
        return f"reserved {count} seats"

    if asynchronous:

        async def reserve_async(ctx: RunContext, count: Annotated[int, Field(ge=1)]) -> str:
            """Reserve seats for the customer."""
            return reserve(ctx, count)

        reserve_seats = function_tool(reserve_async, name="reserve_seats")
    else:
        reserve_seats = function_tool(reserve, name="reserve_seats")

    agent = Agent(
        instructions="Search for reservations, then use call_tool.",
        tools=[ToolProxyToolset(id="reservations", tools=[reserve_seats])],
        llm=ScriptedLLM(
            FunctionToolCall(
                name="tool_search", arguments='{"query":"reserve seats"}', call_id="s"
            ),
            FunctionToolCall(
                name="call_tool",
                arguments='{"name":"reserve_seats","parameters":{"count":' + str(count) + "}}",
                call_id="r",
            ),
            "Reservation processed.",
        ),
    )
    result = await agent.run("Reserve seats.")
    output = next(
        i for i in result.new_messages() if i.type == "function_call_output" and i.call_id == "r"
    )
    if count:
        assert booked == [2]
        assert output.output == "reserved 2 seats" and not output.is_error
    else:
        assert booked == []
        assert output.is_error and "count" in output.output
    assert result.status == "completed" and result.output == "Reservation processed."
