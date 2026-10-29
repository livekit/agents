"""What a tool on a call reads from its ``RunContext`` (agents.md "RunContext")."""

import asyncio

import pytest

from livekit.agents import Agent, RunContext, VoiceSession
from livekit.agents.llm import FunctionToolCall, Toolset, function_tool

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit


def _calls(*inputs: str) -> FakeLLM:
    """A model that answers each input with one ``check`` call, then with "done"."""
    responses = []
    for n, text in enumerate(inputs):
        responses += [
            FakeLLMResponse(
                input=text,
                content="",
                ttft=0,
                duration=0,
                tool_calls=[FunctionToolCall(name="check", arguments="{}", call_id=f"c{n}")],
            ),
            FakeLLMResponse(input=f"checked {n}", content="done", ttft=0, duration=0),
        ]
    return FakeLLM(fake_responses=responses)


async def test_a_call_without_storage_has_an_in_memory_thread() -> None:
    agent = Agent(instructions="Check.", name="desk")
    seen: list[str] = []

    @agent.tool
    async def check(ctx: RunContext) -> str:
        """Check the thread."""
        assert ctx.thread is ctx.live_session.thread
        await ctx.thread.chat_history()
        seen.append(ctx.thread.id)
        return "checked 0"

    async with VoiceSession(llm=_calls("hello"), vad=None, turn_detection="manual") as session:
        await session.start(agent)
        await session._run(user_input="hello")
    assert len(seen) == 1 and seen[0]


async def test_a_call_keeps_one_execution_id_across_its_turns() -> None:
    agent = Agent(instructions="Check.", name="desk")
    seen: list[tuple[str, str]] = []

    @agent.tool
    async def check(ctx: RunContext) -> str:
        """Check the caller's request."""
        seen.append((ctx.execution_id, ctx.speech_handle.id))
        return f"checked {len(seen) - 1}"

    async with VoiceSession(
        llm=_calls("hello", "again"), vad=None, turn_detection="manual"
    ) as session:
        await session.start(agent)
        await session._run(user_input="hello")
        await session._run(user_input="again")
    (first, first_speech), (second, second_speech) = seen
    assert first == second and first_speech != second_speech


async def test_a_delegated_run_adds_to_the_call_usage() -> None:
    helper = Agent(
        instructions="Answer.",
        llm=FakeLLM(
            fake_responses=[FakeLLMResponse(input="look", content="found", ttft=0, duration=0)]
        ),
    )
    agent = Agent(instructions="Check.", name="desk")
    requests: list[int] = []

    @agent.tool
    async def check(ctx: RunContext) -> str:
        """Check the caller's request."""
        if not requests:
            await helper.run("look", usage=ctx.usage)
        requests.append(ctx.usage.requests)
        return f"checked {len(requests) - 1}"

    async with VoiceSession(
        llm=_calls("hello", "again"), vad=None, turn_detection="manual"
    ) as session:
        await session.start(agent)
        await session._run(user_input="hello")
        await session._run(user_input="again")
    assert requests == [1, 1]


async def test_a_duplicate_call_on_a_call_is_refused_with_the_agents_template() -> None:
    agent = Agent(
        instructions="Check.",
        name="desk",
        tool_handling={"async_options": {"duplicate_reject_template": "{function_name} runs"}},
    )
    runs: list[str] = []

    @agent.tool(on_duplicate="reject")
    async def check(ctx: RunContext) -> str:
        """Check the caller's request."""
        runs.append(ctx.function_call.call_id)
        await asyncio.sleep(0.05)
        return "checked"

    calls = [FunctionToolCall(name="check", arguments="{}", call_id=f"c{n}") for n in range(2)]
    model = FakeLLM(
        fake_responses=[
            FakeLLMResponse(input="hello", content="", ttft=0, duration=0, tool_calls=calls),
            FakeLLMResponse(input="checked", content="done", ttft=0, duration=0),
            FakeLLMResponse(input="check runs", content="done", ttft=0, duration=0),
        ]
    )
    async with VoiceSession(llm=model, vad=None, turn_detection="manual") as session:
        await session.start(agent)
        await session._run(user_input="hello")
        outputs = {
            item.call_id: item.output
            for item in session.history.items
            if item.type == "function_call_output"
        }
    assert runs == ["c0"]
    assert outputs == {"c0": "checked", "c1": "check runs"}


@pytest.mark.xfail(
    strict=True,
    reason="the voice frame's definition is built before the agent's toolsets set up: tools a "
    "toolset lists in setup() (MCP, RemoteAgent) lose their policies and expert identity",
)
async def test_tools_a_toolset_lists_at_setup_keep_their_policies_on_a_call() -> None:
    runs: list[str] = []

    @function_tool(on_duplicate="reject")
    async def check(ctx: RunContext) -> str:
        """Check the caller's request."""
        runs.append(ctx.function_call.call_id)
        await asyncio.sleep(0.05)
        return "checked"

    class Listed(Toolset):
        async def setup(self) -> "Listed":
            self._tools = [check]
            return self

    agent = Agent(instructions="Check.", name="desk", tools=[Listed(id="listed")])
    calls = [FunctionToolCall(name="check", arguments="{}", call_id=f"c{n}") for n in range(2)]
    model = FakeLLM(
        fake_responses=[
            FakeLLMResponse(input="hello", content="", ttft=0, duration=0, tool_calls=calls),
            FakeLLMResponse(input="checked", content="done", ttft=0, duration=0),
        ]
    )
    async with VoiceSession(llm=model, vad=None, turn_detection="manual") as session:
        await session.start(agent)
        await session._run(user_input="hello")
    assert runs == ["c0"]
