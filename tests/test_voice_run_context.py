"""What a tool on a call reads from its ``RunContext`` (agents.md "RunContext")."""

import asyncio

import pytest

from livekit.agents import Agent, RunContext, VoiceSession, utils
from livekit.agents.llm import (
    FunctionCall,
    FunctionToolCall,
    GenerationCreatedEvent,
    MessageGeneration,
    Toolset,
    function_tool,
)

from .fake_io import FakeAudioOutput
from .fake_llm import FakeLLM, FakeLLMResponse
from .fake_realtime import FakeRealtimeModel

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


async def test_a_delegated_run_reports_its_own_usage_apart_from_the_call() -> None:
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
            requests.append((await helper.run("look")).usage.requests)
        return "checked"

    async with VoiceSession(
        llm=_calls("hello", "again"), vad=None, turn_detection="manual"
    ) as session:
        await session.start(agent)
        await session._run(user_input="hello")
        await session._run(user_input="again")
    assert requests == [1]
    assert session.usage.settled
    # Each turn's `check` body started once: the call counts tool bodies (usage.md).
    assert session.usage.tool_calls == 2


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


# Catches: tools a toolset lists in setup() losing their policies on a call, on start and when
# their agent arrives by update_agent (the handoff rebuilds the frame on its own path).
@pytest.mark.parametrize("handoff", [False, True])
async def test_tools_a_toolset_lists_at_setup_keep_their_policies_on_a_call(
    handoff: bool,
) -> None:
    runs: list[str] = []
    entered = asyncio.Event()

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

    @agent.on_enter
    async def enter(session: VoiceSession) -> None:
        entered.set()

    calls = [FunctionToolCall(name="check", arguments="{}", call_id=f"c{n}") for n in range(2)]
    model = FakeLLM(
        fake_responses=[
            FakeLLMResponse(input="hello", content="", ttft=0, duration=0, tool_calls=calls),
            FakeLLMResponse(input="checked", content="done", ttft=0, duration=0),
        ]
    )
    async with VoiceSession(llm=model, vad=None, turn_detection="manual") as session:
        if handoff:
            await session.start(Agent(instructions="Greet.", name="greeter"))
            session.update_agent(agent)
        else:
            await session.start(agent)
        await asyncio.wait_for(entered.wait(), 5)
        await session._run(user_input="hello")
    assert runs == ["c0"]


# Catches: a call's model request interrupted before its metrics record never counted at all:
# a request counts when it starts, and one that never reported stays unmeasured (usage.md).
async def test_an_interrupted_model_request_counts_as_unmeasured() -> None:
    llm = FakeLLM(
        fake_responses=[FakeLLMResponse(input="hello", content="hi", ttft=30, duration=30)]
    )
    async with VoiceSession(llm=llm, vad=None, turn_detection="manual") as session:
        await session.start(Agent(instructions="Answer.", name="desk"))
        handle = session.generate_reply(user_input="hello")
        await asyncio.sleep(0.2)
        handle.interrupt()
        await handle
    usage = session.usage
    assert (usage.requests, usage.unmeasured_requests) == (1, 1), usage
    assert not usage.complete


# Catches: a realtime response counted only by a metrics record it may never send (gpt_live sends
# connection rows and its backend's records), so the call reported no request for the model.
async def test_a_realtime_response_counts_as_a_request_when_created() -> None:
    model = FakeRealtimeModel()
    async with VoiceSession(llm=model) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(Agent(instructions="Answer.", name="desk"))
        session.generate_reply()
        while not model.active_session._reply_futs:
            await asyncio.sleep(0)
        messages, functions = utils.aio.Chan[MessageGeneration](), utils.aio.Chan[FunctionCall]()
        messages.close()
        functions.close()
        model.active_session._reply_futs[0].set_result(
            GenerationCreatedEvent(
                message_stream=messages,
                function_stream=functions,
                user_initiated=True,
                response_id="response-id",
            )
        )
        async with asyncio.timeout(5):
            while not [m for m in session.usage.models if m.type == "llm"]:
                await asyncio.sleep(0.01)
        (llm,) = [m for m in session.usage.models if m.type == "llm"]
        assert (llm.requests, llm.unmeasured_requests) == (1, 1), session.usage
