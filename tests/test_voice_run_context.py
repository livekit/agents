"""What a tool on a call reads from its ``RunContext`` (agents.md "RunContext")."""

import pytest

from livekit.agents import Agent, RunContext, VoiceSession
from livekit.agents.llm import FunctionToolCall

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
