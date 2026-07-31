from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, function_tool
from livekit.agents.llm import FunctionTool, FunctionToolCall, RawFunctionTool

from .fake_session import FakeActions, create_session, run_session

pytestmark = pytest.mark.unit

SESSION_TIMEOUT = 30


@function_tool
async def secret() -> str:
    """A tool that is switched on at runtime."""
    return "secret ran"


class _LoaderAgent(Agent):
    """Mirrors the load-on-demand pattern: a resident loader switches the rest on."""

    def __init__(self) -> None:
        loader = function_tool(self._load, name="load", description="switch the secret tool on")
        super().__init__(instructions="you are helpful", tools=[loader, secret])
        self._resident = [t for t in self._tools if t.info.name == "load"]
        self._tools = list(self._resident)

    async def _load(self) -> str:
        await self.update_tools([*self._resident, secret])
        return "loaded"


def _tool_recorder(session) -> list[set[str]]:  # type: ignore[no-untyped-def]
    captured: list[set[str]] = []
    orig_chat = session.llm.chat

    def _recording_chat(*, chat_ctx, tools=None, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(
            {t.info.name for t in (tools or []) if isinstance(t, (FunctionTool, RawFunctionTool))}
        )
        return orig_chat(chat_ctx=chat_ctx, tools=tools, **kwargs)

    session.llm.chat = _recording_chat  # type: ignore[method-assign]
    return captured


async def test_tool_added_by_update_tools_is_callable_in_the_same_turn() -> None:
    """A tool switched on inside a function tool must be usable by the tool-response
    step, not only on the next user turn."""
    actions = FakeActions()
    actions.add_user_speech(0.0, 1.0, "load it")
    actions.add_llm("", tool_calls=[FunctionToolCall(name="load", arguments="{}", call_id="1")])
    # tool-response step, keyed on the loader's return: reaches for the just-added tool
    actions.add_llm(
        "",
        tool_calls=[FunctionToolCall(name="secret", arguments="{}", call_id="2")],
        input="loaded",
    )
    actions.add_llm("All set.", input="secret ran")
    actions.add_tts(1.0)

    session = create_session(actions)
    captured = _tool_recorder(session)
    await asyncio.wait_for(run_session(session, _LoaderAgent()), timeout=SESSION_TIMEOUT)

    # the turn opens with only the loader, and the tool-response step sees the new tool
    assert captured[0] == {"load"}
    assert "secret" in captured[1]

    outputs = {
        item.call_id: item for item in session.history.items if item.type == "function_call_output"
    }
    assert outputs["2"].output == "secret ran"
    assert outputs["2"].is_error is False


async def test_unchanged_tools_are_not_rebuilt_mid_turn() -> None:
    """The refresh is gated on the agent changing its own set: a turn whose tools never
    change must pass the same list to the tool-response step."""
    actions = FakeActions()
    actions.add_user_speech(0.0, 1.0, "just run it")
    actions.add_llm("", tool_calls=[FunctionToolCall(name="noop", arguments="{}", call_id="1")])
    actions.add_llm("Done.", input="noop ran")
    actions.add_tts(1.0)

    @function_tool
    async def noop() -> str:
        """Does not touch the tool set."""
        return "noop ran"

    class _StaticAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="you are helpful", tools=[noop])

    session = create_session(actions)
    captured = _tool_recorder(session)
    await asyncio.wait_for(run_session(session, _StaticAgent()), timeout=SESSION_TIMEOUT)

    assert captured[0] == {"noop"}
    assert captured[1] == {"noop"}
