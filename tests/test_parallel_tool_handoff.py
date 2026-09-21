from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.agents.llm import ChatMessage, FunctionToolCall

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

HANDOFF_CALL = FunctionToolCall(name="handoff", arguments="{}", call_id="call_handoff")
SAVE_CALL = FunctionToolCall(name="save_note", arguments="{}", call_id="call_save")


class Greeter(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="greeter")

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="greet")


class Router(Agent):
    """Hands off, and has an ordinary tool the LLM may call in the same batch."""

    def __init__(self, *, save_delay: float) -> None:
        super().__init__(instructions="router")
        self._save_delay = save_delay

    @function_tool
    async def handoff(self, ctx: RunContext) -> Agent:
        return Greeter()

    @function_tool
    async def save_note(self, ctx: RunContext) -> str:
        await asyncio.sleep(self._save_delay)
        return "saved"


async def _messages(session: AgentSession, agent: Agent) -> list[ChatMessage]:
    events: list = []
    session.on("conversation_item_added", events.append)
    await run_session(session, agent, drain_delay=3.0)
    return [ev.item for ev in events if ev.item.type == "message"]


def _assistant_text(messages: list[ChatMessage]) -> list[str]:
    return [m.text_content for m in messages if m.role == "assistant"]


@pytest.mark.parametrize("save_delay", [0.0, 0.5], ids=["sibling_immediate", "sibling_delayed"])
async def test_handoff_survives_a_parallel_tool(save_delay: float) -> None:
    """A handoff batched with an ordinary tool must still switch agents.

    Tool outputs are collected in completion order, and the loop that reads them
    used to assign ``new_agent_task`` unconditionally — so whichever tool
    finished last decided the handoff, and an ordinary tool contributes None.
    Both cases here dropped the switch silently: no error, no handoff event,
    the session just carried on with the previous agent and answered the user
    as if nothing had happened.
    """
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "go")
    actions.add_llm("", tool_calls=[HANDOFF_CALL, SAVE_CALL])
    actions.add_llm("hello from the greeter", input="greet")
    actions.add_tts(1.0)

    session = create_session(actions)
    messages = await _messages(session, Router(save_delay=save_delay))

    assert isinstance(session.current_agent, Greeter)
    assert "hello from the greeter" in _assistant_text(messages)
