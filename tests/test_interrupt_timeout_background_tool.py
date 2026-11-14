"""A tool that outlives the interruption timeout of a speech waiting on it in the background.

Once the reply that called a tool has played out, its speech waits on the tool in the
background. A user turn or ``session.interrupt()`` interrupts that speech, and when the tool is
still running ``INTERRUPTION_TIMEOUT`` later, the speech is cancelled arbitrarily and the turn is
answered without the tool. The speech keeps waiting on the tool, so a handoff it returns still
applies, and its result gets a deferred reply, as a tool released at the timeout does (#7759).
"""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import (
    Agent,
    AgentSession,
    AgentStateChangedEvent,
    RunContext,
    ToolExecutionUpdatedEvent,
    function_tool,
    llm,
)
from livekit.agents.llm import FunctionToolCall
from livekit.agents.voice.speech_handle import INTERRUPTION_TIMEOUT
from livekit.agents.voice.tool_executor import REPLY_INSTRUCTIONS_AT_TAIL

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

# "Let me check." plays 3.2s -> 4.2s, the barge-in's turn ends at 7.5s and times out at 12.5s
TOOL_DURATION = INTERRUPTION_TIMEOUT + 8


def _ask_and_barge_in(actions: FakeActions, *, barge_in_at: float = 6.0) -> None:
    actions.add_user_speech(0.5, 2.5, "Look up my order")
    actions.add_llm(
        content="Let me check.",
        tool_calls=[FunctionToolCall(name="lookup_order", arguments="{}", call_id="1")],
    )
    actions.add_tts(1.0)
    # the caller checks in while the tool runs and the agent is silent ("thinking")
    actions.add_user_speech(barge_in_at, barge_in_at + 1.0, "Hello, are you there?")
    actions.add_llm("Yes, I am still checking.")
    actions.add_tts(2.0)


def _add_deferred_reply(actions: FakeActions, call_ids: list[str], content: str) -> None:
    actions.add_llm(content, input=REPLY_INSTRUCTIONS_AT_TAIL.format(call_ids=call_ids))
    actions.add_tts(2.0)


def _assistant_messages(chat_ctx: llm.ChatContext) -> list[str | None]:
    return [i.text_content for i in chat_ctx.items if i.type == "message" and i.role == "assistant"]


def _outputs(chat_ctx: llm.ChatContext) -> dict[str, list[str]]:
    outputs: dict[str, list[str]] = {}
    for item in chat_ctx.items:
        if item.type == "function_call_output":
            outputs.setdefault(item.call_id, []).append(item.output)
    return outputs


class _SlowToolAgent(Agent):
    def __init__(self, *, duration: float = TOOL_DURATION) -> None:
        super().__init__(instructions="test")
        self._duration = duration

    @function_tool
    async def lookup_order(self) -> str:
        """Look up the order."""
        await asyncio.sleep(self._duration)
        return "order 42 shipped"


def _record_tool_replies(session: AgentSession) -> list[tuple[list[str], str]]:
    replies: list[tuple[list[str], str]] = []

    def _on_update(ev: ToolExecutionUpdatedEvent) -> None:
        if ev.update.type == "tool_reply_updated":
            replies.append((ev.update.update_ids, ev.update.status))

    session.on("tool_execution_updated", _on_update)
    return replies


async def test_barge_in_while_a_background_tool_runs_speaks_its_result() -> None:
    actions = FakeActions()
    _ask_and_barge_in(actions)
    _add_deferred_reply(actions, ["1"], "Your order 42 has shipped.")

    session = create_session(actions)
    agent = _SlowToolAgent()
    replies = _record_tool_replies(session)
    await asyncio.wait_for(run_session(session, agent, drain_delay=TOOL_DURATION + 5), timeout=60)

    assert _assistant_messages(agent.chat_ctx) == [
        "Let me check.",
        "Yes, I am still checking.",
        "Your order 42 has shipped.",
    ]
    # the tool is not released: its call is answered once, with its result
    assert _outputs(agent.chat_ctx) == {"1": ["order 42 shipped"]}
    assert replies == [(["1"], "scheduled"), (["1"], "completed")]


async def test_barge_in_right_after_the_filler_speaks_the_result() -> None:
    actions = FakeActions()
    _ask_and_barge_in(actions, barge_in_at=4.3)
    _add_deferred_reply(actions, ["1"], "Your order 42 has shipped.")

    session = create_session(actions)
    agent = _SlowToolAgent()
    await asyncio.wait_for(run_session(session, agent, drain_delay=TOOL_DURATION + 5), timeout=60)

    assert _assistant_messages(agent.chat_ctx) == [
        "Let me check.",
        "Yes, I am still checking.",
        "Your order 42 has shipped.",
    ]
    assert _outputs(agent.chat_ctx) == {"1": ["order 42 shipped"]}


async def test_programmatic_interrupt_while_a_background_tool_runs_speaks_its_result() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Look up my order")
    actions.add_llm(
        content="Let me check.",
        tool_calls=[FunctionToolCall(name="lookup_order", arguments="{}", call_id="1")],
    )
    actions.add_tts(1.0)
    _add_deferred_reply(actions, ["1"], "Your order 42 has shipped.")

    session = create_session(actions)
    agent = _SlowToolAgent()

    def _interrupt_once_playout_ends(ev: AgentStateChangedEvent) -> None:
        if ev.old_state == "speaking" and ev.new_state == "thinking":
            session.off("agent_state_changed", _interrupt_once_playout_ends)
            asyncio.get_running_loop().call_later(1.0, session.interrupt)

    session.on("agent_state_changed", _interrupt_once_playout_ends)
    await asyncio.wait_for(run_session(session, agent, drain_delay=TOOL_DURATION + 5), timeout=60)

    assert _assistant_messages(agent.chat_ctx) == ["Let me check.", "Your order 42 has shipped."]
    assert _outputs(agent.chat_ctx) == {"1": ["order 42 shipped"]}


async def test_tool_returning_just_after_the_timeout_speaks_its_result() -> None:
    actions = FakeActions()
    _ask_and_barge_in(actions)
    _add_deferred_reply(actions, ["1"], "Your order 42 has shipped.")

    class JustLateAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def lookup_order(self, ctx: RunContext) -> str:
            """Look up the order."""
            timed_out = asyncio.Event()
            ctx.speech_handle.add_done_callback(lambda _: timed_out.set())
            await timed_out.wait()
            # the reply to the barge-in is already playing without the result
            await asyncio.sleep(0.3)
            return "order 42 shipped"

    session = create_session(actions)
    agent = JustLateAgent()
    await asyncio.wait_for(run_session(session, agent, drain_delay=TOOL_DURATION + 5), timeout=60)

    assert _assistant_messages(agent.chat_ctx) == [
        "Let me check.",
        "Yes, I am still checking.",
        "Your order 42 has shipped.",
    ]
    assert _outputs(agent.chat_ctx) == {"1": ["order 42 shipped"]}


async def test_parallel_tools_get_one_deferred_reply() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Look up my order and my refund")
    actions.add_llm(
        content="Let me check.",
        tool_calls=[
            FunctionToolCall(name="lookup_refund", arguments="{}", call_id="1"),
            FunctionToolCall(name="lookup_order", arguments="{}", call_id="2"),
        ],
    )
    actions.add_tts(1.0)
    actions.add_user_speech(6.0, 7.0, "Hello, are you there?")
    actions.add_llm("Yes, I am still checking.")
    actions.add_tts(2.0)
    _add_deferred_reply(actions, ["1", "2"], "Your refund is on its way and order 42 shipped.")

    class TwoToolsAgent(_SlowToolAgent):
        @function_tool
        async def lookup_refund(self) -> str:
            """Look up the refund."""
            # done before the barge-in, but answered with its slow sibling
            await asyncio.sleep(1.0)
            return "refund sent"

    session = create_session(actions)
    agent = TwoToolsAgent()
    replies = _record_tool_replies(session)
    await asyncio.wait_for(run_session(session, agent, drain_delay=TOOL_DURATION + 5), timeout=60)

    assert _assistant_messages(agent.chat_ctx) == [
        "Let me check.",
        "Yes, I am still checking.",
        "Your refund is on its way and order 42 shipped.",
    ]
    assert _outputs(agent.chat_ctx) == {"1": ["refund sent"], "2": ["order 42 shipped"]}
    assert replies == [(["1", "2"], "scheduled"), (["1", "2"], "completed")]


async def test_tool_returning_before_the_timeout_is_answered_by_the_barge_in_reply() -> None:
    actions = FakeActions()
    _ask_and_barge_in(actions)
    _add_deferred_reply(actions, ["1"], "Your order 42 has shipped.")

    session = create_session(actions)
    # returns at ~10.0s: the turn that interrupted the speech is still waiting for it
    agent = _SlowToolAgent(duration=7.0)
    replies = _record_tool_replies(session)
    await asyncio.wait_for(run_session(session, agent, drain_delay=TOOL_DURATION + 5), timeout=60)

    # the reply to the barge-in already sees the result, so nothing else is said
    assert _assistant_messages(agent.chat_ctx) == ["Let me check.", "Yes, I am still checking."]
    assert _outputs(agent.chat_ctx) == {"1": ["order 42 shipped"]}
    assert replies == []


@pytest.mark.parametrize("with_message", [False, True])
async def test_slow_handoff_tool_still_hands_off(with_message: bool) -> None:
    actions = FakeActions()
    _ask_and_barge_in(actions)
    _add_deferred_reply(actions, ["1"], "Transferring you now.")

    entered: list[Agent] = []

    class BillingAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="billing")

        async def on_enter(self) -> None:
            entered.append(self)

    class FrontDeskAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def lookup_order(self) -> Agent | tuple[Agent, str]:
            """Look up the order."""
            await asyncio.sleep(TOOL_DURATION)
            billing = BillingAgent()
            return (billing, "transferring to billing") if with_message else billing

    session = create_session(actions)
    agent = FrontDeskAgent()
    replies = _record_tool_replies(session)
    await asyncio.wait_for(run_session(session, agent, drain_delay=TOOL_DURATION + 5), timeout=60)

    assert len(entered) == 1
    outputs = _outputs(agent.chat_ctx)["1"]
    assert len(outputs) == 1
    assert "did not happen" not in outputs[0]
    # the agent that handed off doesn't answer the tool afterwards
    assert _assistant_messages(agent.chat_ctx) == ["Let me check.", "Yes, I am still checking."]
    assert replies == []
