from __future__ import annotations

import asyncio
from typing import Any

import pytest

from livekit.agents import Agent
from livekit.agents.llm import CacheBreakpoint, ChatContext
from livekit.agents.llm.chat_context import Instructions
from livekit.agents.voice.generation import (
    INSTRUCTIONS_MESSAGE_ID,
    mark_instructions_cache_boundary,
    update_instructions,
)

from .fake_llm import FakeLLM, FakeLLMResponse
from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

SESSION_TIMEOUT = 60
INSTRUCTIONS = "You are the Riverside Clinic voice agent. Follow the clinic rules."
DYNAMIC = "Current time: 09:01. Caller number: +15551234567."


def _instructions_ctx(*content: Any) -> ChatContext:
    ctx = ChatContext()
    ctx.add_message(role="system", content=list(content), id=INSTRUCTIONS_MESSAGE_ID)
    return ctx


def _content(ctx: ChatContext) -> list[Any]:
    msg = ctx.get_by_id(INSTRUCTIONS_MESSAGE_ID)
    assert msg is not None and msg.type == "message"
    return msg.content


def test_helper_marks_the_instructions_message():
    ctx = ChatContext()
    update_instructions(ctx, instructions=INSTRUCTIONS, add_if_missing=True)

    mark_instructions_cache_boundary(ctx)

    assert _content(ctx) == [INSTRUCTIONS, CacheBreakpoint()]


def test_helper_leaves_the_stored_history_alone():
    stored = ChatContext()
    update_instructions(stored, instructions=INSTRUCTIONS, add_if_missing=True)
    working = stored.copy()

    mark_instructions_cache_boundary(working)

    assert _content(stored) == [INSTRUCTIONS]
    assert _content(working) == [INSTRUCTIONS, CacheBreakpoint()]


def test_helper_does_not_double_a_trailing_marker():
    ctx = _instructions_ctx(INSTRUCTIONS, CacheBreakpoint())

    mark_instructions_cache_boundary(ctx)

    assert _content(ctx) == [INSTRUCTIONS, CacheBreakpoint()]


def test_helper_keeps_an_inner_marker_and_adds_the_trailing_one():
    ctx = _instructions_ctx(INSTRUCTIONS, CacheBreakpoint(), DYNAMIC)

    mark_instructions_cache_boundary(ctx)

    assert _content(ctx) == [INSTRUCTIONS, CacheBreakpoint(), DYNAMIC, CacheBreakpoint()]


def test_helper_without_instructions_is_a_noop():
    ctx = ChatContext()
    ctx.add_message(role="user", content="Hi, I need to reschedule.")

    mark_instructions_cache_boundary(ctx)

    assert [item.type for item in ctx.items] == ["message"]
    assert ctx.get_by_id(INSTRUCTIONS_MESSAGE_ID) is None


class _CapturingLLM(FakeLLM):
    def __init__(self, fake_responses: list[FakeLLMResponse]) -> None:
        super().__init__(fake_responses=fake_responses)
        self.seen: list[ChatContext] = []

    def chat(self, *, chat_ctx: ChatContext, **kwargs: Any) -> Any:
        self.seen.append(chat_ctx)
        return super().chat(chat_ctx=chat_ctx, **kwargs)


async def test_user_turn_reaches_the_llm_with_a_marked_prefix():
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Hi, I need to reschedule.", stt_delay=0.2)
    actions.add_llm("Sure, what day works for you?", ttft=0.1, duration=0.3)
    actions.add_tts(2.0, ttfb=0.2, duration=0.3)
    session = create_session(actions)
    capturing = _CapturingLLM(actions.get_llm_responses())
    agent = Agent(instructions=INSTRUCTIONS, llm=capturing)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert capturing.seen, "the LLM never ran"
    assert _content(capturing.seen[-1]) == [INSTRUCTIONS, CacheBreakpoint()]
    assert _content(agent.chat_ctx) == [INSTRUCTIONS]


async def test_modality_instructions_are_marked_after_rendering():
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Hi, I need to reschedule.", stt_delay=0.2)
    actions.add_llm("Sure, what day works for you?", ttft=0.1, duration=0.3)
    actions.add_tts(2.0, ttfb=0.2, duration=0.3)
    session = create_session(actions)
    capturing = _CapturingLLM(actions.get_llm_responses())
    agent = Agent(instructions=Instructions(INSTRUCTIONS, audio="Keep it short."), llm=capturing)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert _content(capturing.seen[-1]) == [f"{INSTRUCTIONS}\n\nKeep it short.", CacheBreakpoint()]


class _GreetingAgent(Agent):
    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="Greet the caller.")


async def test_per_turn_instructions_follow_the_marked_prefix():
    actions = FakeActions()
    actions.add_llm("Hello, thanks for calling Riverside Clinic.", input="Greet the caller.")
    actions.add_tts(2.0, ttfb=0.2, duration=0.3)
    session = create_session(actions, with_stt=False)
    capturing = _CapturingLLM(actions.get_llm_responses())
    agent = _GreetingAgent(instructions=INSTRUCTIONS, llm=capturing)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert capturing.seen, "the LLM never ran"
    sent = [(m.role, m.content) for m in capturing.seen[-1].items if m.type == "message"]
    assert sent[0] == ("system", [INSTRUCTIONS, CacheBreakpoint()])
    assert sent[1] == ("system", ["Greet the caller."])
    assert _content(agent.chat_ctx) == [INSTRUCTIONS]
