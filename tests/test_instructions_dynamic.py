from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, function_tool
from livekit.agents.llm import CacheBreakpoint, ChatContext, Tool
from livekit.agents.llm.chat_context import Instructions
from livekit.agents.voice.agent_session import _append_instructions
from livekit.agents.voice.generation import (
    DYNAMIC_INSTRUCTIONS_MESSAGE_ID,
    INSTRUCTIONS_MESSAGE_ID,
    mark_instructions_cache_boundary,
    remove_instructions,
    update_instructions,
)

from .fake_llm import FakeLLM, FakeLLMResponse
from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

SESSION_TIMEOUT = 60
COMMON = "You are the Riverside Clinic voice agent. Follow the clinic rules."
AUDIO = "Keep it short."
DYNAMIC = "Current time: 09:01. Caller number: +15551234567."
BREAKPOINT = {"mode": "explicit"}


def test_render_appends_dynamic_after_the_modality_addition():
    instr = Instructions(COMMON, audio=AUDIO, dynamic=DYNAMIC)

    assert instr.render(modality="audio") == f"{COMMON}\n\n{AUDIO}\n{DYNAMIC}"


def test_render_without_modality_still_includes_dynamic():
    assert Instructions(COMMON, dynamic=DYNAMIC).render() == f"{COMMON}\n{DYNAMIC}"


def test_render_without_dynamic_is_unchanged():
    assert Instructions(COMMON, audio=AUDIO).render(modality="audio") == f"{COMMON}\n\n{AUDIO}"


def test_render_static_leaves_dynamic_out():
    instr = Instructions(COMMON, audio=AUDIO, dynamic=DYNAMIC)

    assert instr.render_static(modality="audio") == f"{COMMON}\n\n{AUDIO}"


def test_render_fills_data_in_both_parts():
    instr = Instructions("Agent for {clinic}.", dynamic="Caller: {caller}.")

    assert instr.render(data={"clinic": "Riverside", "caller": "Alex"}) == (
        "Agent for Riverside.\nCaller: Alex."
    )


def test_instructions_with_dynamic_do_not_equal_their_common_text():
    assert Instructions(COMMON, dynamic=DYNAMIC) != COMMON
    assert COMMON != Instructions(COMMON, dynamic=DYNAMIC)


def test_instructions_with_a_modality_addition_do_not_equal_their_common_text():
    assert Instructions(COMMON, audio=AUDIO) != COMMON


def test_section_free_instructions_equal_their_text():
    assert Instructions(COMMON) == COMMON
    assert Instructions(COMMON, audio="", dynamic="") == COMMON


def test_str_stays_the_common_text():
    assert str(Instructions(COMMON, dynamic=DYNAMIC)) == COMMON


def test_equality_and_hash_include_dynamic():
    with_dynamic = Instructions(COMMON, dynamic=DYNAMIC)
    without = Instructions(COMMON)

    assert with_dynamic == Instructions(COMMON, dynamic=DYNAMIC)
    assert with_dynamic != without
    assert hash(with_dynamic) != hash(without)


def test_repr_shows_dynamic():
    assert (
        repr(Instructions(COMMON, dynamic=DYNAMIC))
        == f"Instructions({COMMON!r}, dynamic={DYNAMIC!r})"
    )
    assert repr(Instructions(COMMON)) == f"Instructions({COMMON!r})"


def test_add_message_stores_the_rendered_text():
    ctx = ChatContext()
    ctx.add_message(role="system", content=Instructions(COMMON, dynamic=DYNAMIC))

    assert ctx.items[0].content == [f"{COMMON}\n{DYNAMIC}"]  # type: ignore[union-attr]


def test_add_message_without_dynamic_stores_the_common_text():
    ctx = ChatContext()
    ctx.add_message(role="system", content=Instructions(COMMON, audio=AUDIO))

    assert ctx.items[0].content == [COMMON]  # type: ignore[union-attr]


def test_append_instructions_keeps_dynamic():
    appended = _append_instructions(Instructions(COMMON, dynamic=DYNAMIC), "Extra rule.")

    assert appended == Instructions(f"{COMMON}\n\nExtra rule.", dynamic=DYNAMIC)


def _messages(ctx: ChatContext) -> list[tuple[str | None, str, list]]:
    return [(m.id, m.role, list(m.content)) for m in ctx.items if m.type == "message"]


def test_update_instructions_stores_dynamic_as_its_own_message():
    ctx = ChatContext()

    update_instructions(
        ctx, instructions=Instructions(COMMON, dynamic=DYNAMIC), add_if_missing=True
    )

    assert _messages(ctx) == [
        (INSTRUCTIONS_MESSAGE_ID, "system", [COMMON]),
        (DYNAMIC_INSTRUCTIONS_MESSAGE_ID, "system", [DYNAMIC]),
    ]


def test_update_instructions_renders_the_modality_into_the_static_message():
    ctx = ChatContext()

    update_instructions(
        ctx,
        instructions=Instructions(COMMON, audio=AUDIO, dynamic=DYNAMIC),
        add_if_missing=True,
        modality="audio",
    )

    assert _messages(ctx) == [
        (INSTRUCTIONS_MESSAGE_ID, "system", [f"{COMMON}\n\n{AUDIO}"]),
        (DYNAMIC_INSTRUCTIONS_MESSAGE_ID, "system", [DYNAMIC]),
    ]


def test_update_instructions_keeps_dynamic_right_after_the_instructions():
    ctx = ChatContext()
    ctx.add_message(role="user", content="Hi, I need to reschedule.")

    update_instructions(
        ctx, instructions=Instructions(COMMON, dynamic=DYNAMIC), add_if_missing=True
    )

    assert [m.id for m in ctx.items[:2]] == [
        INSTRUCTIONS_MESSAGE_ID,
        DYNAMIC_INSTRUCTIONS_MESSAGE_ID,
    ]
    assert ctx.items[2].role == "user"  # type: ignore[union-attr]


def test_update_instructions_replaces_the_dynamic_message():
    ctx = ChatContext()
    update_instructions(
        ctx, instructions=Instructions(COMMON, dynamic=DYNAMIC), add_if_missing=True
    )

    update_instructions(
        ctx, instructions=Instructions(COMMON, dynamic="Current time: 09:02."), add_if_missing=True
    )

    assert _messages(ctx) == [
        (INSTRUCTIONS_MESSAGE_ID, "system", [COMMON]),
        (DYNAMIC_INSTRUCTIONS_MESSAGE_ID, "system", ["Current time: 09:02."]),
    ]


def test_update_instructions_removes_the_dynamic_message_when_dropped():
    ctx = ChatContext()
    update_instructions(
        ctx, instructions=Instructions(COMMON, dynamic=DYNAMIC), add_if_missing=True
    )

    update_instructions(ctx, instructions=Instructions(COMMON), add_if_missing=True)

    assert _messages(ctx) == [(INSTRUCTIONS_MESSAGE_ID, "system", [COMMON])]


def test_update_instructions_with_a_plain_string_is_unchanged():
    ctx = ChatContext()

    update_instructions(ctx, instructions=COMMON, add_if_missing=True)

    assert _messages(ctx) == [(INSTRUCTIONS_MESSAGE_ID, "system", [COMMON])]


def test_update_instructions_adds_nothing_without_an_instructions_message():
    ctx = ChatContext()

    update_instructions(
        ctx, instructions=Instructions(COMMON, dynamic=DYNAMIC), add_if_missing=False
    )

    assert ctx.items == []


def test_remove_instructions_removes_both_messages():
    ctx = ChatContext()
    update_instructions(
        ctx, instructions=Instructions(COMMON, dynamic=DYNAMIC), add_if_missing=True
    )
    ctx.add_message(role="user", content="Hi.")

    remove_instructions(ctx)

    assert [m.role for m in ctx.items] == ["user"]  # type: ignore[union-attr]


def test_stored_text_joins_back_to_render():
    instr = Instructions(COMMON, audio=AUDIO, dynamic=DYNAMIC)
    ctx = ChatContext()
    update_instructions(ctx, instructions=instr, add_if_missing=True, modality="audio")

    stored = "\n".join(m.text_content or "" for m in ctx.items if m.type == "message")

    assert stored == instr.render(modality="audio")


def test_openai_wire_shape_for_dynamic_instructions():
    ctx = ChatContext()
    update_instructions(
        ctx, instructions=Instructions(COMMON, dynamic=DYNAMIC), add_if_missing=True
    )
    mark_instructions_cache_boundary(ctx)

    messages, _ = ctx.to_provider_format("openai", prompt_cache_breakpoints=True)

    assert messages[0]["content"] == [
        {"type": "text", "text": COMMON, "prompt_cache_breakpoint": BREAKPOINT}
    ]
    assert messages[1] == {"role": "system", "content": DYNAMIC}


class _CapturingLLM(FakeLLM):
    def __init__(self, fake_responses: list[FakeLLMResponse]) -> None:
        super().__init__(fake_responses=fake_responses)
        self.seen: list[ChatContext] = []
        self.seen_tools: list[list[Tool]] = []

    def chat(self, *, chat_ctx: ChatContext, tools: list[Tool] | None = None, **kwargs):  # type: ignore[no-untyped-def]
        self.seen.append(chat_ctx)
        self.seen_tools.append(tools or [])
        return super().chat(chat_ctx=chat_ctx, tools=tools, **kwargs)


@function_tool
async def lookup_appointment(phone: str) -> str:
    """Find the caller's next appointment."""
    return "Tuesday at 10am"


async def test_agent_with_tools_sends_dynamic_after_the_marked_instructions():
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Hi, I need to reschedule.", stt_delay=0.2)
    actions.add_llm("Sure, what day works for you?", ttft=0.1, duration=0.3)
    actions.add_tts(2.0, ttfb=0.2, duration=0.3)
    session = create_session(actions)
    capturing = _CapturingLLM(actions.get_llm_responses())
    agent = Agent(
        instructions=Instructions(COMMON, dynamic=DYNAMIC),
        llm=capturing,
        tools=[lookup_appointment],
    )

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert capturing.seen, "the LLM never ran"
    assert capturing.seen_tools[-1], "the request carried no tools"
    sent = _messages(capturing.seen[-1])
    assert sent[0] == (INSTRUCTIONS_MESSAGE_ID, "system", [COMMON, CacheBreakpoint()])
    assert sent[1] == (DYNAMIC_INSTRUCTIONS_MESSAGE_ID, "system", [DYNAMIC])
    wire, _ = capturing.seen[-1].to_provider_format("openai", prompt_cache_breakpoints=True)
    assert wire[0]["content"] == [
        {"type": "text", "text": COMMON, "prompt_cache_breakpoint": BREAKPOINT}
    ]
    assert wire[1] == {"role": "system", "content": DYNAMIC}
    assert _messages(agent.chat_ctx)[:2] == [
        (INSTRUCTIONS_MESSAGE_ID, "system", [COMMON]),
        (DYNAMIC_INSTRUCTIONS_MESSAGE_ID, "system", [DYNAMIC]),
    ]


class _GreetingAgent(Agent):
    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=Instructions("Greet the caller.", dynamic="It is 09:01.")
        )


async def test_per_reply_instructions_render_dynamic_inline():
    actions = FakeActions()
    actions.add_llm("Hello, thanks for calling Riverside Clinic.", input="Greet the caller.")
    actions.add_tts(2.0, ttfb=0.2, duration=0.3)
    session = create_session(actions, with_stt=False)
    capturing = _CapturingLLM(actions.get_llm_responses())
    agent = _GreetingAgent(instructions=COMMON, llm=capturing)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert capturing.seen, "the LLM never ran"
    sent = _messages(capturing.seen[-1])
    assert sent[0] == (INSTRUCTIONS_MESSAGE_ID, "system", [COMMON, CacheBreakpoint()])
    assert sent[1][1:] == ("system", ["Greet the caller.\nIt is 09:01."])
    assert capturing.seen[-1].get_by_id(DYNAMIC_INSTRUCTIONS_MESSAGE_ID) is None
