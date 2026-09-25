from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent
from livekit.agents.llm import CacheBreakpoint, ChatContext
from livekit.agents.llm.chat_context import Instructions
from livekit.agents.voice.agent_session import _append_instructions
from livekit.agents.voice.generation import INSTRUCTIONS_MESSAGE_ID, update_instructions

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


def test_render_content_puts_a_breakpoint_before_dynamic():
    instr = Instructions(COMMON, audio=AUDIO, dynamic=DYNAMIC)

    assert instr.render_content(modality="audio") == [
        f"{COMMON}\n\n{AUDIO}",
        CacheBreakpoint(),
        DYNAMIC,
    ]


def test_render_content_without_dynamic_is_one_string():
    assert Instructions(COMMON, audio=AUDIO).render_content(modality="audio") == [
        f"{COMMON}\n\n{AUDIO}"
    ]


def test_render_content_with_empty_dynamic_is_one_string():
    assert Instructions(COMMON, dynamic="").render_content() == [COMMON]


def test_render_content_fills_data_in_both_parts():
    instr = Instructions("Agent for {clinic}.", dynamic="Caller: {caller}.")

    assert instr.render_content(data={"clinic": "Riverside", "caller": "Alex"}) == [
        "Agent for Riverside.",
        CacheBreakpoint(),
        "Caller: Alex.",
    ]


def test_render_content_joins_back_to_render():
    instr = Instructions(COMMON, audio=AUDIO, dynamic=DYNAMIC)

    items = instr.render_content(modality="audio")
    joined = "\n".join(item for item in items if isinstance(item, str))

    assert joined == instr.render(modality="audio")


def test_stored_text_content_matches_render():
    ctx = ChatContext()
    update_instructions(
        ctx, instructions=Instructions(COMMON, dynamic=DYNAMIC), add_if_missing=True
    )

    msg = ctx.get_by_id(INSTRUCTIONS_MESSAGE_ID)
    assert msg is not None and msg.type == "message"
    assert msg.text_content == Instructions(COMMON, dynamic=DYNAMIC).render()


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


def test_add_message_stores_dynamic_behind_a_breakpoint():
    ctx = ChatContext()
    ctx.add_message(role="system", content=Instructions(COMMON, dynamic=DYNAMIC))

    assert ctx.items[0].content == [COMMON, CacheBreakpoint(), DYNAMIC]  # type: ignore[union-attr]


def test_add_message_without_dynamic_stores_the_common_text():
    ctx = ChatContext()
    ctx.add_message(role="system", content=Instructions(COMMON, audio=AUDIO))

    assert ctx.items[0].content == [COMMON]  # type: ignore[union-attr]


def test_update_instructions_stores_three_items():
    ctx = ChatContext()

    update_instructions(
        ctx, instructions=Instructions(COMMON, dynamic=DYNAMIC), add_if_missing=True
    )

    msg = ctx.get_by_id(INSTRUCTIONS_MESSAGE_ID)
    assert msg is not None and msg.type == "message"
    assert msg.content == [COMMON, CacheBreakpoint(), DYNAMIC]


def test_update_instructions_with_a_plain_string_is_unchanged():
    ctx = ChatContext()

    update_instructions(ctx, instructions=COMMON, add_if_missing=True)

    msg = ctx.get_by_id(INSTRUCTIONS_MESSAGE_ID)
    assert msg is not None and msg.type == "message"
    assert msg.content == [COMMON]


def test_append_instructions_keeps_dynamic():
    appended = _append_instructions(Instructions(COMMON, dynamic=DYNAMIC), "Extra rule.")

    assert appended == Instructions(f"{COMMON}\n\nExtra rule.", dynamic=DYNAMIC)


def test_openai_wire_shape_for_dynamic_instructions():
    ctx = ChatContext()
    update_instructions(
        ctx, instructions=Instructions(COMMON, dynamic=DYNAMIC), add_if_missing=True
    )

    messages, _ = ctx.to_provider_format("openai", prompt_cache_breakpoints=True)

    assert messages[0]["content"] == [
        {"type": "text", "text": COMMON, "prompt_cache_breakpoint": BREAKPOINT},
        {"type": "text", "text": f"\n{DYNAMIC}"},
    ]


class _CapturingLLM(FakeLLM):
    def __init__(self, fake_responses: list[FakeLLMResponse]) -> None:
        super().__init__(fake_responses=fake_responses)
        self.seen: list[ChatContext] = []

    def chat(self, *, chat_ctx: ChatContext, **kwargs):  # type: ignore[no-untyped-def]
        self.seen.append(chat_ctx)
        return super().chat(chat_ctx=chat_ctx, **kwargs)


async def test_agent_with_dynamic_instructions_reaches_the_llm_with_two_boundaries():
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Hi, I need to reschedule.", stt_delay=0.2)
    actions.add_llm("Sure, what day works for you?", ttft=0.1, duration=0.3)
    actions.add_tts(2.0, ttfb=0.2, duration=0.3)
    session = create_session(actions)
    capturing = _CapturingLLM(actions.get_llm_responses())
    agent = Agent(instructions=Instructions(COMMON, dynamic=DYNAMIC), llm=capturing)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert capturing.seen, "the LLM never ran"
    sent = capturing.seen[-1].get_by_id(INSTRUCTIONS_MESSAGE_ID)
    stored = agent.chat_ctx.get_by_id(INSTRUCTIONS_MESSAGE_ID)
    assert sent is not None and sent.type == "message"
    assert stored is not None and stored.type == "message"
    # the static boundary from Instructions plus the trailing one the pipeline adds per turn
    assert sent.content == [COMMON, CacheBreakpoint(), DYNAMIC, CacheBreakpoint()]
    assert stored.content == [COMMON, CacheBreakpoint(), DYNAMIC]
