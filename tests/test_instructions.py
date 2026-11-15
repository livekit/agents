from __future__ import annotations

import asyncio
from typing import Any

import pytest

from livekit.agents import Agent, function_tool
from livekit.agents.llm import CacheBreakpoint, ChatContext, Tool
from livekit.agents.llm.chat_context import DYNAMIC_INSTRUCTIONS_MESSAGE_ID, Instructions
from livekit.agents.voice.agent_session import _append_instructions
from livekit.agents.voice.generation import (
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


# ---------------------------------------------------------------------------
# Instructions value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("instr", "kwargs", "expected"),
    [
        (
            Instructions(COMMON, audio=AUDIO, dynamic=DYNAMIC),
            {"modality": "audio"},
            f"{COMMON}\n\n{AUDIO}\n{DYNAMIC}",
        ),
        (Instructions(COMMON, dynamic=DYNAMIC), {}, f"{COMMON}\n{DYNAMIC}"),
        (Instructions(COMMON, audio=AUDIO), {"modality": "audio"}, f"{COMMON}\n\n{AUDIO}"),
        (
            Instructions("Agent for {clinic}.", dynamic="Caller: {caller}."),
            {"data": {"clinic": "Riverside", "caller": "Alex"}},
            "Agent for Riverside.\nCaller: Alex.",
        ),
        (
            Instructions(COMMON, audio=AUDIO, text="Use markdown."),
            {"modality": "text"},
            f"{COMMON}\n\nUse markdown.",
        ),
        (Instructions(COMMON, audio=AUDIO), {"modality": "text"}, COMMON),
        (Instructions("", audio="audio only", text="text only"), {"modality": "text"}, "text only"),
        (Instructions("Hello {name}, total {}"), {"data": {"name": "Alex"}}, "Hello Alex, total "),
        (Instructions("Hello {name}, item {0}"), {"data": {"name": "Alex"}}, "Hello Alex, item "),
        (Instructions("Hi {name}, {price:.2f}"), {"data": {"name": "Alex"}}, "Hi Alex, "),
        (Instructions("Hi {name}, {caller!s}"), {"data": {"name": "Alex"}}, "Hi Alex, "),
    ],
    ids=[
        "modality-then-dynamic",
        "dynamic-only",
        "no-dynamic",
        "data-fills-both-parts",
        "text-modality",
        "other-modality-addition-left-out",
        "empty-common",
        "missing-positional-placeholder",
        "missing-index-placeholder",
        "missing-format-spec-placeholder",
        "missing-conversion-placeholder",
    ],
)
def test_render(instr: Instructions, kwargs: dict, expected: str) -> None:
    assert instr.render(**kwargs) == expected


def test_resolve_template_renders_each_modality_once() -> None:
    instr = Instructions.resolve_template(
        "{persona}\n\n{modality_specific}",
        persona="You are a helpful assistant.",
        modality_specific=Instructions(
            audio="Handle noisy voice input.", text="Handle typed input."
        ),
    )

    assert (
        instr.render(modality="audio")
        == "You are a helpful assistant.\n\nHandle noisy voice input."
    )
    assert instr.render(modality="text") == "You are a helpful assistant.\n\nHandle typed input."


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"name": "Alex"}, "Hello Alex"),
        # identical modality variants collapse to common-only
        ({"name": Instructions("Alex")}, "Hello Alex"),
    ],
)
def test_resolve_template_without_modality_parts_is_common_only(
    kwargs: dict, expected: str
) -> None:
    instr = Instructions.resolve_template("Hello {name}", **kwargs)

    assert (instr.common, instr.audio, instr.text) == (expected, None, None)
    assert instr.render(modality="audio") == expected


def test_render_static_leaves_dynamic_out() -> None:
    instr = Instructions(COMMON, audio=AUDIO, dynamic=DYNAMIC)

    assert instr.render_static(modality="audio") == f"{COMMON}\n\n{AUDIO}"


def test_equality_str_hash_and_repr() -> None:
    with_dynamic = Instructions(COMMON, dynamic=DYNAMIC)

    assert with_dynamic != COMMON and COMMON != with_dynamic
    assert Instructions(COMMON, audio=AUDIO) != COMMON
    assert Instructions(COMMON) == COMMON
    assert Instructions(COMMON, audio="", dynamic="") == COMMON
    assert with_dynamic == Instructions(COMMON, dynamic=DYNAMIC)
    assert with_dynamic != Instructions(COMMON)
    assert hash(with_dynamic) != hash(Instructions(COMMON))
    assert str(with_dynamic) == COMMON
    assert repr(with_dynamic) == f"Instructions({COMMON!r}, dynamic={DYNAMIC!r})"
    assert repr(Instructions(COMMON)) == f"Instructions({COMMON!r})"


@pytest.mark.parametrize(
    ("instr", "stored"),
    [
        (Instructions(COMMON, dynamic=DYNAMIC), f"{COMMON}\n{DYNAMIC}"),
        (Instructions(COMMON, audio=AUDIO), COMMON),
    ],
)
def test_add_message_stores_the_rendered_text(instr: Instructions, stored: str) -> None:
    ctx = ChatContext()
    ctx.add_message(role="system", content=instr)

    assert ctx.items[0].content == [stored]  # type: ignore[union-attr]


def test_append_instructions_keeps_dynamic() -> None:
    appended = _append_instructions(Instructions(COMMON, dynamic=DYNAMIC), "Extra rule.")

    assert appended == Instructions(f"{COMMON}\n\nExtra rule.", dynamic=DYNAMIC)


# ---------------------------------------------------------------------------
# update_instructions / remove_instructions
# ---------------------------------------------------------------------------


def _messages(ctx: ChatContext) -> list[tuple[str | None, str, list]]:
    return [(m.id, m.role, list(m.content)) for m in ctx.items if m.type == "message"]


def _dynamic_ctx() -> ChatContext:
    ctx = ChatContext()
    update_instructions(
        ctx, instructions=Instructions(COMMON, dynamic=DYNAMIC), add_if_missing=True
    )
    ctx.add_message(role="user", content="Hi, I need to reschedule.")
    return ctx


def test_update_instructions_stores_dynamic_as_its_own_message() -> None:
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


def test_update_instructions_keeps_dynamic_right_after_the_instructions() -> None:
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


@pytest.mark.parametrize(
    ("update", "expected"),
    [
        (
            Instructions(COMMON, dynamic="Current time: 09:02."),
            [
                (INSTRUCTIONS_MESSAGE_ID, "system", [COMMON]),
                (DYNAMIC_INSTRUCTIONS_MESSAGE_ID, "system", ["Current time: 09:02."]),
            ],
        ),
        (Instructions(COMMON), [(INSTRUCTIONS_MESSAGE_ID, "system", [COMMON])]),
        (COMMON, [(INSTRUCTIONS_MESSAGE_ID, "system", [COMMON])]),
    ],
    ids=["replaces-dynamic", "drops-dynamic", "plain-string"],
)
def test_update_instructions_rewrites_the_dynamic_message(
    update: Instructions | str, expected: list
) -> None:
    ctx = ChatContext()
    update_instructions(
        ctx, instructions=Instructions(COMMON, dynamic=DYNAMIC), add_if_missing=True
    )

    update_instructions(ctx, instructions=update, add_if_missing=True)

    assert _messages(ctx) == expected


def test_update_instructions_adds_nothing_without_an_instructions_message() -> None:
    ctx = ChatContext()

    update_instructions(
        ctx, instructions=Instructions(COMMON, dynamic=DYNAMIC), add_if_missing=False
    )

    assert ctx.items == []


def test_remove_instructions_removes_both_messages() -> None:
    ctx = _dynamic_ctx()

    remove_instructions(ctx)

    assert [m.role for m in ctx.items] == ["user"]  # type: ignore[union-attr]


def test_stored_text_joins_back_to_render() -> None:
    instr = Instructions(COMMON, audio=AUDIO, dynamic=DYNAMIC)
    ctx = ChatContext()
    update_instructions(ctx, instructions=instr, add_if_missing=True, modality="audio")

    stored = "\n".join(m.text_content or "" for m in ctx.items if m.type == "message")

    assert stored == instr.render(modality="audio")


# ---------------------------------------------------------------------------
# mark_instructions_cache_boundary
# ---------------------------------------------------------------------------


def _instructions_content(ctx: ChatContext) -> list[Any]:
    msg = ctx.get_by_id(INSTRUCTIONS_MESSAGE_ID)
    assert msg is not None and msg.type == "message"
    return msg.content


def test_cache_boundary_marks_a_copy_and_leaves_the_stored_history_alone() -> None:
    stored = ChatContext()
    update_instructions(stored, instructions=COMMON, add_if_missing=True)
    working = stored.copy()

    mark_instructions_cache_boundary(working)

    assert _instructions_content(stored) == [COMMON]
    assert _instructions_content(working) == [COMMON, CacheBreakpoint()]


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ([COMMON, CacheBreakpoint()], [COMMON, CacheBreakpoint()]),
        (
            [COMMON, CacheBreakpoint(), DYNAMIC],
            [COMMON, CacheBreakpoint(), DYNAMIC, CacheBreakpoint()],
        ),
    ],
    ids=["no-double-trailing-marker", "keeps-inner-marker"],
)
def test_cache_boundary_with_existing_markers(content: list, expected: list) -> None:
    ctx = ChatContext()
    ctx.add_message(role="system", content=content, id=INSTRUCTIONS_MESSAGE_ID)

    mark_instructions_cache_boundary(ctx)

    assert _instructions_content(ctx) == expected


def test_cache_boundary_without_instructions_is_a_noop() -> None:
    ctx = ChatContext()
    ctx.add_message(role="user", content="Hi, I need to reschedule.")

    mark_instructions_cache_boundary(ctx)

    assert [item.type for item in ctx.items] == ["message"]
    assert ctx.get_by_id(INSTRUCTIONS_MESSAGE_ID) is None


# ---------------------------------------------------------------------------
# provider formats
# ---------------------------------------------------------------------------


def test_openai_wire_shape_for_dynamic_instructions() -> None:
    ctx = _dynamic_ctx()
    mark_instructions_cache_boundary(ctx)

    messages, _ = ctx.to_provider_format("openai", prompt_cache_breakpoints=True)

    assert messages[0]["content"] == [
        {"type": "text", "text": COMMON, "prompt_cache_breakpoint": BREAKPOINT}
    ]
    assert messages[1] == {"role": "system", "content": DYNAMIC}


@pytest.mark.parametrize("fmt", ["openai", "openai.responses"])
def test_format_without_breakpoints_folds_dynamic_into_one_system_message(fmt: str) -> None:
    # gpt-4.1 and every inference google/* model: the old single-system-prompt bytes
    ctx = _dynamic_ctx()
    mark_instructions_cache_boundary(ctx)

    messages, _ = ctx.to_provider_format(fmt, prompt_cache_breakpoints=False)

    assert messages[0] == {"role": "system", "content": f"{COMMON}\n{DYNAMIC}"}
    assert [m["role"] for m in messages] == ["system", "user"]


def test_openai_format_does_not_fold_per_turn_instructions() -> None:
    ctx = ChatContext()
    ctx.add_message(role="system", content=COMMON, id=INSTRUCTIONS_MESSAGE_ID)
    ctx.add_message(role="system", content="Greet the caller.")

    messages, _ = ctx.to_provider_format("openai", prompt_cache_breakpoints=False)

    assert messages == [
        {"role": "system", "content": COMMON},
        {"role": "system", "content": "Greet the caller."},
    ]


@pytest.mark.parametrize("fmt", ["google", "anthropic", "aws", "mistralai"])
def test_format_folds_dynamic_into_the_system_instruction(fmt: str) -> None:
    messages, extra = _dynamic_ctx().to_provider_format(fmt)

    system = extra.instructions if fmt == "mistralai" else extra.system_messages
    assert system == (f"{COMMON}\n{DYNAMIC}" if fmt == "mistralai" else [f"{COMMON}\n{DYNAMIC}"])
    assert [m["role"] for m in messages] == ["user"]
    assert "<instructions>" not in str(messages)


def test_google_format_can_keep_dynamic_as_a_turn_for_cached_content() -> None:
    # Gemini cached_content requests carry no system_instruction at all
    turns, extra = _dynamic_ctx().to_provider_format("google", merge_dynamic_instructions=False)

    assert extra.system_messages == [COMMON]
    assert [t["role"] for t in turns] == ["user"]
    assert [p["text"] for p in turns[0]["parts"]] == [
        f"<instructions>\n{DYNAMIC}\n</instructions>",
        "Hi, I need to reschedule.",
    ]


def test_per_turn_instructions_after_dynamic_still_become_a_user_turn() -> None:
    # generate_reply(instructions=...) on turn 1: two leading system messages, no user yet
    ctx = ChatContext()
    update_instructions(
        ctx, instructions=Instructions(COMMON, dynamic=DYNAMIC), add_if_missing=True
    )
    ctx.add_message(role="system", content="Greet the caller.")

    entries, extra = ctx.to_provider_format("mistralai")

    assert extra.instructions == f"{COMMON}\n{DYNAMIC}"
    assert entries == [
        {
            "type": "message.input",
            "role": "user",
            "content": "<instructions>\nGreet the caller.\n</instructions>",
        }
    ]


@pytest.mark.parametrize("with_preamble", [True, False])
def test_dynamic_message_away_from_the_preamble_is_converted_like_any_other(
    with_preamble: bool,
) -> None:
    ctx = ChatContext()
    if with_preamble:
        ctx.add_message(role="system", content=COMMON, id=INSTRUCTIONS_MESSAGE_ID)
    ctx.add_message(role="user", content="Hi.")
    ctx.add_message(role="system", content=DYNAMIC, id=DYNAMIC_INSTRUCTIONS_MESSAGE_ID)

    entries, extra = ctx.to_provider_format("mistralai")

    assert extra.instructions == (COMMON if with_preamble else None)
    assert [e["content"] for e in entries] == ["Hi.", f"<instructions>\n{DYNAMIC}\n</instructions>"]


# ---------------------------------------------------------------------------
# truncate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tail", "max_items", "kept_tail_roles"),
    [
        (["assistant", "user", "assistant"], 2, ["user", "assistant"]),
        ([], 3, ["user"]),
        (["assistant"], 0, []),
    ],
    ids=["keeps-dynamic-with-static", "no-duplicate-in-tail", "truncate-to-zero"],
)
def test_truncate_keeps_both_instruction_messages(
    tail: list[str], max_items: int, kept_tail_roles: list[str]
) -> None:
    ctx = _dynamic_ctx()
    for role in tail:
        ctx.add_message(role=role, content="...")  # type: ignore[arg-type]

    ctx.truncate(max_items=max_items)

    assert [m.id for m in ctx.items[:2]] == [
        INSTRUCTIONS_MESSAGE_ID,
        DYNAMIC_INSTRUCTIONS_MESSAGE_ID,
    ]
    assert [m.role for m in ctx.items[2:]] == kept_tail_roles  # type: ignore[union-attr]


def test_truncate_without_dynamic_keeps_only_the_first_instructions() -> None:
    ctx = ChatContext()
    update_instructions(ctx, instructions=COMMON, add_if_missing=True)
    ctx.add_message(role="system", content="Not the dynamic message.")
    for text in ("u1", "u2", "u3"):
        ctx.add_message(role="user", content=text)

    ctx.truncate(max_items=2)

    assert ctx.items[0].id == INSTRUCTIONS_MESSAGE_ID
    assert len(ctx.items) == 3


# ---------------------------------------------------------------------------
# agent sessions: the LLM sees a marked prefix, the stored history stays unmarked
# ---------------------------------------------------------------------------


class _CapturingLLM(FakeLLM):
    def __init__(self, fake_responses: list[FakeLLMResponse]) -> None:
        super().__init__(fake_responses=fake_responses)
        self.seen: list[ChatContext] = []
        self.seen_tools: list[list[Tool]] = []

    def chat(self, *, chat_ctx: ChatContext, tools: list[Tool] | None = None, **kwargs):  # type: ignore[no-untyped-def]
        self.seen.append(chat_ctx)
        self.seen_tools.append(tools or [])
        return super().chat(chat_ctx=chat_ctx, tools=tools, **kwargs)


def _user_turn() -> FakeActions:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Hi, I need to reschedule.", stt_delay=0.2)
    actions.add_llm("Sure, what day works for you?", ttft=0.1, duration=0.3)
    actions.add_tts(2.0, ttfb=0.2, duration=0.3)
    return actions


@function_tool
async def lookup_appointment(phone: str) -> str:
    """Find the caller's next appointment."""
    return "Tuesday at 10am"


async def test_agent_with_tools_sends_dynamic_after_the_marked_instructions() -> None:
    actions = _user_turn()
    capturing = _CapturingLLM(actions.get_llm_responses())
    agent = Agent(
        instructions=Instructions(COMMON, dynamic=DYNAMIC),
        llm=capturing,
        tools=[lookup_appointment],
    )

    await asyncio.wait_for(run_session(create_session(actions), agent), timeout=SESSION_TIMEOUT)

    assert capturing.seen_tools[-1], "the request carried no tools"
    assert _messages(capturing.seen[-1])[:2] == [
        (INSTRUCTIONS_MESSAGE_ID, "system", [COMMON, CacheBreakpoint()]),
        (DYNAMIC_INSTRUCTIONS_MESSAGE_ID, "system", [DYNAMIC]),
    ]
    assert _messages(agent.chat_ctx)[:2] == [
        (INSTRUCTIONS_MESSAGE_ID, "system", [COMMON]),
        (DYNAMIC_INSTRUCTIONS_MESSAGE_ID, "system", [DYNAMIC]),
    ]


async def test_modality_instructions_are_marked_after_rendering() -> None:
    actions = _user_turn()
    capturing = _CapturingLLM(actions.get_llm_responses())
    agent = Agent(instructions=Instructions(COMMON, audio=AUDIO), llm=capturing)

    await asyncio.wait_for(run_session(create_session(actions), agent), timeout=SESSION_TIMEOUT)

    assert _instructions_content(capturing.seen[-1]) == [f"{COMMON}\n\n{AUDIO}", CacheBreakpoint()]


async def test_per_reply_instructions_follow_the_marked_prefix_with_dynamic_inline() -> None:
    class _GreetingAgent(Agent):
        async def on_enter(self) -> None:
            self.session.generate_reply(
                instructions=Instructions("Greet the caller.", dynamic="It is 09:01.")
            )

    actions = FakeActions()
    actions.add_llm("Hello, thanks for calling Riverside Clinic.", input="Greet the caller.")
    actions.add_tts(2.0, ttfb=0.2, duration=0.3)
    capturing = _CapturingLLM(actions.get_llm_responses())
    agent = _GreetingAgent(instructions=COMMON, llm=capturing)

    await asyncio.wait_for(
        run_session(create_session(actions, with_stt=False), agent), timeout=SESSION_TIMEOUT
    )

    sent = _messages(capturing.seen[-1])
    assert sent[0] == (INSTRUCTIONS_MESSAGE_ID, "system", [COMMON, CacheBreakpoint()])
    assert sent[1][1:] == ("system", ["Greet the caller.\nIt is 09:01."])
    assert capturing.seen[-1].get_by_id(DYNAMIC_INSTRUCTIONS_MESSAGE_ID) is None
