from __future__ import annotations

import pytest

from livekit.agents.llm.chat_context import ChatContext, FunctionCall, FunctionCallOutput
from livekit.agents.voice.generation import (
    _RUNNING_PLACEHOLDER_KEY,
    _inject_running_tool_calls,
    _strip_running_tool_calls,
)

pytestmark = pytest.mark.unit


def test_inject_adds_flagged_pair_for_inflight_call() -> None:
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content="book me a flight")

    running = FunctionCall(call_id="call_1", name="book_flight", arguments="{}", created_at=1.0)
    _inject_running_tool_calls(chat_ctx, [running])

    calls = [i for i in chat_ctx.items if i.type == "function_call" and i.call_id == "call_1"]
    outs = [i for i in chat_ctx.items if i.type == "function_call_output" and i.call_id == "call_1"]
    assert len(calls) == 1 and len(outs) == 1
    assert calls[0].extra.get(_RUNNING_PLACEHOLDER_KEY) is True
    assert outs[0].output and outs[0].is_error is False
    # the call must precede its output so the pair is valid for the LLM
    assert chat_ctx.items.index(calls[0]) < chat_ctx.items.index(outs[0])


def test_inject_does_not_flag_the_live_running_call() -> None:
    # the flag goes on a copy; the FunctionCall held by _RunningTasks stays clean
    running = FunctionCall(call_id="call_1", name="book_flight", arguments="{}")
    _inject_running_tool_calls(ChatContext.empty(), [running])
    assert _RUNNING_PLACEHOLDER_KEY not in running.extra


def test_inject_skips_call_already_present() -> None:
    chat_ctx = ChatContext.empty()
    chat_ctx.insert(FunctionCall(call_id="call_1", name="book_flight", arguments="{}"))
    chat_ctx.insert(
        FunctionCallOutput(call_id="call_1", name="book_flight", output="booked", is_error=False)
    )

    _inject_running_tool_calls(
        chat_ctx, [FunctionCall(call_id="call_1", name="book_flight", arguments="{}")]
    )

    outs = [i for i in chat_ctx.items if i.type == "function_call_output" and i.call_id == "call_1"]
    assert len(outs) == 1 and outs[0].output == "booked"


def test_strip_removes_injected_pairs_only() -> None:
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content="hi")
    # a real, completed call/output pair (unflagged) must survive
    chat_ctx.insert(FunctionCall(call_id="real", name="t", arguments="{}"))
    chat_ctx.insert(FunctionCallOutput(call_id="real", name="t", output="done", is_error=False))

    _inject_running_tool_calls(chat_ctx, [FunctionCall(call_id="run", name="t2", arguments="{}")])
    _strip_running_tool_calls(chat_ctx)

    assert not any(
        i.type in ("function_call", "function_call_output") and i.call_id == "run"
        for i in chat_ctx.items
    )
    assert any(i.type == "function_call" and i.call_id == "real" for i in chat_ctx.items)
    assert any(i.type == "function_call_output" and i.call_id == "real" for i in chat_ctx.items)
    assert any(i.type == "message" for i in chat_ctx.items)


def test_inject_then_strip_preserves_user_added_items() -> None:
    # mirrors llm_node mutating the ctx: the LLM sees the placeholder, the user appends an
    # item, then we strip before forwarding -> the user's item survives, placeholder gone
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content="hi")

    _inject_running_tool_calls(chat_ctx, [FunctionCall(call_id="run", name="t", arguments="{}")])
    chat_ctx.add_message(role="system", content="extra context from llm_node")
    _strip_running_tool_calls(chat_ctx)

    assert not any(getattr(i, "call_id", None) == "run" for i in chat_ctx.items)
    assert "extra context from llm_node" in [
        i.text_content for i in chat_ctx.items if i.type == "message"
    ]


def test_strip_is_noop_without_placeholders() -> None:
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content="hi")
    before = list(chat_ctx.items)
    _strip_running_tool_calls(chat_ctx)
    assert chat_ctx.items == before
