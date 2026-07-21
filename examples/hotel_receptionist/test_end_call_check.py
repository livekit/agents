from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, cast

import pytest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common import Userdata
from end_call_check import (
    _render_transcript,
    _verdict_from_tool_calls,
    find_missing_action,
    run_goodbye_gate,
)
from hotel_db import HotelDB

from livekit.agents import llm
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

POLICY = "Offer to book whatever was discussed before the call winds down."


def _may_end_call() -> llm.FunctionToolCall:
    return llm.FunctionToolCall(call_id="v1", name="may_end", arguments="{}")


def _missing_call(arguments: str) -> llm.FunctionToolCall:
    return llm.FunctionToolCall(call_id="v1", name="missing", arguments=arguments)


class ScriptedLLM(llm.LLM):
    """Replies with scripted tool calls (or plain text); can raise or stall to
    exercise the fail-open paths."""

    def __init__(
        self,
        *,
        tool_calls: list[llm.FunctionToolCall] | None = None,
        content: str = "",
        error: bool = False,
        delay: float = 0.0,
    ) -> None:
        super().__init__()
        self.scripted_tool_calls = tool_calls or []
        self.content = content
        self.error = error
        self.delay = delay
        self.calls = 0
        self.last_chat_ctx: llm.ChatContext | None = None
        self.last_tools: list[llm.Tool] = []

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> llm.LLMStream:
        self.calls += 1
        self.last_chat_ctx = chat_ctx
        self.last_tools = tools or []
        return _ScriptedStream(
            self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options
        )


class _ScriptedStream(llm.LLMStream):
    def __init__(self, scripted: ScriptedLLM, **kwargs: Any) -> None:
        super().__init__(scripted, **kwargs)
        self._scripted = scripted

    async def _run(self) -> None:
        if self._scripted.error:
            raise RuntimeError("scripted failure")
        await asyncio.sleep(self._scripted.delay)
        self._event_ch.send_nowait(
            llm.ChatChunk(
                id="scripted",
                delta=llm.ChoiceDelta(
                    role="assistant",
                    content=self._scripted.content or None,
                    tool_calls=self._scripted.scripted_tool_calls,
                ),
            )
        )


def _transcript_ctx() -> llm.ChatContext:
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="system", content="internal system prompt")
    chat_ctx.add_message(role="user", content="hey - what are your room rates?")
    chat_ctx.add_message(role="assistant", content="Queen, king, or double queen.")
    chat_ctx.items.append(llm.FunctionCall(call_id="c1", name="take_guest_message", arguments="{}"))
    chat_ctx.items.append(
        llm.FunctionCallOutput(
            call_id="c1", name="take_guest_message", output="secret tool output", is_error=False
        )
    )
    return chat_ctx


def _userdata() -> Userdata:
    return Userdata(db=cast(HotelDB, None))


# --- verdict extraction from tool calls ---------------------------------------


def test_verdict_may_end() -> None:
    assert _verdict_from_tool_calls([_may_end_call()]) is None


def test_verdict_missing() -> None:
    call = _missing_call('{"action": "offer to add breakfast"}')
    assert _verdict_from_tool_calls([call]) == "offer to add breakfast"


def test_verdict_no_tool_calls_fails_open() -> None:
    assert _verdict_from_tool_calls([]) is None


def test_verdict_missing_with_empty_action_fails_open() -> None:
    assert _verdict_from_tool_calls([_missing_call('{"action": "   "}')]) is None


def test_verdict_missing_with_malformed_arguments_fails_open() -> None:
    assert _verdict_from_tool_calls([_missing_call("not json")]) is None


def test_verdict_unknown_tool_ignored() -> None:
    unknown = llm.FunctionToolCall(call_id="v0", name="something_else", arguments="{}")
    call = _missing_call('{"action": "book the dinner table"}')
    assert _verdict_from_tool_calls([unknown, call]) == "book the dinner table"


def test_verdict_first_relevant_call_wins() -> None:
    calls = [_may_end_call(), _missing_call('{"action": "offer breakfast"}')]
    assert _verdict_from_tool_calls(calls) is None


# --- transcript rendering ----------------------------------------------------


def test_render_transcript_speakers_and_tools() -> None:
    text = _render_transcript(_transcript_ctx())
    assert "Caller: hey - what are your room rates?" in text
    assert "Agent: Queen, king, or double queen." in text
    assert "[tool] take_guest_message" in text
    # internal machinery stays out of the audit prompt
    assert "internal system prompt" not in text
    assert "secret tool output" not in text


# --- find_missing_action -----------------------------------------------------


@pytest.mark.asyncio
async def test_find_missing_action_returns_missing() -> None:
    scripted = ScriptedLLM(tool_calls=[_missing_call('{"action": "offer to add breakfast"}')])
    missing = await find_missing_action(scripted, instructions=POLICY, chat_ctx=_transcript_ctx())
    assert missing == "offer to add breakfast"
    # the verdict tools are offered to the audit LLM
    assert {t.info.name for t in scripted.last_tools if isinstance(t, llm.FunctionTool)} == {
        "may_end",
        "missing",
    }
    # the audit prompt carries both the policy and the transcript
    assert scripted.last_chat_ctx is not None
    prompt = "\n".join(
        item.text_content or "" for item in scripted.last_chat_ctx.items if item.type == "message"
    )
    assert POLICY in prompt
    assert "Caller: hey - what are your room rates?" in prompt


@pytest.mark.asyncio
async def test_find_missing_action_may_end() -> None:
    scripted = ScriptedLLM(tool_calls=[_may_end_call()])
    assert (
        await find_missing_action(scripted, instructions=POLICY, chat_ctx=_transcript_ctx()) is None
    )


@pytest.mark.asyncio
async def test_find_missing_action_text_only_reply_fails_open() -> None:
    scripted = ScriptedLLM(content="I think the agent did well overall.")
    assert (
        await find_missing_action(scripted, instructions=POLICY, chat_ctx=_transcript_ctx()) is None
    )


@pytest.mark.asyncio
async def test_find_missing_action_fails_open_on_error() -> None:
    scripted = ScriptedLLM(error=True)
    assert (
        await find_missing_action(scripted, instructions=POLICY, chat_ctx=_transcript_ctx()) is None
    )


@pytest.mark.asyncio
async def test_find_missing_action_fails_open_on_timeout() -> None:
    scripted = ScriptedLLM(tool_calls=[_may_end_call()], delay=5.0)
    assert (
        await find_missing_action(
            scripted, instructions=POLICY, chat_ctx=_transcript_ctx(), timeout=0.1
        )
        is None
    )


# --- the one-shot goodbye gate ------------------------------------------------


@pytest.mark.asyncio
async def test_gate_nudges_once_then_allows_close() -> None:
    userdata = _userdata()
    scripted = ScriptedLLM(tool_calls=[_missing_call('{"action": "offer to add breakfast"}')])

    nudge = await run_goodbye_gate(
        userdata, scripted, instructions=POLICY, chat_ctx=_transcript_ctx()
    )
    assert nudge is not None and "offer to add breakfast" in nudge
    assert userdata.end_call_nudged is True

    # second attempt goes through unconditionally: no nudge, no second audit call
    assert (
        await run_goodbye_gate(userdata, scripted, instructions=POLICY, chat_ctx=_transcript_ctx())
        is None
    )
    assert scripted.calls == 1


@pytest.mark.asyncio
async def test_gate_allows_close_when_clean() -> None:
    userdata = _userdata()
    scripted = ScriptedLLM(tool_calls=[_may_end_call()])
    assert (
        await run_goodbye_gate(userdata, scripted, instructions=POLICY, chat_ctx=_transcript_ctx())
        is None
    )
    assert userdata.end_call_nudged is False


@pytest.mark.asyncio
async def test_gate_allows_close_when_llm_unavailable() -> None:
    # realtime models / no LLM: skip the audit rather than block the goodbye
    userdata = _userdata()
    assert (
        await run_goodbye_gate(userdata, None, instructions=POLICY, chat_ctx=_transcript_ctx())
        is None
    )
