"""Unit tests for the tool-call preamble flush signal in ``inference.llm.LLMStream``.

Regression coverage for #5826: when a model streams a text preamble followed by a
tool call in the same turn, ``_parse_choice`` used to emit nothing while the tool
arguments serialized (~1s), leaving the preamble buffered in TTS and producing
audible dead air. It now emits a ``tool_call_started`` marker chunk at the tool
boundary so the preamble can be flushed immediately.

These tests drive ``_parse_choice`` directly with fake OpenAI-shaped deltas, so no
network or API key is required.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from livekit.agents.inference.llm import LLMStream

pytestmark = pytest.mark.unit


def _stream() -> LLMStream:
    # Build without __init__ — these tests only exercise _parse_choice and its state.
    stream = LLMStream.__new__(LLMStream)
    stream._tool_call_id = None
    stream._fnc_name = None
    stream._fnc_raw_arguments = None
    stream._tool_extra = None
    stream._tool_index = None
    stream._tool_start_signaled = False
    return stream


def _tool(*, name: str | None = None, arguments: str | None = None) -> Any:
    return SimpleNamespace(
        function=SimpleNamespace(name=name, arguments=arguments),
        id="call_1" if name else None,
        index=0,
        type="function",
        extra_content=None,
    )


def _choice(*, content: str | None = None, tools: list[Any] | None = None, finish=None) -> Any:
    return SimpleNamespace(
        delta=SimpleNamespace(content=content, tool_calls=tools, extra_content=None),
        finish_reason=finish,
    )


def test_marker_emitted_at_tool_boundary() -> None:
    stream = _stream()
    thinking = asyncio.Event()

    preamble = stream._parse_choice("c", _choice(content="Let me check"), thinking)
    assert preamble is not None and preamble.delta is not None
    assert preamble.delta.content == "Let me check"
    assert preamble.delta.tool_call_started is False

    marker = stream._parse_choice(
        "c", _choice(tools=[_tool(name="get_balance", arguments="")]), thinking
    )
    assert marker is not None and marker.delta is not None
    assert marker.delta.tool_call_started is True
    # The marker must not carry an executable tool call, or the pipeline would run the
    # tool with incomplete arguments.
    assert not marker.delta.tool_calls
    assert marker.delta.content is None


def test_arguments_stream_without_extra_markers() -> None:
    stream = _stream()
    thinking = asyncio.Event()

    stream._parse_choice("c", _choice(tools=[_tool(name="get_balance", arguments="")]), thinking)
    assert stream._parse_choice("c", _choice(tools=[_tool(arguments='{"acc')]), thinking) is None
    assert stream._parse_choice("c", _choice(tools=[_tool(arguments='ount":1}')]), thinking) is None

    final = stream._parse_choice("c", _choice(finish="tool_calls"), thinking)
    assert final is not None and final.delta is not None
    assert final.delta.tool_call_started is False
    assert len(final.delta.tool_calls) == 1
    call = final.delta.tool_calls[0]
    assert call.name == "get_balance"
    assert call.arguments == '{"account":1}'


def test_marker_emitted_once_per_turn() -> None:
    stream = _stream()
    thinking = asyncio.Event()

    first = stream._parse_choice("c", _choice(tools=[_tool(name="a", arguments="")]), thinking)
    assert first is not None and first.delta is not None and first.delta.tool_call_started is True

    # Argument fragments for the same call never re-signal.
    assert stream._parse_choice("c", _choice(tools=[_tool(arguments="{}")]), thinking) is None
