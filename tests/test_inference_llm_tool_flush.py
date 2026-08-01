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

from types import SimpleNamespace
from typing import Any

import pytest

from livekit.agents.inference.llm import LLMStream
from livekit.agents.llm.utils import ThinkingTokenFilter

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


def _tool(
    *,
    name: str | None = None,
    arguments: str | None = None,
    id: str | None = None,
    index: int = 0,
) -> Any:
    return SimpleNamespace(
        function=SimpleNamespace(name=name, arguments=arguments),
        id=id if id is not None else ("call_1" if name else None),
        index=index,
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
    thinking = ThinkingTokenFilter()

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
    thinking = ThinkingTokenFilter()

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
    thinking = ThinkingTokenFilter()

    first = stream._parse_choice("c", _choice(tools=[_tool(name="a", arguments="")]), thinking)
    assert first is not None and first.delta is not None and first.delta.tool_call_started is True

    # Argument fragments for the same call never re-signal.
    assert stream._parse_choice("c", _choice(tools=[_tool(arguments="{}")]), thinking) is None

    # A second tool call later in the turn flushes the first as a call chunk,
    # but the start signal was already emitted and must not fire again.
    second = stream._parse_choice(
        "c", _choice(tools=[_tool(name="b", arguments="", id="call_2", index=1)]), thinking
    )
    assert second is not None and second.delta is not None
    assert second.delta.tool_call_started is False
    assert len(second.delta.tool_calls) == 1
    assert second.delta.tool_calls[0].name == "a"


def test_content_in_same_delta_as_tool_survives_on_marker() -> None:
    # Some providers pack the last text token into the same delta as the first
    # named tool call; that content must ride the marker chunk, not be dropped.
    stream = _stream()
    thinking = ThinkingTokenFilter()

    marker = stream._parse_choice(
        "c",
        _choice(content="One moment.", tools=[_tool(name="get_balance", arguments="")]),
        thinking,
    )
    assert marker is not None and marker.delta is not None
    assert marker.delta.tool_call_started is True
    assert marker.delta.content == "One moment."
    assert not marker.delta.tool_calls

    final = stream._parse_choice("c", _choice(finish="tool_calls"), thinking)
    assert final is not None and final.delta is not None
    assert len(final.delta.tool_calls) == 1
    assert final.delta.tool_calls[0].name == "get_balance"


def test_parallel_tool_calls_in_one_delta_both_accumulated() -> None:
    # Two named tool calls batched into a single delta: the first is flushed as a
    # call chunk when the second starts, and the second is assembled at finish.
    # The flushed call preempts the post-loop marker, so it must carry the start
    # signal itself or the preamble flush would be lost for the whole turn.
    stream = _stream()
    thinking = ThinkingTokenFilter()

    chunk = stream._parse_choice(
        "c",
        _choice(
            tools=[
                _tool(name="get_balance", arguments='{"account":1}', id="call_1", index=0),
                _tool(name="get_history", arguments='{"account":2}', id="call_2", index=1),
            ]
        ),
        thinking,
    )
    assert chunk is not None and chunk.delta is not None
    assert chunk.delta.tool_call_started is True
    assert len(chunk.delta.tool_calls) == 1
    assert chunk.delta.tool_calls[0].name == "get_balance"
    assert chunk.delta.tool_calls[0].arguments == '{"account":1}'

    final = stream._parse_choice("c", _choice(finish="tool_calls"), thinking)
    assert final is not None and final.delta is not None
    assert len(final.delta.tool_calls) == 1
    assert final.delta.tool_calls[0].name == "get_history"
    assert final.delta.tool_calls[0].arguments == '{"account":2}'


def test_finish_reason_in_same_delta_still_emits_call() -> None:
    # A provider may send the whole tool call and finish_reason="tool_calls" in
    # one delta; the marker must never preempt the assembled call.
    stream = _stream()
    thinking = ThinkingTokenFilter()

    final = stream._parse_choice(
        "c",
        _choice(tools=[_tool(name="get_balance", arguments='{"account":1}')], finish="tool_calls"),
        thinking,
    )
    assert final is not None and final.delta is not None
    assert final.delta.tool_call_started is False
    assert len(final.delta.tool_calls) == 1
    assert final.delta.tool_calls[0].name == "get_balance"
    assert final.delta.tool_calls[0].arguments == '{"account":1}'
