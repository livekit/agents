"""Unit tests for Anthropic LLMStream event parsing (no network)."""

from __future__ import annotations

import pytest
from anthropic import types as at

from livekit.agents import APIConnectOptions, llm
from livekit.plugins.anthropic import LLM
from livekit.plugins.anthropic.llm import LLMStream

pytestmark = [pytest.mark.plugin("anthropic"), pytest.mark.asyncio]


def _stream() -> LLMStream:
    async def _create():  # pragma: no cover - never called
        raise AssertionError

    return LLMStream(
        LLM(api_key="sk-ant-test"),
        create_anthropic_stream=_create,
        chat_ctx=llm.ChatContext.empty(),
        tools=[],
        conn_options=APIConnectOptions(),
    )


async def test_output_tokens_not_double_counted() -> None:
    s = _stream()
    await s.aclose()
    start = at.RawMessageStartEvent(
        type="message_start",
        message=at.Message(
            id="m",
            type="message",
            role="assistant",
            model="claude",
            content=[],
            stop_reason=None,
            stop_sequence=None,
            usage=at.Usage(input_tokens=10, output_tokens=5),
        ),
    )
    delta = at.RawMessageDeltaEvent(
        type="message_delta",
        delta=at.raw_message_delta_event.Delta(stop_reason="end_turn", stop_sequence=None),
        usage=at.MessageDeltaUsage(output_tokens=42),  # cumulative per API docs
    )
    s._parse_event(start)
    s._parse_event(delta)
    assert s._output_tokens == 42
