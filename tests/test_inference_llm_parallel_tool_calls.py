from __future__ import annotations

import json

import pytest
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from livekit.agents.inference.llm import LLMStream
from livekit.agents.llm.utils import ThinkingTokenFilter

pytestmark = pytest.mark.unit

CITIES = ["Paris", "London", "Rome"]


def _stream() -> LLMStream:
    stream = LLMStream.__new__(LLMStream)
    stream._tool_call_id = None
    stream._fnc_name = None
    stream._fnc_raw_arguments = None
    stream._tool_extra = None
    stream._tool_index = None
    return stream


def _all_calls_in_one_delta() -> list[ChoiceDeltaToolCall]:
    # Some OpenAI-compatible providers (e.g. Ollama) send every parallel call,
    # complete, in a single chunk.
    return [
        ChoiceDeltaToolCall(
            index=i,
            id=f"call-{i}",
            type="function",
            function=ChoiceDeltaToolCallFunction(
                name="get_weather", arguments=json.dumps({"city": city})
            ),
        )
        for i, city in enumerate(CITIES)
    ]


def _cities(chunks) -> list[str]:
    return [
        json.loads(call.arguments)["city"]
        for chunk in chunks
        if chunk is not None and chunk.delta is not None
        for call in chunk.delta.tool_calls
    ]


def test_parallel_calls_in_one_chunk_are_all_emitted() -> None:
    stream = _stream()
    choices = [
        Choice(index=0, delta=ChoiceDelta(tool_calls=_all_calls_in_one_delta())),
        Choice(index=0, finish_reason="tool_calls", delta=ChoiceDelta()),
    ]

    chunks = [stream._parse_choice("chunk", choice, ThinkingTokenFilter()) for choice in choices]

    assert _cities(chunks) == CITIES


@pytest.mark.parametrize("finish_reason", ["tool_calls", "stop"])
def test_parallel_calls_in_the_finishing_chunk_are_all_emitted(finish_reason: str) -> None:
    stream = _stream()
    choice = Choice(
        index=0,
        finish_reason=finish_reason,
        delta=ChoiceDelta(tool_calls=_all_calls_in_one_delta()),
    )

    chunk = stream._parse_choice("chunk", choice, ThinkingTokenFilter())

    assert _cities([chunk]) == CITIES
