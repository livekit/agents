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
    stream._tool_calls = {}
    stream._tool_key = None
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


def _calls_with_index(index: int | None) -> list[ChoiceDeltaToolCall]:
    # Parallel calls sent whole, all with the same index or none at all. The OpenAI client
    # builds a delta without an index as `index=None`, which validation would reject here.
    return [
        ChoiceDeltaToolCall.model_construct(
            index=index,
            id=f"call-{i}",
            type="function",
            function=ChoiceDeltaToolCallFunction(
                name="get_weather", arguments=json.dumps({"city": city})
            ),
        )
        for i, city in enumerate(CITIES)
    ]


@pytest.mark.parametrize("index", [None, 0])
def test_parallel_calls_with_the_same_or_no_index_are_told_apart_by_id(index: int | None) -> None:
    stream = _stream()
    choices = [
        Choice(index=0, delta=ChoiceDelta(tool_calls=_calls_with_index(index))),
        Choice(index=0, finish_reason="tool_calls", delta=ChoiceDelta()),
    ]

    chunks = [stream._parse_choice("chunk", choice, ThinkingTokenFilter()) for choice in choices]

    assert _cities(chunks) == CITIES


@pytest.mark.parametrize("index", [None, 0])
def test_parallel_calls_without_distinct_index_in_separate_chunks(index: int | None) -> None:
    stream = _stream()
    choices = [
        Choice(index=0, delta=ChoiceDelta(tool_calls=[call])) for call in _calls_with_index(index)
    ] + [Choice(index=0, finish_reason="tool_calls", delta=ChoiceDelta())]

    chunks = [stream._parse_choice("chunk", choice, ThinkingTokenFilter()) for choice in choices]

    assert _cities(chunks) == CITIES


def test_a_call_whose_id_is_repeated_on_every_delta_stays_one_call() -> None:
    stream = _stream()
    deltas = [
        ChoiceDeltaToolCall(
            index=0,
            id="call-0",
            type="function",
            function=ChoiceDeltaToolCallFunction(name=name, arguments=args),
        )
        for name, args in (("get_weather", '{"city": '), (None, '"Paris"}'))
    ]
    choices = [Choice(index=0, delta=ChoiceDelta(tool_calls=[d])) for d in deltas] + [
        Choice(index=0, finish_reason="tool_calls", delta=ChoiceDelta())
    ]

    chunks = [stream._parse_choice("chunk", choice, ThinkingTokenFilter()) for choice in choices]

    assert _cities(chunks) == ["Paris"]


def test_a_different_index_still_starts_a_new_call_even_with_the_same_id() -> None:
    stream = _stream()
    deltas = [
        ChoiceDeltaToolCall(
            index=i,
            id="call-0",
            type="function",
            function=ChoiceDeltaToolCallFunction(
                name="get_weather", arguments=json.dumps({"city": city})
            ),
        )
        for i, city in enumerate(["Paris", "London"])
    ]
    choices = [Choice(index=0, delta=ChoiceDelta(tool_calls=[d])) for d in deltas] + [
        Choice(index=0, finish_reason="tool_calls", delta=ChoiceDelta())
    ]

    chunks = [stream._parse_choice("chunk", choice, ThinkingTokenFilter()) for choice in choices]

    assert _cities(chunks) == ["Paris", "London"]


def _delta(id: str | None, name: str | None, args: str, index: int | None = None):
    return ChoiceDeltaToolCall.model_construct(
        index=index,
        id=id,
        type="function",
        function=ChoiceDeltaToolCallFunction(name=name, arguments=args),
    )


def _run(deltas: list[ChoiceDeltaToolCall]) -> list[str]:
    stream = _stream()
    choices = [Choice(index=0, delta=ChoiceDelta(tool_calls=[d])) for d in deltas] + [
        Choice(index=0, finish_reason="tool_calls", delta=ChoiceDelta())
    ]
    chunks = [stream._parse_choice("chunk", choice, ThinkingTokenFilter()) for choice in choices]
    return _cities(chunks)


@pytest.mark.parametrize("index", [None, 0])
def test_interleaved_calls_without_distinct_index_keep_their_own_arguments(
    index: int | None,
) -> None:
    deltas = [
        _delta("a", "get_weather", '{"city":', index),
        _delta("b", "get_weather", '{"city":', index),
        _delta("a", None, '"Paris"}', index),
        _delta("b", None, '"London"}', index),
    ]

    assert _run(deltas) == ["Paris", "London"]


def test_a_continuation_without_id_extends_the_latest_call() -> None:
    deltas = [
        _delta("a", "get_weather", '{"city":'),
        _delta("b", "get_weather", '{"city":'),
        _delta(None, None, '"London"}'),
    ]

    # call "a" never got its arguments, so it is emitted as sent
    stream = _stream()
    choices = [Choice(index=0, delta=ChoiceDelta(tool_calls=[d])) for d in deltas] + [
        Choice(index=0, finish_reason="tool_calls", delta=ChoiceDelta())
    ]
    chunks = [stream._parse_choice("chunk", c, ThinkingTokenFilter()) for c in choices]
    calls = [c for chunk in chunks if chunk for c in chunk.delta.tool_calls]
    assert [(c.call_id, c.arguments) for c in calls] == [
        ("a", '{"city":'),
        ("b", '{"city":"London"}'),
    ]


def test_arguments_without_a_started_call_are_ignored() -> None:
    assert _run([_delta(None, None, '"Paris"}')]) == []
