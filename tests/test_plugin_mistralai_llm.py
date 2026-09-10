from __future__ import annotations

import types

import pytest
from mistralai.client.models import (
    ToolExecutionDeltaEvent,
    ToolExecutionDoneEvent,
    ToolExecutionStartedEvent,
)

from livekit.agents import llm
from livekit.plugins.mistralai.llm import LLMStream

pytestmark = pytest.mark.plugin("mistralai")


class _RecordingLLM:
    """Captures the events the stream emits on its parent LLM."""

    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def emit(self, name: str, payload: object) -> None:
        self.events.append((name, payload))


def _stream() -> LLMStream:
    # bypass __init__/network; _parse_event only needs the arg buffer and the parent LLM
    stream = LLMStream.__new__(LLMStream)
    stream._provider_tool_args = {}
    stream._llm = _RecordingLLM()  # type: ignore[assignment]
    return stream


def _event(data: object) -> object:
    # _parse_event only reads `ev.data`
    return types.SimpleNamespace(data=data)


class TestProviderToolLifecycle:
    def test_started_emits_started_event(self) -> None:
        stream = _stream()

        chunks = stream._parse_event(
            _event(ToolExecutionStartedEvent(id="t1", name="web_search", arguments='{"q":"x"}')),
            {},
        )

        # provider tools flow via the LLM event, not the chunk stream
        assert chunks == []
        assert len(stream._llm.events) == 1
        name, call = stream._llm.events[0]
        assert name == "provider_tool_call"
        assert isinstance(call, llm.ProviderToolCall)
        assert call.phase == "started"
        assert call.call_id == "t1"
        assert call.name == "web_search"
        assert call.arguments == '{"q":"x"}'

    def test_delta_accumulates_without_emitting(self) -> None:
        stream = _stream()
        stream._parse_event(
            _event(ToolExecutionStartedEvent(id="t1", name="web_search", arguments="{")),
            {},
        )
        stream._llm.events.clear()

        chunks = stream._parse_event(
            _event(ToolExecutionDeltaEvent(id="t1", name="web_search", arguments='"q":"x"}')),
            {},
        )

        assert chunks == []
        assert stream._llm.events == []
        assert stream._provider_tool_args["t1"] == '{"q":"x"}'

    def test_done_emits_ended_event_with_accumulated_args_and_result(self) -> None:
        stream = _stream()
        stream._parse_event(
            _event(ToolExecutionStartedEvent(id="t1", name="web_search", arguments="{")),
            {},
        )
        stream._parse_event(
            _event(ToolExecutionDeltaEvent(id="t1", name="web_search", arguments='"q":"x"}')),
            {},
        )
        stream._llm.events.clear()

        stream._parse_event(
            _event(ToolExecutionDoneEvent(id="t1", name="web_search", info={"answer": 42})),
            {},
        )

        assert len(stream._llm.events) == 1
        name, call = stream._llm.events[0]
        assert name == "provider_tool_call"
        assert call.phase == "done"
        assert call.call_id == "t1"
        assert call.name == "web_search"
        assert call.arguments == '{"q":"x"}'
        assert call.result == str({"answer": 42})
        # state is popped so a later turn can't leak args
        assert "t1" not in stream._provider_tool_args

    def test_done_without_start_is_safe(self) -> None:
        stream = _stream()

        stream._parse_event(
            _event(ToolExecutionDoneEvent(id="ghost", name="web_search", info=None)),
            {},
        )

        assert len(stream._llm.events) == 1
        _, call = stream._llm.events[0]
        assert call.phase == "done"
        assert call.arguments == ""
        assert call.result is None
