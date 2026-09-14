from __future__ import annotations

import asyncio
import types
from typing import Literal

import pytest
from mistralai.client.models import (
    ToolExecutionDeltaEvent,
    ToolExecutionDoneEvent,
    ToolExecutionStartedEvent,
)

from livekit import rtc
from livekit.agents import APIConnectionError, llm
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.plugins.mistralai.llm import LLMStream

pytestmark = pytest.mark.plugin("mistralai")


class _RecordingLLM:
    """Captures provider-tool events emitted by the stream."""

    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def emit(self, name: str, payload: object) -> None:
        self.events.append((name, payload))


def _stream() -> LLMStream:
    # Bypass __init__/network; initialize only the emitter state needed by _parse_event.
    stream = LLMStream.__new__(LLMStream)
    rtc.EventEmitter.__init__(stream)
    stream._pending_provider_tool_calls = {}
    stream._llm = _RecordingLLM()  # type: ignore[assignment]
    stream.on("provider_tool_call", lambda call: stream._llm.emit("provider_tool_call", call))
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

        # Provider tools flow via the stream event, not the chunk stream.
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
        assert stream._pending_provider_tool_calls["t1"].arguments == '{"q":"x"}'

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
        assert call.status == "done"
        assert call.call_id == "t1"
        assert call.name == "web_search"
        assert call.arguments == '{"q":"x"}'
        assert call.result == str({"answer": 42})
        # state is popped so a later turn can't leak args
        assert "t1" not in stream._pending_provider_tool_calls

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

    @pytest.mark.parametrize("status", ["error", "cancelled"])
    def test_unfinished_call_emits_terminal_update(
        self, status: Literal["error", "cancelled"]
    ) -> None:
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

        stream._end_pending_provider_tool_calls(status=status)

        assert len(stream._llm.events) == 1
        name, call = stream._llm.events[0]
        assert name == "provider_tool_call"
        assert call.phase == "done"
        assert call.status == status
        assert call.call_id == "t1"
        assert call.name == "web_search"
        assert call.arguments == '{"q":"x"}'
        assert call.result is None
        assert not stream._pending_provider_tool_calls

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", ["error", "cancelled"])
    async def test_stream_failure_ends_outstanding_call(
        self, status: Literal["error", "cancelled"]
    ) -> None:
        stream = _stream()
        stream._chat_ctx = llm.ChatContext.empty()
        stream._tool_ctx = llm.ToolContext([])
        stream._model = "test-model"
        stream._conn_options = DEFAULT_API_CONNECT_OPTIONS
        stream._extra_kwargs = {}

        async def _events():
            yield _event(
                ToolExecutionStartedEvent(id="t1", name="web_search", arguments='{"q":"x"}')
            )
            if status == "cancelled":
                raise asyncio.CancelledError
            raise RuntimeError("connection lost")

        async def _start_stream_async(**kwargs: object) -> object:
            return _events()

        stream._client = types.SimpleNamespace(
            beta=types.SimpleNamespace(
                conversations=types.SimpleNamespace(start_stream_async=_start_stream_async)
            )
        )

        expected_error = asyncio.CancelledError if status == "cancelled" else APIConnectionError
        with pytest.raises(expected_error):
            await stream._run()

        calls = [payload for name, payload in stream._llm.events if name == "provider_tool_call"]
        assert [(call.phase, call.status) for call in calls] == [
            ("started", None),
            ("done", status),
        ]
