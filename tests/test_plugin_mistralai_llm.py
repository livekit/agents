from __future__ import annotations

import asyncio
import types
from collections.abc import Callable
from typing import Literal

import httpx
import pytest
from mistralai.client.errors import (
    HTTPValidationError,
    HTTPValidationErrorData,
    MistralError,
    SDKError,
)
from mistralai.client.models import (
    MessageOutputEvent,
    ToolExecutionDeltaEvent,
    ToolExecutionDoneEvent,
    ToolExecutionStartedEvent,
)

from livekit import rtc
from livekit.agents import APIConnectionError, APIStatusError, llm
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.plugins.mistralai.llm import LLM, ApiMode, LLMStream
from livekit.plugins.mistralai.tools import WebSearch

pytestmark = pytest.mark.plugin("mistralai")


def _event(data: object) -> object:
    return types.SimpleNamespace(data=data)


class _FakeConversations:
    def __init__(self, handler: Callable[[int], object]) -> None:
        self._handler = handler
        self.calls = 0

    async def start_stream_async(self, **kwargs: object) -> object:
        self.calls += 1
        result = self._handler(self.calls)
        if isinstance(result, BaseException):
            raise result
        return result


class _FakeChat:
    def __init__(self, handler: Callable[[int], object]) -> None:
        self._handler = handler
        self.calls = 0

    async def stream_async(self, **kwargs: object) -> object:
        self.calls += 1
        result = self._handler(self.calls)
        if isinstance(result, BaseException):
            raise result
        return result


def _model(
    conversations: _FakeConversations | None = None,
    chat: _FakeChat | None = None,
    api_mode: ApiMode = ApiMode.CONVERSATIONS,
) -> LLM:
    client = types.SimpleNamespace(
        beta=types.SimpleNamespace(
            conversations=conversations or _FakeConversations(lambda _: (_ for _ in ())),
        ),
        chat=chat or _FakeChat(lambda _: (_ for _ in ())),
    )
    return LLM(client=client, model="test-model", api_mode=api_mode)  # type: ignore[arg-type]


def _sdk_error(status_code: int) -> SDKError:
    body = '{"message":"provider error"}'
    response = httpx.Response(
        status_code,
        headers={"content-type": "application/json", "x-request-id": "req_test"},
        content=body,
        request=httpx.Request("POST", "https://api.mistral.ai/v1/conversations"),
    )
    return SDKError("provider error", response, body)


def _validation_error() -> HTTPValidationError:
    body = '{"detail":[{"msg":"invalid request"}]}'
    response = httpx.Response(
        422,
        headers={"content-type": "application/json", "x-request-id": "req_validation"},
        content=body,
        request=httpx.Request("POST", "https://api.mistral.ai/v1/conversations"),
    )
    return HTTPValidationError(HTTPValidationErrorData(), response, body)


class TestLLMErrorHandling:
    @pytest.mark.asyncio
    async def test_client_error_is_not_retried(self) -> None:
        conversations = _FakeConversations(lambda _: _sdk_error(400))
        model = _model(conversations=conversations)
        errors: list[llm.LLMError] = []
        model.on("error", errors.append)

        try:
            with pytest.raises(APIStatusError) as excinfo:
                async with model.chat(
                    chat_ctx=llm.ChatContext.empty(),
                    conn_options=APIConnectOptions(max_retry=3, retry_interval=0.0, timeout=1.0),
                ) as stream:
                    async for _ in stream:
                        pass

            error = excinfo.value
            assert conversations.calls == 1
            assert error.status_code == 400
            assert error.request_id == "req_test"
            assert error.body == '{"message":"provider error"}'
            assert error.retryable is False
            assert len(errors) == 1
            assert errors[0].error is error
            assert errors[0].recoverable is False
        finally:
            await model.aclose()

    @pytest.mark.asyncio
    async def test_validation_error_is_not_retried(self) -> None:
        conversations = _FakeConversations(lambda _: _validation_error())
        model = _model(conversations=conversations)

        try:
            with pytest.raises(APIStatusError) as excinfo:
                async with model.chat(
                    chat_ctx=llm.ChatContext.empty(),
                    conn_options=APIConnectOptions(max_retry=3, retry_interval=0.0, timeout=1.0),
                ) as stream:
                    async for _ in stream:
                        pass

            error = excinfo.value
            assert conversations.calls == 1
            assert error.status_code == 422
            assert error.request_id == "req_validation"
            assert error.body == '{"detail":[{"msg":"invalid request"}]}'
            assert error.retryable is False
        finally:
            await model.aclose()

    @pytest.mark.asyncio
    async def test_status_error_after_output_is_not_retried(self) -> None:
        async def _events():
            yield _event(MessageOutputEvent(id="msg_1", content="partial"))
            raise _sdk_error(500)

        conversations = _FakeConversations(lambda _: _events())
        model = _model(conversations=conversations)
        errors: list[llm.LLMError] = []
        model.on("error", errors.append)
        chunks: list[llm.ChatChunk] = []

        try:
            with pytest.raises(APIStatusError) as excinfo:
                async with model.chat(
                    chat_ctx=llm.ChatContext.empty(),
                    conn_options=APIConnectOptions(max_retry=3, retry_interval=0.0, timeout=1.0),
                ) as stream:
                    async for chunk in stream:
                        chunks.append(chunk)

            assert [chunk.delta.content for chunk in chunks if chunk.delta] == ["partial"]
            assert conversations.calls == 1
            assert excinfo.value.status_code == 500
            assert excinfo.value.retryable is False
            assert len(errors) == 1
            assert errors[0].recoverable is False
        finally:
            await model.aclose()


def _completion_event(
    chunk_id: str = "cmpl_1",
    content: object = None,
    tool_calls: object = None,
    finish_reason: str | None = None,
    usage: object = None,
) -> object:
    delta = types.SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = types.SimpleNamespace(delta=delta, finish_reason=finish_reason)
    chunk = types.SimpleNamespace(id=chunk_id, choices=[choice], usage=usage)
    return types.SimpleNamespace(data=chunk)


def _mistral_error(status_code: int, message: str = "error") -> MistralError:
    response = httpx.Response(
        status_code,
        headers={"content-type": "application/json"},
        content=message,
        request=httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions"),
    )
    return MistralError(message, response, message)


class TestChatCompletionsMode:
    @pytest.mark.asyncio
    async def test_text_streaming(self) -> None:
        async def _events():
            yield _completion_event(content="Hello")
            yield _completion_event(content=" world")

        chat = _FakeChat(lambda _: _events())
        model = _model(chat=chat, api_mode=ApiMode.CHAT_COMPLETIONS)
        chunks: list[llm.ChatChunk] = []

        try:
            async with model.chat(
                chat_ctx=llm.ChatContext.empty(),
                conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
            ) as stream:
                async for chunk in stream:
                    chunks.append(chunk)

            texts = [c.delta.content for c in chunks if c.delta and c.delta.content]
            assert texts == ["Hello", " world"]
            assert chat.calls == 1
        finally:
            await model.aclose()

    @pytest.mark.asyncio
    async def test_tool_call_accumulation(self) -> None:
        tc_part1 = types.SimpleNamespace(
            index=0,
            id="call_1",
            function=types.SimpleNamespace(name="get_weather", arguments='{"loc'),
        )
        tc_part2 = types.SimpleNamespace(
            index=0,
            id=None,
            function=types.SimpleNamespace(name="", arguments='ation": "paris"}'),
        )

        async def _events():
            yield _completion_event(tool_calls=[tc_part1])
            yield _completion_event(tool_calls=[tc_part2], finish_reason="tool_calls")

        chat = _FakeChat(lambda _: _events())
        model = _model(chat=chat, api_mode=ApiMode.CHAT_COMPLETIONS)
        chunks: list[llm.ChatChunk] = []

        try:
            async with model.chat(
                chat_ctx=llm.ChatContext.empty(),
                conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
            ) as stream:
                async for chunk in stream:
                    chunks.append(chunk)

            tool_chunks = [c for c in chunks if c.delta and c.delta.tool_calls]
            assert len(tool_chunks) == 1
            tc = tool_chunks[0].delta.tool_calls[0]
            assert tc.name == "get_weather"
            assert tc.arguments == '{"location": "paris"}'
            assert tc.call_id == "call_1"
        finally:
            await model.aclose()

    @pytest.mark.asyncio
    async def test_usage_chunk(self) -> None:
        usage = types.SimpleNamespace(completion_tokens=10, prompt_tokens=5, total_tokens=15)

        async def _events():
            yield _completion_event(content="hi", usage=usage)

        chat = _FakeChat(lambda _: _events())
        model = _model(chat=chat, api_mode=ApiMode.CHAT_COMPLETIONS)
        chunks: list[llm.ChatChunk] = []

        try:
            async with model.chat(
                chat_ctx=llm.ChatContext.empty(),
                conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
            ) as stream:
                async for chunk in stream:
                    chunks.append(chunk)

            usage_chunks = [c for c in chunks if c.usage]
            assert len(usage_chunks) == 1
            assert usage_chunks[0].usage.completion_tokens == 10
            assert usage_chunks[0].usage.prompt_tokens == 5
            assert usage_chunks[0].usage.total_tokens == 15
        finally:
            await model.aclose()

    @pytest.mark.asyncio
    async def test_default_mode_uses_conversations(self) -> None:
        async def _events():
            yield _event(MessageOutputEvent(id="msg_1", content="ok"))

        conversations = _FakeConversations(lambda _: _events())
        chat = _FakeChat(lambda _: (_ for _ in ()))
        model = _model(conversations=conversations, chat=chat)

        try:
            async with model.chat(
                chat_ctx=llm.ChatContext.empty(),
                conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
            ) as stream:
                async for _ in stream:
                    pass

            assert conversations.calls == 1
            assert chat.calls == 0
        finally:
            await model.aclose()

    @pytest.mark.asyncio
    async def test_mistral_error_mapped(self) -> None:
        chat = _FakeChat(lambda _: _mistral_error(503, "overloaded"))
        model = _model(chat=chat, api_mode=ApiMode.CHAT_COMPLETIONS)

        try:
            with pytest.raises(APIStatusError) as excinfo:
                async with model.chat(
                    chat_ctx=llm.ChatContext.empty(),
                    conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
                ) as stream:
                    async for _ in stream:
                        pass

            assert excinfo.value.status_code == 503
        finally:
            await model.aclose()

    @pytest.mark.asyncio
    async def test_error_after_output_not_retried(self) -> None:
        async def _events():
            yield _completion_event(content="partial")
            raise _mistral_error(500, "server error")

        chat = _FakeChat(lambda _: _events())
        model = _model(chat=chat, api_mode=ApiMode.CHAT_COMPLETIONS)
        chunks: list[llm.ChatChunk] = []

        try:
            with pytest.raises(APIStatusError) as excinfo:
                async with model.chat(
                    chat_ctx=llm.ChatContext.empty(),
                    conn_options=APIConnectOptions(max_retry=3, retry_interval=0.0, timeout=1.0),
                ) as stream:
                    async for chunk in stream:
                        chunks.append(chunk)

            texts = [c.delta.content for c in chunks if c.delta and c.delta.content]
            assert texts == ["partial"]
            assert chat.calls == 1
            assert excinfo.value.retryable is False
        finally:
            await model.aclose()

    @pytest.mark.asyncio
    async def test_provider_tools_rejected(self) -> None:
        chat = _FakeChat(lambda _: (_ for _ in ()))
        model = _model(chat=chat, api_mode=ApiMode.CHAT_COMPLETIONS)

        try:
            with pytest.raises(ValueError, match="Provider tools"):
                async with model.chat(
                    chat_ctx=llm.ChatContext.empty(),
                    conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
                    tools=[WebSearch()],
                ) as stream:
                    async for _ in stream:
                        pass
        finally:
            await model.aclose()


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
            await stream._run_conversations()

        calls = [payload for name, payload in stream._llm.events if name == "provider_tool_call"]
        assert [(call.phase, call.status) for call in calls] == [
            ("started", None),
            ("done", status),
        ]
