from __future__ import annotations

import types
from collections.abc import Callable

import httpx
import pytest
from mistralai.client.errors import (
    HTTPValidationError,
    HTTPValidationErrorData,
    MistralError,
    SDKError,
)
from mistralai.client.models import MessageOutputEvent

from livekit.agents import APIStatusError, llm
from livekit.agents.types import APIConnectOptions
from livekit.plugins.mistralai.llm import LLM, ApiMode
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
