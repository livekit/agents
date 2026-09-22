from __future__ import annotations

import types
from collections.abc import Callable

import httpx
import pytest
from mistralai.client.errors import HTTPValidationError, HTTPValidationErrorData, SDKError
from mistralai.client.models import MessageOutputEvent

from livekit.agents import APIStatusError, llm
from livekit.agents.types import APIConnectOptions
from livekit.plugins.mistralai.llm import LLM

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


def _model(conversations: _FakeConversations) -> LLM:
    client = types.SimpleNamespace(
        beta=types.SimpleNamespace(conversations=conversations),
    )
    return LLM(client=client, model="test-model")  # type: ignore[arg-type]


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
        model = _model(conversations)
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
        model = _model(conversations)

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
        model = _model(conversations)
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
