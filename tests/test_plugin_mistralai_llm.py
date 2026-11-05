from __future__ import annotations

import types
from collections.abc import Callable
from typing import Any

import httpx2 as httpx
import pytest
from mistralai.client.errors import (
    HTTPValidationError,
    HTTPValidationErrorData,
    SDKError,
)
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
        self.requests: list[dict[str, object]] = []

    async def start_stream_async(self, **kwargs: object) -> object:
        self.calls += 1
        self.requests.append(kwargs)
        result = self._handler(self.calls)
        if isinstance(result, BaseException):
            raise result
        return result


def _model(conversations: _FakeConversations | None = None, **kwargs: Any) -> LLM:
    client = types.SimpleNamespace(
        beta=types.SimpleNamespace(
            conversations=conversations or _FakeConversations(lambda _: (_ for _ in ())),
        ),
    )
    return LLM(client=client, model="test-model", **kwargs)  # type: ignore[arg-type]


async def _no_events():
    return
    yield


async def _run(model: LLM, **chat_kwargs: Any) -> None:
    async with model.chat(chat_ctx=llm.ChatContext.empty(), **chat_kwargs) as stream:
        async for _ in stream:
            pass


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


class TestStore:
    @pytest.mark.asyncio
    async def test_store_is_not_sent_by_default(self) -> None:
        conversations = _FakeConversations(lambda _: _no_events())
        model = _model(conversations=conversations)

        try:
            await _run(model)
            assert "store" not in conversations.requests[0]
        finally:
            await model.aclose()

    @pytest.mark.asyncio
    async def test_store_is_sent_with_each_request(self) -> None:
        conversations = _FakeConversations(lambda _: _no_events())
        model = _model(conversations=conversations, store=False)

        try:
            await _run(model)
            assert conversations.requests[0]["store"] is False
        finally:
            await model.aclose()

    @pytest.mark.asyncio
    async def test_update_options_changes_store(self) -> None:
        conversations = _FakeConversations(lambda _: _no_events())
        model = _model(conversations=conversations)

        try:
            model.update_options(store=False)
            await _run(model)
            model.update_options(store=True)
            await _run(model)
            assert [request["store"] for request in conversations.requests] == [False, True]
        finally:
            await model.aclose()

    @pytest.mark.asyncio
    async def test_extra_kwargs_override_store(self) -> None:
        conversations = _FakeConversations(lambda _: _no_events())
        model = _model(conversations=conversations, store=False)

        try:
            await _run(model, extra_kwargs={"store": True})
            assert conversations.requests[0]["store"] is True
        finally:
            await model.aclose()
