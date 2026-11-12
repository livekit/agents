from __future__ import annotations

import json
from typing import Any

import anthropic
import httpx2 as httpx
import pytest

from livekit.agents import APIConnectOptions, llm
from livekit.plugins.anthropic import LLM

pytestmark = pytest.mark.plugin("anthropic")


def _hello_response(start_usage: dict[str, Any], delta_usage: dict[str, Any]) -> httpx.Response:
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-test",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": start_usage,
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hello"},
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": delta_usage,
        },
        {"type": "message_stop"},
    ]
    body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


async def test_retry_opens_a_fresh_request_without_mutating_the_injected_client() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            raise httpx.ConnectError("connection refused", request=request)
        return _hello_response({"input_tokens": 1, "output_tokens": 0}, {"output_tokens": 1})

    async with anthropic.AsyncClient(
        api_key="test-key",
        max_retries=5,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ) as client:
        model = LLM(api_key="test-key", client=client)
        context = llm.ChatContext.empty()
        context.add_message(role="user", content="hi")
        try:
            async with model.chat(
                chat_ctx=context,
                conn_options=APIConnectOptions(max_retry=1, retry_interval=0),
            ) as stream:
                response = await stream.collect()
        finally:
            await model.aclose()

        assert response.text == "hello"
        assert len(requests) == 2
        assert client.max_retries == 5


async def test_usage_counts_cache_inside_the_input_and_output_once() -> None:
    # Anthropic reports cache reads and writes outside input_tokens, and message_delta's
    # output_tokens is cumulative (it already includes message_start's).
    def handler(request: httpx.Request) -> httpx.Response:
        return _hello_response(
            {
                "input_tokens": 20,
                "output_tokens": 1,
                "cache_read_input_tokens": 60,
                "cache_creation_input_tokens": 20,
            },
            {"output_tokens": 30},
        )

    async with anthropic.AsyncClient(
        api_key="test-key", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as client:
        model = LLM(api_key="test-key", client=client)
        context = llm.ChatContext.empty()
        context.add_message(role="user", content="hi")
        try:
            async with model.chat(chat_ctx=context) as stream:
                response = await stream.collect()
        finally:
            await model.aclose()

    assert response.usage is not None
    assert response.usage.input_tokens == 100
    assert response.usage.output_tokens == 30
    assert response.usage.cache_read_tokens == 60
    assert response.usage.cache_write_tokens == 20
