from __future__ import annotations

import json

import anthropic
import httpx2 as httpx
import pytest

from livekit.agents import APIConnectOptions, llm
from livekit.plugins.anthropic import LLM

pytestmark = pytest.mark.plugin("anthropic")


async def test_retry_opens_a_fresh_request_without_mutating_the_injected_client() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            raise httpx.ConnectError("connection refused", request=request)
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
                    "usage": {"input_tokens": 1, "output_tokens": 0},
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
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        ]
        body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

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
