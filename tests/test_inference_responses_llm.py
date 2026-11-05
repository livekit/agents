from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx2 as httpx
import openai
import pytest

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    llm as agents_llm,
)
from livekit.agents.inference import responses as inference_responses
from livekit.agents.inference.responses import LLM as ResponsesLLM

pytestmark = pytest.mark.unit

_RESPONSE = {
    "id": "resp_1",
    "created_at": 0,
    "model": "gpt-4.1",
    "object": "response",
    "output": [],
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
}
_USAGE = {
    "input_tokens": 120,
    "output_tokens": 30,
    "total_tokens": 150,
    "input_tokens_details": {"cached_tokens": 64, "cache_write_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0},
}
_FRAMES = [
    {"type": "response.created", "sequence_number": 0, "response": _RESPONSE},
    {
        "type": "response.output_text.delta",
        "sequence_number": 1,
        "output_index": 0,
        "content_index": 0,
        "item_id": "msg_1",
        "delta": "hi",
        "logprobs": [],
    },
    {
        "type": "response.completed",
        "sequence_number": 2,
        "response": {**_RESPONSE, "status": "completed", "usage": _USAGE},
    },
]


def _sse(frames: list[dict]) -> bytes:
    return "".join(f"event: {f['type']}\ndata: {json.dumps(f)}\n\n" for f in frames).encode()


class _GatewayTransport(httpx.AsyncBaseTransport):
    def __init__(self, response: httpx.Response | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self._response = response

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        self.requests.append(request)
        if self._response is not None:
            return httpx.Response(
                self._response.status_code,
                headers=self._response.headers,
                content=self._response.content,
                request=request,
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(_FRAMES),
            request=request,
        )

    def body(self, index: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[index].content)


def _make_llm(
    monkeypatch: pytest.MonkeyPatch,
    transport: _GatewayTransport,
    model: str = "openai/gpt-4.1",
    **kwargs: Any,
) -> ResponsesLLM:
    def _make_client(self: ResponsesLLM) -> openai.AsyncClient:
        return openai.AsyncClient(
            api_key="test-key",
            base_url=self._opts.base_url,
            max_retries=0,
            http_client=httpx.AsyncClient(transport=transport),
        )

    monkeypatch.setattr(ResponsesLLM, "_make_client", _make_client)
    return ResponsesLLM(model, base_url="https://gw.test/v1", api_key="k", api_secret="s", **kwargs)


@agents_llm.function_tool
async def _get_weather(location: str) -> str:
    """Get the weather for a location."""
    return "sunny"


def _ctx(*texts: str) -> agents_llm.ChatContext:
    chat_ctx = agents_llm.ChatContext()
    for text in texts or ("hi",):
        chat_ctx.add_message(role="user", content=text)
    return chat_ctx


async def _drain(
    model: ResponsesLLM, chat_ctx: agents_llm.ChatContext | None = None, **kwargs: Any
) -> list[agents_llm.ChatChunk]:
    stream = model.chat(chat_ctx=chat_ctx or _ctx(), **kwargs)
    try:
        return [chunk async for chunk in stream]
    finally:
        await stream.aclose()


async def test_request_targets_gateway_responses_path(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _make_llm(monkeypatch, transport)

    await _drain(model)

    assert str(transport.requests[0].url) == "https://gw.test/v1/responses"


async def test_token_minted_per_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def _counting_token(api_key: str | None, api_secret: str | None, ttl: float = 600) -> str:
        nonlocal calls
        calls += 1
        return f"token-{calls}"

    monkeypatch.setattr(inference_responses, "create_access_token", _counting_token)
    model = ResponsesLLM(
        "openai/gpt-4.1", base_url="https://gw.test/v1", api_key="k", api_secret="s"
    )
    transport = _GatewayTransport()
    model._client = openai.AsyncClient(
        api_key="stale",
        base_url="https://gw.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=transport),
    )

    await _drain(model)
    await _drain(model)

    assert calls == 3
    assert transport.requests[-1].headers["authorization"] == "Bearer token-3"


async def test_inference_headers_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _make_llm(monkeypatch, transport)

    await _drain(model)

    assert transport.requests[0].headers["user-agent"].startswith("LiveKit Agents/")


async def test_provider_header_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _make_llm(monkeypatch, transport, provider="azure")

    await _drain(model)

    assert transport.requests[0].headers["x-livekit-inference-provider"] == "azure"


async def test_inference_class_header_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _make_llm(monkeypatch, transport, inference_class="priority")

    await _drain(model)

    assert transport.requests[0].headers["x-livekit-inference-priority"] == "priority"


async def test_extra_kwargs_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _make_llm(monkeypatch, transport, extra_kwargs={"max_output_tokens": 64})

    await _drain(model)

    assert transport.body()["max_output_tokens"] == 64


async def test_per_call_parallel_tool_calls_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _make_llm(monkeypatch, transport, extra_kwargs={"parallel_tool_calls": False})

    await _drain(model, extra_kwargs={"parallel_tool_calls": True})

    assert transport.body()["parallel_tool_calls"] is True


async def test_per_call_extra_kwargs_tool_choice_converted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _GatewayTransport()
    model = _make_llm(monkeypatch, transport, extra_kwargs={"tool_choice": "none"})

    await _drain(
        model,
        tools=[_get_weather],
        extra_kwargs={"tool_choice": {"type": "function", "function": {"name": "get_weather"}}},
    )

    assert transport.body()["tool_choice"] == {"type": "function", "name": "get_weather"}


async def test_tool_choice_dropped_without_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _make_llm(monkeypatch, transport, extra_kwargs={"tool_choice": "required"})

    await _drain(model, tools=[])

    assert "tool_choice" not in transport.body()


async def test_tool_choice_sent_with_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _make_llm(monkeypatch, transport, extra_kwargs={"tool_choice": "required"})

    await _drain(model, tools=[_get_weather])

    assert transport.body()["tool_choice"] == "required"


class _TimeoutMidStream(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield _sse(_FRAMES[:1])
        raise httpx.ReadTimeout("read timed out")


class _StallingTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_TimeoutMidStream(),
            request=request,
        )


async def test_read_timeout_mid_stream_is_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _make_client(self: ResponsesLLM) -> openai.AsyncClient:
        return openai.AsyncClient(
            api_key="test-key",
            base_url=self._opts.base_url,
            max_retries=0,
            http_client=httpx.AsyncClient(transport=_StallingTransport()),
        )

    monkeypatch.setattr(ResponsesLLM, "_make_client", _make_client)
    model = ResponsesLLM(
        "openai/gpt-4.1", base_url="https://gw.test/v1", api_key="k", api_secret="s"
    )

    with pytest.raises(APITimeoutError):
        await _drain(model, conn_options=APIConnectOptions(max_retry=0))


async def test_prompt_cache_ttl_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _make_llm(monkeypatch, transport, extra_kwargs={"prompt_cache_options": {"ttl": "30m"}})

    await _drain(model)

    assert transport.body()["prompt_cache_options"] == {"ttl": "30m"}


async def test_breakpoints_sent_for_gpt_5_6(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _make_llm(monkeypatch, transport, model="openai/gpt-5.6-luna")
    chat_ctx = agents_llm.ChatContext()
    chat_ctx.add_message(role="system", content=["static prompt", agents_llm.CacheBreakpoint()])
    chat_ctx.add_message(role="user", content="hi")

    await _drain(model, chat_ctx)

    assert "prompt_cache_breakpoint" in json.dumps(transport.body()["input"])


async def test_breakpoints_not_sent_for_gpt_4_1(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _make_llm(monkeypatch, transport, model="openai/gpt-4.1")
    chat_ctx = agents_llm.ChatContext()
    chat_ctx.add_message(role="system", content=["static prompt", agents_llm.CacheBreakpoint()])
    chat_ctx.add_message(role="user", content="hi")

    await _drain(model, chat_ctx)

    assert "prompt_cache_breakpoint" not in json.dumps(transport.body()["input"])


async def test_previous_response_id_never_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _make_llm(monkeypatch, transport)
    chat_ctx = _ctx("hi")

    await _drain(model, chat_ctx)
    chat_ctx.add_message(role="assistant", content="hi")
    chat_ctx.add_message(role="user", content="and again")
    await _drain(model, chat_ctx)

    assert len(transport.requests) == 2
    assert "previous_response_id" not in transport.body(0)
    assert "previous_response_id" not in transport.body(1)


async def test_text_and_usage_streamed(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _make_llm(monkeypatch, transport)

    chunks = await _drain(model)

    text = "".join(c.delta.content or "" for c in chunks if c.delta)
    usage = next(c.usage for c in chunks if c.usage)
    assert text == "hi"
    assert usage.prompt_tokens == 120
    assert usage.completion_tokens == 30
    assert usage.total_tokens == 150


async def test_rate_limit_logs_quota_headers(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.agents")
    transport = _GatewayTransport(
        httpx.Response(
            429,
            headers={
                "content-type": "application/json",
                "X-LiveKit-Inference-RPM-Limit": "10",
            },
            json={"error": "rate limited"},
        )
    )
    model = _make_llm(monkeypatch, transport)

    with pytest.raises(APIStatusError) as exc_info:
        await _drain(model, conn_options=APIConnectOptions(max_retry=0))

    records = [
        r
        for r in caplog.records
        if r.getMessage() == "LLM request rate limited by inference gateway"
    ]
    assert exc_info.value.status_code == 429
    assert len(records) == 1
    assert records[0].rpm_limit == "10"  # type: ignore[attr-defined]


async def test_terminal_quota_denial_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport(
        httpx.Response(
            429,
            headers={"content-type": "application/json", "x-should-retry": "False"},
            json={"error": "credit quota exhausted", "category": "MaxGatewayCredits"},
        )
    )
    model = _make_llm(monkeypatch, transport)

    with pytest.raises(APIStatusError) as exc_info:
        await _drain(model, conn_options=APIConnectOptions(max_retry=1, retry_interval=0))

    assert exc_info.value.status_code == 429
    assert exc_info.value.retryable is False
    assert len(transport.requests) == 1


async def test_quota_denial_without_should_retry_is_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _GatewayTransport(
        httpx.Response(
            429,
            headers={"content-type": "application/json"},
            json={"error": "rate limited", "category": "MaxConcurrentGatewayLLMRpm"},
        )
    )
    model = _make_llm(monkeypatch, transport)

    with pytest.raises(APIConnectionError, match="after 2 attempts"):
        await _drain(model, conn_options=APIConnectOptions(max_retry=1, retry_interval=0))

    assert len(transport.requests) == 2


def test_provider_property_is_livekit(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_llm(monkeypatch, _GatewayTransport())

    assert model.provider == "livekit"
    assert model.model == "openai/gpt-4.1"


async def test_rest_read_timeout_defaults_to_conn_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _make_llm(monkeypatch, transport)

    await _drain(model, conn_options=APIConnectOptions(max_retry=0, timeout=3.0))

    assert transport.requests[0].extensions["timeout"]["read"] == 3.0
