"""Responses-API inference LLM over REST and the gateway websocket."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import aiohttp
import httpx2 as httpx
import openai
import pytest
from multidict import CIMultiDict, CIMultiDictProxy
from openai.types import Reasoning
from pydantic import BaseModel
from yarl import URL

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    llm as agents_llm,
    utils,
)
from livekit.agents.inference import responses as inference_responses
from livekit.agents.inference.responses import LLM as ResponsesLLM
from livekit.agents.llm._responses_engine import ResponsesWebsocket

pytestmark = pytest.mark.unit

# long enough to keep PyJWT's short-key warning out of the suite output
_SECRET = "s" * 32

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
_CONNECTED = {
    "type": "gateway.connected",
    "deployment": "standard_openai",
    "model": "gpt-4.1",
    "session_id": "sess_1",
}


def _gateway_error(code: str, message: str = "", **extra: Any) -> dict[str, Any]:
    return {"type": "gateway.error", "code": code, "message": message, **extra}


class _Answer(BaseModel):
    city: str
    temperature: float


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
    async with model.chat(chat_ctx=chat_ctx or _ctx(), **kwargs) as stream:
        return [chunk async for chunk in stream]


def _text(chunks: list[agents_llm.ChatChunk]) -> str:
    return "".join(c.delta.content or "" for c in chunks if c.delta)


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------


class _GatewayTransport(httpx.AsyncBaseTransport):
    def __init__(self, response: httpx.Response | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self._response = response

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        self.requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"object": "list", "data": []}, request=request)
        if self._response is not None:
            return httpx.Response(
                self._response.status_code,
                headers=self._response.headers,
                content=self._response.content,
                request=request,
            )
        content = "".join(f"event: {f['type']}\ndata: {json.dumps(f)}\n\n" for f in _FRAMES)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=content.encode(),
            request=request,
        )

    def body(self, index: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[index].content)


def _rest_llm(
    monkeypatch: pytest.MonkeyPatch,
    transport: httpx.AsyncBaseTransport,
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
    return ResponsesLLM(
        model, base_url="https://gw.test/v1", api_key="k", api_secret=_SECRET, **kwargs
    )


async def _rest_body(
    monkeypatch: pytest.MonkeyPatch,
    model: str = "openai/gpt-4.1",
    *,
    update: dict[str, Any] | None = None,
    chat: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    transport = _GatewayTransport()
    llm_model = _rest_llm(monkeypatch, transport, model, **kwargs)
    if update:
        llm_model.update_options(**update)
    await _drain(llm_model, **(chat or {}))
    return transport.body()


async def test_text_and_usage_streamed(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    chunks = await _drain(_rest_llm(monkeypatch, transport))

    usage = next(c.usage for c in chunks if c.usage)
    assert _text(chunks) == "hi"
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (120, 30, 150)
    assert str(transport.requests[0].url) == "https://gw.test/v1/responses"


@pytest.mark.parametrize(
    ("kwargs", "header", "value"),
    [
        ({}, "user-agent", "LiveKit Agents/"),
        ({"provider": "azure"}, "x-livekit-inference-provider", "azure"),
        ({"inference_class": "priority"}, "x-livekit-inference-priority", "priority"),
    ],
)
async def test_rest_headers(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict, header: str, value: str
) -> None:
    transport = _GatewayTransport()
    await _drain(_rest_llm(monkeypatch, transport, **kwargs))

    assert transport.requests[0].headers[header].startswith(value)


async def test_token_minted_per_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def _counting_token(api_key: str | None, api_secret: str | None, ttl: float = 600) -> str:
        nonlocal calls
        calls += 1
        return f"token-{calls}"

    monkeypatch.setattr(inference_responses, "create_access_token", _counting_token)
    model = ResponsesLLM(
        "openai/gpt-4.1", base_url="https://gw.test/v1", api_key="k", api_secret=_SECRET
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


@pytest.mark.parametrize(
    ("kwargs", "chat", "update", "field", "expected"),
    [
        ({"extra_kwargs": {"max_output_tokens": 64}}, {}, None, "max_output_tokens", 64),
        (
            {"extra_kwargs": {"max_output_tokens": 64}},
            {},
            {"extra_kwargs": {}},
            "max_output_tokens",
            None,
        ),
        ({}, {}, {"model": "openai/gpt-4.1-mini"}, "model", "openai/gpt-4.1-mini"),
        (
            {"extra_kwargs": {"prompt_cache_options": {"ttl": "30m"}}},
            {},
            None,
            "prompt_cache_options",
            {"ttl": "30m"},
        ),
        (
            {"extra_kwargs": {"parallel_tool_calls": False}},
            {"extra_kwargs": {"parallel_tool_calls": True}},
            None,
            "parallel_tool_calls",
            True,
        ),
        (
            {"extra_kwargs": {"tool_choice": "none"}},
            {
                "tools": [_get_weather],
                "extra_kwargs": {
                    "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
                },
            },
            None,
            "tool_choice",
            {"type": "function", "name": "get_weather"},
        ),
        ({"extra_kwargs": {"tool_choice": "required"}}, {"tools": []}, None, "tool_choice", None),
        (
            {"extra_kwargs": {"tool_choice": "required"}},
            {"tools": [_get_weather]},
            None,
            "tool_choice",
            "required",
        ),
    ],
    ids=[
        "extra-kwargs",
        "update-replaces-extra-kwargs",
        "update-model",
        "prompt-cache-ttl",
        "per-call-parallel-tool-calls",
        "per-call-tool-choice-converted",
        "tool-choice-dropped-without-tools",
        "tool-choice-sent-with-tools",
    ],
)
async def test_request_options(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict,
    chat: dict,
    update: dict | None,
    field: str,
    expected: Any,
) -> None:
    body = await _rest_body(monkeypatch, update=update, chat=chat, **kwargs)

    assert body.get(field) == expected


async def test_previous_response_id_never_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _rest_llm(monkeypatch, transport)
    chat_ctx = _ctx("hi")

    await _drain(model, chat_ctx)
    chat_ctx.add_message(role="assistant", content="hi")
    chat_ctx.add_message(role="user", content="and again")
    await _drain(model, chat_ctx)

    assert len(transport.requests) == 2
    assert all("previous_response_id" not in transport.body(i) for i in (0, 1))


@pytest.mark.parametrize(
    ("model", "sent"), [("openai/gpt-5.6-luna", True), ("openai/gpt-4.1", False)]
)
async def test_cache_breakpoints_sent_only_where_supported(
    monkeypatch: pytest.MonkeyPatch, model: str, sent: bool
) -> None:
    transport = _GatewayTransport()
    chat_ctx = agents_llm.ChatContext()
    chat_ctx.add_message(role="system", content=["static prompt", agents_llm.CacheBreakpoint()])
    chat_ctx.add_message(role="user", content="hi")

    await _drain(_rest_llm(monkeypatch, transport, model), chat_ctx)

    assert ("prompt_cache_breakpoint" in json.dumps(transport.body()["input"])) == sent


async def test_rest_read_timeout_defaults_to_conn_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _rest_llm(monkeypatch, transport)

    await _drain(model, conn_options=APIConnectOptions(max_retry=0, timeout=3.0))

    assert transport.requests[0].extensions["timeout"]["read"] == 3.0


async def test_read_timeout_mid_stream_is_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _TimeoutMidStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield f"event: response.created\ndata: {json.dumps(_FRAMES[0])}\n\n".encode()
            raise httpx.ReadTimeout("read timed out")

    transport = httpx.MockTransport(
        lambda _: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=_TimeoutMidStream()
        )
    )
    model = _rest_llm(monkeypatch, transport)

    with pytest.raises(APITimeoutError):
        await _drain(model, conn_options=APIConnectOptions(max_retry=0))


async def test_rate_limit_logs_quota_headers(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.agents")
    transport = _GatewayTransport(
        httpx.Response(
            429,
            headers={"content-type": "application/json", "X-LiveKit-Inference-RPM-Limit": "10"},
            json={"error": "rate limited"},
        )
    )

    with pytest.raises(APIStatusError) as exc_info:
        await _drain(_rest_llm(monkeypatch, transport), conn_options=APIConnectOptions(max_retry=0))

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

    with pytest.raises(APIStatusError) as exc_info:
        await _drain(
            _rest_llm(monkeypatch, transport),
            conn_options=APIConnectOptions(max_retry=1, retry_interval=0),
        )

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

    with pytest.raises(APIConnectionError, match="after 2 attempts"):
        await _drain(
            _rest_llm(monkeypatch, transport),
            conn_options=APIConnectOptions(max_retry=1, retry_interval=0),
        )

    assert len(transport.requests) == 2


async def test_prewarm_rest_lists_models(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _rest_llm(monkeypatch, transport)

    await model._prewarm_impl()

    assert transport.requests[0].method == "GET"
    assert str(transport.requests[0].url) == "https://gw.test/v1/models"


# ---------------------------------------------------------------------------
# response_format -> text.format
# ---------------------------------------------------------------------------


async def test_response_format_maps_to_text_format(monkeypatch: pytest.MonkeyPatch) -> None:
    body = await _rest_body(
        monkeypatch, chat={"response_format": _Answer}, extra_kwargs={"text": {"verbosity": "low"}}
    )

    fmt = body["text"]["format"]
    assert body["text"]["verbosity"] == "low"
    assert (fmt["type"], fmt["name"], fmt["strict"]) == ("json_schema", "_Answer", True)
    assert set(fmt["schema"]["properties"]) == {"city", "temperature"}
    assert "json_schema" not in fmt


_SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}}


@pytest.mark.parametrize(
    ("response_format", "expected"),
    [
        (
            {
                "type": "json_schema",
                "json_schema": {"name": "Answer", "schema": _SCHEMA, "strict": True},
            },
            {"type": "json_schema", "name": "Answer", "schema": _SCHEMA, "strict": True},
        ),
        (
            {"type": "json_schema", "name": "Answer", "schema": {"type": "object"}},
            {"type": "json_schema", "name": "Answer", "schema": {"type": "object"}},
        ),
        ({"type": "json_object"}, {"type": "json_object"}),
    ],
    ids=["chat-completions-shape", "responses-shape", "json-object"],
)
async def test_response_format_dict(
    monkeypatch: pytest.MonkeyPatch, response_format: dict, expected: dict
) -> None:
    body = await _rest_body(monkeypatch, chat={"response_format": response_format})

    assert body["text"]["format"] == expected


@pytest.mark.parametrize(
    ("response_format", "match"),
    [
        ({"type": "json_schema", "name": "Answer"}, "name and a schema"),
        ({"type": "xml"}, "unsupported response_format type"),
    ],
)
async def test_invalid_response_format_dict_raises(
    monkeypatch: pytest.MonkeyPatch, response_format: dict, match: str
) -> None:
    model = _rest_llm(monkeypatch, _GatewayTransport())

    with pytest.raises(TypeError, match=match):
        model.chat(chat_ctx=_ctx(), response_format=response_format)


# ---------------------------------------------------------------------------
# reasoning: OpenAI models default to their lowest effort
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "kwargs", "update", "chat", "expected"),
    [
        ("openai/gpt-4.1", {}, None, None, None),
        ("openai/gpt-4.1", {"reasoning": "low"}, None, None, {"effort": "low"}),
        (
            "openai/gpt-4.1",
            {"reasoning": {"effort": "low", "summary": "auto"}},
            None,
            None,
            {"effort": "low", "summary": "auto"},
        ),
        (
            "openai/gpt-4.1",
            {"reasoning": Reasoning(effort="low", summary="auto")},
            None,
            None,
            {"effort": "low", "summary": "auto"},
        ),
        (
            "openai/gpt-4.1",
            {"reasoning": "high", "extra_kwargs": {"reasoning": {"effort": "low"}}},
            None,
            None,
            {"effort": "high"},
        ),
        (
            "openai/gpt-4.1",
            {"extra_kwargs": {"reasoning": {"effort": "low"}}},
            None,
            None,
            {"effort": "low"},
        ),
        (
            "openai/gpt-4.1",
            {"reasoning": "low"},
            None,
            {"extra_kwargs": {"reasoning": {"effort": "high"}}},
            {"effort": "high"},
        ),
        ("openai/gpt-4.1", {"reasoning": "low"}, {"reasoning": "high"}, None, {"effort": "high"}),
        ("openai/gpt-4.1", {"reasoning": "low"}, {"reasoning": None}, None, None),
        (
            "openai/gpt-4.1",
            {"reasoning": "low"},
            {"extra_kwargs": {"max_output_tokens": 64}},
            None,
            {"effort": "low"},
        ),
        ("openai/gpt-6-luna", {}, None, None, {"effort": "none"}),
        ("openai/gpt-6.1-sol", {}, None, None, {"effort": "low"}),
        ("openai/gpt-5-mini", {}, None, None, {"effort": "minimal"}),
        ("google/gemini-3-flash", {}, None, None, None),
        ("openai/gpt-6-luna", {"reasoning": None}, None, None, None),
        ("openai/gpt-6-luna", {}, {"reasoning": None}, None, None),
        (
            "openai/gpt-6-luna",
            {"extra_kwargs": {"reasoning": {"effort": "high", "summary": "auto"}}},
            None,
            None,
            {"effort": "high", "summary": "auto"},
        ),
        (
            "openai/gpt-6-luna",
            {},
            None,
            {"extra_kwargs": {"reasoning": {"effort": "high"}}},
            {"effort": "high"},
        ),
        ("openai/gpt-6-luna", {}, {"model": "openai/gpt-6.1-sol"}, None, {"effort": "low"}),
    ],
)
async def test_reasoning_on_the_wire(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    kwargs: dict,
    update: dict | None,
    chat: dict | None,
    expected: dict | None,
) -> None:
    body = await _rest_body(monkeypatch, model, update=update, chat=chat, **kwargs)

    assert body.get("reasoning") == expected


async def test_reasoning_update_keeps_extra_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    body = await _rest_body(
        monkeypatch, extra_kwargs={"max_output_tokens": 64}, update={"reasoning": "high"}
    )

    assert body["max_output_tokens"] == 64


async def test_default_reasoning_keeps_temperature_at_effort_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # gpt-6 accepts sampling params again at effort "none", so the default lets them through
    body = await _rest_body(monkeypatch, "openai/gpt-6-luna", extra_kwargs={"temperature": 0.3})

    assert body["temperature"] == 0.3


@pytest.mark.parametrize(
    ("model", "kwargs", "update", "warning"),
    [
        (
            "openai/gpt-4.1",
            {"reasoning": "high", "extra_kwargs": {"reasoning": {"effort": "low"}}},
            None,
            "extra_kwargs['reasoning']",
        ),
        (
            "openai/gpt-4.1",
            {"reasoning": "low", "extra_kwargs": {"reasoning": Reasoning(effort="low")}},
            None,
            None,
        ),
        ("openai/gpt-6.1-sol", {"reasoning": "none"}, None, "lowest"),
        ("openai/gpt-6-luna", {"reasoning": "none"}, "openai/gpt-6.1-sol", "lowest"),
    ],
)
async def test_reasoning_warnings(
    caplog: pytest.LogCaptureFixture,
    model: str,
    kwargs: dict,
    update: str | None,
    warning: str | None,
) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.agents")

    llm_model = ResponsesLLM(model, api_key="k", api_secret=_SECRET, **kwargs)
    if update:
        llm_model.update_options(model=update)
    await llm_model.aclose()

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    if warning is None:
        assert warnings == []
    else:
        assert any(warning in m for m in warnings)


# ---------------------------------------------------------------------------
# websocket
# ---------------------------------------------------------------------------


class _WsMsg:
    def __init__(self, data: str) -> None:
        self.type = aiohttp.WSMsgType.TEXT
        self.data = data
        self.extra = None


class _ScriptedWs:
    def __init__(self, frames: list[dict] | None = None) -> None:
        self.sent: list[str] = []
        self.closed = False
        self._frames = list(frames or [])

    async def send_str(self, data: str) -> None:
        self.sent.append(data)

    async def receive(self, timeout: float | None = None) -> _WsMsg:
        return _WsMsg(json.dumps(self._frames.pop(0)))

    async def close(self) -> None:
        self.closed = True


class _Session:
    """Answers each dial with a socket that completes one turn; can fail the first dial."""

    def __init__(self, fail_first: Callable[[], None] | None = None) -> None:
        self.urls: list[str] = []
        self.headers: list[dict[str, str]] = []
        self.heartbeats: list[float] = []
        self.sockets: list[_ScriptedWs] = []
        self._fail_first = fail_first

    async def ws_connect(
        self, url: str, *, headers: dict[str, str], heartbeat: float
    ) -> _ScriptedWs:
        self.urls.append(url)
        self.headers.append(headers)
        self.heartbeats.append(heartbeat)
        if self._fail_first is not None:
            fail, self._fail_first = self._fail_first, None
            fail()
            raise aiohttp.ClientConnectionError("dial failed")
        sock = _ScriptedWs(_FRAMES)
        self.sockets.append(sock)
        return sock


class _RefusingSession:
    def __init__(self, status: int, headers: dict[str, str] | None = None) -> None:
        self._status = status
        self._headers = headers or {}

    async def ws_connect(self, url: str, *, headers: dict[str, str], heartbeat: float) -> None:
        request_url = URL(url)
        request_info = aiohttp.RequestInfo(
            url=request_url,
            method="GET",
            headers=CIMultiDictProxy(CIMultiDict()),
            real_url=request_url,
        )
        raise aiohttp.WSServerHandshakeError(
            request_info,
            (),
            status=self._status,
            message="Invalid response status",
            headers=CIMultiDict(self._headers),
        )


def _ws_llm(**kwargs: Any) -> ResponsesLLM:
    return ResponsesLLM(
        "openai/gpt-4.1",
        base_url="https://gw.test/v1",
        api_key="k",
        api_secret=_SECRET,
        use_websocket=True,
        **kwargs,
    )


def _transport(model: ResponsesLLM) -> inference_responses._GatewayResponsesWebsocket:
    transport = model._ws
    assert isinstance(transport, inference_responses._GatewayResponsesWebsocket)
    return transport


def _with_session(model: ResponsesLLM, session: Any = None) -> Any:
    _transport(model)._session = session = session or _Session()
    return session


def _use_connection(model: ResponsesLLM, make_ws: Callable[[], _ScriptedWs]) -> list[float]:
    """Serves every pool dial from `make_ws`; returns the connect timeouts it was given."""
    dials: list[float] = []

    async def _connect(timeout: float) -> _ScriptedWs:
        dials.append(timeout)
        return make_ws()

    _transport(model)._pool._connect_cb = _connect  # type: ignore[assignment]
    return dials


async def _consume(stream: agents_llm.LLMStream) -> None:
    async with stream:
        async for _ in stream:
            pass


async def test_ws_dial_url_and_fresh_token(monkeypatch: pytest.MonkeyPatch) -> None:
    tokens = iter(("token-1", "token-2"))
    monkeypatch.setattr(inference_responses, "create_access_token", lambda *a, **k: next(tokens))
    model = _ws_llm()
    session = _with_session(model)

    await _transport(model)._create_ws(1.0)
    await _transport(model)._create_ws(1.0)

    assert session.urls[0] == str(
        URL("wss://gw.test/v1/responses").update_query(model="openai/gpt-4.1")
    )
    assert session.heartbeats[0] == 30.0
    assert [h["Authorization"] for h in session.headers] == ["Bearer token-1", "Bearer token-2"]
    assert session.headers[0]["User-Agent"].startswith("LiveKit Agents/")


@pytest.mark.parametrize(
    ("kwargs", "header", "value"),
    [
        ({"provider": "azure"}, "X-LiveKit-Inference-Provider", "azure"),
        ({"inference_class": "priority"}, "X-LiveKit-Inference-Priority", "priority"),
        ({"extra_kwargs": {"extra_headers": {"X-Deployment": "blue"}}}, "X-Deployment", "blue"),
    ],
)
async def test_ws_dial_headers(kwargs: dict, header: str, value: str) -> None:
    model = _ws_llm(**kwargs)
    session = _with_session(model)

    await _transport(model)._create_ws(1.0)

    assert session.headers[0][header] == value
    await model.aclose()


async def test_ws_per_call_inference_class_warns(caplog: pytest.LogCaptureFixture) -> None:
    model = _ws_llm()
    session = _with_session(model)

    with caplog.at_level(logging.WARNING):
        stream = model.chat(chat_ctx=_ctx(), inference_class="priority")
    await stream.aclose()
    await _transport(model)._create_ws(1.0)

    records = [r for r in caplog.records if "cannot change per request" in r.getMessage()]
    assert len(records) == 1
    assert records[0].requested == "priority"  # type: ignore[attr-defined]
    assert records[0].configured is None  # type: ignore[attr-defined]
    assert "X-LiveKit-Inference-Priority" not in session.headers[0]
    await model.aclose()


async def test_ws_per_call_extra_headers_warns(caplog: pytest.LogCaptureFixture) -> None:
    model = _ws_llm()

    with caplog.at_level(logging.WARNING):
        stream = model.chat(chat_ctx=_ctx(), extra_kwargs={"extra_headers": {"X-Deployment": "b"}})
    await stream.aclose()

    assert len([r for r in caplog.records if "extra_headers cannot change" in r.getMessage()]) == 1
    await model.aclose()


async def test_ws_payload_carries_options_but_no_extra_headers() -> None:
    model = _ws_llm(provider="azure", inference_class="priority", reasoning="low")
    ws = _ScriptedWs([_CONNECTED, *_FRAMES])
    _use_connection(model, lambda: ws)

    await _drain(model, conn_options=APIConnectOptions(max_retry=0))

    payload = json.loads(ws.sent[0])
    assert payload["type"] == "response.create"
    assert payload["reasoning"] == {"effort": "low"}
    assert "extra_headers" not in payload
    await model.aclose()


@pytest.mark.parametrize(
    "extra",
    [[], [{"type": "gateway.session_expiring", "seconds_until_close": 300}]],
    ids=["connected", "session-expiring"],
)
async def test_gateway_control_frames_are_ignored(extra: list[dict]) -> None:
    model = _ws_llm()
    created, *rest = _FRAMES
    frames = [_CONNECTED, *extra, created, {"type": "gateway.keepalive"}, *rest]
    _use_connection(model, lambda: _ScriptedWs(frames))

    assert _text(await _drain(model, conn_options=APIConnectOptions(max_retry=0))) == "hi"
    await model.aclose()


async def test_ws_dial_uses_request_connect_timeout() -> None:
    model = _ws_llm()
    dials = _use_connection(model, lambda: _ScriptedWs([_CONNECTED, *_FRAMES]))

    await _drain(model, conn_options=APIConnectOptions(max_retry=0, timeout=1.5))

    assert dials == [1.5]
    await model.aclose()


@pytest.mark.parametrize(
    ("frames", "exc_type", "status", "retryable"),
    [
        (
            [_CONNECTED, _gateway_error("tool_not_allowed", "hosted tool not allowed")],
            APIStatusError,
            400,
            False,
        ),
        ([_gateway_error("upstream_connect_failed")], APIStatusError, 502, True),
        ([_CONNECTED, _gateway_error("upstream_timeout")], APITimeoutError, None, True),
    ],
    ids=["tool-not-allowed", "upstream-connect-failed", "upstream-timeout"],
)
async def test_gateway_errors(
    frames: list[dict], exc_type: type[Exception], status: int | None, retryable: bool
) -> None:
    model = _ws_llm()
    _use_connection(model, lambda: _ScriptedWs(frames))

    with pytest.raises(exc_type) as exc_info:
        await _drain(model, conn_options=APIConnectOptions(max_retry=0))

    assert getattr(exc_info.value, "status_code", None) == status
    assert exc_info.value.retryable is retryable  # type: ignore[attr-defined]
    await model.aclose()


async def test_gateway_upstream_timeout_after_text_is_not_retried() -> None:
    model = _ws_llm()
    created, delta, _ = _FRAMES
    dials = _use_connection(
        model,
        lambda: _ScriptedWs([_CONNECTED, created, delta, _gateway_error("upstream_timeout")]),
    )

    text = ""
    with pytest.raises(APITimeoutError) as exc_info:
        async with model.chat(chat_ctx=_ctx(), conn_options=APIConnectOptions(max_retry=1)) as s:
            async for chunk in s:
                text += chunk.delta.content or "" if chunk.delta else ""

    assert text == "hi"
    assert exc_info.value.retryable is False
    assert len(dials) == 1
    await model.aclose()


async def test_gateway_terminal_quota_exceeded_is_not_retried() -> None:
    model = _ws_llm()
    denied = _gateway_error(
        "quota_exceeded", "credit quota exhausted", category="MaxGatewayCredits", should_retry=False
    )
    dials = _use_connection(model, lambda: _ScriptedWs([_CONNECTED, denied]))

    with pytest.raises(APIStatusError) as exc_info:
        await _drain(model, conn_options=APIConnectOptions(max_retry=1, retry_interval=0))

    assert exc_info.value.status_code == 429
    assert exc_info.value.retryable is False
    assert "MaxGatewayCredits" in exc_info.value.message
    assert len(dials) == 1
    await model.aclose()


@pytest.mark.virtual_time
async def test_gateway_quota_exceeded_without_should_retry_is_retryable() -> None:
    model = _ws_llm()
    dials = _use_connection(
        model, lambda: _ScriptedWs([_CONNECTED, _gateway_error("quota_exceeded")])
    )

    with pytest.raises(APIConnectionError, match="after 2 attempts"):
        await _drain(model, conn_options=APIConnectOptions(max_retry=1, retry_interval=0))

    assert len(dials) == 2
    await model.aclose()


@pytest.mark.parametrize(
    ("status", "headers", "retryable"),
    [(401, {}, False), (429, {"X-Should-Retry": "false"}, False)],
)
async def test_handshake_refusal_is_not_retryable(
    status: int, headers: dict, retryable: bool
) -> None:
    model = _ws_llm()
    _with_session(model, _RefusingSession(status, headers))

    with pytest.raises(APIStatusError) as exc_info:
        await _drain(model, conn_options=APIConnectOptions(max_retry=0))

    assert exc_info.value.status_code == status
    assert exc_info.value.retryable is retryable
    await model.aclose()


async def test_handshake_429_logs_quota_headers(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.agents")
    model = _ws_llm()
    _with_session(model, _RefusingSession(429, {"X-LiveKit-Inference-RPM-Limit": "10"}))

    with pytest.raises(APIStatusError) as exc_info:
        await _transport(model)._create_ws(1.0)

    records = [
        r
        for r in caplog.records
        if r.getMessage() == "LLM request rate limited by inference gateway"
    ]
    assert exc_info.value.status_code == 429
    assert exc_info.value.retryable is True
    assert len(records) == 1
    assert records[0].rpm_limit == "10"  # type: ignore[attr-defined]
    assert records[0].model == "openai/gpt-4.1"  # type: ignore[attr-defined]
    await model.aclose()


@pytest.mark.virtual_time
async def test_ws_receive_timeout_raises_api_timeout_and_drops_socket() -> None:
    class _SilentWs(_ScriptedWs):
        async def receive(self, timeout: float | None = None) -> _WsMsg:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    model = _ws_llm()
    ws = _SilentWs()
    _use_connection(model, lambda: ws)

    with pytest.raises(APITimeoutError):
        await _drain(model, conn_options=APIConnectOptions(max_retry=0, timeout=0.05))

    assert ws not in _transport(model)._pool._available
    assert ws not in _transport(model)._pool._connections
    await model.aclose()


@pytest.mark.virtual_time
async def test_ws_pongs_do_not_reset_idle_timeout() -> None:
    class _PongingWs(_ScriptedWs):
        pongs = 0

        async def receive(self, timeout: float | None = None) -> _WsMsg:
            # aiohttp consumes pongs inside receive() and restarts its own timer on each
            while True:
                await asyncio.sleep(0.01)
                self.pongs += 1

    model = _ws_llm()
    ws = _PongingWs()
    _use_connection(model, lambda: ws)

    with pytest.raises(APITimeoutError):
        await _drain(model, conn_options=APIConnectOptions(max_retry=0, timeout=0.1))

    assert ws.pongs > 0
    assert ws not in _transport(model)._pool._available
    await model.aclose()


@pytest.mark.virtual_time
async def test_ws_text_frame_resets_idle_timeout() -> None:
    class _SlowWs(_ScriptedWs):
        async def receive(self, timeout: float | None = None) -> _WsMsg:
            # each frame lands inside the window, though together they outlast it
            await asyncio.sleep(0.08)
            return await super().receive(timeout)

    model = _ws_llm()
    _use_connection(model, lambda: _SlowWs([_CONNECTED, *_FRAMES]))

    chunks = await _drain(model, conn_options=APIConnectOptions(max_retry=0, timeout=0.2))

    assert _text(chunks) == "hi"
    await model.aclose()


@pytest.mark.virtual_time
async def test_ws_consumer_time_does_not_count_toward_idle_timeout() -> None:
    model = _ws_llm()
    _use_connection(model, lambda: _ScriptedWs([_CONNECTED, *_FRAMES]))
    seen: list[str] = []

    async for event in _transport(model).generate_response(
        {"type": "response.create"}, receive_timeout=0.1
    ):
        seen.append(event["type"])
        await asyncio.sleep(0.15)

    assert seen[-1] == "response.completed"
    await model.aclose()


async def test_gateway_error_socket_leaves_pool() -> None:
    model = _ws_llm()
    transport = _transport(model)
    ws = _ScriptedWs([_CONNECTED, _gateway_error("invalid_event", "unsupported event type")])
    _use_connection(model, lambda: ws)

    # the stream stops reading at the error and drops the generator; close it the same way
    events = transport.generate_response({"type": "response.create"})
    assert (await events.__anext__())["type"] == "gateway.connected"
    frame = await events.__anext__()
    parsed = inference_responses._GatewayResponsesLLMStream._parse_ws_event(object(), frame)  # type: ignore[arg-type]
    await events.aclose()

    assert parsed is not None and parsed.type == "error"
    assert ws not in transport._pool._available
    assert ws not in transport._pool._connections
    await model.aclose()


async def test_ws_mode_pool_and_session_defaults() -> None:
    model = _ws_llm()
    engine = ResponsesWebsocket(api_key="k", timeout=1.0, model="gpt-4.1")

    assert model._client is None
    assert _transport(model)._session is None
    assert _transport(model)._pool._max_session_duration == 3000
    assert engine._pool._max_session_duration == 3600
    await model.aclose()
    await engine.aclose()


async def test_prewarm_ws_prewarms_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _ws_llm()
    calls: list[bool] = []
    monkeypatch.setattr(_transport(model)._pool, "prewarm", lambda: calls.append(True))

    await model._prewarm_impl()

    assert calls == [True]


# ---------------------------------------------------------------------------
# websocket: option updates and model swaps
# ---------------------------------------------------------------------------


async def test_update_model_builds_new_ws_transport_and_closes_the_idle_one() -> None:
    model = _ws_llm()
    transport = _transport(model)
    sock = _ScriptedWs()
    _use_connection(model, lambda: sock)
    transport._pool.put(await transport._pool.get(timeout=1.0))

    model.update_options(model="openai/gpt-4.1-mini")
    await asyncio.gather(*model._ws_close_tasks)

    assert model._ws is not None and model._ws is not transport
    assert model._ws._base_url == str(
        URL("wss://gw.test/v1/responses").update_query(model="openai/gpt-4.1-mini")
    )
    assert sock.closed is True
    assert transport not in model._retired_ws


async def test_stream_created_before_model_swap_dials_its_model() -> None:
    model = _ws_llm()
    session = _with_session(model)

    stream = model.chat(chat_ctx=_ctx(), conn_options=APIConnectOptions(max_retry=0))
    model.update_options(model="openai/gpt-4.1-mini")
    await _consume(stream)

    assert URL(session.urls[0]).query["model"] == "openai/gpt-4.1"
    assert json.loads(session.sockets[0].sent[0])["model"] == "openai/gpt-4.1"


@pytest.mark.virtual_time
async def test_retry_after_model_swap_dials_its_model() -> None:
    model = _ws_llm()
    session = _with_session(
        model, _Session(fail_first=lambda: model.update_options(model="openai/gpt-4.1-mini"))
    )

    await _consume(
        model.chat(chat_ctx=_ctx(), conn_options=APIConnectOptions(max_retry=1, retry_interval=0))
    )

    assert [URL(u).query["model"] for u in session.urls] == ["openai/gpt-4.1", "openai/gpt-4.1"]
    assert json.loads(session.sockets[0].sent[0])["model"] == "openai/gpt-4.1"


async def test_retired_ws_transport_closes_after_its_last_stream() -> None:
    model = _ws_llm()
    transport = _transport(model)
    session = _with_session(model)

    stream = model.chat(chat_ctx=_ctx(), conn_options=APIConnectOptions(max_retry=0))
    model.update_options(model="openai/gpt-4.1-mini")
    assert transport in model._retired_ws

    await _consume(stream)
    await asyncio.sleep(0)
    await asyncio.gather(*model._ws_close_tasks)

    assert transport not in model._retired_ws
    assert session.sockets[0].closed is True


async def test_aclose_survives_retired_stream_finishing_mid_close() -> None:
    model = _ws_llm()
    first = _transport(model)
    first._holders = 1
    model.update_options(model="openai/gpt-4.1-mini")
    second = _transport(model)
    second._holders = 1
    model.update_options(model="openai/gpt-4.1-nano")
    closes: list[inference_responses._GatewayResponsesWebsocket] = []

    def _closing_releases(
        ws: inference_responses._GatewayResponsesWebsocket,
    ) -> Callable[[], Awaitable[None]]:
        async def _aclose() -> None:
            closes.append(ws)
            model._release_ws(ws)  # its last stream finishes during shutdown

        return _aclose

    first.aclose = _closing_releases(first)  # type: ignore[method-assign]
    second.aclose = _closing_releases(second)  # type: ignore[method-assign]

    await model.aclose()

    assert sorted(map(id, closes)) == sorted([id(first), id(second)])
    assert model._retired_ws == set()


@pytest.mark.parametrize(
    ("update", "invalidated"),
    [
        ({"model": "openai/gpt-4.1"}, False),
        ({"extra_kwargs": {"max_output_tokens": 64}}, False),
        ({"extra_kwargs": {"extra_headers": {"X-Deployment": "blue"}}}, True),
    ],
)
async def test_update_invalidates_ws_pool_only_on_header_change(
    monkeypatch: pytest.MonkeyPatch, update: dict, invalidated: bool
) -> None:
    model = _ws_llm()
    calls: list[bool] = []
    monkeypatch.setattr(_transport(model)._pool, "invalidate", lambda: calls.append(True))

    model.update_options(**update)

    assert calls == ([True] if invalidated else [])


@pytest.mark.parametrize("mutate", [False, True], ids=["reassigned", "mutated"])
async def test_update_with_changed_caller_headers_invalidates_ws_pool(
    monkeypatch: pytest.MonkeyPatch, mutate: bool
) -> None:
    kw: dict[str, Any] = {"extra_headers": {"X-A": "1"}}
    model = _ws_llm()
    model.update_options(extra_kwargs=kw)
    calls: list[bool] = []
    monkeypatch.setattr(_transport(model)._pool, "invalidate", lambda: calls.append(True))

    if mutate:
        kw["extra_headers"]["X-A"] = "2"
    else:
        kw["extra_headers"] = {"X-A": "2"}
    model.update_options(extra_kwargs=kw)

    assert calls == [True]


@pytest.mark.parametrize(
    ("prewarm", "close_first", "prewarms_new_pool"),
    [(True, False, True), (True, True, False), (False, False, False)],
    ids=["prewarmed", "after-aclose", "never-prewarmed"],
)
async def test_model_swap_prewarms_new_pool_only_when_prewarmed(
    monkeypatch: pytest.MonkeyPatch, prewarm: bool, close_first: bool, prewarms_new_pool: bool
) -> None:
    model = _ws_llm()
    prewarmed: list[object] = []
    monkeypatch.setattr(utils.ConnectionPool, "prewarm", lambda pool: prewarmed.append(pool))
    if prewarm:
        model.prewarm()
        await asyncio.sleep(0)
    if close_first:
        await model.aclose()
    prewarmed.clear()

    model.update_options(model="openai/gpt-4.1-mini")

    assert prewarmed == ([_transport(model)._pool] if prewarms_new_pool else [])
    await model.aclose()
