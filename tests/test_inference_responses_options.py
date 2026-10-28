from __future__ import annotations

import json
from typing import Any

import httpx2 as httpx
import openai
import pytest
from pydantic import BaseModel
from yarl import URL

from livekit.agents import llm as agents_llm
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
_FRAMES = [
    {"type": "response.created", "sequence_number": 0, "response": _RESPONSE},
    {
        "type": "response.completed",
        "sequence_number": 1,
        "response": {**_RESPONSE, "status": "completed"},
    },
]


class _Answer(BaseModel):
    city: str
    temperature: float


def _sse(frames: list[dict]) -> bytes:
    return "".join(f"event: {f['type']}\ndata: {json.dumps(f)}\n\n" for f in frames).encode()


class _GatewayTransport(httpx.AsyncBaseTransport):
    def __init__(self, models: bool = False) -> None:
        self.requests: list[httpx.Request] = []
        self._models = models

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        self.requests.append(request)
        if self._models:
            return httpx.Response(200, json={"object": "list", "data": []}, request=request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(_FRAMES),
            request=request,
        )

    def body(self, index: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[index].content)


def _make_rest_llm(
    monkeypatch: pytest.MonkeyPatch, transport: _GatewayTransport, **kwargs: Any
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
        "openai/gpt-4.1", base_url="https://gw.test/v1", api_key="k", api_secret="s", **kwargs
    )


def _make_ws_llm() -> ResponsesLLM:
    return ResponsesLLM(
        "openai/gpt-4.1",
        base_url="https://gw.test/v1",
        api_key="k",
        api_secret="s",
        use_websocket=True,
    )


def _ctx() -> agents_llm.ChatContext:
    chat_ctx = agents_llm.ChatContext()
    chat_ctx.add_message(role="user", content="hi")
    return chat_ctx


async def _drain(model: ResponsesLLM, **kwargs: Any) -> None:
    stream = model.chat(chat_ctx=_ctx(), **kwargs)
    try:
        async for _ in stream:
            pass
    finally:
        await stream.aclose()


class _FakeSocket:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


async def test_update_model_rebuilds_ws_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_ws_llm()
    transport = model._ws
    assert transport is not None
    calls: list[bool] = []
    monkeypatch.setattr(transport._pool, "invalidate", lambda: calls.append(True))

    model.update_options(model="openai/gpt-4.1-mini")

    assert model._ws is transport
    expected = str(URL("wss://gw.test/v1/responses").update_query(model="openai/gpt-4.1-mini"))
    assert transport._base_url == expected
    assert calls == [True]


async def test_update_model_keeps_in_flight_ws_socket() -> None:
    model = _make_ws_llm()
    transport = model._ws
    assert transport is not None
    sock = _FakeSocket()

    async def _connect(_timeout: float) -> _FakeSocket:
        return sock

    transport._pool._connect_cb = _connect  # type: ignore[assignment]
    checked_out = await transport._pool.get(timeout=1.0)

    model.update_options(model="openai/gpt-4.1-mini")

    assert checked_out is sock
    assert sock.closed is False
    assert sock in transport._pool._retired


async def test_update_same_model_keeps_ws_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_ws_llm()
    assert model._ws is not None
    calls: list[bool] = []
    monkeypatch.setattr(model._ws._pool, "invalidate", lambda: calls.append(True))

    model.update_options(model="openai/gpt-4.1")

    assert calls == []


async def test_update_model_rest_uses_new_model(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _make_rest_llm(monkeypatch, transport)

    model.update_options(model="openai/gpt-4.1-mini")
    await _drain(model)

    assert transport.body()["model"] == "openai/gpt-4.1-mini"


async def test_update_extra_kwargs_replaces(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _make_rest_llm(monkeypatch, transport, extra_kwargs={"max_output_tokens": 64})

    model.update_options(extra_kwargs={})
    await _drain(model)

    assert "max_output_tokens" not in transport.body()


async def test_response_format_maps_to_text_format(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _make_rest_llm(monkeypatch, transport)

    await _drain(model, response_format=_Answer)

    fmt = transport.body()["text"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["name"] == "_Answer"
    assert fmt["strict"] is True
    assert set(fmt["schema"]["properties"]) == {"city", "temperature"}
    assert "json_schema" not in fmt


async def test_response_format_keeps_other_text_options(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _make_rest_llm(monkeypatch, transport, extra_kwargs={"text": {"verbosity": "low"}})

    await _drain(model, response_format=_Answer)

    text = transport.body()["text"]
    assert text["verbosity"] == "low"
    assert text["format"]["type"] == "json_schema"


async def test_prewarm_ws_prewarms_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_ws_llm()
    assert model._ws is not None
    calls: list[bool] = []
    monkeypatch.setattr(model._ws._pool, "prewarm", lambda: calls.append(True))

    await model._prewarm_impl()

    assert calls == [True]


async def test_prewarm_rest_lists_models(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport(models=True)
    model = _make_rest_llm(monkeypatch, transport)

    await model._prewarm_impl()

    assert transport.requests[0].method == "GET"
    assert str(transport.requests[0].url) == "https://gw.test/v1/models"


def test_to_text_format_flattens_chat_shape() -> None:
    fmt = inference_responses._to_text_format(_Answer)

    assert list(fmt) == ["type", "name", "schema", "strict"]
