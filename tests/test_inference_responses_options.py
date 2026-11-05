from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp
import httpx2 as httpx
import openai
import pytest
from openai.types import Reasoning
from pydantic import BaseModel
from yarl import URL

from livekit.agents import llm as agents_llm, utils
from livekit.agents.inference import responses as inference_responses
from livekit.agents.inference.responses import LLM as ResponsesLLM
from livekit.agents.types import APIConnectOptions

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


def _make_ws_llm(**kwargs: Any) -> ResponsesLLM:
    return ResponsesLLM(
        "openai/gpt-4.1",
        base_url="https://gw.test/v1",
        api_key="k",
        api_secret="s",
        use_websocket=True,
        **kwargs,
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


class _WsMsg:
    def __init__(self, data: str) -> None:
        self.type = aiohttp.WSMsgType.TEXT
        self.data = data
        self.extra = None


class _ScriptedSocket(_FakeSocket):
    def __init__(self) -> None:
        super().__init__()
        self.sent: list[str] = []
        self._frames = list(_FRAMES)

    async def send_str(self, data: str) -> None:
        self.sent.append(data)

    async def receive(self, timeout: float | None = None) -> _WsMsg:
        return _WsMsg(json.dumps(self._frames.pop(0)))


class _ScriptedSession:
    """Answers each dial with a socket that completes one turn; can fail the first dial."""

    def __init__(self, fail_first: Callable[[], None] | None = None) -> None:
        self.urls: list[str] = []
        self.sockets: list[_ScriptedSocket] = []
        self._fail_first = fail_first

    async def ws_connect(
        self, url: str, *, headers: dict[str, str], heartbeat: float
    ) -> _ScriptedSocket:
        self.urls.append(url)
        if self._fail_first is not None:
            fail, self._fail_first = self._fail_first, None
            fail()
            raise aiohttp.ClientConnectionError("dial failed")
        sock = _ScriptedSocket()
        self.sockets.append(sock)
        return sock


def _gateway_ws(model: ResponsesLLM) -> inference_responses._GatewayResponsesWebsocket:
    transport = model._ws
    assert isinstance(transport, inference_responses._GatewayResponsesWebsocket)
    return transport


async def _consume(stream: agents_llm.LLMStream) -> None:
    try:
        async for _ in stream:
            pass
    finally:
        await stream.aclose()


async def test_update_model_builds_new_ws_transport() -> None:
    model = _make_ws_llm()
    transport = model._ws

    model.update_options(model="openai/gpt-4.1-mini")

    assert model._ws is not None and model._ws is not transport
    expected = str(URL("wss://gw.test/v1/responses").update_query(model="openai/gpt-4.1-mini"))
    assert model._ws._base_url == expected


async def test_update_model_closes_idle_ws_transport() -> None:
    model = _make_ws_llm()
    transport = _gateway_ws(model)
    sock = _FakeSocket()

    async def _connect(_timeout: float) -> _FakeSocket:
        return sock

    transport._pool._connect_cb = _connect  # type: ignore[assignment]
    transport._pool.put(await transport._pool.get(timeout=1.0))

    model.update_options(model="openai/gpt-4.1-mini")
    await asyncio.gather(*model._ws_close_tasks)

    assert sock.closed is True
    assert transport not in model._retired_ws


async def test_stream_created_before_model_swap_dials_its_model() -> None:
    model = _make_ws_llm()
    session = _ScriptedSession()
    _gateway_ws(model)._session = session  # type: ignore[assignment]

    stream = model.chat(chat_ctx=_ctx(), conn_options=APIConnectOptions(max_retry=0))
    model.update_options(model="openai/gpt-4.1-mini")
    await _consume(stream)

    assert URL(session.urls[0]).query["model"] == "openai/gpt-4.1"
    assert json.loads(session.sockets[0].sent[0])["model"] == "openai/gpt-4.1"


async def test_retry_after_model_swap_dials_its_model() -> None:
    model = _make_ws_llm()
    session = _ScriptedSession(fail_first=lambda: model.update_options(model="openai/gpt-4.1-mini"))
    _gateway_ws(model)._session = session  # type: ignore[assignment]

    stream = model.chat(
        chat_ctx=_ctx(), conn_options=APIConnectOptions(max_retry=1, retry_interval=0)
    )
    await _consume(stream)

    assert [URL(u).query["model"] for u in session.urls] == ["openai/gpt-4.1", "openai/gpt-4.1"]
    assert json.loads(session.sockets[0].sent[0])["model"] == "openai/gpt-4.1"


async def test_retired_ws_transport_closes_after_its_last_stream() -> None:
    model = _make_ws_llm()
    transport = _gateway_ws(model)
    session = _ScriptedSession()
    transport._session = session  # type: ignore[assignment]

    stream = model.chat(chat_ctx=_ctx(), conn_options=APIConnectOptions(max_retry=0))
    model.update_options(model="openai/gpt-4.1-mini")
    assert transport in model._retired_ws

    await _consume(stream)
    await asyncio.sleep(0)
    await asyncio.gather(*model._ws_close_tasks)

    assert transport not in model._retired_ws
    assert session.sockets[0].closed is True


async def test_aclose_survives_retired_stream_finishing_mid_close() -> None:
    model = _make_ws_llm()
    first = _gateway_ws(model)
    first._holders = 1
    model.update_options(model="openai/gpt-4.1-mini")
    second = _gateway_ws(model)
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


async def test_update_same_model_keeps_ws_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_ws_llm()
    assert model._ws is not None
    calls: list[bool] = []
    monkeypatch.setattr(model._ws._pool, "invalidate", lambda: calls.append(True))

    model.update_options(model="openai/gpt-4.1")

    assert calls == []


async def test_update_extra_headers_invalidates_ws_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_ws_llm()
    assert model._ws is not None
    calls: list[bool] = []
    monkeypatch.setattr(model._ws._pool, "invalidate", lambda: calls.append(True))

    model.update_options(extra_kwargs={"extra_headers": {"X-Deployment": "blue"}})

    assert calls == [True]


async def test_update_extra_kwargs_without_header_change_keeps_ws_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _make_ws_llm()
    assert model._ws is not None
    calls: list[bool] = []
    monkeypatch.setattr(model._ws._pool, "invalidate", lambda: calls.append(True))

    model.update_options(extra_kwargs={"max_output_tokens": 64})

    assert calls == []


async def test_update_with_reassigned_caller_headers_invalidates_ws_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kw: dict[str, Any] = {"extra_headers": {"X-A": "1"}}
    model = ResponsesLLM(
        "openai/gpt-4.1",
        base_url="https://gw.test/v1",
        api_key="k",
        api_secret="s",
        extra_kwargs=kw,
        use_websocket=True,
    )
    assert model._ws is not None
    calls: list[bool] = []
    monkeypatch.setattr(model._ws._pool, "invalidate", lambda: calls.append(True))

    kw["extra_headers"] = {"X-A": "2"}
    model.update_options(extra_kwargs=kw)

    assert calls == [True]


async def test_update_with_mutated_caller_headers_invalidates_ws_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kw: dict[str, Any] = {"extra_headers": {"X-A": "1"}}
    model = _make_ws_llm()
    model.update_options(extra_kwargs=kw)
    assert model._ws is not None
    calls: list[bool] = []
    monkeypatch.setattr(model._ws._pool, "invalidate", lambda: calls.append(True))

    kw["extra_headers"]["X-A"] = "2"
    model.update_options(extra_kwargs=kw)

    assert calls == [True]


async def test_model_swap_prewarms_new_pool_when_prewarmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _make_ws_llm()
    prewarmed: list[object] = []
    monkeypatch.setattr(utils.ConnectionPool, "prewarm", lambda pool: prewarmed.append(pool))
    model.prewarm()
    await asyncio.sleep(0)

    model.update_options(model="openai/gpt-4.1-mini")

    assert model._ws is not None and prewarmed[-1] is model._ws._pool
    await model.aclose()


async def test_model_swap_after_aclose_skips_prewarm(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_ws_llm()
    prewarmed: list[object] = []
    monkeypatch.setattr(utils.ConnectionPool, "prewarm", lambda pool: prewarmed.append(pool))
    model.prewarm()
    await asyncio.sleep(0)
    await model.aclose()
    prewarmed.clear()

    model.update_options(model="openai/gpt-4.1-mini")

    assert prewarmed == []
    await model.aclose()


async def test_model_swap_skips_prewarm_when_never_prewarmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _make_ws_llm()
    prewarmed: list[object] = []
    monkeypatch.setattr(utils.ConnectionPool, "prewarm", lambda pool: prewarmed.append(pool))

    model.update_options(model="openai/gpt-4.1-mini")

    assert prewarmed == []
    await model.aclose()


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


async def test_response_format_chat_completions_dict_maps_to_text_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _GatewayTransport()
    model = _make_rest_llm(monkeypatch, transport)
    schema = {"type": "object", "properties": {"city": {"type": "string"}}}

    await _drain(
        model,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "Answer", "schema": schema, "strict": True},
        },
    )

    assert transport.body()["text"]["format"] == {
        "type": "json_schema",
        "name": "Answer",
        "schema": schema,
        "strict": True,
    }


async def test_response_format_responses_dict_passes_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _GatewayTransport()
    model = _make_rest_llm(monkeypatch, transport)
    fmt = {"type": "json_schema", "name": "Answer", "schema": {"type": "object"}}

    await _drain(model, response_format=fmt)

    assert transport.body()["text"]["format"] == fmt


async def test_response_format_json_object_dict_passes_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _GatewayTransport()
    model = _make_rest_llm(monkeypatch, transport)

    await _drain(model, response_format={"type": "json_object"})

    assert transport.body()["text"]["format"] == {"type": "json_object"}


async def test_response_format_json_schema_dict_without_schema_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _make_rest_llm(monkeypatch, _GatewayTransport())

    with pytest.raises(TypeError, match="name and a schema"):
        model.chat(chat_ctx=_ctx(), response_format={"type": "json_schema", "name": "Answer"})


async def test_response_format_dict_with_unknown_type_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _make_rest_llm(monkeypatch, _GatewayTransport())

    with pytest.raises(TypeError, match="unsupported response_format type"):
        model.chat(chat_ctx=_ctx(), response_format={"type": "xml"})


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


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


async def test_no_reasoning_sends_no_reasoning_field(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _make_rest_llm(monkeypatch, transport)

    await _drain(model)

    assert "reasoning" not in transport.body()


async def test_reasoning_string_sends_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _make_rest_llm(monkeypatch, transport, reasoning="low")

    await _drain(model)

    assert transport.body()["reasoning"] == {"effort": "low"}


async def test_reasoning_dict_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _make_rest_llm(monkeypatch, transport, reasoning={"effort": "low", "summary": "auto"})

    await _drain(model)

    assert transport.body()["reasoning"] == {"effort": "low", "summary": "auto"}


async def test_reasoning_object_sends_only_its_set_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _GatewayTransport()
    model = _make_rest_llm(
        monkeypatch, transport, reasoning=Reasoning(effort="low", summary="auto")
    )

    await _drain(model)

    assert transport.body()["reasoning"] == {"effort": "low", "summary": "auto"}


async def test_reasoning_overrides_extra_kwargs_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _GatewayTransport()
    model = _make_rest_llm(
        monkeypatch, transport, reasoning="high", extra_kwargs={"reasoning": {"effort": "low"}}
    )

    await _drain(model)

    assert transport.body()["reasoning"] == {"effort": "high"}


async def test_extra_kwargs_reasoning_still_sent_without_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _GatewayTransport()
    model = _make_rest_llm(monkeypatch, transport, extra_kwargs={"reasoning": {"effort": "low"}})

    await _drain(model)

    assert transport.body()["reasoning"] == {"effort": "low"}


async def test_per_call_extra_kwargs_reasoning_overrides_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _GatewayTransport()
    model = _make_rest_llm(monkeypatch, transport, reasoning="low")

    await _drain(model, extra_kwargs={"reasoning": {"effort": "high"}})

    assert transport.body()["reasoning"] == {"effort": "high"}


async def test_reasoning_conflicting_with_extra_kwargs_logs_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.agents")

    model = _make_rest_llm(
        monkeypatch,
        _GatewayTransport(),
        reasoning="high",
        extra_kwargs={"reasoning": {"effort": "low"}},
    )
    await model.aclose()

    assert any("extra_kwargs['reasoning']" in m for m in _warnings(caplog))


async def test_reasoning_matching_extra_kwargs_logs_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.agents")

    model = _make_rest_llm(
        monkeypatch,
        _GatewayTransport(),
        reasoning="low",
        extra_kwargs={"reasoning": Reasoning(effort="low")},
    )
    await model.aclose()

    assert _warnings(caplog) == []


async def test_update_options_reasoning_changes_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _make_rest_llm(monkeypatch, transport, reasoning="low")

    model.update_options(reasoning="high")
    await _drain(model)

    assert transport.body()["reasoning"] == {"effort": "high"}


async def test_update_options_reasoning_keeps_extra_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _GatewayTransport()
    model = _make_rest_llm(monkeypatch, transport, extra_kwargs={"max_output_tokens": 64})

    model.update_options(reasoning="high")
    await _drain(model)

    assert transport.body()["max_output_tokens"] == 64


async def test_update_options_reasoning_none_clears_it(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _make_rest_llm(monkeypatch, transport, reasoning="low")

    model.update_options(reasoning=None)
    await _drain(model)

    assert "reasoning" not in transport.body()


async def test_update_options_without_reasoning_keeps_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _GatewayTransport()
    model = _make_rest_llm(monkeypatch, transport, reasoning="low")

    model.update_options(extra_kwargs={"max_output_tokens": 64})
    await _drain(model)

    assert transport.body()["reasoning"] == {"effort": "low"}


async def test_reasoning_rides_the_ws_payload() -> None:
    model = _make_ws_llm(reasoning="low")
    session = _ScriptedSession()
    _gateway_ws(model)._session = session  # type: ignore[assignment]

    await _consume(model.chat(chat_ctx=_ctx(), conn_options=APIConnectOptions(max_retry=0)))

    assert json.loads(session.sockets[0].sent[0])["reasoning"] == {"effort": "low"}


async def test_effort_below_model_minimum_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.agents")

    model = ResponsesLLM("openai/gpt-6.1-sol", api_key="k", api_secret="s", reasoning="none")
    await model.aclose()

    assert any("lowest" in m for m in _warnings(caplog))


async def test_update_options_model_below_effort_minimum_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.agents")
    model = ResponsesLLM("openai/gpt-6-luna", api_key="k", api_secret="s", reasoning="none")

    model.update_options(model="openai/gpt-6.1-sol")
    await model.aclose()

    assert any("lowest" in m for m in _warnings(caplog))


async def test_default_reasoning_is_the_models_lowest_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _GatewayTransport()
    model = _make_rest_llm(monkeypatch, transport, "openai/gpt-6-luna")

    await _drain(model)

    assert transport.body()["reasoning"] == {"effort": "none"}


async def test_default_reasoning_follows_a_low_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _make_rest_llm(monkeypatch, transport, "openai/gpt-6.1-sol")

    await _drain(model)

    assert transport.body()["reasoning"] == {"effort": "low"}


async def test_default_reasoning_follows_a_minimal_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _GatewayTransport()
    model = _make_rest_llm(monkeypatch, transport, "openai/gpt-5-mini")

    await _drain(model)

    assert transport.body()["reasoning"] == {"effort": "minimal"}


async def test_no_default_reasoning_for_non_openai_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _GatewayTransport()
    model = _make_rest_llm(monkeypatch, transport, "google/gemini-3-flash")

    await _drain(model)

    assert "reasoning" not in transport.body()


async def test_reasoning_none_skips_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _make_rest_llm(monkeypatch, transport, "openai/gpt-6-luna", reasoning=None)

    await _drain(model)

    assert "reasoning" not in transport.body()


async def test_update_options_reasoning_none_drops_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _GatewayTransport()
    model = _make_rest_llm(monkeypatch, transport, "openai/gpt-6-luna")

    model.update_options(reasoning=None)
    await _drain(model)

    assert "reasoning" not in transport.body()


async def test_extra_kwargs_reasoning_replaces_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _GatewayTransport()
    model = _make_rest_llm(
        monkeypatch,
        transport,
        "openai/gpt-6-luna",
        extra_kwargs={"reasoning": {"effort": "high", "summary": "auto"}},
    )

    await _drain(model)

    assert transport.body()["reasoning"] == {"effort": "high", "summary": "auto"}


async def test_per_call_extra_kwargs_reasoning_replaces_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _GatewayTransport()
    model = _make_rest_llm(monkeypatch, transport, "openai/gpt-6-luna")

    await _drain(model, extra_kwargs={"reasoning": {"effort": "high"}})

    assert transport.body()["reasoning"] == {"effort": "high"}


async def test_default_reasoning_follows_a_model_swap(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _GatewayTransport()
    model = _make_rest_llm(monkeypatch, transport, "openai/gpt-6-luna")

    model.update_options(model="openai/gpt-6.1-sol")
    await _drain(model)

    assert transport.body()["reasoning"] == {"effort": "low"}


async def test_default_reasoning_keeps_temperature_at_effort_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # gpt-6 accepts sampling params again at effort "none", so the default lets them through
    transport = _GatewayTransport()
    model = _make_rest_llm(
        monkeypatch, transport, "openai/gpt-6-luna", extra_kwargs={"temperature": 0.3}
    )

    await _drain(model)

    assert transport.body()["temperature"] == 0.3
