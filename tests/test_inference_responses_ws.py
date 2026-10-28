from __future__ import annotations

import json
from typing import Any

import aiohttp
import pytest
from yarl import URL

from livekit.agents import APIConnectOptions, APIStatusError, llm as agents_llm
from livekit.agents.inference import responses as inference_responses
from livekit.agents.inference.responses import LLM as ResponsesLLM
from livekit.agents.llm._responses_engine import ResponsesWebsocket

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
_CONNECTED = {
    "type": "gateway.connected",
    "deployment": "standard_openai",
    "model": "gpt-4.1",
    "session_id": "sess_1",
}
_TEXT_FRAMES = [
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
_TOOL_NOT_ALLOWED = {
    "type": "gateway.error",
    "code": "tool_not_allowed",
    "message": 'tools[0].type "mcp" is not allowed',
}


class _FakeWsMsg:
    def __init__(self, data: str) -> None:
        self.type = aiohttp.WSMsgType.TEXT
        self.data = data
        self.extra = None


class _ScriptedWs:
    def __init__(self, frames: list[dict]) -> None:
        self.sent: list[str] = []
        self.closed = False
        self._frames = list(frames)

    async def send_str(self, data: str) -> None:
        self.sent.append(data)

    async def receive(self) -> _FakeWsMsg:
        return _FakeWsMsg(json.dumps(self._frames.pop(0)))

    async def close(self) -> None:
        self.closed = True


class _DialRecord:
    def __init__(self, url: str, headers: dict[str, str], heartbeat: float) -> None:
        self.url = url
        self.headers = headers
        self.heartbeat = heartbeat


class _FakeSession:
    def __init__(self) -> None:
        self.dials: list[_DialRecord] = []

    async def ws_connect(
        self, url: str, *, headers: dict[str, str], heartbeat: float
    ) -> _ScriptedWs:
        self.dials.append(_DialRecord(url, headers, heartbeat))
        return _ScriptedWs([])


def _make_llm(**kwargs: Any) -> ResponsesLLM:
    return ResponsesLLM(
        "openai/gpt-4.1",
        base_url="https://gw.test/v1",
        api_key="k",
        api_secret="s",
        use_websocket=True,
        **kwargs,
    )


def _transport(llm_model: ResponsesLLM) -> inference_responses._GatewayResponsesWebsocket:
    transport = llm_model._ws
    assert isinstance(transport, inference_responses._GatewayResponsesWebsocket)
    return transport


def _dial_session(llm_model: ResponsesLLM) -> _FakeSession:
    session = _FakeSession()
    _transport(llm_model)._session = session  # type: ignore[assignment]
    return session


def _use_connection(llm_model: ResponsesLLM, ws: _ScriptedWs) -> None:
    async def _connect(_timeout: float) -> _ScriptedWs:
        return ws

    _transport(llm_model)._pool._connect_cb = _connect  # type: ignore[assignment]


def _ctx() -> agents_llm.ChatContext:
    chat_ctx = agents_llm.ChatContext()
    chat_ctx.add_message(role="user", content="hi")
    return chat_ctx


async def _drain(llm_model: ResponsesLLM) -> str:
    stream = llm_model.chat(chat_ctx=_ctx(), conn_options=APIConnectOptions(max_retry=0))
    try:
        chunks = [chunk async for chunk in stream]
    finally:
        await stream.aclose()
    return "".join(c.delta.content or "" for c in chunks if c.delta)


async def test_ws_url_is_gateway_responses_with_model() -> None:
    llm_model = _make_llm()
    session = _dial_session(llm_model)

    await _transport(llm_model)._create_ws(1.0)

    expected = str(URL("wss://gw.test/v1/responses").update_query(model="openai/gpt-4.1"))
    assert session.dials[0].url == expected
    assert session.dials[0].heartbeat == 30.0


async def test_ws_dial_sends_fresh_token_and_inference_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def _counting_token(api_key: str | None, api_secret: str | None, ttl: float = 600) -> str:
        nonlocal calls
        calls += 1
        return f"token-{calls}"

    monkeypatch.setattr(inference_responses, "create_access_token", _counting_token)
    llm_model = _make_llm()
    session = _dial_session(llm_model)

    await _transport(llm_model)._create_ws(1.0)
    await _transport(llm_model)._create_ws(1.0)

    assert [d.headers["Authorization"] for d in session.dials] == [
        "Bearer token-1",
        "Bearer token-2",
    ]
    assert session.dials[0].headers["User-Agent"].startswith("LiveKit Agents/")


async def test_ws_dial_sends_provider_header() -> None:
    llm_model = _make_llm(provider="azure")
    session = _dial_session(llm_model)

    await _transport(llm_model)._create_ws(1.0)

    assert session.dials[0].headers["X-LiveKit-Inference-Provider"] == "azure"


async def test_ws_dial_sends_inference_class_header() -> None:
    llm_model = _make_llm(inference_class="priority")
    session = _dial_session(llm_model)

    await _transport(llm_model)._create_ws(1.0)

    assert session.dials[0].headers["X-LiveKit-Inference-Priority"] == "priority"


async def test_gateway_error_raises_api_status_error() -> None:
    llm_model = _make_llm()
    _use_connection(llm_model, _ScriptedWs([_CONNECTED, _TOOL_NOT_ALLOWED]))

    with pytest.raises(APIStatusError) as exc_info:
        await _drain(llm_model)

    assert exc_info.value.message == _TOOL_NOT_ALLOWED["message"]
    assert exc_info.value.status_code == -1
    assert exc_info.value.retryable is False
    await llm_model.aclose()


async def test_gateway_error_socket_leaves_pool() -> None:
    llm_model = _make_llm()
    transport = _transport(llm_model)
    ws = _ScriptedWs([_CONNECTED, _TOOL_NOT_ALLOWED])
    _use_connection(llm_model, ws)

    # the stream stops reading at the error and drops the generator; close it the same way
    events = transport.generate_response({"type": "response.create"})
    assert (await events.__anext__())["type"] == "gateway.connected"
    frame = await events.__anext__()
    parsed = inference_responses._GatewayResponsesLLMStream._parse_ws_event(object(), frame)  # type: ignore[arg-type]
    await events.aclose()

    assert parsed is not None and parsed.type == "error"
    assert ws not in transport._pool._available
    assert ws not in transport._pool._connections
    await llm_model.aclose()


async def test_gateway_connected_is_ignored() -> None:
    llm_model = _make_llm()
    _use_connection(llm_model, _ScriptedWs([_CONNECTED, *_TEXT_FRAMES]))

    text = await _drain(llm_model)

    assert text == "hi"
    await llm_model.aclose()


async def test_gateway_session_expiring_is_ignored() -> None:
    llm_model = _make_llm()
    expiring = {"type": "gateway.session_expiring", "seconds_until_close": 300}
    _use_connection(llm_model, _ScriptedWs([_CONNECTED, expiring, *_TEXT_FRAMES]))

    text = await _drain(llm_model)

    assert text == "hi"
    await llm_model.aclose()


async def test_ws_pool_session_is_55_minutes() -> None:
    llm_model = _make_llm()

    assert _transport(llm_model)._pool._max_session_duration == 3300
    await llm_model.aclose()


async def test_ws_payload_has_no_extra_headers() -> None:
    llm_model = _make_llm(provider="azure", inference_class="priority")
    ws = _ScriptedWs([_CONNECTED, *_TEXT_FRAMES])
    _use_connection(llm_model, ws)

    await _drain(llm_model)

    payload = json.loads(ws.sent[0])
    assert payload["type"] == "response.create"
    assert "extra_headers" not in payload
    await llm_model.aclose()


async def test_websocket_mode_builds_no_rest_client() -> None:
    llm_model = _make_llm()

    assert llm_model._opts.use_websocket is True
    assert llm_model._client is None
    assert _transport(llm_model)._session is None
    await llm_model.aclose()


async def test_engine_default_session_duration_unchanged() -> None:
    transport = ResponsesWebsocket(api_key="k", timeout=1.0, model="gpt-4.1")

    assert transport._pool._max_session_duration == 3600
    await transport.aclose()
