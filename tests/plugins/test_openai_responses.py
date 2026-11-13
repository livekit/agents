"""OpenAI Responses LLM: the websocket and HTTP transports against a local Responses peer."""

from __future__ import annotations

import copy
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Literal, ParamSpec, TypedDict

import openai
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from openai.types import Reasoning
from pydantic import JsonValue

from livekit.agents import (
    NOT_GIVEN,
    APIConnectOptions,
    APIError,
    APIStatusError,
    NotGivenOr,
    http,
    llm,
)
from livekit.plugins.openai import responses
from livekit.plugins.openai.tools import WebSearch
from tests.fakes import until

NO_RETRY = APIConnectOptions(max_retry=0)
P = ParamSpec("P")
RESPONSE: dict[str, object] = {
    "id": "resp_1",
    "created_at": 0,
    "model": "gpt-4.1",
    "object": "response",
    "output": [],
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
}
CREATED = {"type": "response.created", "sequence_number": 0, "response": RESPONSE}
TEXT = {
    "type": "response.output_text.delta",
    "sequence_number": 1,
    "content_index": 0,
    "delta": "hello",
    "item_id": "msg_1",
    "output_index": 0,
    "logprobs": [],
}
DROP = "drop the connection"
Frame = dict[str, object] | str  # a JSON event, or DROP


class _Request(TypedDict, total=False):
    """The parts of a Responses request the tests read."""

    reasoning: dict[str, object]
    previous_response_id: str
    input: list[dict[str, object]]
    text: dict[str, object]
    store: bool
    tools: list[dict[str, object]]


def _completed(**response: object) -> dict[str, object]:
    done = {**RESPONSE, "status": "completed", **response}
    return {"type": "response.completed", "sequence_number": 2, "response": done}


def _incomplete(reason: str | None) -> dict[str, object]:
    details = {"incomplete_details": {"reason": reason}} if reason else {}
    response = {**RESPONSE, "status": "incomplete", **details}
    return {"type": "response.incomplete", "sequence_number": 2, "response": response}


class _Peer:
    """A Responses endpoint over websocket and HTTP streaming. Each request takes the next
    scripted reply (the last one repeats); ``DROP`` in a reply ends the connection there."""

    def __init__(self, *replies: list[Frame]) -> None:
        self.replies = list(replies) or [[CREATED, TEXT, _completed()]]
        self.requests: list[_Request] = []
        self.connections: list[web.WebSocketResponse] = []
        self.queries: list[dict[str, str]] = []

    def reply_to(self, payload: _Request) -> list[Frame]:
        self.requests.append(payload)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        response_id = f"resp_{len(self.requests)}"
        return [_with_id(frame, response_id) for frame in reply]

    async def socket(self, request: web.Request) -> web.WebSocketResponse:
        self.queries.append(dict(request.query))
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        self.connections.append(socket)
        async for message in socket:
            for frame in self.reply_to(json.loads(message.data)):
                if frame == DROP:
                    await socket.close()
                    return socket
                await socket.send_json(frame)
        return socket

    async def stream(self, request: web.Request) -> web.StreamResponse:
        reply = self.reply_to(await request.json())
        response = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await response.prepare(request)
        for frame in reply:
            await response.write(f"data: {json.dumps(frame)}\n\n".encode())
        return response


def _with_id(frame: Frame, response_id: str) -> Frame:
    if isinstance(frame, dict):
        response = frame.get("response")
        if isinstance(response, dict) and response.get("id") == "resp_1":
            return {**frame, "response": {**response, "id": response_id}}
    return frame


@asynccontextmanager
async def _serve(peer: _Peer) -> AsyncIterator[str]:
    """The peer's base url, with the Responses route on both transports."""
    app = web.Application()
    app.router.add_get("/v1/responses", peer.socket)
    app.router.add_post("/v1/responses", peer.stream)
    async with TestServer(app) as server, http.open():
        yield str(server.make_url("/v1"))


def _llm(
    base_url: str,
    transport: str = "websocket",
    *,
    model: str = "gpt-4.1",
    reasoning: NotGivenOr[Reasoning] = NOT_GIVEN,
    store: NotGivenOr[bool] = NOT_GIVEN,
    verbosity: NotGivenOr[Literal["low", "medium", "high"]] = NOT_GIVEN,
) -> responses.LLM:
    websocket = transport == "websocket"
    return responses.LLM(
        api_key="test",
        base_url=f"{base_url}/responses" if websocket else base_url,
        use_websocket=websocket,
        model=model,
        reasoning=reasoning,
        store=store,
        verbosity=verbosity,
    )


def _hi() -> llm.ChatContext:
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="user", content="hi")
    return ctx


async def _reply(
    model: responses.LLM,
    ctx: llm.ChatContext,
    conn: APIConnectOptions = NO_RETRY,
    *,
    tools: list[llm.Tool] | None = None,
    extra_kwargs: NotGivenOr[dict[str, object]] = NOT_GIVEN,
) -> list[llm.ChatChunk]:
    async with model.chat(
        chat_ctx=ctx, conn_options=conn, tools=tools, extra_kwargs=extra_kwargs
    ) as stream:
        return [chunk async for chunk in stream]


def _kept(
    make: Callable[P, openai.AsyncClient], made: list[openai.AsyncClient]
) -> Callable[P, openai.AsyncClient]:
    """``make``, keeping every client it makes in ``made``."""

    def make_and_keep(*args: P.args, **kwargs: P.kwargs) -> openai.AsyncClient:
        made.append(make(*args, **kwargs))
        return made[-1]

    return make_and_keep


async def test_reasoning_is_sent_without_null_fields() -> None:
    """A reasoning setting goes out with only the fields set; the API rejects explicit nulls.

    https://github.com/livekit/agents/pull/6385
    """
    peer = _Peer()
    async with _serve(peer) as base_url:
        model = _llm(base_url, model="gpt-5.4", reasoning=Reasoning(effort="none"))
        try:
            await _reply(model, _hi())
        finally:
            await model.aclose()

    assert peer.requests[0]["reasoning"] == {"effort": "none"}


async def test_an_error_frame_fails_the_request_with_its_message() -> None:
    """A top-level error frame, which has no sequence_number, surfaces the API's message.

    https://github.com/livekit/agents/pull/6385
    """
    message = "Invalid type for 'reasoning.mode': expected one of 'standard' or 'pro'."
    error = {"type": "error", "message": message, "code": "invalid_type", "status": 400}
    peer = _Peer([error])
    async with _serve(peer) as base_url:
        model = _llm(base_url)
        try:
            with pytest.raises(APIError, match="reasoning.mode"):
                await _reply(model, _hi())
        finally:
            await model.aclose()


async def test_a_socket_closed_while_idle_is_replaced_without_a_retry() -> None:
    """A pooled socket the server closed is discarded and the request goes out on a new one.

    https://github.com/livekit/agents/issues/6513
    """
    peer = _Peer()
    async with _serve(peer) as base_url:
        model = _llm(base_url)
        try:
            await _reply(model, _hi())
            await peer.connections[0].close()

            chunks = await _reply(model, _hi())
        finally:
            await model.aclose()

    assert "".join(c.delta.content or "" for c in chunks if c.delta) == "hello"
    assert len(peer.connections) == 2


@pytest.mark.parametrize("transport", ["websocket", "http"])
@pytest.mark.parametrize("reason", ["max_output_tokens", None])
async def test_an_incomplete_response_fails_once_and_is_not_continued(
    transport: str, reason: str | None
) -> None:
    """An incomplete response is a final error, and the next request does not build on it.

    https://github.com/livekit/agents/issues/7013
    """
    peer = _Peer([CREATED, _incomplete(reason)])
    async with _serve(peer) as base_url:
        model = _llm(base_url, transport)
        retrying = APIConnectOptions(max_retry=2, retry_interval=0)
        try:
            with pytest.raises(APIStatusError) as caught:
                await _reply(model, _hi(), retrying)
            followup = _hi()
            followup.add_message(role="user", content="follow up")
            with pytest.raises(APIStatusError):
                await _reply(model, followup)
        finally:
            await model.aclose()

    assert caught.value.retryable is False
    assert (reason or "reason unavailable") in caught.value.message
    assert len(peer.requests) == 2
    assert "previous_response_id" not in peer.requests[1]


@pytest.mark.parametrize(
    ("before_drop", "requests"),
    [([CREATED], 3), ([CREATED, TEXT], 1)],
    ids=["after_response_created", "after_text"],
)
async def test_a_dropped_stream_is_retried_only_before_any_text(
    before_drop: list[Frame], requests: int
) -> None:
    """A connection lost before any output is retried; text already delivered is not redone.

    https://github.com/livekit/agents/pull/6697
    """
    peer = _Peer([*before_drop, DROP])
    async with _serve(peer) as base_url:
        model = _llm(base_url)
        try:
            with pytest.raises(APIError):
                await _reply(model, _hi(), APIConnectOptions(max_retry=2, retry_interval=0))
        finally:
            await model.aclose()

    assert len(peer.requests) == requests


async def test_a_completed_response_is_continued_with_only_the_new_input() -> None:
    """The next turn sends previous_response_id and only the messages added since."""
    peer = _Peer()
    first = _hi()
    second = first.copy()
    second.add_message(role="user", content="second")
    async with _serve(peer) as base_url:
        model = _llm(base_url)
        try:
            await _reply(model, first)
            await _reply(model, second)
        finally:
            await model.aclose()

    assert "previous_response_id" not in peer.requests[0]
    assert peer.requests[1]["previous_response_id"] == "resp_1"
    assert [item["content"] for item in peer.requests[1]["input"]] == ["second"]


async def test_verbosity_and_a_json_schema_format_share_the_text_field() -> None:
    """Verbosity joins a caller's json_schema format, which keeps its wire field names."""
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    text = {"format": {"type": "json_schema", "name": "Answer", "schema": schema}}
    peer = _Peer()
    async with _serve(peer) as base_url:
        model = _llm(base_url, verbosity="low")
        try:
            await _reply(model, _hi(), extra_kwargs={"text": text})
        finally:
            await model.aclose()

    assert peer.requests[0]["text"] == {**text, "verbosity": "low"}


@pytest.mark.parametrize("transport", ["websocket", "http"])
async def test_store_and_tool_schemas_survive_repeated_requests(transport: str) -> None:
    """`store` and a raw-schema tool are sent intact on every request, the schema unchanged.

    https://github.com/livekit/agents/issues/4963
    """
    raw: dict[str, JsonValue] = {
        "name": "lookup",
        "description": "Look up a record",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}}},
    }
    original = copy.deepcopy(raw)

    @llm.function_tool(raw_schema=raw)
    async def lookup(args: dict[str, object]) -> str:
        return "found"

    peer = _Peer([CREATED, _completed()])
    async with _serve(peer) as base_url:
        model = _llm(base_url, transport, store=False)
        try:
            for _ in range(2):
                await _reply(model, _hi(), tools=[lookup, WebSearch()])
        finally:
            await model.aclose()

    assert raw == original
    for request in peer.requests:
        assert request["store"] is False
        assert "previous_response_id" not in request
        function, search = request["tools"]
        assert function["parameters"] == raw["parameters"]
        assert search["type"] == "web_search"


async def test_an_unanswered_tool_call_is_left_out_of_the_request() -> None:
    """A call with no output is not replayed, and the caller's context is left as it was.

    https://github.com/livekit/agents/issues/5092
    """
    peer = _Peer()
    ctx = _hi()
    ctx.insert(
        llm.FunctionCall(id="pending", call_id="call-pending", name="lookup", arguments="{}")
    )
    ctx.add_message(role="user", content="never mind")
    async with _serve(peer) as base_url:
        model = _llm(base_url)
        try:
            await _reply(model, ctx)
        finally:
            await model.aclose()

    assert [item["role"] for item in peer.requests[0]["input"]] == ["user", "user"]
    assert "call-pending" not in json.dumps(peer.requests[0])
    assert len(ctx.items) == 3


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="https://github.com/livekit/agents/pull/7710 reproduces: Responses drops reported "
    "cache-write token counts",
)
async def test_usage_keeps_reasoning_and_cache_counts() -> None:
    """Reported usage keeps its cache-read, cache-write and reasoning token counts.

    https://github.com/livekit/agents/pull/7710
    """
    usage = {
        "input_tokens": 100,
        "output_tokens": 30,
        "total_tokens": 130,
        "input_tokens_details": {"cached_tokens": 40, "cache_write_tokens": 20},
        "output_tokens_details": {"reasoning_tokens": 15},
    }
    peer = _Peer([CREATED, _completed(usage=usage)])
    async with _serve(peer) as base_url:
        model = _llm(base_url)
        try:
            chunks = await _reply(model, _hi())
        finally:
            await model.aclose()

    [tokens] = [chunk.usage for chunk in chunks if chunk.usage]
    assert (tokens.cache_read_tokens, tokens.output_reasoning_tokens) == (40, 15)
    assert tokens.cache_write_tokens == 20


async def test_a_gateway_socket_url_keeps_its_query_and_names_the_model() -> None:
    """A gateway's websocket URL keeps its query and carries the model for routing.

    https://github.com/livekit/agents/pull/6405
    """
    peer = _Peer()
    async with _serve(peer) as base_url:
        model = responses.LLM(
            api_key="test", base_url=f"{base_url}/responses?model=old&tenant=a", model="routed"
        )
        try:
            await _reply(model, _hi())
        finally:
            await model.aclose()

    assert peer.queries == [{"model": "routed", "tenant": "a"}]


@pytest.mark.parametrize("transport", ["websocket", "http"])
async def test_prewarm_opens_one_connection_and_close_settles_it(transport: str) -> None:
    """Repeated prewarms open the transport once, and closing the model closes it.

    https://github.com/livekit/agents/issues/2867
    """
    peer = _Peer()
    listed: list[str] = []

    async def models(request: web.Request) -> web.Response:
        listed.append(request.path)
        return web.json_response({"data": [], "object": "list"})

    app = web.Application()
    app.router.add_get("/v1/responses", peer.socket)
    app.router.add_get("/v1/models", models)
    async with TestServer(app) as server, http.open():
        model = _llm(str(server.make_url("/v1")), transport)
        model.prewarm()
        model.prewarm()
        await until(lambda: peer.connections or listed, "the prewarm reached the server")

        await model.aclose()

        if transport == "websocket":
            assert len(peer.connections) == 1
            await until(lambda: peer.connections[0].closed, "the prewarmed socket closed")
        else:
            assert listed == ["/v1/models"]


async def test_closing_the_model_closes_the_client_it_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model closes the SDK client it made, and leaves a supplied one open.

    https://github.com/livekit/agents/pull/4735
    """
    created: list[openai.AsyncClient] = []
    make_client = openai.AsyncClient

    monkeypatch.setattr(openai, "AsyncClient", _kept(make_client, created))
    supplied = make_client(api_key="test")
    owned = responses.LLM(api_key="test", use_websocket=False)
    borrowed = responses.LLM(client=supplied)

    await owned.aclose()
    await borrowed.aclose()

    assert [client.is_closed() for client in created] == [True]
    assert not supplied.is_closed()
    await supplied.close()
