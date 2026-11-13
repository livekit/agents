"""Anthropic wire regressions, using the installed SDK and recorded SSE events."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, Literal

import anthropic
import httpx2 as httpx
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import NOT_GIVEN, Agent, APIConnectOptions, RunContext, llm
from livekit.plugins.anthropic import LLM

# Any: decoded wire JSON, indexed by the provider's documented shape
Wire = dict[str, Any]


def _response(blocks: list[Wire]) -> httpx.Response:
    events: list[Wire] = [
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
        }
    ]
    for index, block in enumerate(blocks):
        kind = block["type"]
        if kind == "tool_use" and block["input"]:  # a call without arguments streams no JSON
            initial = {**block, "input": {}}
            deltas = [
                {"type": "input_json_delta", "partial_json": part}
                for part in (json.dumps(block["input"])[:9], json.dumps(block["input"])[9:])
            ]
        elif kind == "text":
            initial = {"type": "text", "text": ""}
            deltas = [{"type": "text_delta", "text": block["text"]}]
        elif kind == "thinking":
            initial = {"type": "thinking", "thinking": "", "signature": ""}
            deltas = [
                {"type": "thinking_delta", "thinking": block["thinking"]},
                {"type": "signature_delta", "signature": block["signature"]},
            ]
        else:
            initial, deltas = block, []
        events.append({"type": "content_block_start", "index": index, "content_block": initial})
        events.extend({"type": "content_block_delta", "index": index, "delta": d} for d in deltas)
        events.append({"type": "content_block_stop", "index": index})
    events.extend(
        [
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 3},
            },
            {"type": "message_stop"},
        ]
    )
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text="".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events),
    )


def _context() -> llm.ChatContext:
    context = llm.ChatContext.empty()
    context.add_message(role="user", content="Find both visitors.")
    return context


@pytest.mark.parametrize(
    "model_name,needs_filler",
    [
        ("claude-3-5-sonnet-20241022", False),
        ("claude-sonnet-4-6", True),
        ("claude-opus-4-6-20260205", True),
        pytest.param(
            "claude-sonnet-4-7",
            True,
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="https://github.com/livekit/agents/issues/7217 reproduces: newer models miss trailing user",
            ),
        ),
        pytest.param(
            "claude-opus-5",
            True,
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="https://github.com/livekit/agents/issues/7217 reproduces: newer models miss trailing user",
            ),
        ),
    ],
)
async def test_assistant_ending_requests_use_nonblank_filler_only_when_required(
    model_name: str,
    needs_filler: bool,
) -> None:
    """Assistant continuations remain valid; https://github.com/livekit/agents/issues/4907 https://github.com/livekit/agents/issues/5254 https://github.com/livekit/agents/issues/6213 https://github.com/livekit/agents/issues/7217."""
    sent = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _response([{"type": "text", "text": "done"}])

    async with anthropic.AsyncClient(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
    ) as client:
        model = LLM(model=model_name, client=client)
        context = _context()
        context.add_message(role="assistant", content="Hello")
        try:
            result = await model.chat(
                chat_ctx=context, conn_options=APIConnectOptions(max_retry=0)
            ).collect()
        finally:
            await model.aclose()
    assert result.text == "done"
    messages = sent[0]["messages"]
    assert [m["role"] for m in messages] == (
        ["user", "assistant", "user"] if needs_filler else ["user", "assistant"]
    )
    if needs_filler:
        assert messages[-1]["content"][0]["text"].strip()
    last = context.items[-1]
    assert last.type == "message" and last.role == "assistant"


async def test_parallel_tools_and_followup_preserve_text_arguments_and_pairing() -> None:
    """Tool continuations retain exact blocks; https://github.com/livekit/agents/issues/699 https://github.com/livekit/agents/issues/1017 https://github.com/livekit/agents/issues/1048 https://github.com/livekit/agents/pull/1441 https://github.com/livekit/agents/issues/1059 https://github.com/livekit/agents/issues/5959."""
    sent = []
    visitors = [
        {"name": "Juan", "apto": "203", "surname": "Varela", "ci": ""},
        {"name": "Jane", "apto": "204", "surname": "Doe", "ci": "key"},
    ]

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        if len(sent) == 1:
            return _response(
                [
                    {"type": "text", "text": "Checking."},
                    *[
                        {"type": "tool_use", "id": f"call_{i}", "name": "lookup", "input": args}
                        for i, args in enumerate(visitors)
                    ],
                ]
            )
        return _response([{"type": "text", "text": "Both found."}])

    async with anthropic.AsyncClient(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
    ) as client:
        model = LLM(client=client)
        agent = Agent(instructions="Find the visitors.", llm=model)

        @agent.tool
        def lookup(ctx: RunContext, name: str, apto: str, surname: str, ci: str) -> str:
            """Look up a visitor."""
            return f"Found {name} {surname} in {apto}"

        try:
            result = await agent.run("Find both visitors.")
        finally:
            await model.aclose()
    assert result.status == "completed"
    assert result.output == "Both found."
    history = sent[1]["messages"]
    assert history[1]["content"] == [
        {"type": "text", "text": "Checking."},
        *[
            {"type": "tool_use", "id": f"call_{i}", "name": "lookup", "input": args}
            for i, args in enumerate(visitors)
        ],
    ]
    assert [(b["tool_use_id"], b["content"], b["is_error"]) for b in history[2]["content"]] == [
        ("call_0", "Found Juan Varela in 203", False),
        ("call_1", "Found Jane Doe in 204", False),
    ]
    assert sent[1]["tools"] == sent[0]["tools"]
    assert [
        i.text_content
        for i in result.new_messages()
        if i.type == "message" and i.role == "assistant"
    ] == [
        "Checking.",
        "Both found.",
    ]


@pytest.mark.parametrize(
    "model_name,choice",
    [
        pytest.param(
            "claude-sonnet-4-6",
            None,
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="https://github.com/livekit/agents/issues/8077 reproduces: parallel restriction omitted without choice",
            ),
        ),
        ("claude-sonnet-4-6", "auto"),
        pytest.param(
            "claude-opus-5-5",
            "required",
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="https://github.com/livekit/agents/pull/7511 reproduces: unsupported forced tool choice",
            ),
        ),
    ],
)
async def test_tool_choice_keeps_parallel_restrictions(
    model_name: str, choice: Literal["auto", "required"] | None
) -> None:
    """Tool restrictions survive translation; https://github.com/livekit/agents/issues/8077 https://github.com/livekit/agents/pull/7511."""
    sent = []

    @llm.function_tool
    def lookup() -> str:
        """Look up a visitor."""
        return "found"

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _response([{"type": "text", "text": "done"}])

    async with anthropic.AsyncClient(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
    ) as client:
        model = LLM(model=model_name, client=client, parallel_tool_calls=False)
        try:
            result = await model.chat(
                chat_ctx=_context(), tools=[lookup], tool_choice=choice or NOT_GIVEN
            ).collect()
        finally:
            await model.aclose()
    assert result.text == "done"
    assert sent[0].get("tool_choice") == {"type": "auto", "disable_parallel_tool_use": True}


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="https://github.com/livekit/agents/issues/7908 reproduces: signed thinking blocks are dropped",
)
async def test_tool_result_request_preserves_signed_and_redacted_thinking() -> None:
    """Thinking survives tool-result replay; https://github.com/livekit/agents/issues/7908."""
    thinking = [
        {
            "type": "thinking",
            "thinking": "I need the visitor record.",
            "signature": "signed-reasoning",
        },
        {"type": "redacted_thinking", "data": "opaque-reasoning"},
    ]
    sent = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        if len(sent) == 1:
            return _response(
                [*thinking, {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}}]
            )
        return _response([{"type": "text", "text": "found"}])

    async with anthropic.AsyncClient(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
    ) as client:
        model = LLM(client=client)
        agent = Agent(instructions="Find the visitor.", llm=model)

        @agent.tool
        def lookup(ctx: RunContext) -> str:
            """Find the visitor."""
            return "found"

        try:
            result = await agent.run("Find the visitor.")
        finally:
            await model.aclose()
    assert result.status == "completed"
    assert result.output == "found"
    assert sent[1]["messages"][1]["content"][:2] == thinking


async def test_default_client_can_send_a_request_with_the_installed_sdk() -> None:
    """The installed SDK accepts the owned client; https://github.com/livekit/agents/issues/7235."""

    async def complete(request: web.Request) -> web.Response:
        await request.read()
        response = _response([{"type": "text", "text": "hello"}])
        return web.Response(body=response.content, content_type="text/event-stream")

    app = web.Application()
    app.router.add_post("/v1/messages", complete)
    async with TestServer(app) as server:
        model = LLM(api_key="test", base_url=str(server.make_url("/")))
        try:
            result = await model.chat(chat_ctx=_context()).collect()
            assert result.text == "hello"
        finally:
            await model.aclose()


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/pull/7970 reproduces: user is passed as unsupported SDK keyword",
)
async def test_user_is_sent_as_metadata_without_losing_other_metadata() -> None:
    """User metadata reaches the provider; https://github.com/livekit/agents/pull/7970."""
    sent = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _response([{"type": "text", "text": "hello"}])

    async with anthropic.AsyncClient(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
    ) as client:
        model = LLM(client=client, user="visitor-17")
        try:
            result = await model.chat(
                chat_ctx=_context(),
                conn_options=APIConnectOptions(max_retry=0),
                extra_kwargs={"metadata": {"trace_id": "trace-1"}},
            ).collect()
        finally:
            await model.aclose()
    assert result.text == "hello"
    assert sent[0]["metadata"] == {"user_id": "visitor-17", "trace_id": "trace-1"}
    assert "user" not in sent[0]


@pytest.mark.parametrize("model_name", ["claude-sonnet-5", "claude-haiku-5-5"])
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="https://github.com/livekit/agents/issues/7726 reproduces: default thinking eats max_tokens",
)
async def test_default_request_to_a_thinking_model_leaves_room_for_text(model_name: str) -> None:
    """Think-by-default models are not starved of text; https://github.com/livekit/agents/issues/7726 https://github.com/livekit/agents/issues/5508."""
    sent = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _response([{"type": "text", "text": "done"}])

    async with anthropic.AsyncClient(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
    ) as client:
        model = LLM(model=model_name, client=client)
        try:
            await model.chat(chat_ctx=_context()).collect()
        finally:
            await model.aclose()
    assert sent[0].get("thinking") == {"type": "disabled"}


async def test_a_tool_call_without_arguments_carries_an_empty_object() -> None:
    """A tool_use block streams no input_json_delta when the tool takes no arguments: the call's
    arguments are ``{}`` and its replay sends ``"input": {}``. Catches arguments left as ``""``,
    which fails to parse as the tool's arguments and replays as invalid input."""
    sent: list[Wire] = []
    replies: Iterator[list[Wire]] = iter(
        [
            [{"type": "tool_use", "id": "call_1", "name": "ring", "input": {}}],
            [{"type": "text", "text": "rung"}],
        ]
    )

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _response(next(replies))

    async with anthropic.AsyncClient(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
    ) as client:
        model = LLM(model="claude-sonnet-4-6", client=client)
        context = _context()
        try:
            first = await model.chat(
                chat_ctx=context, conn_options=APIConnectOptions(max_retry=0)
            ).collect()
            [tool_call] = first.tool_calls
            call = llm.FunctionCall(
                call_id=tool_call.call_id, name=tool_call.name, arguments=tool_call.arguments
            )
            context.insert(call)
            context.insert(
                llm.FunctionCallOutput(
                    call_id=call.call_id, name="ring", output="ok", is_error=False
                )
            )
            await model.chat(
                chat_ctx=context, conn_options=APIConnectOptions(max_retry=0)
            ).collect()
        finally:
            await model.aclose()

    assert call.arguments == "{}"
    replayed = [b for m in sent[1]["messages"] for b in m["content"] if b["type"] == "tool_use"]
    assert replayed[0]["input"] == {}
