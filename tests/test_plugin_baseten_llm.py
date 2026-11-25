"""Baseten LLM inlines mid-conversation system messages only for models whose templates need it."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx2 as httpx
import openai
import pytest

from livekit.agents.llm import ChatContext, FunctionCall, FunctionCallOutput
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.plugins.baseten import LLM

pytestmark = pytest.mark.unit

PREAMBLE = "You are a helpful assistant."
INSTRUCTIONS = "Ask the caller for the year they were born."
INLINED = f"<instructions>\n{INSTRUCTIONS}\n</instructions>"


class _OneTokenStream(httpx.AsyncByteStream):
    """A minimal successful chat.completions SSE stream."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"


def _llm(model: str, *, inline: NotGivenOr[bool] = NOT_GIVEN) -> tuple[LLM, list[dict[str, Any]]]:
    """Build a Baseten LLM whose HTTP layer records every request body."""
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=_OneTokenStream()
        )

    client = openai.AsyncClient(
        api_key="test",
        base_url="http://baseten.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    llm_model = LLM(
        model=model, api_key="test", client=client, inline_mid_conversation_instructions=inline
    )
    return llm_model, requests


async def _sent_messages(
    model: str, chat_ctx: ChatContext, *, inline: NotGivenOr[bool] = NOT_GIVEN
) -> list[dict[str, Any]]:
    llm_model, requests = _llm(model, inline=inline)
    try:
        async with llm_model.chat(chat_ctx=chat_ctx) as stream:
            async for _ in stream:
                pass
    finally:
        await llm_model.aclose()

    assert len(requests) == 1
    return cast(list[dict[str, Any]], requests[0]["messages"])


def _per_turn_ctx() -> ChatContext:
    # the shape generate_reply(instructions=...) produces: a trailing system message
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="system", content=[PREAMBLE])
    chat_ctx.add_message(role="assistant", content=["Hello! How can I help you?"])
    chat_ctx.add_message(role="user", content=["I'd like to refill my prescription."])
    chat_ctx.add_message(role="system", content=[INSTRUCTIONS])
    return chat_ctx


def _tool_ctx() -> ChatContext:
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="system", content=[PREAMBLE])
    chat_ctx.add_message(role="user", content=["What's the weather in SF?"])
    chat_ctx.items.append(
        FunctionCall(call_id="call_1", name="get_weather", arguments='{"city": "SF"}')
    )
    chat_ctx.items.append(
        FunctionCallOutput(call_id="call_1", name="get_weather", output="sunny", is_error=False)
    )
    chat_ctx.add_message(role="system", content=[INSTRUCTIONS])
    return chat_ctx


def _plain_ctx() -> ChatContext:
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="system", content=[PREAMBLE])
    chat_ctx.add_message(role="user", content=["Hi!"])
    chat_ctx.add_message(role="assistant", content=["Hello! How can I help you?"])
    chat_ctx.add_message(role="user", content=["Tell me a joke."])
    return chat_ctx


@pytest.mark.parametrize(
    ("model", "inline", "inlined"),
    [
        ("Qwen/Qwen3.8-27B", NOT_GIVEN, True),
        ("qwen/qwen3.8-27b", NOT_GIVEN, True),
        # an explicit allow-list: other generations and sizes of the family are not opted in
        ("Qwen/Qwen3.5-35B-A3B-FP8", NOT_GIVEN, False),
        ("Qwen/Qwen3.5-122B-A10B", NOT_GIVEN, False),
        ("Qwen/Qwen3-235B-A22B-Instruct-2507", NOT_GIVEN, False),
        ("qwen3-dedicated", NOT_GIVEN, False),
        ("google/gemma-4-31B-it", NOT_GIVEN, False),
        ("openai/gpt-oss-120b", NOT_GIVEN, False),
        ("zai-org/GLM-5.2", NOT_GIVEN, False),
        ("meta-llama/Llama-4-Maverick-17B-128E-Instruct", NOT_GIVEN, False),
        ("deepseek-ai/DeepSeek-V3-0324", NOT_GIVEN, False),
        ("moonshotai/Kimi-K2-Instruct", NOT_GIVEN, False),
        ("my-dedicated-model", NOT_GIVEN, False),
        # the caller's override wins either way
        ("google/gemma-4-31B-it", True, True),
        ("my-dedicated-model", True, True),
        ("Qwen/Qwen3.8-27B", False, False),
    ],
)
async def test_per_turn_instructions_inlined_by_model(
    model: str, inline: NotGivenOr[bool], inlined: bool
) -> None:
    messages = await _sent_messages(model, _per_turn_ctx(), inline=inline)

    assert messages[:3] == [
        {"role": "system", "content": PREAMBLE},
        {"role": "assistant", "content": "Hello! How can I help you?"},
        {"role": "user", "content": "I'd like to refill my prescription."},
    ]
    if inlined:
        assert messages[3:] == [{"role": "user", "content": INLINED}]
    else:
        assert messages[3:] == [{"role": "system", "content": INSTRUCTIONS}]


async def test_inlining_preserves_tool_call_history() -> None:
    messages = await _sent_messages("Qwen/Qwen3.8-27B", _tool_ctx())

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "user"]
    assert messages[2]["tool_calls"][0]["id"] == "call_1"
    assert messages[2]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert messages[3] == {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
    assert messages[-1] == {"role": "user", "content": INLINED}


async def test_inlining_does_not_mutate_the_callers_chat_ctx() -> None:
    chat_ctx = _per_turn_ctx()
    snapshot = chat_ctx.to_dict()

    await _sent_messages("Qwen/Qwen3.8-27B", chat_ctx)

    assert chat_ctx.to_dict() == snapshot
    last = chat_ctx.items[-1]
    assert last.type == "message" and last.role == "system"


async def test_empty_mid_conversation_system_message_is_dropped_when_inlining() -> None:
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="system", content=[PREAMBLE])
    chat_ctx.add_message(role="user", content=["Hi!"])
    chat_ctx.add_message(role="system", content=[""])

    messages = await _sent_messages("Qwen/Qwen3.8-27B", chat_ctx)

    assert [m["role"] for m in messages] == ["system", "user"]


async def test_tool_history_passes_through_for_openai_style_models() -> None:
    messages = await _sent_messages("openai/gpt-oss-120b", _tool_ctx())

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "system"]
    assert messages[3] == {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}


async def test_conversation_without_mid_system_messages_is_identical_across_models() -> None:
    # when there is nothing to inline, both code paths must produce the same request
    inlined = await _sent_messages("Qwen/Qwen3.8-27B", _plain_ctx())
    passthrough = await _sent_messages("openai/gpt-oss-120b", _plain_ctx())

    assert inlined == passthrough
    assert [m["role"] for m in inlined] == ["system", "user", "assistant", "user"]
