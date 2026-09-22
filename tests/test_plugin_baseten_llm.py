"""Baseten LLM: per-model handling of mid-conversation system messages.

LiveKit appends ``generate_reply(instructions=...)`` (and the expressive TTS guide) to
the chat context as a trailing system message. Gemma's chat template rejects any system
turn after the first and Qwen's was not trained on one, so for those families the plugin
inlines such messages as ``<instructions>``-wrapped user messages. OpenAI-style models
(gpt-oss, GLM, Llama, DeepSeek, Kimi) accept system messages anywhere and must keep
receiving the request untouched.

These tests capture the JSON body the plugin would POST, so they exercise the real
serializer path without network access.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx
import openai
import pytest

from livekit.agents.llm import ChatContext, FunctionCall, FunctionCallOutput
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.plugins.baseten import LLM
from livekit.plugins.baseten.llm import _needs_inline_instructions

pytestmark = pytest.mark.unit

PREAMBLE = "You are a helpful assistant."
INSTRUCTIONS = "Ask the caller for the year they were born."
INLINED = f"<instructions>\n{INSTRUCTIONS}\n</instructions>"

INLINE_MODELS = ["google/gemma-4-31B-it", "google/gemma-4-E4B-it", "Qwen/Qwen3.5-35B-A3B-FP8"]
PASSTHROUGH_MODELS = [
    "openai/gpt-oss-120b",
    "zai-org/GLM-5.2",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "deepseek-ai/DeepSeek-V3-0324",
    "moonshotai/Kimi-K2-Instruct",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Model-family detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("google/gemma-4-31B-it", True),
        ("GOOGLE/GEMMA-4-E2B-IT", True),
        ("Qwen/Qwen3.5-122B-A10B", True),
        ("qwen3-dedicated", True),
        ("openai/gpt-oss-120b", False),
        ("zai-org/GLM-5.2", False),
        ("meta-llama/Llama-4-Scout-17B-16E-Instruct", False),
        ("my-dedicated-model", False),
    ],
)
def test_inline_instructions_inferred_from_model_id(model: str, expected: bool) -> None:
    assert _needs_inline_instructions(model) is expected


# ---------------------------------------------------------------------------
# New behaviour: Gemma / Qwen get inlined instructions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", INLINE_MODELS)
async def test_per_turn_instructions_are_inlined_for_single_system_models(model: str) -> None:
    messages = await _sent_messages(model, _per_turn_ctx())

    assert [m["role"] for m in messages] == ["system", "assistant", "user", "user"]
    assert messages[0] == {"role": "system", "content": PREAMBLE}
    assert messages[-1] == {"role": "user", "content": INLINED}


async def test_inlining_preserves_tool_call_history() -> None:
    messages = await _sent_messages("google/gemma-4-31B-it", _tool_ctx())

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "user"]
    assert messages[2]["tool_calls"][0]["id"] == "call_1"
    assert messages[2]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert messages[3] == {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
    assert messages[-1] == {"role": "user", "content": INLINED}


async def test_inlining_does_not_mutate_the_callers_chat_ctx() -> None:
    chat_ctx = _per_turn_ctx()
    snapshot = chat_ctx.to_dict()

    await _sent_messages("google/gemma-4-31B-it", chat_ctx)

    assert chat_ctx.to_dict() == snapshot
    last = chat_ctx.items[-1]
    assert last.type == "message" and last.role == "system"


async def test_empty_mid_conversation_system_message_is_dropped_when_inlining() -> None:
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="system", content=[PREAMBLE])
    chat_ctx.add_message(role="user", content=["Hi!"])
    chat_ctx.add_message(role="system", content=[""])

    messages = await _sent_messages("google/gemma-4-31B-it", chat_ctx)

    assert [m["role"] for m in messages] == ["system", "user"]


# ---------------------------------------------------------------------------
# Regression: OpenAI-style models are untouched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", PASSTHROUGH_MODELS)
async def test_per_turn_instructions_pass_through_for_openai_style_models(model: str) -> None:
    messages = await _sent_messages(model, _per_turn_ctx())

    assert [m["role"] for m in messages] == ["system", "assistant", "user", "system"]
    assert messages[0] == {"role": "system", "content": PREAMBLE}
    assert messages[-1] == {"role": "system", "content": INSTRUCTIONS}


async def test_tool_history_passes_through_for_openai_style_models() -> None:
    messages = await _sent_messages("openai/gpt-oss-120b", _tool_ctx())

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "system"]
    assert messages[3] == {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}


async def test_conversation_without_mid_system_messages_is_identical_across_models() -> None:
    # when there is nothing to inline, both code paths must produce the same request
    inlined = await _sent_messages("google/gemma-4-31B-it", _plain_ctx())
    passthrough = await _sent_messages("openai/gpt-oss-120b", _plain_ctx())

    assert inlined == passthrough
    assert [m["role"] for m in inlined] == ["system", "user", "assistant", "user"]


# ---------------------------------------------------------------------------
# Explicit override
# ---------------------------------------------------------------------------


async def test_override_enables_inlining_for_unrecognised_model() -> None:
    messages = await _sent_messages("my-dedicated-model", _per_turn_ctx(), inline=True)

    assert [m["role"] for m in messages] == ["system", "assistant", "user", "user"]
    assert messages[-1] == {"role": "user", "content": INLINED}


async def test_override_disables_inlining_for_gemma() -> None:
    messages = await _sent_messages("google/gemma-4-31B-it", _per_turn_ctx(), inline=False)

    assert [m["role"] for m in messages] == ["system", "assistant", "user", "system"]
    assert messages[-1] == {"role": "system", "content": INSTRUCTIONS}
