from __future__ import annotations

import pytest

from livekit.agents.llm import ChatContext
from livekit.plugins.cerebras import LLM as CerebrasLLM
from livekit.plugins.openai import LLM as OpenAILLM

pytestmark = pytest.mark.unit


def _chat_ctx() -> ChatContext:
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="hi")
    return chat_ctx


@pytest.mark.asyncio
async def test_cerebras_reasoning_format_in_request() -> None:
    """``reasoning_format`` is forwarded to the request body via ``extra_body``."""
    llm = CerebrasLLM(
        model="gpt-oss-120b",
        api_key="test-key",
        reasoning_format="hidden",
        gzip_compression=False,
        msgpack_encoding=False,
    )
    stream = llm.chat(chat_ctx=_chat_ctx())
    try:
        extra_body = stream._extra_kwargs.get("extra_body", {})
        assert extra_body.get("reasoning_format") == "hidden"
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_cerebras_reasoning_format_omitted_by_default() -> None:
    """No ``reasoning_format`` is sent when the option is not set."""
    llm = CerebrasLLM(
        model="gpt-oss-120b",
        api_key="test-key",
        gzip_compression=False,
        msgpack_encoding=False,
    )
    stream = llm.chat(chat_ctx=_chat_ctx())
    try:
        extra_body = stream._extra_kwargs.get("extra_body", {})
        assert "reasoning_format" not in extra_body
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_openai_with_cerebras_reasoning_format_in_request() -> None:
    """``LLM.with_cerebras`` forwards ``reasoning_format`` to the request body."""
    llm = OpenAILLM.with_cerebras(
        model="gpt-oss-120b",
        api_key="test-key",
        reasoning_format="hidden",
    )
    stream = llm.chat(chat_ctx=_chat_ctx())
    try:
        extra_body = stream._extra_kwargs.get("extra_body", {})
        assert extra_body.get("reasoning_format") == "hidden"
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_xai_reasoning_format_in_request() -> None:
    """``LLM.with_x_ai`` forwards ``reasoning_format`` to the request body."""
    llm = OpenAILLM.with_x_ai(
        model="grok-4-1-fast-reasoning",
        api_key="test-key",
        reasoning_format="parsed",
    )
    stream = llm.chat(chat_ctx=_chat_ctx())
    try:
        extra_body = stream._extra_kwargs.get("extra_body", {})
        assert extra_body.get("reasoning_format") == "parsed"
    finally:
        await stream.aclose()
