from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit.agents.inference.llm import LLMStream
from livekit.agents.llm import ChatContext
from livekit.agents.types import APIConnectOptions

pytestmark = pytest.mark.unit


class _FakeOpenAIStream:
    """Stands in for the openai streaming response: async context manager + async iterator."""

    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self._chunks = chunks

    async def __aenter__(self) -> _FakeOpenAIStream:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


def _usage_chunk(
    *,
    completion_tokens: int | None,
    prompt_tokens: int | None,
    total_tokens: int | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id="chunk-id",
        choices=[],
        usage=SimpleNamespace(
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
            prompt_tokens_details=None,
        ),
    )


def _make_stream(chunk: SimpleNamespace) -> LLMStream:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_FakeOpenAIStream([chunk]))

    llm_v = MagicMock()
    llm_v._label = "test-llm"
    llm_v.model = "gpt-4o-mini"
    llm_v.provider = "openai"

    return LLMStream(
        llm_v,
        model="gpt-4o-mini",
        strict_tool_schema=False,
        client=client,
        chat_ctx=ChatContext.empty(),
        tools=[],
        conn_options=APIConnectOptions(max_retry=0, timeout=5.0),
        extra_kwargs={},
    )


@pytest.mark.asyncio
async def test_null_completion_tokens_does_not_crash_stream() -> None:
    """A provider that omits completion_tokens must not kill the session."""
    stream = _make_stream(_usage_chunk(completion_tokens=None, prompt_tokens=7, total_tokens=7))

    chunks = [chunk async for chunk in stream]

    usages = [chunk.usage for chunk in chunks if chunk.usage is not None]
    assert len(usages) == 1
    assert usages[0].completion_tokens == 0
    assert usages[0].prompt_tokens == 7
    assert usages[0].total_tokens == 7


@pytest.mark.asyncio
async def test_null_prompt_and_total_tokens_are_coerced() -> None:
    stream = _make_stream(_usage_chunk(completion_tokens=5, prompt_tokens=None, total_tokens=None))

    chunks = [chunk async for chunk in stream]

    usages = [chunk.usage for chunk in chunks if chunk.usage is not None]
    assert len(usages) == 1
    assert usages[0].completion_tokens == 5
    assert usages[0].prompt_tokens == 0
    assert usages[0].total_tokens == 0


@pytest.mark.asyncio
async def test_complete_usage_is_passed_through() -> None:
    stream = _make_stream(_usage_chunk(completion_tokens=11, prompt_tokens=13, total_tokens=24))

    chunks = [chunk async for chunk in stream]

    usages = [chunk.usage for chunk in chunks if chunk.usage is not None]
    assert len(usages) == 1
    assert usages[0].completion_tokens == 11
    assert usages[0].prompt_tokens == 13
    assert usages[0].total_tokens == 24
