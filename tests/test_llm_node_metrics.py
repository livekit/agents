from __future__ import annotations

import asyncio

import pytest

from livekit.agents.llm import ChatChunk, ChatContext, ChoiceDelta, CompletionUsage
from livekit.agents.llm.tool_context import ToolContext
from livekit.agents.tokenize import blingfire
from livekit.agents.utils import aio
from livekit.agents.voice.agent import ModelSettings
from livekit.agents.voice.generation import (
    _SENTENCE_TOKENIZER,
    _llm_inference_task,
    _LLMGenerationData,
)

pytestmark = pytest.mark.unit


def _fake_node(chunks: list[ChatChunk]):
    # matches the io.LLMNode signature: (chat_ctx, tools, model_settings) -> AsyncIterable
    async def node(chat_ctx, tools, model_settings):  # type: ignore[no-untyped-def]
        for chunk in chunks:
            await asyncio.sleep(0)  # yield so the ttfs consumer task can run
            yield chunk

    return node


async def _run_inference(chunks: list[ChatChunk]) -> _LLMGenerationData:
    data = _LLMGenerationData(text_ch=aio.Chan(), function_ch=aio.Chan())
    await _llm_inference_task(
        _fake_node(chunks),
        ChatContext.empty(),
        ToolContext.empty(),
        ModelSettings(),
        data,
    )
    return data


def _content(text: str) -> ChatChunk:
    return ChatChunk(id="c", delta=ChoiceDelta(content=text))


def _usage_chunk(completion_tokens: int) -> ChatChunk:
    return ChatChunk(
        id="c",
        usage=CompletionUsage(
            completion_tokens=completion_tokens,
            prompt_tokens=5,
            total_tokens=completion_tokens + 5,
        ),
    )


class TestLLMNodeTps:
    async def test_tps_set_when_usage_reported(self) -> None:
        data = await _run_inference([_content("Hello there, friend."), _usage_chunk(30)])
        assert data.tps is not None
        assert data.tps > 0

    async def test_tps_zero_when_zero_usage_is_reported(self) -> None:
        data = await _run_inference([_content("Hello there, friend."), _usage_chunk(0)])
        assert data.tps == 0

    async def test_tps_none_when_no_usage_reported(self) -> None:
        data = await _run_inference([_content("Hello there, friend.")])  # no usage chunk
        assert data.tps is None


class TestLLMNodeTtfs:
    async def test_ttfs_set_for_nonempty_generation(self) -> None:
        data = await _run_inference([_content("First sentence. "), _content("Second one.")])
        assert data.ttfs is not None
        assert data.ttfs > 0

    async def test_ttfs_none_for_whitespace_only(self) -> None:
        data = await _run_inference([_content("   ")])
        assert data.ttfs is None


class TestSentenceTokenizerConfig:
    def test_short_first_sentence_is_counted(self) -> None:
        # regression: ttfs must see short openers. blingfire's TTS-oriented default
        # (min_sentence_len=20) drops them, which silently inflated ttfs to the whole turn.
        text = "Hi there. How can I help?"
        assert len(_SENTENCE_TOKENIZER.tokenize(text)) == 2
