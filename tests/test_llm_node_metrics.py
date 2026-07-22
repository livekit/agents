from __future__ import annotations

import asyncio

import pytest

from livekit.agents.llm import ChatChunk, ChatContext, ChoiceDelta, CompletionUsage
from livekit.agents.llm.tool_context import ToolContext
from livekit.agents.tokenize import blingfire
from livekit.agents.utils import aio
from livekit.agents.voice.agent import ModelSettings
from livekit.agents.voice.generation import (
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


async def _run_inference(
    chunks: list[ChatChunk],
    sentence_tokenizer: blingfire.SentenceTokenizer | None = None,
) -> _LLMGenerationData:
    data = _LLMGenerationData(text_ch=aio.Chan(), function_ch=aio.Chan())
    await _llm_inference_task(
        _fake_node(chunks),
        ChatContext.empty(),
        ToolContext.empty(),
        ModelSettings(),
        data,
        sentence_tokenizer=sentence_tokenizer,
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


class _RecordingTokenizer(blingfire.SentenceTokenizer):
    """Counts stream() calls so tests can assert the caller-provided tokenizer is used."""

    def __init__(self) -> None:
        super().__init__(min_sentence_len=1)
        self.stream_calls = 0

    def stream(self, *, language: str | None = None):  # type: ignore[no-untyped-def]
        self.stream_calls += 1
        return super().stream(language=language)


class TestLLMNodeTtfs:
    async def test_ttfs_set_for_nonempty_generation(self) -> None:
        data = await _run_inference([_content("First sentence. "), _content("Second one.")])
        assert data.ttfs is not None
        assert data.ttfs > 0

    async def test_ttfs_none_for_whitespace_only(self) -> None:
        data = await _run_inference([_content("   ")])
        assert data.ttfs is None

    async def test_ttfs_uses_provided_tokenizer(self) -> None:
        # the tokenizer must be resolved from the actual TTS path
        tokenizer = _RecordingTokenizer()
        data = await _run_inference(
            [_content("First sentence here, quite long. "), _content("Second one.")],
            sentence_tokenizer=tokenizer,
        )
        assert tokenizer.stream_calls == 1
        assert data.ttfs is not None


class TestTtsSentenceTokenizerResolution:
    """`AgentActivity._tts_sentence_tokenizer` is the single source of truth the default
    `tts_node` and the ttfs metric both read, so the segmentation ttfs times matches what
    TTS actually applies."""

    async def test_reuses_live_stream_adapter_instance(self) -> None:
        # a StreamAdapter already owns the tokenizer TTS will run; reuse that exact
        # instance rather than building a parallel one that could drift from it.
        from types import SimpleNamespace

        from livekit.agents import tts
        from livekit.agents.voice.agent_activity import AgentActivity

        from .fake_tts import FakeTTS

        adapter = tts.StreamAdapter(tts=FakeTTS())
        activity = SimpleNamespace(tts=adapter, _resolve_expressive_options=lambda: None)
        resolved = AgentActivity._tts_sentence_tokenizer(activity)  # type: ignore[arg-type]
        assert resolved is adapter._sentence_tokenizer
