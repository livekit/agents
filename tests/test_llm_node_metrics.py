from __future__ import annotations

import asyncio

import pytest

from livekit.agents.llm import ChatChunk, ChatContext, ChoiceDelta, CompletionUsage
from livekit.agents.llm.tool_context import ToolContext
from livekit.agents.utils import aio
from livekit.agents.voice.agent import ModelSettings
from livekit.agents.voice.generation import (
    _llm_inference_task,
    _LLMGenerationData,
    _time_to_first_sentence,
    _TTSGenerationData,
)

pytestmark = pytest.mark.unit


def _fake_node(chunks: list[ChatChunk]):
    # matches the io.LLMNode signature: (chat_ctx, tools, model_settings) -> AsyncIterable
    async def node(chat_ctx, tools, model_settings):  # type: ignore[no-untyped-def]
        for chunk in chunks:
            await asyncio.sleep(0)
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


def _tts_data(synthesis_started_at: float | None) -> _TTSGenerationData:
    return _TTSGenerationData(
        audio_ch=aio.Chan(),
        timed_texts_fut=asyncio.Future(),
        synthesis_started_at=synthesis_started_at,
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


class TestLLMNodeStartedAt:
    async def test_started_at_recorded(self) -> None:
        # ttfs is measured from here, so the inference task must always stamp it
        data = await _run_inference([_content("Hello there, friend.")])
        assert data.started_at is not None


class TestTimeToFirstSentence:
    """ttfs is read from the instant the TTS stream handed its first segment to the
    provider, never reconstructed by re-tokenizing the LLM stream, so it can't drift from
    the segmentation the TTS actually applies (custom plugin tokenizer, inference gateway
    per-provider defaults, StreamAdapter, or provider-side splitting)."""

    async def test_measured_from_llm_start_to_synthesis_start(self) -> None:
        llm_data = _LLMGenerationData(text_ch=aio.Chan(), function_ch=aio.Chan(), started_at=100.0)
        assert _time_to_first_sentence(llm_data, _tts_data(102.5)) == pytest.approx(2.5)

    async def test_none_without_tts(self) -> None:
        # text-only session: nothing was ever segmented for synthesis
        llm_data = _LLMGenerationData(text_ch=aio.Chan(), function_ch=aio.Chan(), started_at=100.0)
        assert _time_to_first_sentence(llm_data, None) is None

    async def test_none_when_tts_published_no_stamp(self) -> None:
        # frames that never came from a LiveKit TTS carry no started-time userdata (a
        # tts_node synthesizing its own audio); ttfs stays unreported rather than falling
        # back to a different anchor the way ttfb does
        llm_data = _LLMGenerationData(text_ch=aio.Chan(), function_ch=aio.Chan(), started_at=100.0)
        assert _time_to_first_sentence(llm_data, _tts_data(None)) is None

    async def test_none_when_llm_never_started(self) -> None:
        llm_data = _LLMGenerationData(text_ch=aio.Chan(), function_ch=aio.Chan())
        assert _time_to_first_sentence(llm_data, _tts_data(102.5)) is None
