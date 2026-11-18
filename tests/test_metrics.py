"""Metric computation: end-of-turn delays, LLM tokens per second and ttfs, usage aggregation,
and the metrics consumer that adapter streams tee off."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from livekit.agents import APIConnectionError, APIConnectOptions
from livekit.agents.llm import (
    ChatChunk,
    ChatContext,
    ChoiceDelta,
    CompletionUsage,
    FallbackAdapter as LLMFallbackAdapter,
)
from livekit.agents.llm.tool_context import ToolContext
from livekit.agents.metrics import (
    LLMMetrics,
    LLMModelUsage,
    ModelUsageCollector,
    STTMetrics,
    STTModelUsage,
)
from livekit.agents.metrics.base import Metadata
from livekit.agents.tts import FallbackAdapter as TTSFallbackAdapter, StreamAdapter
from livekit.agents.utils import aio
from livekit.agents.voice.agent import ModelSettings
from livekit.agents.voice.audio_recognition import _compute_end_of_turn_metrics
from livekit.agents.voice.generation import (
    _llm_inference_task,
    _LLMGenerationData,
    _time_to_first_sentence,
    _TTSGenerationData,
)

from .fake_llm import FakeLLM, FakeLLMResponse
from .fake_tts import FakeTTS

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time]


# -- end of turn --


def test_end_of_turn_delays_are_measured_from_the_anchors() -> None:
    metrics = _compute_end_of_turn_metrics(
        speech_start_time=1000.0,
        last_speaking_time=1005.0,
        last_final_transcript_time=1005.2,
        now=1005.4,
    )
    assert metrics.started_speaking_at == 1000.0
    assert metrics.stopped_speaking_at == 1005.0
    assert metrics.transcription_delay == pytest.approx(0.2)
    assert metrics.end_of_turn_delay == pytest.approx(0.4)


def test_anchor_equal_to_start_is_accepted() -> None:
    metrics = _compute_end_of_turn_metrics(
        speech_start_time=2000.0,
        last_speaking_time=2000.0,
        last_final_transcript_time=2000.0,
        now=2000.3,
    )
    assert metrics.stopped_speaking_at == 2000.0
    assert metrics.transcription_delay == 0.0
    assert metrics.end_of_turn_delay == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("speech_start_time", "last_speaking_time", "last_final_transcript_time", "now"),
    [
        # a stale anchor from an earlier segment, ~220 s before the turn started
        (1781342804.815377, 1781342584.6181495, 1781342804.9027314, 1781342804.9027314),
        (None, 1005.0, 1005.2, 1006.0),
        (1000.0, None, 1005.2, 1006.0),
        (1000.0, 1005.0, None, 1006.0),
    ],
    ids=["stale_anchor", "no_start", "no_stop", "no_final"],
)
def test_unreliable_anchors_skip_the_metrics(
    speech_start_time: float | None,
    last_speaking_time: float | None,
    last_final_transcript_time: float | None,
    now: float,
) -> None:
    metrics = _compute_end_of_turn_metrics(
        speech_start_time=speech_start_time,
        last_speaking_time=last_speaking_time,
        last_final_transcript_time=last_final_transcript_time,
        now=now,
    )
    assert metrics.started_speaking_at is None
    assert metrics.stopped_speaking_at is None
    assert metrics.transcription_delay is None
    assert metrics.end_of_turn_delay is None


# -- llm node --


async def _run_inference(*chunks: ChatChunk) -> _LLMGenerationData:
    async def node(chat_ctx, tools, model_settings):  # type: ignore[no-untyped-def]
        for chunk in chunks:
            await asyncio.sleep(0)
            yield chunk

    data = _LLMGenerationData(text_ch=aio.Chan(), function_ch=aio.Chan())
    await _llm_inference_task(node, ChatContext.empty(), ToolContext.empty(), ModelSettings(), data)
    return data


def _content(text: str) -> ChatChunk:
    return ChatChunk(id="c", delta=ChoiceDelta(content=text))


def _usage(completion_tokens: int) -> ChatChunk:
    return ChatChunk(
        id="c",
        usage=CompletionUsage(
            completion_tokens=completion_tokens,
            prompt_tokens=5,
            total_tokens=completion_tokens + 5,
        ),
    )


async def test_tps_set_when_usage_reported() -> None:
    data = await _run_inference(_content("Hello there, "), _content("friend."), _usage(30))
    assert data.tps is not None and data.tps > 0
    # ttfs is measured from here, so the inference task must always stamp it
    assert data.started_at is not None


async def test_tps_zero_when_zero_usage_is_reported() -> None:
    data = await _run_inference(_content("Hello there, "), _content("friend."), _usage(0))
    assert data.tps == 0


async def test_tps_none_when_no_usage_reported() -> None:
    data = await _run_inference(_content("Hello there, friend."))
    assert data.tps is None


async def test_tps_none_when_reply_arrived_in_one_chunk() -> None:
    # nothing streamed, so there is no rate: dividing by prefill or a round trip would invent one
    data = await _run_inference(_content("Sure!"), _usage(3))
    assert data.tps is None


async def test_tps_excludes_time_to_first_token() -> None:
    # 10 tokens over a 50 ms stream after a 200 ms prefill: ~200 tok/s, not ~40
    async def slow_first_token(chat_ctx, tools, model_settings):  # type: ignore[no-untyped-def]
        await asyncio.sleep(0.2)
        yield _content("Hello there, ")
        await asyncio.sleep(0.05)
        yield _content("friend.")
        yield _usage(10)

    data = _LLMGenerationData(text_ch=aio.Chan(), function_ch=aio.Chan())
    await _llm_inference_task(
        slow_first_token, ChatContext.empty(), ToolContext.empty(), ModelSettings(), data
    )
    assert data.tps is not None
    assert data.tps > 100


@pytest.mark.parametrize(
    ("llm_started_at", "with_tts", "synthesis_started_at", "expected"),
    [
        (100.0, True, 102.5, 2.5),
        # text-only session: nothing was ever segmented for synthesis
        (100.0, False, None, None),
        # audio from a custom tts_node carries no synthesis stamp; no other anchor is used
        (100.0, True, None, None),
        (None, True, 102.5, None),
    ],
    ids=["measured", "no_tts", "no_stamp", "llm_not_started"],
)
async def test_time_to_first_sentence(
    llm_started_at: float | None,
    with_tts: bool,
    synthesis_started_at: float | None,
    expected: float | None,
) -> None:
    """ttfs runs from LLM start to the instant the TTS handed its first segment to the provider."""
    llm_data = _LLMGenerationData(
        text_ch=aio.Chan(), function_ch=aio.Chan(), started_at=llm_started_at
    )
    tts_data = (
        _TTSGenerationData(
            audio_ch=aio.Chan(),
            timed_texts_fut=asyncio.Future(),
            synthesis_started_at=synthesis_started_at,
        )
        if with_tts
        else None
    )
    ttfs = _time_to_first_sentence(llm_data, tts_data)
    assert ttfs == (pytest.approx(expected) if expected is not None else None)


# -- usage aggregation --


def _llm_metrics(**overrides: object) -> LLMMetrics:
    base: dict[str, object] = {
        "label": "test.LLM",
        "request_id": "req-1",
        "timestamp": 0.0,
        "duration": 1.0,
        "ttft": 0.1,
        "cancelled": False,
        "completion_tokens": 10,
        "prompt_tokens": 100,
        "prompt_cached_tokens": 20,
        "total_tokens": 110,
        "tokens_per_second": 10.0,
        "metadata": Metadata(model_provider="anthropic", model_name="claude-sonnet-4"),
    }
    base.update(overrides)
    return LLMMetrics(**base)


def test_llm_metrics_default_optional_token_counts_to_zero() -> None:
    m = _llm_metrics()
    assert m.cache_creation_tokens == 0
    assert m.reasoning_tokens == 0


def test_collector_aggregates_cache_creation_tokens() -> None:
    collector = ModelUsageCollector()
    collector.collect(_llm_metrics(cache_creation_tokens=42))
    collector.collect(_llm_metrics(cache_creation_tokens=8))

    [usage] = collector.flatten()
    assert isinstance(usage, LLMModelUsage)
    assert usage.input_cache_creation_tokens == 50


def test_collector_aggregates_reasoning_tokens() -> None:
    collector = ModelUsageCollector()
    collector.collect(_llm_metrics(completion_tokens=100, reasoning_tokens=64))
    collector.collect(_llm_metrics(completion_tokens=50, reasoning_tokens=8))

    [usage] = collector.flatten()
    assert isinstance(usage, LLMModelUsage)
    assert usage.output_reasoning_tokens == 72
    # reasoning is a subset of the output tokens, never added on top of them
    assert usage.output_tokens == 150


def _stt_metrics(**overrides: object) -> STTMetrics:
    base: dict[str, object] = {
        "label": "test.STT",
        "request_id": "req-1",
        "timestamp": 0.0,
        "duration": 0.0,
        "audio_duration": 1.0,
        "streamed": True,
        "metadata": Metadata(model_provider="mistralai", model_name="voxtral-mini"),
    }
    base.update(overrides)
    return STTMetrics(**base)


def test_stt_total_tokens_defaults_to_input_plus_output() -> None:
    assert _stt_metrics(input_tokens=80, output_tokens=20).total_tokens == 100
    assert _stt_metrics(input_tokens=80, output_tokens=20, total_tokens=99).total_tokens == 99


def test_collector_aggregates_streaming_stt_token_usage() -> None:
    collector = ModelUsageCollector()
    collector.collect(_stt_metrics(input_tokens=80, input_audio_tokens=60, output_tokens=20))
    collector.collect(_stt_metrics(input_tokens=40, input_audio_tokens=30, output_tokens=10))

    [usage] = collector.flatten()
    assert isinstance(usage, STTModelUsage)
    assert usage.input_tokens == 120
    assert usage.input_audio_tokens == 90
    assert usage.output_tokens == 30
    assert usage.audio_duration == 2.0


# -- adapter metrics consumer --


@dataclass
class _Probe:
    adapter: Any
    stream: Any
    metrics: list[Any]


def _record_metrics(adapter: Any) -> list[Any]:
    metrics: list[Any] = []
    adapter.on("metrics_collected", metrics.append)
    return metrics


def _stream_adapter_probe() -> _Probe:
    adapter = StreamAdapter(tts=FakeTTS(fake_audio_duration=1.0))
    metrics = _record_metrics(adapter)
    stream = adapter.stream(conn_options=APIConnectOptions(max_retry=0))
    stream.push_text("hello world.")
    stream.end_input()
    return _Probe(adapter=adapter, stream=stream, metrics=metrics)


def _tts_fallback_chunked_probe() -> _Probe:
    adapter = TTSFallbackAdapter([FakeTTS(fake_audio_duration=1.0)], max_retry_per_tts=0)
    metrics = _record_metrics(adapter)
    stream = adapter.synthesize("hello world.")
    return _Probe(adapter=adapter, stream=stream, metrics=metrics)


def _tts_fallback_stream_probe() -> _Probe:
    adapter = TTSFallbackAdapter([FakeTTS(fake_audio_duration=1.0)], max_retry_per_tts=0)
    metrics = _record_metrics(adapter)
    stream = adapter.stream()
    stream.push_text("hello world.")
    stream.end_input()
    return _Probe(adapter=adapter, stream=stream, metrics=metrics)


def _llm_fallback_probe() -> _Probe:
    response = FakeLLMResponse(
        input="hello",
        content="response " * 20,
        ttft=0.0,
        duration=0.01,
    )
    adapter = LLMFallbackAdapter([FakeLLM(fake_responses=[response])])
    metrics = _record_metrics(adapter)
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content="hello")
    return _Probe(adapter=adapter, stream=adapter.chat(chat_ctx=chat_ctx), metrics=metrics)


PROBES: list[tuple[str, Callable[[], _Probe]]] = [
    ("tts-stream-adapter", _stream_adapter_probe),
    ("tts-fallback-chunked", _tts_fallback_chunked_probe),
    ("tts-fallback-stream", _tts_fallback_stream_probe),
    ("llm-fallback", _llm_fallback_probe),
]


def _tee(stream: Any) -> Any:
    return getattr(stream, "_tee", getattr(stream, "_tee_aiter", None))


@pytest.mark.parametrize(
    "probe_factory", [pytest.param(factory, id=name) for name, factory in PROBES]
)
async def test_unused_metrics_consumer_stays_drained(
    probe_factory: Callable[[], _Probe],
) -> None:
    probe = probe_factory()
    tee = _tee(probe.stream)
    metrics_buffer = tee._buffers[1]
    max_buffered = 0
    event_count = 0

    try:
        async with probe.stream:
            async for _ in probe.stream:
                event_count += 1
                max_buffered = max(max_buffered, len(metrics_buffer))

        assert event_count > 1
        assert max_buffered <= 1
        assert not tee._buffers
        assert len(probe.metrics) == 1
    finally:
        await probe.adapter.aclose()


@pytest.mark.parametrize(
    "probe_factory", [pytest.param(factory, id=name) for name, factory in PROBES]
)
async def test_close_after_first_event_releases_events(
    probe_factory: Callable[[], _Probe],
) -> None:
    probe = probe_factory()
    tee = _tee(probe.stream)

    try:
        await anext(probe.stream)
        await probe.stream.aclose()

        assert probe.stream._event_ch.qsize() == 0
        assert not tee._buffers
    finally:
        await probe.stream.aclose()
        await probe.adapter.aclose()


@pytest.mark.parametrize(
    "probe_factory", [pytest.param(factory, id=name) for name, factory in PROBES]
)
async def test_cancelled_consumer_releases_events(probe_factory: Callable[[], _Probe]) -> None:
    probe = probe_factory()
    tee = _tee(probe.stream)
    received = asyncio.Event()
    blocker = asyncio.Event()

    async def _consume() -> None:
        async for _ in probe.stream:
            received.set()
            await blocker.wait()

    consumer = asyncio.create_task(_consume())
    try:
        await asyncio.wait_for(received.wait(), timeout=5.0)
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer

        await probe.stream.aclose()
        assert probe.stream._event_ch.qsize() == 0
        assert not tee._buffers
    finally:
        consumer.cancel()
        await probe.stream.aclose()
        await probe.adapter.aclose()


async def test_failed_stream_releases_events() -> None:
    adapter = StreamAdapter(
        tts=FakeTTS(
            fake_audio_duration=1.0,
            fake_exception=APIConnectionError("probe failure"),
        )
    )
    stream = adapter.stream(conn_options=APIConnectOptions(max_retry=0))
    tee = _tee(stream)
    event_count = 0
    stream.push_text("hello world.")
    stream.end_input()

    try:
        with pytest.raises(APIConnectionError, match="probe failure"):
            async with stream:
                async for _ in stream:
                    event_count += 1

        assert event_count > 1
        assert stream._event_ch.qsize() == 0
        assert not tee._buffers
    finally:
        await adapter.aclose()
