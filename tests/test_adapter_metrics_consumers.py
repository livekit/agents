from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from livekit.agents import APIConnectionError, APIConnectOptions
from livekit.agents.llm import ChatContext, FallbackAdapter as LLMFallbackAdapter
from livekit.agents.tts import FallbackAdapter as TTSFallbackAdapter, StreamAdapter

from .fake_llm import FakeLLM, FakeLLMResponse
from .fake_tts import FakeTTS

pytestmark = pytest.mark.unit


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
