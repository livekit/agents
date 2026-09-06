"""STT telemetry uses the optional metadata hook, not configured identity."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from livekit.agents import Agent, AgentSession, APIConnectionError
from livekit.agents.llm.chat_context import MetricsMetadata
from livekit.agents.metrics import STTMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.stt import (
    STT,
    FallbackAdapter,
    MultiSpeakerAdapter,
    RecognizeStream,
    SpeechEvent,
    SpeechEventType,
    StreamAdapter,
)
from livekit.agents.stt.stt import RecognitionUsage
from livekit.agents.telemetry import trace_types
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils.audio import silence_frame
from livekit.agents.voice import audio_recognition
from livekit.agents.voice.amd import detector as amd_detector
from livekit.agents.voice.events import ErrorEvent

from .fake_llm import FakeLLM
from .fake_realtime import FakeRealtimeModel
from .fake_stt import FakeSTT
from .fake_tts import FakeTTS
from .fake_vad import FakeVAD

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

METADATA: list[MetricsMetadata | None] = [
    None,
    {"model_name": "safe-model", "model_provider": "safe-provider"},
    {"model_name": "safe-model"},
    {"model_provider": "safe-provider"},
    {},
]


class _IdentifiedSTT(FakeSTT):
    @property
    def model(self) -> str:
        return "configured-private-model"

    @property
    def provider(self) -> str:
        return "configured-private-provider"


class _MetadataSTT(_IdentifiedSTT):
    def __init__(self, metadata: MetricsMetadata) -> None:
        super().__init__(fake_transcript="hello")
        self._metadata = metadata

    @property
    def metrics_metadata(self) -> MetricsMetadata:
        return self._metadata


def _provider(metadata: MetricsMetadata | None) -> _IdentifiedSTT:
    if metadata is None:
        return _IdentifiedSTT(fake_transcript="hello")
    return _MetadataSTT(metadata)


class _MetricsStream(RecognizeStream):
    async def _run(self) -> None:
        self._report_connection_acquired(acquire_time=0.25, connection_reused=True)
        self._event_ch.send_nowait(
            SpeechEvent(
                type=SpeechEventType.RECOGNITION_USAGE,
                recognition_usage=RecognitionUsage(audio_duration=0.1),
            )
        )


@pytest.mark.parametrize("metadata", METADATA)
@pytest.mark.parametrize("sink", ["batch", "connection", "usage"])
async def test_core_metrics_use_hook(metadata: MetricsMetadata | None, sink: str) -> None:
    provider = _provider(metadata)
    expected = Metadata(**provider.metrics_metadata)
    events: list[STTMetrics] = []
    provider.on("metrics_collected", events.append)
    if sink == "batch":
        await provider.recognize(silence_frame(duration=0.1, sample_rate=16000))
        assert len(events) == 1
        metric = events[0]
        assert not metric.streamed
    else:
        stream = _MetricsStream(stt=provider, conn_options=DEFAULT_API_CONNECT_OPTIONS)
        try:
            await asyncio.wait_for(stream._task, timeout=5)
            await asyncio.wait_for(stream._metrics_task, timeout=5)
            assert len(events) == 2
            metric = next(
                event for event in events if (event.audio_duration == 0) == (sink == "connection")
            )
            assert metric.streamed
        finally:
            await stream.aclose()
    assert metric.metadata == expected
    if metadata is not None:
        assert provider.model not in metric.model_dump_json()
        assert provider.provider not in metric.model_dump_json()


@pytest.mark.parametrize("metadata", METADATA)
@pytest.mark.parametrize("wrapper", ["stream", "multi-speaker", "fallback"])
async def test_wrappers_delegate_metadata_without_changing_identity(
    metadata: MetricsMetadata | None, wrapper: str
) -> None:
    provider = _provider(metadata)
    provider._capabilities.diarization = True
    if wrapper == "stream":
        wrapped = StreamAdapter(stt=provider, vad=FakeVAD())
    elif wrapper == "multi-speaker":
        wrapped = MultiSpeakerAdapter(stt=provider)
    else:
        wrapped = FallbackAdapter([provider])
    try:
        assert wrapped.model == ("FallbackAdapter" if wrapper == "fallback" else provider.model)
        assert wrapped.provider == ("livekit" if wrapper == "fallback" else provider.provider)
        assert wrapped.metrics_metadata == provider.metrics_metadata
        source = ErrorEvent(error=RuntimeError("safe error"), source=wrapped).model_dump()["source"]
        assert source == {
            "model": provider.metrics_metadata.get("model_name"),
            "provider": provider.metrics_metadata.get("model_provider"),
        }
    finally:
        await wrapped.aclose()


@pytest.mark.parametrize("metadata", METADATA)
async def test_stt_error_serialization_uses_hook(metadata: MetricsMetadata | None) -> None:
    provider = _provider(metadata)
    event = ErrorEvent(error=RuntimeError("safe error"), source=provider)
    assert event.model_dump()["source"] == {
        "model": provider.metrics_metadata.get("model_name"),
        "provider": provider.metrics_metadata.get("model_provider"),
    }
    if metadata is not None:
        assert provider.model not in event.model_dump_json()
        assert provider.provider not in event.model_dump_json()


async def test_other_error_sources_keep_existing_serialization() -> None:
    for provider in (FakeLLM(), FakeTTS(), FakeRealtimeModel()):
        event = ErrorEvent(error=RuntimeError("safe error"), source=provider)
        assert event.model_dump()["source"] == {
            "model": provider.model,
            "provider": provider.provider,
        }

    class Source(BaseModel):
        name: str = "custom"

    assert ErrorEvent(error=None, source=Source()).model_dump()["source"] == {"name": "custom"}
    assert ErrorEvent(error=None, source=42).model_dump()["source"] == "42"


@pytest.mark.parametrize("metadata", METADATA)
@pytest.mark.parametrize("phase", ["initial", "update"])
async def test_voice_trace_initialization_and_update_use_hook(
    metadata: MetricsMetadata | None, phase: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _provider(metadata)
    initial = (
        provider
        if phase == "initial"
        else _MetadataSTT({"model_name": "old-safe-model", "model_provider": "old-safe-provider"})
    )
    agent = Agent(instructions="test", stt=initial, vad=FakeVAD(), llm=FakeLLM(), tts=FakeTTS())
    session = AgentSession(turn_handling={"turn_detection": None})
    await session.start(agent)
    try:
        if phase == "update":
            agent.update_options(stt=provider)
        assert session._activity is not None
        recognition = session._activity._audio_recognition
        assert recognition is not None
        recognition._end_user_turn_span()
        tracer = MagicMock()
        monkeypatch.setattr(audio_recognition, "tracer", tracer)
        recognition._ensure_user_turn_span()
        attrs = dict(
            call.args for call in tracer.start_span.return_value.set_attribute.call_args_list
        )
        assert attrs.get(trace_types.ATTR_GEN_AI_REQUEST_MODEL) == provider.metrics_metadata.get(
            "model_name"
        )
        assert attrs.get(trace_types.ATTR_GEN_AI_PROVIDER_NAME) == provider.metrics_metadata.get(
            "model_provider"
        )
        recognition._end_user_turn_span()
    finally:
        await session.aclose()


@pytest.mark.parametrize("metadata", METADATA[1:])
@pytest.mark.parametrize("dev_mode", [False, True])
async def test_amd_warning_redacts_identity_but_keeps_compatibility_comparison(
    metadata: MetricsMetadata,
    dev_mode: bool,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("LIVEKIT_DEV_MODE", "1" if dev_mode else "0")
    monkeypatch.delenv("LIVEKIT_REMOTE_EOT_URL", raising=False)
    for name in (
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "LIVEKIT_INFERENCE_API_KEY",
        "LIVEKIT_INFERENCE_API_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    provider = _MetadataSTT(metadata)
    session = AgentSession(turn_handling={"turn_detection": None})
    # Only AMD's compatibility check is under test, not session setup warnings.
    caplog.clear()
    monkeypatch.setattr(amd_detector, "EVALUATED_STT_MODELS", {provider.model})
    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        amd_detector.AMD(session, llm=FakeLLM(), stt=provider)
    assert not caplog.records

    monkeypatch.setattr(amd_detector, "EVALUATED_STT_MODELS", set())
    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        amd_detector.AMD(session, llm=FakeLLM(), stt=provider)
    assert "hasn't been evaluated" in caplog.text
    assert provider.model not in caplog.text
    assert (metadata.get("model_name") or "unknown") in caplog.text


def test_amd_llm_warning_keeps_existing_identity(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        amd_detector._warn_if_not_evaluated("configured-llm", set(), model_kind="llm")
    assert "configured-llm" in caplog.text


@pytest.mark.parametrize("metadata", METADATA)
@pytest.mark.parametrize("phase", ["initial", "update"])
@pytest.mark.parametrize(
    ("wrapper", "traffic"),
    [
        ("none", "batch"),
        ("none", "stream"),
        ("stream", "batch"),
        ("stream", "stream"),
        ("multi-speaker", "batch"),
        ("multi-speaker", "stream"),
        ("non-streaming", "batch"),
    ],
)
async def test_voice_trace_follows_real_fallback_transition(
    metadata: MetricsMetadata | None,
    phase: str,
    wrapper: str,
    traffic: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = _MetadataSTT(
        {"model_name": "primary-safe-model", "model_provider": "primary-safe-provider"}
    )
    primary.update_options(fake_transcript=None)
    secondary = _provider(metadata)
    primary._capabilities.diarization = secondary._capabilities.diarization = True
    if wrapper == "non-streaming":
        primary._capabilities.streaming = secondary._capabilities.streaming = False
    fallback = FallbackAdapter([primary, secondary], vad=FakeVAD(), max_retry_per_stt=0)
    wrapped: STT = fallback
    if wrapper == "stream":
        wrapped = StreamAdapter(stt=fallback, vad=FakeVAD())
    elif wrapper == "multi-speaker":
        wrapped = MultiSpeakerAdapter(stt=fallback)

    old = _MetadataSTT({"model_name": "old-safe-model", "model_provider": "old-safe-provider"})
    agent = Agent(
        instructions="test",
        stt=wrapped if phase == "initial" else old,
        vad=FakeVAD(),
        llm=FakeLLM(),
        tts=FakeTTS(),
    )
    session = AgentSession(turn_handling={"turn_detection": None})
    exporter = InMemorySpanExporter()
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(exporter))
    await session.start(agent)
    try:
        assert session._activity is not None
        recognition = session._activity._audio_recognition
        assert recognition is not None
        recognition._end_user_turn_span()
        monkeypatch.setattr(audio_recognition, "tracer", trace_provider.get_tracer(__name__))

        def assert_fresh_trace(expected: MetricsMetadata) -> None:
            # Background STT events may have started a span during the request.
            recognition._end_user_turn_span()
            recognition._ensure_user_turn_span()
            recognition._end_user_turn_span()
            spans = exporter.get_finished_spans()
            attrs = spans[-1].attributes
            assert spans[-1].name == "user_turn"
            assert attrs is not None
            assert attrs.get(trace_types.ATTR_GEN_AI_REQUEST_MODEL) == expected.get("model_name")
            assert attrs.get(trace_types.ATTR_GEN_AI_PROVIDER_NAME) == expected.get(
                "model_provider"
            )
            assert len({span.context.span_id for span in spans}) == len(spans)
            if metadata is not None:
                assert secondary.model not in str(attrs)
                assert secondary.provider not in str(attrs)

        if phase == "update":
            assert_fresh_trace(old.metrics_metadata)
            agent.update_options(stt=wrapped)
            old._metadata = {"model_name": "detached-old-model"}
        assert_fresh_trace(primary.metrics_metadata)

        primary.update_options(fake_exception=APIConnectionError("primary unavailable"))
        if traffic == "batch":
            event = await asyncio.wait_for(fallback.recognize([]), timeout=5)
            assert event.alternatives[0].text == "hello"
        else:

            async def drain_stream() -> list[SpeechEvent]:
                async with fallback.stream() as stream:
                    stream.end_input()
                    return [event async for event in stream]

            events = await asyncio.wait_for(drain_stream(), timeout=5)
            assert any(event.alternatives[0].text == "hello" for event in events)

        # Public traffic, not assignment to _active_instance, caused the transition.
        assert fallback.metrics_metadata == secondary.metrics_metadata
        assert wrapped.metrics_metadata == secondary.metrics_metadata
        assert_fresh_trace(secondary.metrics_metadata)

        # An adapter may change its labels again without being replaced.
        if isinstance(secondary, _MetadataSTT):
            secondary._metadata = {}
            assert_fresh_trace({})
        agent.update_options(stt=None)
        assert_fresh_trace({})
        # Disabling STT queues pipeline cleanup; finish it before closing the session.
        await asyncio.wait_for(asyncio.gather(*recognition._tasks), timeout=5)
    finally:
        await session.aclose()
        await wrapped.aclose()
        if wrapped is not fallback:
            await fallback.aclose()
        trace_provider.shutdown()


async def test_voice_trace_keeps_legacy_static_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    session = AgentSession(turn_handling={"turn_detection": None})
    recognition = audio_recognition.AudioRecognition(
        session,
        hooks=MagicMock(),
        endpointing=MagicMock(),
        stt=None,
        vad=None,
        interruption_detection=None,
        turn_detection=None,
        stt_model="legacy-model",
        stt_provider="legacy-provider",
    )
    exporter = InMemorySpanExporter()
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(audio_recognition, "tracer", trace_provider.get_tracer(__name__))
    try:
        recognition._ensure_user_turn_span()
        recognition._end_user_turn_span()
        first = exporter.get_finished_spans()[-1].attributes
        assert first[trace_types.ATTR_GEN_AI_REQUEST_MODEL] == "legacy-model"
        assert first[trace_types.ATTR_GEN_AI_PROVIDER_NAME] == "legacy-provider"
        recognition._update_stt(None, model="updated-legacy-model", provider=None)
        recognition._ensure_user_turn_span()
        recognition._end_user_turn_span()
        second = exporter.get_finished_spans()[-1].attributes
        assert second[trace_types.ATTR_GEN_AI_REQUEST_MODEL] == "updated-legacy-model"
        assert trace_types.ATTR_GEN_AI_PROVIDER_NAME not in second
    finally:
        recognition._end_user_turn_span()
        await session.aclose()
        trace_provider.shutdown()
