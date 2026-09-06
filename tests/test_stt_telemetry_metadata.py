"""STT telemetry uses the optional metadata hook, not configured identity."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from livekit.agents import Agent, AgentSession
from livekit.agents.llm.chat_context import MetricsMetadata
from livekit.agents.metrics import STTMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.stt import (
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
