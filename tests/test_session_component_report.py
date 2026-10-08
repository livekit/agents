from __future__ import annotations

import json
from enum import Enum
from typing import Any
from unittest.mock import MagicMock

import pytest

from livekit.agents import AgentSession, JobContext, inference, stt, tts
from livekit.agents.telemetry.traces import _serialize_session_components
from livekit.agents.types import NOT_GIVEN
from livekit.agents.voice.report import SessionReport

from .fake_stt import FakeSTT
from .fake_tts import FakeTTS
from .fake_vad import FakeVAD

pytestmark = pytest.mark.unit


def _report(session: AgentSession) -> SessionReport:
    ctx = MagicMock(spec=JobContext)
    ctx.job.id = "job-1"
    ctx.job.room.sid = "room-1"
    ctx.job.room.name = "test-room"
    return JobContext.make_session_report(ctx, session)


def test_report_snapshots_inference_settings_without_credentials() -> None:
    vad = inference.VAD(activation_threshold=0.7, min_silence_duration=0.4)
    stt_model = inference.STT(
        "deepgram/nova-3",
        language="en",
        api_key="secret-key",
        api_secret="secret-value",
        base_url="https://private-endpoint.example.com",
        extra_kwargs={"endpointing": 50, "keyterm": ["private-customer"]},
        fallback={
            "model": "cartesia/ink-whisper",
            "extra_kwargs": {"min_volume": 0.2, "callback": "private-callback"},
        },
    )
    tts_model = inference.TTS(
        "cartesia/sonic-3",
        voice="voice-1",
        language="de",
        api_key="secret-key",
        api_secret="secret-value",
        extra_kwargs={"speed": 1.2},
        fallback={
            "model": "inworld/inworld-tts-1.5-max",
            "voice": "fallback-voice",
            "extra_kwargs": {"speaking_rate": 0.9, "api_key": "private-api-key"},
        },
    )
    session = AgentSession(vad=vad, stt=stt_model, tts=tts_model)
    report = _report(session)

    assert set(report.components) == {"vad", "stt", "tts"}
    assert report.components["vad"]["activation_threshold"] == 0.7
    assert report.components["vad"]["min_silence_duration"] == 0.4
    assert report.components["stt"]["model"] == "deepgram/nova-3"
    assert report.components["stt"]["language"] == "en"
    assert report.components["stt"]["extra_kwargs"] == {"endpointing": 50}
    assert report.components["tts"]["voice"] == "voice-1"
    assert report.components["tts"]["language"] == "de"
    assert report.components["tts"]["extra_kwargs"] == {"speed": 1.2}
    assert report.components["stt"]["fallback"] == [
        {"model": "cartesia/ink-whisper", "extra_kwargs": {"min_volume": 0.2}}
    ]
    assert report.components["tts"]["fallback"] == [
        {
            "model": "inworld/inworld-tts-1.5-max",
            "voice": "fallback-voice",
            "extra_kwargs": {"speaking_rate": 0.9},
        }
    ]
    assert report.to_dict()["components"] == report.components

    encoded = json.dumps(report.to_dict())
    for sensitive in (
        "secret-key",
        "secret-value",
        "private-endpoint",
        "private-customer",
        "private-callback",
        "private-api-key",
    ):
        assert sensitive not in encoded

    vad.update_options(activation_threshold=0.9)
    stt_model.update_options(language="fr", extra={"endpointing": 100})
    tts_model.update_options(voice="voice-2", extra_kwargs={"speed": 1.5})
    current = _report(session).components
    assert current["vad"]["activation_threshold"] == 0.9
    assert current["stt"]["language"] == "fr"
    assert current["tts"]["voice"] == "voice-2"
    assert report.components["vad"]["activation_threshold"] == 0.7
    assert report.components["stt"]["language"] == "en"
    assert report.components["tts"]["voice"] == "voice-1"
    assert report.components["tts"]["extra_kwargs"] == {"speed": 1.2}


def test_disabled_components_and_existing_report_constructor() -> None:
    session = AgentSession(vad=None)
    assert _report(session).components == {}
    report = SessionReport(
        job_id="job-1",
        room_id="room-1",
        room="test-room",
        options=session.options,
        events=[],
        chat_history=session.history,
    )
    assert report.to_dict()["components"] == {}


def test_default_session_reports_vad_settings() -> None:
    components = _report(AgentSession()).components
    assert set(components) == {"vad"}
    assert components["vad"]["model"] == "silero"
    assert components["vad"]["activation_threshold"] == 0.5
    assert components["vad"]["sample_rate"] == 16000


def test_custom_component_options_are_normalized() -> None:
    class Mode(Enum):
        FAST = "fast"

    class CustomVAD(FakeVAD):
        def describe_options(self) -> dict[str, Any]:
            return {
                "mode": Mode.FAST,
                "thresholds": [0.3, 0.7],
                "unset": NOT_GIVEN,
                "nested": {"unset": NOT_GIVEN, "instructions": "private-prompt", "enabled": True},
            }

    report = _report(AgentSession(vad=CustomVAD()))
    assert report.components["vad"]["mode"] == "fast"
    assert report.components["vad"]["thresholds"] == [0.3, 0.7]
    assert report.components["vad"]["nested"] == {"enabled": True}
    assert "unset" not in report.components["vad"]
    json.dumps(report.to_dict())


def test_broken_component_description_keeps_identity() -> None:
    class BrokenVAD(FakeVAD):
        def describe_options(self) -> dict[str, Any]:
            raise RuntimeError("unavailable")

    report = _report(AgentSession(vad=BrokenVAD()))
    assert report.components["vad"] == {
        "type": f"{__name__}.BrokenVAD",
        "model": "unknown",
        "provider": "unknown",
    }


async def test_adapters_include_underlying_settings() -> None:
    class ConfiguredSTT(FakeSTT):
        def describe_options(self) -> dict[str, Any]:
            return {"language": "fr"}

    class ConfiguredTTS(FakeTTS):
        def describe_options(self) -> dict[str, Any]:
            return {"voice": "test-voice"}

    vad = inference.VAD(activation_threshold=0.6)
    stt_adapter = stt.FallbackAdapter([stt.StreamAdapter(stt=ConfiguredSTT(), vad=vad)])
    tts_adapter = tts.FallbackAdapter([tts.StreamAdapter(tts=ConfiguredTTS())])
    try:
        report = _report(AgentSession(vad=None, stt=stt_adapter, tts=tts_adapter))
        assert report.components["stt"]["stt"][0]["stt"]["language"] == "fr"
        assert report.components["stt"]["stt"][0]["vad"]["activation_threshold"] == 0.6
        assert report.components["tts"]["tts"][0]["tts"]["voice"] == "test-voice"
        json.dumps(report.to_dict())
    finally:
        await stt_adapter.aclose()
        await tts_adapter.aclose()


def test_elevenlabs_nested_settings_omit_unset_fields_and_customer_text() -> None:
    elevenlabs = pytest.importorskip("livekit.plugins.elevenlabs")
    session = AgentSession(
        vad=None,
        stt=elevenlabs.STT(
            api_key="private-api-key",
            previous_text="private-transcript",
            server_vad={"vad_threshold": 0.6, "min_speech_duration_ms": 200},
        ),
        tts=elevenlabs.TTS(
            api_key="private-api-key",
            voice_settings=elevenlabs.VoiceSettings(stability=0.4, similarity_boost=0.8, speed=1.2),
        ),
    )
    components = _serialize_session_components(session)
    assert components["stt"]["server_vad"]["vad_threshold"] == 0.6
    assert components["tts"]["voice_settings"] == {
        "stability": 0.4,
        "similarity_boost": 0.8,
        "speed": 1.2,
    }
    encoded = json.dumps(components)
    assert "private-api-key" not in encoded
    assert "private-transcript" not in encoded
    assert "NotGiven" not in encoded


def test_openai_turn_detection_and_prompt_omission() -> None:
    openai = pytest.importorskip("livekit.plugins.openai")
    session = AgentSession(
        vad=None,
        stt=openai.STT(
            api_key="private-api-key", turn_detection={"threshold": 0.8}, prompt="private-prompt"
        ),
        tts=openai.TTS(api_key="private-api-key", speed=1.2, instructions="private-instructions"),
    )
    components = _serialize_session_components(session)
    assert components["stt"]["languages"] == ["en"]
    assert components["stt"]["turn_detection"]["threshold"] == 0.8
    assert components["tts"]["speed"] == 1.2
    assert components["tts"]["sample_rate"] > 0
    encoded = json.dumps(components)
    assert "private-" not in encoded


def test_silero_reports_updated_thresholds() -> None:
    silero = pytest.importorskip("livekit.plugins.silero")
    vad = silero.VAD.load(activation_threshold=0.7, min_silence_duration=0.4)
    session = AgentSession(vad=vad)
    report = _report(session)
    vad.update_options(activation_threshold=0.9)
    assert report.components["vad"]["activation_threshold"] == 0.7
    assert _report(session).components["vad"]["activation_threshold"] == 0.9
