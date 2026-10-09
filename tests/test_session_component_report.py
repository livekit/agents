from __future__ import annotations

import json
from collections import UserDict
from enum import Enum
from typing import Any, TypedDict
from unittest.mock import MagicMock

import pytest

from livekit.agents import AgentSession, JobContext, inference, stt, tts
from livekit.agents._reporting import Reportable, reportable_option_names
from livekit.agents.types import NOT_GIVEN
from livekit.agents.voice.report import SessionReport, _serialize_session_components

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


def test_reportable_options_require_explicit_opt_in() -> None:
    class Options(TypedDict):
        speed: Reportable[float]
        prompt: str
        future_setting: float

    assert reportable_option_names(Options) == {"speed"}


def test_report_snapshots_inference_settings_without_credentials() -> None:
    vad = inference.VAD(activation_threshold=0.7, min_silence_duration=0.4)
    stt_model = inference.STT(
        "deepgram/nova-3",
        language="en",
        api_key="secret-key",
        api_secret="secret-value",
        base_url="https://private-endpoint.example.com",
        extra_kwargs={
            "endpointing": 50,
            "keyterm": ["private-customer"],
            "future_setting": "private-future-setting",
        },
        fallback={
            "model": "cartesia/ink-whisper",
            "extra_kwargs": {
                "min_volume": 0.2,
                "callback": "private-callback",
                "future_setting": "private-future-setting",
            },
        },
    )
    tts_model = inference.TTS(
        "cartesia/sonic-3",
        voice="voice-1",
        language="de",
        api_key="secret-key",
        api_secret="secret-value",
        extra_kwargs={"speed": 1.2, "future_setting": "private-future-setting"},
        fallback={
            "model": "inworld/inworld-tts-1.5-max",
            "voice": "fallback-voice",
            "extra_kwargs": {
                "speaking_rate": 0.9,
                "api_key": "private-api-key",
                "future_setting": "private-future-setting",
            },
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
        "private-future-setting",
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


def test_broken_component_metadata_does_not_break_report() -> None:
    class BrokenVAD(FakeVAD):
        @property
        def provider(self) -> str:
            raise RuntimeError("unavailable")

    report = _report(AgentSession(vad=BrokenVAD(), stt=FakeSTT()))
    assert report.components["vad"] == {"type": f"{__name__}.BrokenVAD"}
    assert report.components["stt"]["model"] == "unknown"


@pytest.mark.parametrize("kind", ["stt", "tts"])
@pytest.mark.parametrize("wrap_stream", [False, True])
@pytest.mark.parametrize("broken_metadata", [False, True])
async def test_adapter_snapshots_isolate_child_failures(
    kind: str, wrap_stream: bool, broken_metadata: bool
) -> None:
    class HealthySTT(FakeSTT):
        def describe_options(self) -> UserDict[str, Any]:
            return UserDict(language="fr", optional=None, unset=NOT_GIVEN)

    class BrokenSTT(FakeSTT):
        @property
        def provider(self) -> str:
            if broken_metadata:
                raise RuntimeError("unavailable")
            return super().provider

        def describe_options(self) -> dict[str, Any]:
            raise RuntimeError("unavailable")

    class HealthyTTS(FakeTTS):
        def describe_options(self) -> UserDict[str, Any]:
            return UserDict(voice="voice-1", optional=None, unset=NOT_GIVEN)

    class BrokenTTS(FakeTTS):
        @property
        def provider(self) -> str:
            if broken_metadata:
                raise RuntimeError("unavailable")
            return super().provider

        def describe_options(self) -> dict[str, Any]:
            raise RuntimeError("unavailable")

    if kind == "stt":
        children = [BrokenSTT(), HealthySTT()]
        if wrap_stream:
            children = [stt.StreamAdapter(stt=child, vad=FakeVAD()) for child in children]
        adapter = stt.FallbackAdapter(children, max_retry_per_stt=3)
        expected = {"language": "fr"}
    else:
        children = [BrokenTTS(), HealthyTTS()]
        if wrap_stream:
            children = [tts.StreamAdapter(tts=child) for child in children]
        adapter = tts.FallbackAdapter(children, max_retry_per_tts=3)
        expected = {"voice": "voice-1"}

    try:
        reported = _report(AgentSession(vad=None, **{kind: adapter})).components[kind]
        assert reported[f"max_retry_per_{kind}"] == 3
        broken, healthy = reported[kind]
        if wrap_stream:
            healthy, broken = healthy[kind], broken[kind]
        assert healthy == {
            "type": f"{__name__}.Healthy{kind.upper()}",
            "model": "unknown",
            "provider": "unknown",
            **expected,
        }
        assert broken == {
            "type": f"{__name__}.Broken{kind.upper()}",
            **({} if broken_metadata else {"model": "unknown", "provider": "unknown"}),
        }
    finally:
        await adapter.aclose()


@pytest.mark.parametrize(
    "provider, kind, nested_key, field, value",
    [
        ("openai", "stt", "turn_detection", "threshold", 0.8),
        ("elevenlabs", "stt", "server_vad", "vad_threshold", 0.6),
        ("hume", "tts", "voice", "name", "test-voice"),
    ],
)
async def test_nested_provider_settings_require_opt_in(
    provider: str, kind: str, nested_key: str, field: str, value: Any
) -> None:
    plugin = pytest.importorskip(f"livekit.plugins.{provider}")
    nested = {field: value, "api_key": "private-key", "future_setting": "private-value"}
    component = getattr(plugin, kind.upper())(api_key="private-key", **{nested_key: nested})
    try:
        report = _report(AgentSession(vad=None, **{kind: component}))
        assert report.components[kind][nested_key][field] == value
        assert "api_key" not in report.components[kind][nested_key]
        assert "future_setting" not in report.components[kind][nested_key]
        assert "private-" not in json.dumps(report.to_dict())
        nested[field] = "changed"
        assert report.components[kind][nested_key][field] == value
    finally:
        await component.aclose()


@pytest.mark.parametrize("model", ["orpheus", "qwen3-tts"])
async def test_baseten_reports_both_backends_without_customer_content(model: str) -> None:
    baseten = pytest.importorskip("livekit.plugins.baseten")
    component = baseten.TTS(
        model=model,
        api_key="private-key",
        model_endpoint="wss://private-endpoint.example.com",
        voice="voice-1",
        language="English",
        max_new_tokens=512,
        instructions="private-prompt",
        ref_audio="private-audio",
        ref_text="private-text",
        extra_config={"api_key": "private-config-key"},
    )
    try:
        session = AgentSession(vad=None, tts=component)
        report = _report(session)
        options = report.components["tts"]
        assert options["model"] == model
        assert options["voice"] == "voice-1"
        assert options["language"] == "English"
        assert options["sample_rate"] == 24000
        assert options["num_channels"] == 1
        if model == "qwen3-tts":
            assert options["max_new_tokens"] == 512
            assert options["task_type"] == "Base"
            assert options["word_timestamps"] is False
        else:
            assert options["max_tokens"] == 2000
        assert "private-" not in json.dumps(report.to_dict())
        component.update_options(voice="voice-2")
        assert options["voice"] == "voice-1"
        assert _report(session).components["tts"]["voice"] == "voice-2"
    finally:
        await component.aclose()


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
