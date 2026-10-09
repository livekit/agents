from __future__ import annotations

import json
from collections import UserDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypedDict
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ConfigDict
from typing_extensions import Required

from livekit.agents import AgentSession, JobContext, inference, stt, tts
from livekit.agents._reporting import Sensitive, report_options
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.voice.report import SessionReport, _serialize_session_models

from .fake_stt import FakeSTT
from .fake_tts import FakeTTS
from .fake_vad import FakeVAD

pytestmark = pytest.mark.unit


class _NestedDictOptions(TypedDict, total=False):
    language: str
    prompt: Required[Sensitive[str]]
    endpoint: str
    optional: NotGivenOr[str | None]


@dataclass
class _NestedDataclassOptions:
    language: str = "en"
    prompt: Sensitive[str] = "private-prompt"
    optional: NotGivenOr[str | None] = NOT_GIVEN


class _NestedModelOptions(BaseModel):
    model_config = ConfigDict(extra="allow")
    language: str = "en"
    prompt: Sensitive[str] = "private-prompt"
    optional: NotGivenOr[str | None] = None


class _NestedOptions(TypedDict):
    dictionary: _NestedDictOptions | None
    dataclass: _NestedDataclassOptions
    model: _NestedModelOptions
    items: NotGivenOr[list[_NestedDictOptions | None]]
    untyped: dict[str, Any]
    optional: str | None
    unset: NotGivenOr[str]


def _report(session: AgentSession) -> SessionReport:
    ctx = MagicMock(spec=JobContext)
    ctx.job.id = "job-1"
    ctx.job.room.sid = "room-1"
    ctx.job.room.name = "test-room"
    return JobContext.make_session_report(ctx, session)


def test_reportable_options_exclude_sensitive_fields() -> None:
    class Options(TypedDict):
        speed: float
        prompt: Sensitive[str]
        future_setting: float

    class OtherOptions(TypedDict):
        prompt: str

    config = {"speed": 1.0, "prompt": "private-prompt", "future_setting": 2.0, "unknown": "private"}
    assert report_options(config, Options) == {"speed": 1.0, "future_setting": 2.0}
    assert report_options(config, Options, OtherOptions, exclude=["speed"]) == {
        "future_setting": 2.0
    }
    assert report_options(config) == {}


@pytest.mark.parametrize("pydantic", [False, True])
def test_report_options_uses_declared_fields_and_exclusions(pydantic: bool) -> None:
    @dataclass
    class DataclassOptions:
        language: str = "en"
        prompt: Sensitive[str] = "private-prompt"
        endpoint: str = "private-endpoint"

    class ModelOptions(BaseModel):
        model_config = ConfigDict(extra="allow")
        language: str = "en"
        prompt: Sensitive[str] = "private-prompt"
        endpoint: str = "private-endpoint"

    config = ModelOptions() if pydantic else DataclassOptions()
    config.unknown = "private-unknown"

    class CustomSTT(FakeSTT):
        def describe_options(self) -> dict[str, Any]:
            return report_options(config, exclude=["endpoint"])

    session = AgentSession(vad=None, stt=CustomSTT())
    reported = _report(session).models["stt"]
    assert reported["language"] == "en"
    assert "private" not in json.dumps(reported)
    config.language = "fr"
    assert reported["language"] == "en"
    assert _report(session).models["stt"]["language"] == "fr"


@pytest.mark.parametrize("config", [None, NOT_GIVEN])
def test_report_options_accepts_absent_configs(config: Any) -> None:
    assert report_options(config) == {}


def test_report_options_filters_nested_schemas_and_unset_values() -> None:
    nested = {
        "language": "en",
        "prompt": "private-prompt",
        "endpoint": "private-endpoint",
        "unknown": "private-unknown",
        "optional": NOT_GIVEN,
    }
    config = {
        "dictionary": nested,
        "dataclass": _NestedDataclassOptions(),
        "model": _NestedModelOptions(unknown="private-unknown"),
        "items": [nested, None, NOT_GIVEN],
        "untyped": {"unknown": "private-unknown"},
        "optional": None,
        "unset": NOT_GIVEN,
    }
    reported = report_options(
        config, _NestedOptions, exclude={"dictionary": {"endpoint"}, "items": {"endpoint"}}
    )
    assert reported == {
        "dictionary": {"language": "en"},
        "dataclass": {"language": "en"},
        "model": {"language": "en"},
        "items": [{"language": "en"}],
        "untyped": {},
    }
    nested["language"] = "fr"
    assert reported["items"] == [{"language": "en"}]


@pytest.mark.parametrize(
    "model, safe, sensitive",
    [
        (
            "assemblyai/u3-rt-pro",
            {"format_turns": True},
            {"prompt": "private-prompt", "keyterms_prompt": ["private-name"]},
        ),
        (
            "speechmatics/enhanced",
            {"max_delay": 1.0},
            {
                "additional_vocab": [{"content": "private-name"}],
                "transcript_filtering_config": {"payload": "private-payload"},
            },
        ),
    ],
)
def test_inference_omits_sensitive_primary_and_fallback_options(
    model: str, safe: dict[str, Any], sensitive: dict[str, Any]
) -> None:
    options = {**safe, **sensitive, "unknown_option": "private-unknown"}
    component = inference.STT(
        model,
        api_key="private-key",
        api_secret="private-secret",
        extra_kwargs=options,
        fallback={"model": model, "extra_kwargs": options},
    )
    report = _report(AgentSession(vad=None, stt=component))
    assert report.models["stt"]["extra_kwargs"] == safe
    assert report.models["stt"]["fallback"][0]["extra_kwargs"] == safe
    assert "private-" not in json.dumps(report.to_dict())


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

    assert set(report.models) == {"vad", "stt", "tts"}
    assert report.models["vad"]["activation_threshold"] == 0.7
    assert report.models["vad"]["min_silence_duration"] == 0.4
    assert report.models["stt"]["model"] == "deepgram/nova-3"
    assert report.models["stt"]["language"] == "en"
    assert report.models["stt"]["extra_kwargs"] == {"endpointing": 50}
    assert report.models["tts"]["voice"] == "voice-1"
    assert report.models["tts"]["language"] == "de"
    assert report.models["tts"]["extra_kwargs"] == {"speed": 1.2}
    assert report.models["stt"]["fallback"] == [
        {"model": "cartesia/ink-whisper", "extra_kwargs": {"min_volume": 0.2}}
    ]
    assert report.models["tts"]["fallback"] == [
        {
            "model": "inworld/inworld-tts-1.5-max",
            "voice": "fallback-voice",
            "extra_kwargs": {"speaking_rate": 0.9},
        }
    ]
    assert report.to_dict()["models"] == report.models

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
    current = _report(session).models
    assert current["vad"]["activation_threshold"] == 0.9
    assert current["stt"]["language"] == "fr"
    assert current["tts"]["voice"] == "voice-2"
    assert report.models["vad"]["activation_threshold"] == 0.7
    assert report.models["stt"]["language"] == "en"
    assert report.models["tts"]["voice"] == "voice-1"
    assert report.models["tts"]["extra_kwargs"] == {"speed": 1.2}


def test_disabled_models_and_existing_report_constructor() -> None:
    session = AgentSession(vad=None)
    assert _report(session).models == {}
    report = SessionReport(
        job_id="job-1",
        room_id="room-1",
        room="test-room",
        options=session.options,
        events=[],
        chat_history=session.history,
    )
    assert report.to_dict()["models"] == {}


def test_default_session_reports_vad_settings() -> None:
    models = _report(AgentSession()).models
    assert set(models) == {"vad"}
    assert models["vad"]["model"] == "silero"
    assert models["vad"]["activation_threshold"] == 0.5
    assert models["vad"]["sample_rate"] == 16000


def test_custom_model_options_are_normalized() -> None:
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
    assert report.models["vad"]["mode"] == "fast"
    assert report.models["vad"]["thresholds"] == [0.3, 0.7]
    assert report.models["vad"]["nested"] == {"enabled": True}
    assert "unset" not in report.models["vad"]
    json.dumps(report.to_dict())


def test_broken_model_description_keeps_identity() -> None:
    class BrokenVAD(FakeVAD):
        def describe_options(self) -> dict[str, Any]:
            raise RuntimeError("unavailable")

    report = _report(AgentSession(vad=BrokenVAD()))
    assert report.models["vad"] == {
        "type": f"{__name__}.BrokenVAD",
        "model": "unknown",
        "provider": "unknown",
    }


def test_model_properties_take_precedence_over_reported_options() -> None:
    class CustomTTS(FakeTTS):
        def describe_options(self) -> dict[str, Any]:
            return {
                "model": "config-model",
                "provider": "config-provider",
                "sample_rate": 8000,
                "num_channels": 2,
                "nested": {"model": "nested-model", "provider": "nested-provider"},
            }

    report = _report(AgentSession(vad=None, tts=CustomTTS()))
    assert report.models["tts"] == {
        "type": f"{__name__}.CustomTTS",
        "model": "unknown",
        "provider": "unknown",
        "sample_rate": 24000,
        "num_channels": 1,
        "nested": {"model": "nested-model", "provider": "nested-provider"},
    }


def test_broken_model_metadata_does_not_break_report() -> None:
    class BrokenVAD(FakeVAD):
        @property
        def provider(self) -> str:
            raise RuntimeError("unavailable")

        def describe_options(self) -> dict[str, Any]:
            return {"activation_threshold": 0.6}

    report = _report(AgentSession(vad=BrokenVAD(), stt=FakeSTT()))
    assert report.models["vad"] == {
        "type": f"{__name__}.BrokenVAD",
        "activation_threshold": 0.6,
    }
    assert report.models["stt"]["model"] == "unknown"


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
        reported = _report(AgentSession(vad=None, **{kind: adapter})).models[kind]
        assert reported[f"max_retry_per_{kind}"] == 3
        broken, healthy = reported[kind]
        if wrap_stream:
            healthy, broken = healthy[kind], broken[kind]
        metadata = {"model": "unknown", "provider": "unknown"}
        if kind == "tts":
            metadata.update(sample_rate=24000, num_channels=1)
        assert healthy == {
            "type": f"{__name__}.Healthy{kind.upper()}",
            **metadata,
            **expected,
        }
        assert broken == {
            "type": f"{__name__}.Broken{kind.upper()}",
            **({} if broken_metadata else metadata),
        }
    finally:
        await adapter.aclose()


@pytest.mark.parametrize("version", [1, 2])
async def test_deepgram_reports_settings_without_tags_or_keyterms(version: int) -> None:
    deepgram = pytest.importorskip("livekit.plugins.deepgram")
    factory = deepgram.STT if version == 1 else deepgram.STTv2
    model = factory(
        api_key="private-key",
        tags=["private-customer"],
        keyterm=["private-name"],
        sample_rate=24000,
    )
    try:
        reported = _report(AgentSession(vad=None, stt=model)).models["stt"]
        assert reported["model"] == model.model
        assert reported["sample_rate"] == 24000
        assert "private-" not in json.dumps(reported)
    finally:
        await model.aclose()


async def test_gladia_reports_nested_settings_without_customer_content() -> None:
    gladia = pytest.importorskip("livekit.plugins.gladia")
    model = gladia.STT(
        api_key="private-key",
        translation_enabled=True,
        translation_target_languages=["fr"],
        translation_context="private-context",
        custom_vocabulary=["private-name"],
        custom_spelling={"private-term": ["private-spelling"]},
        pre_processing_speech_threshold=0.8,
    )
    try:
        reported = _report(AgentSession(vad=None, stt=model)).models["stt"]
        assert reported["translation_config"]["enabled"] is True
        assert reported["translation_config"]["target_languages"] == ["fr"]
        assert reported["pre_processing"]["speech_threshold"] == 0.8
        assert "private-" not in json.dumps(reported)
    finally:
        await model.aclose()


@pytest.mark.parametrize(
    "provider, kind, nested_key, field, value",
    [
        ("openai", "stt", "turn_detection", "threshold", 0.8),
        ("elevenlabs", "stt", "server_vad", "vad_threshold", 0.6),
        ("hume", "tts", "voice", "name", "test-voice"),
    ],
)
async def test_nested_provider_settings_exclude_unknown_fields(
    provider: str, kind: str, nested_key: str, field: str, value: Any
) -> None:
    plugin = pytest.importorskip(f"livekit.plugins.{provider}")
    nested = {field: value, "api_key": "private-key", "future_setting": "private-value"}
    component = getattr(plugin, kind.upper())(api_key="private-key", **{nested_key: nested})
    try:
        report = _report(AgentSession(vad=None, **{kind: component}))
        assert report.models[kind][nested_key][field] == value
        assert "api_key" not in report.models[kind][nested_key]
        assert "future_setting" not in report.models[kind][nested_key]
        assert "private-" not in json.dumps(report.to_dict())
        nested[field] = "changed"
        assert report.models[kind][nested_key][field] == value
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
        options = report.models["tts"]
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
        assert _report(session).models["tts"]["voice"] == "voice-2"
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
        assert report.models["stt"]["stt"][0]["stt"]["language"] == "fr"
        assert report.models["stt"]["stt"][0]["vad"]["activation_threshold"] == 0.6
        assert report.models["tts"]["tts"][0]["tts"]["voice"] == "test-voice"
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
    models = _serialize_session_models(session)
    assert models["stt"]["server_vad"]["vad_threshold"] == 0.6
    assert models["tts"]["voice_settings"] == {
        "stability": 0.4,
        "similarity_boost": 0.8,
        "speed": 1.2,
    }
    encoded = json.dumps(models)
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
    models = _serialize_session_models(session)
    assert models["stt"]["languages"] == ["en"]
    assert models["stt"]["turn_detection"]["threshold"] == 0.8
    assert models["tts"]["speed"] == 1.2
    assert models["tts"]["sample_rate"] > 0
    encoded = json.dumps(models)
    assert "private-" not in encoded


def test_silero_reports_updated_thresholds() -> None:
    silero = pytest.importorskip("livekit.plugins.silero")
    vad = silero.VAD.load(activation_threshold=0.7, min_silence_duration=0.4)
    session = AgentSession(vad=vad)
    report = _report(session)
    vad.update_options(activation_threshold=0.9)
    assert report.models["vad"]["activation_threshold"] == 0.7
    assert _report(session).models["vad"]["activation_threshold"] == 0.9
