from __future__ import annotations

import json
from collections import UserDict
from typing import Any

import pytest

from livekit.agents import AgentSession, inference, llm
from livekit.agents.types import NOT_GIVEN
from livekit.agents.voice.report import _serialize_session_models

from .fake_llm import FakeLLM
from .fake_realtime import FakeRealtimeModel

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def disable_prewarm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm.LLM, "prewarm", lambda *args, **kwargs: None)


def _snapshot(model: llm.LLM | llm.RealtimeModel | llm.DuplexModel) -> dict[str, Any]:
    return _serialize_session_models(AgentSession(vad=None, llm=model))["llm"]


async def test_inference_llm_settings_are_detached_and_exclude_sensitive_options() -> None:
    model = inference.LLM(
        "openai/gpt-4.1",
        api_key="private-key",
        api_secret="private-secret",
        base_url="https://private-endpoint.example.com",
        inference_class="standard",
        extra_kwargs={
            "temperature": 0.2,
            "max_completion_tokens": 64,
            "metadata": {"customer": "private-customer"},
            "prediction": {"type": "content", "content": "private-text"},
            "prompt_cache_key": "private-cache",
            "user": "private-user",
            "unknown": "private-unknown",
        },
    )
    try:
        reported = _snapshot(model)
        assert reported["model"] == "openai/gpt-4.1"
        assert reported["provider"] == "livekit"
        assert reported["inference_class"] == "standard"
        assert reported["extra_kwargs"] == {"temperature": 0.2, "max_completion_tokens": 64}
        assert "private-" not in json.dumps(reported)
        model.update_options(extra_kwargs={"temperature": 0.8})
        assert _snapshot(model)["extra_kwargs"] == {"temperature": 0.8}
        assert reported["extra_kwargs"]["temperature"] == 0.2
    finally:
        await model.aclose()


@pytest.mark.parametrize("realtime", [False, True])
@pytest.mark.parametrize("broken_metadata", [False, True])
async def test_llm_fallback_preserves_healthy_children(
    realtime: bool, broken_metadata: bool
) -> None:
    base = FakeRealtimeModel if realtime else FakeLLM

    class HealthyModel(base):
        def describe_options(self) -> UserDict[str, Any]:
            return UserDict(temperature=0.3, optional=None, unset=NOT_GIVEN)

    class BrokenModel(base):
        @property
        def provider(self) -> str:
            if broken_metadata:
                raise RuntimeError("unavailable")
            return super().provider

        def describe_options(self) -> dict[str, Any]:
            raise RuntimeError("unavailable")

    models = [BrokenModel(), HealthyModel()]
    adapter = (
        llm.RealtimeModelFallbackAdapter(models, cooldown=7.0)
        if realtime
        else llm.FallbackAdapter(models, max_retry_per_llm=3, sticky=True)
    )
    try:
        reported = _snapshot(adapter)
        assert reported["cooldown" if realtime else "max_retry_per_llm"] == (7.0 if realtime else 3)
        broken, healthy = reported["llm"]
        assert healthy == {
            "model": "unknown",
            "provider": "unknown",
            "type": f"{__name__}.HealthyModel",
            "temperature": 0.3,
        }
        assert broken["type"] == f"{__name__}.BrokenModel"
        assert "temperature" not in broken
    finally:
        await adapter.aclose()


@pytest.mark.parametrize(
    "provider",
    [
        "openai",
        "anthropic",
        "aws",
        "google",
        "mistralai",
        "groq",
        "baseten",
        "perplexity",
        "sarvam",
        "cerebras",
    ],
)
async def test_llm_plugins_report_generation_settings(provider: str) -> None:
    plugin = pytest.importorskip(f"livekit.plugins.{provider}")
    kwargs: dict[str, Any] = {"api_key": "private-key", "temperature": 0.25}
    if provider == "aws":
        kwargs["api_secret"] = "private-secret"
    if provider == "google":
        kwargs["vertexai"] = False
    model = plugin.LLM(**kwargs)
    try:
        reported = _snapshot(model)
        assert reported["temperature"] == 0.25
        assert reported["model"] == model.model
        assert "private-" not in json.dumps(reported)
    finally:
        await model.aclose()


async def test_openai_responses_excludes_user_metadata_and_unknown_reasoning_fields() -> None:
    plugin = pytest.importorskip("livekit.plugins.openai")
    sdk = pytest.importorskip("openai")
    model = plugin.responses.LLM(
        api_key="private-key",
        user="private-user",
        metadata={"customer": "private-customer"},
        reasoning=sdk.types.Reasoning(effort="low", future_setting="private-unknown"),
        max_output_tokens=64,
    )
    try:
        reported = _snapshot(model)
        assert reported["max_output_tokens"] == 64
        assert reported["reasoning"] == {"effort": "low"}
        assert "private-" not in json.dumps(reported)
    finally:
        await model.aclose()


@pytest.mark.parametrize("hosted", [False, True])
async def test_openai_realtime_reports_safe_nested_settings(hosted: bool) -> None:
    plugin = pytest.importorskip("livekit.plugins.openai")
    sdk = pytest.importorskip("openai")
    kwargs: dict[str, Any] = {
        "api_key": "private-key",
        "voice": "coral",
        "speed": 1.1,
        "turn_detection": sdk.types.realtime.realtime_audio_input_turn_detection.ServerVad(
            type="server_vad", threshold=0.6, future_setting="private-unknown"
        ),
        "input_audio_transcription": sdk.types.realtime.AudioTranscription(
            model="whisper-1", language="en", prompt="private-prompt"
        ),
    }
    model = (
        inference.realtime.RealtimeModel(
            "openai/gpt-realtime", api_secret="private-secret", **kwargs
        )
        if hosted
        else plugin.realtime.RealtimeModel(**kwargs)
    )
    try:
        reported = _snapshot(model)
        assert reported["voice"] == "coral"
        assert reported["speed"] == 1.1
        assert reported["turn_detection"]["threshold"] == 0.6
        assert reported["input_audio_transcription"] == {"model": "whisper-1", "language": "en"}
        assert "private-" not in json.dumps(reported)
    finally:
        await model.aclose()


async def test_google_nested_configs_exclude_customer_content() -> None:
    plugin = pytest.importorskip("livekit.plugins.google")
    types = pytest.importorskip("google.genai.types")
    models = [
        plugin.LLM(
            api_key="private-key",
            vertexai=False,
            cached_content="private-cache",
            thinking_config={"thinking_budget": 64},
        ),
        plugin.realtime.RealtimeModel(
            api_key="private-key",
            vertexai=False,
            instructions="private-prompt",
            thinking_config=types.ThinkingConfig(thinking_budget=64),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    silence_duration_ms=500
                )
            ),
            input_audio_transcription=types.AudioTranscriptionConfig(
                language_codes=["en"], custom_vocabulary=["private-vocabulary"]
            ),
        ),
    ]
    try:
        for model in models:
            reported = _snapshot(model)
            assert reported["thinking_config"]["thinking_budget"] == 64
            assert "private-" not in json.dumps(reported)
        assert (
            reported["realtime_input_config"]["automatic_activity_detection"]["silence_duration_ms"]
            == 500
        )
        assert reported["input_audio_transcription"] == {"language_codes": ["en"]}
    finally:
        for model in models:
            await model.aclose()


async def test_duplex_model_is_reported_through_session_adapter() -> None:
    plugin = pytest.importorskip("livekit.plugins.openai")
    model = plugin.realtime.GPTLiveModel(
        api_key="private-key",
        voice="marin",
        responses_options={
            "model": "gpt-4.1",
            "max_output_tokens": 64,
            "instructions": "private-prompt",
            "text": {"format": {"schema": {"private-schema": "private-content"}}},
            "future_setting": "private-unknown",
        },
    )
    try:
        reported = _snapshot(model)
        assert reported["type"].endswith("DuplexRealtimeAdapter")
        assert reported["llm"]["voice"] == "marin"
        assert reported["llm"]["responses"] == {"model": "gpt-4.1", "max_output_tokens": 64}
        assert "private-" not in json.dumps(reported)
    finally:
        await model.aclose()
