"""Azure Voice Live: the plugin's options as the core's beta-protocol dialect receives them."""

from __future__ import annotations

from typing import Any

import pytest

from livekit.agents import _core
from livekit.plugins.azure.realtime import RealtimeModel

pytestmark = pytest.mark.unit

_ENDPOINT = "https://res.services.ai.azure.com/"


@pytest.fixture
def configs(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    seen: list[Any] = []
    native = _core.OpenAIRealtimeModel

    def record(config: Any) -> Any:
        seen.append(config)
        return native(config)

    monkeypatch.setattr(_core, "OpenAIRealtimeModel", record)
    return seen


def test_connects_to_the_voice_live_endpoint_with_its_model(configs: list[Any]) -> None:
    model = RealtimeModel(endpoint=_ENDPOINT, api_key="key", model="gpt-realtime")

    (config,) = configs
    assert config.dialect == "voice_live"
    assert config.base_url == (
        "https://res.services.ai.azure.com/voice-live/realtime?api-version=2026-04-10"
    )
    assert config.azure_deployment == "gpt-realtime"
    assert config.api_key == "key"
    # Voice Live has no conversation.item.truncate
    assert not model.capabilities.message_truncation


def test_azure_voice_names_become_voice_objects_pinned_to_one_language(
    configs: list[Any],
) -> None:
    RealtimeModel(
        endpoint=_ENDPOINT,
        api_key="key",
        voice="en-US-AvaNeural",
        input_audio_transcription={"model": "azure-speech", "language": "en-US"},
    )
    RealtimeModel(
        endpoint=_ENDPOINT,
        api_key="key",
        voice="en-US-AvaNeural",
        input_audio_transcription={"model": "azure-speech", "language": "en-US,fr-FR"},
    )
    RealtimeModel(endpoint=_ENDPOINT, api_key="key", voice="alloy")

    assert [c.voice for c in configs] == [
        {"type": "azure-standard", "name": "en-US-AvaNeural", "locale": "en-US"},
        {"type": "azure-standard", "name": "en-US-AvaNeural"},
        "alloy",
    ]


@pytest.mark.parametrize(
    ("model", "transcription"),
    [
        ("gpt-realtime", "whisper-1"),
        ("gpt-4.1", "azure-speech"),
        ("phi4-mm-realtime", "azure-speech"),
    ],
)
def test_the_transcription_default_follows_the_model(
    configs: list[Any], model: str, transcription: str
) -> None:
    RealtimeModel(endpoint=_ENDPOINT, api_key="key", model=model)
    assert configs[0].options.input_audio_transcription == {"model": transcription}


def test_requires_an_endpoint_and_a_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AZURE_VOICE_LIVE_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_VOICE_LIVE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="endpoint"):
        RealtimeModel(api_key="key")
    with pytest.raises(ValueError, match="credentials"):
        RealtimeModel(endpoint=_ENDPOINT)
    RealtimeModel(endpoint=_ENDPOINT, entra_token="token")
