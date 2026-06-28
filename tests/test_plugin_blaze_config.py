"""Unit tests for Blaze configuration."""

from __future__ import annotations

import pytest

from livekit.plugins.blaze._config import BlazeConfig

pytestmark = pytest.mark.unit


def test_blaze_config_explicit_values() -> None:
    config = BlazeConfig(
        api_url="https://custom.example.com",
        api_token="secret",
        stt_timeout=12.0,
        tts_timeout=34.0,
        tts_stream_timeout=400.0,
        llm_timeout=56.0,
    )

    assert config.api_url == "https://custom.example.com"
    assert config.api_token == "secret"
    assert config.stt_timeout == 12.0
    assert config.tts_timeout == 34.0
    assert config.tts_stream_timeout == 400.0
    assert config.llm_timeout == 56.0


def test_blaze_config_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLAZE_API_URL", "https://env.example.com")
    monkeypatch.setenv("BLAZE_API_TOKEN", "env-token")
    monkeypatch.setenv("BLAZE_STT_TIMEOUT", "11")
    monkeypatch.setenv("BLAZE_TTS_TIMEOUT", "22")
    monkeypatch.setenv("BLAZE_TTS_STREAM_TIMEOUT", "333")
    monkeypatch.setenv("BLAZE_LLM_TIMEOUT", "44")

    config = BlazeConfig()

    assert config.api_url == "https://env.example.com"
    assert config.api_token == "env-token"
    assert config.stt_timeout == 11.0
    assert config.tts_timeout == 22.0
    assert config.tts_stream_timeout == 333.0
    assert config.llm_timeout == 44.0


def test_blaze_config_defaults_without_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "BLAZE_API_URL",
        "BLAZE_API_TOKEN",
        "BLAZE_STT_TIMEOUT",
        "BLAZE_TTS_TIMEOUT",
        "BLAZE_TTS_STREAM_TIMEOUT",
        "BLAZE_LLM_TIMEOUT",
    ):
        monkeypatch.delenv(key, raising=False)

    config = BlazeConfig()

    assert config.api_url == "https://api.blaze.vn"
    assert config.api_token == ""
    assert config.stt_timeout == 30.0
    assert config.tts_timeout == 60.0
    assert config.tts_stream_timeout == 300.0
    assert config.llm_timeout == 60.0
