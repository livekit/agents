"""Tests for Rime TTS plugin defaults and request controls."""

from __future__ import annotations

import logging
from urllib.parse import parse_qs, urlparse

import pytest

pytestmark = pytest.mark.unit


def test_model_and_speaker_defaults() -> None:
    from livekit.plugins.rime import TTS

    default_tts = TTS(api_key="test-key")
    explicit_coda_tts = TTS(api_key="test-key", model="coda")

    assert default_tts.model == "coda"
    assert default_tts._opts.speaker == "astra"
    assert explicit_coda_tts.model == "coda"
    assert explicit_coda_tts._opts.speaker == "lyra"


def test_arcana_model_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    from livekit.plugins.rime import TTS

    with caplog.at_level(logging.WARNING, logger="livekit.plugins.rime"):
        tts = TTS(api_key="test-key", model="arcana")

    assert 'Rime Arcana is no longer supported. Use model="coda" instead.' in caplog.messages

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="livekit.plugins.rime"):
        tts.update_options(model="arcana")

    assert 'Rime Arcana is no longer supported. Use model="coda" instead.' in caplog.messages


def test_coda_request_controls() -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(
        api_key="test-key",
        model="coda",
        repetition_penalty=1.1,
        temperature=0.5,
        top_p=0.9,
        max_tokens=200,
        time_scale_factor=1.2,
    )

    params = parse_qs(urlparse(tts._ws_url()).query)

    assert params["repetition_penalty"] == ["1.1"]
    assert params["temperature"] == ["0.5"]
    assert params["top_p"] == ["0.9"]
    assert params["max_tokens"] == ["200"]
    assert params["timeScaleFactor"] == ["1.2"]

    tts.update_options(
        repetition_penalty=1.2,
        temperature=0.6,
        top_p=0.8,
        max_tokens=300,
        time_scale_factor=1.1,
    )
    updated_params = parse_qs(urlparse(tts._ws_url()).query)

    assert updated_params["repetition_penalty"] == ["1.2"]
    assert updated_params["temperature"] == ["0.6"]
    assert updated_params["top_p"] == ["0.8"]
    assert updated_params["max_tokens"] == ["300"]
    assert updated_params["timeScaleFactor"] == ["1.1"]
