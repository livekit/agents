"""Tests for Rime TTS plugin defaults."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_model_and_speaker_defaults() -> None:
    from livekit.plugins.rime import TTS

    default_tts = TTS(api_key="test-key")
    explicit_coda_tts = TTS(api_key="test-key", model="coda")
    explicit_arcana_tts = TTS(api_key="test-key", model="arcana")

    assert default_tts.model == "coda"
    assert default_tts._opts.speaker == "astra"
    assert explicit_coda_tts.model == "coda"
    assert explicit_coda_tts._opts.speaker == "lyra"
    assert explicit_arcana_tts.model == "arcana"
    assert explicit_arcana_tts._opts.speaker == "astra"
