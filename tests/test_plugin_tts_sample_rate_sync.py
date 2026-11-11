from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("plugin", ["inworld", "lmnt", "smallestai"])
def test_update_options_syncs_public_sample_rate(plugin: str) -> None:
    """FallbackAdapter resamples from TTS.sample_rate, so it must follow the wire option."""
    tts = importlib.import_module(f"livekit.plugins.{plugin}").TTS(api_key="test-key")
    before = tts.sample_rate

    tts.update_options()
    assert tts.sample_rate == before

    tts.update_options(sample_rate=8000)
    assert tts._opts.sample_rate == 8000
    assert tts.sample_rate == 8000
