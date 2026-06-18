"""Tests for Fish Audio TTS plugin configuration and request building."""

from __future__ import annotations

import pytest

from livekit.agents.types import NOT_GIVEN

pytestmark = pytest.mark.plugin("fishaudio")


def test_default_model_is_s2_1_pro():
    """A bare TTS() defaults to the s2.1-pro model."""
    from livekit.plugins.fishaudio import TTS

    tts = TTS(api_key="test-key")
    assert tts.model == "s2.1-pro"


def test_model_variants_accepted():
    """s1, s2-pro, and s2.1-pro are all accepted."""
    from livekit.plugins.fishaudio import TTS

    for model in ("s1", "s2-pro", "s2.1-pro"):
        assert TTS(api_key="test-key", model=model).model == model


def test_prosody_omitted_by_default():
    """Without speed/volume the request sends prosody=None (unchanged default)."""
    from livekit.plugins.fishaudio.tts import _build_tts_request

    tts = TTS_opts()
    assert tts.speed is NOT_GIVEN
    assert tts.volume is NOT_GIVEN
    assert _build_tts_request(tts, text="hi")["prosody"] is None


def test_speed_sets_prosody():
    """speed populates only prosody.speed."""
    from livekit.plugins.fishaudio.tts import _build_tts_request

    assert _build_tts_request(TTS_opts(speed=0.8), text="hi")["prosody"] == {"speed": 0.8}


def test_volume_sets_prosody():
    """volume populates only prosody.volume."""
    from livekit.plugins.fishaudio.tts import _build_tts_request

    assert _build_tts_request(TTS_opts(volume=-3.0), text="hi")["prosody"] == {"volume": -3.0}


def test_speed_and_volume_set_prosody():
    """speed and volume together populate both prosody fields."""
    from livekit.plugins.fishaudio.tts import _build_tts_request

    prosody = _build_tts_request(TTS_opts(speed=1.2, volume=-3.0), text="hi")["prosody"]
    assert prosody == {"speed": 1.2, "volume": -3.0}


def test_update_options_sets_speed_and_volume():
    """update_options forwards speed/volume onto the request."""
    from livekit.plugins.fishaudio import TTS
    from livekit.plugins.fishaudio.tts import _build_tts_request

    tts = TTS(api_key="test-key")
    assert _build_tts_request(tts._opts, text="hi")["prosody"] is None

    tts.update_options(speed=0.9, volume=2.0)
    assert _build_tts_request(tts._opts, text="hi")["prosody"] == {"speed": 0.9, "volume": 2.0}


def TTS_opts(**overrides):
    """Build a _TTSOptions with sensible test defaults, overriding as needed."""
    from livekit.plugins.fishaudio.tts import DEFAULT_BASE_URL, DEFAULT_MODEL, _TTSOptions

    opts = {
        "model": DEFAULT_MODEL,
        "output_format": "wav",
        "sample_rate": 24000,
        "voice_id": NOT_GIVEN,
        "base_url": DEFAULT_BASE_URL,
        "api_key": "test-key",
        "latency_mode": "balanced",
        "chunk_length": 100,
        "speed": NOT_GIVEN,
        "volume": NOT_GIVEN,
    }
    opts.update(overrides)
    return _TTSOptions(**opts)
