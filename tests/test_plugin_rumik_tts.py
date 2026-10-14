"""Tests for Rumik AI TTS plugin configuration and behavior."""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.plugin("rumik")


def test_tts_requires_api_key():
    """TTS constructor raises when no API key is provided."""
    from livekit.plugins.rumik import TTS

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API key is required"):
            TTS(api_key=None)


def test_tts_accepts_api_key_directly():
    """TTS constructor accepts api_key argument."""
    from livekit.plugins.rumik import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.api_key == "test-key"


def test_tts_accepts_api_key_from_env():
    """TTS constructor reads RUMIK_API_KEY from environment."""
    from livekit.plugins.rumik import TTS

    with patch.dict("os.environ", {"RUMIK_API_KEY": "env-key"}):
        tts = TTS()
        assert tts._opts.api_key == "env-key"


def test_tts_default_model():
    """TTS defaults to 'muga' model."""
    from livekit.plugins.rumik import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.model == "muga"
    assert tts.model == "muga"


def test_tts_custom_model():
    """TTS accepts custom model."""
    from livekit.plugins.rumik import TTS

    tts = TTS(api_key="test-key", model="mulberry")
    assert tts._opts.model == "mulberry"
    assert tts.model == "mulberry"


def test_tts_provider_property():
    """TTS provider property returns 'RumikAI'."""
    from livekit.plugins.rumik import TTS

    tts = TTS(api_key="test-key")
    assert tts.provider == "RumikAI"


def test_tts_capabilities():
    """TTS reports streaming=True."""
    from livekit.plugins.rumik import TTS

    tts = TTS(api_key="test-key")
    assert tts.capabilities.streaming is True


def test_tts_update_options():
    """update_options can change parameters."""
    from livekit.plugins.rumik import TTS

    tts = TTS(api_key="test-key")
    tts.update_options(
        model="mulberry",
        description="warm, friendly voice",
        speaker="speaker_3",
        f0_up_key=2,
    )
    assert tts._opts.model == "mulberry"
    assert tts._opts.description == "warm, friendly voice"
    assert tts._opts.speaker == "speaker_3"
    assert tts._opts.f0_up_key == 2
