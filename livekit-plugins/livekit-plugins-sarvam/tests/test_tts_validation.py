"""Tests for Sarvam TTS parameter validation (speakers, sample rates, model-specific pace)."""

from __future__ import annotations

import pytest

from livekit.plugins.sarvam import TTS

pytestmark = pytest.mark.unit


def test_v3_speaker_validation() -> None:
    """Test that all newly added v3 male speakers pass validation and amelia/sophia are rejected."""
    # Valid v3 male speakers
    for speaker in ["anand", "tarun", "sunny", "mani", "gokul", "vijay", "mohit", "rehan", "soham"]:
        tts = TTS(api_key="test-key", model="bulbul:v3", speaker=speaker)
        assert tts._opts.speaker == speaker

    # Invalid v3 speakers (amelia/sophia belong to v3-beta only)
    for speaker in ["amelia", "sophia"]:
        with pytest.raises(ValueError, match="is not compatible with model 'bulbul:v3'"):
            TTS(api_key="test-key", model="bulbul:v3", speaker=speaker)


@pytest.mark.asyncio
async def test_streaming_sample_rate_gating() -> None:
    """Test that REST-only sample rates (32000, 44100, 48000) are allowed in constructor but fail on stream()."""
    tts_rest = TTS(api_key="test-key", model="bulbul:v3", speech_sample_rate=48000)
    assert tts_rest._opts.speech_sample_rate == 48000

    with pytest.raises(ValueError, match="REST-only"):
        tts_rest.stream()

    # Valid streaming sample rates
    for rate in [8000, 16000, 22050, 24000]:
        tts_stream = TTS(api_key="test-key", model="bulbul:v3", speech_sample_rate=rate)
        # Verify stream creation succeeds without raising ValueError
        stream = tts_stream.stream()
        assert stream is not None


def test_model_specific_pace_validation() -> None:
    """Test that pace is validated per model (0.5-2.0 for v3/v3-beta, 0.3-3.0 for v2)."""
    # bulbul:v3 rejects pace > 2.0
    with pytest.raises(ValueError, match="Pace must be between 0.5 and 2.0"):
        TTS(api_key="test-key", model="bulbul:v3", pace=2.5)

    with pytest.raises(ValueError, match="Pace must be between 0.5 and 2.0"):
        TTS(api_key="test-key", model="bulbul:v3-beta", pace=0.2)

    # bulbul:v3 accepts pace 1.5
    tts_v3 = TTS(api_key="test-key", model="bulbul:v3", pace=1.5)
    assert tts_v3._opts.pace == 1.5

    # bulbul:v2 accepts pace 2.5
    tts_v2 = TTS(api_key="test-key", model="bulbul:v2", pace=2.5)
    assert tts_v2._opts.pace == 2.5
