from __future__ import annotations

import pytest

from livekit.agents import LanguageCode
from livekit.plugins import cartesia

pytestmark = pytest.mark.plugin("cartesia")


def test_cartesia_tts_partial_update_preserves_sibling_options() -> None:
    # Initialize with comprehensive non-default configuration
    tts = cartesia.TTS(
        model="sonic-3",
        voice="test-voice-id-1",
        speed=1.25,
        emotion=["Happy:high"],
        volume=1.5,
        language="en",
        pronunciation_dict_id="dict-123",
        add_phoneme_timestamps=True,
        use_normalized_timestamps=True,
        api_key="cartesia-test-key",
    )

    # 1. Update only voice -> preserves model, speed, emotion, volume, etc.
    tts.update_options(voice="test-voice-id-2")
    assert tts._opts.voice == "test-voice-id-2"
    assert tts._opts.model == "sonic-3"
    assert tts._opts.speed == 1.25
    assert tts._opts.emotion == ["Happy:high"]
    assert tts._opts.volume == 1.5
    assert tts._opts.language == LanguageCode("en")
    assert tts._opts.pronunciation_dict_id == "dict-123"
    assert tts._opts.add_phoneme_timestamps is True
    assert tts._opts.use_normalized_timestamps is True

    # 2. Update only speed -> preserves voice, model, emotion, volume
    tts.update_options(speed=1.75)
    assert tts._opts.speed == 1.75
    assert tts._opts.voice == "test-voice-id-2"
    assert tts._opts.model == "sonic-3"
    assert tts._opts.emotion == ["Happy:high"]
    assert tts._opts.volume == 1.5
    assert tts._opts.pronunciation_dict_id == "dict-123"

    # 3. Update only model -> preserves voice, speed, emotion
    tts.update_options(model="sonic-preview")
    assert tts._opts.model == "sonic-preview"
    assert tts._opts.voice == "test-voice-id-2"
    assert tts._opts.speed == 1.75
    assert tts._opts.emotion == ["Happy:high"]


def test_cartesia_tts_update_options_accepts_integer_speed() -> None:
    # update_options with integer speed (e.g. speed=1 or speed=2) must convert to float
    tts = cartesia.TTS(model="sonic-3", api_key="cartesia-test-key")
    tts.update_options(speed=1)
    assert isinstance(tts._opts.speed, float)
    assert tts._opts.speed == 1.0

    tts.update_options(speed=2)
    assert isinstance(tts._opts.speed, float)
    assert tts._opts.speed == 2.0

    # __init__ with integer speed
    tts_int = cartesia.TTS(model="sonic-3", speed=1, api_key="cartesia-test-key")
    assert isinstance(tts_int._opts.speed, float)
    assert tts_int._opts.speed == 1.0


def test_cartesia_tts_update_options_atomic_preserves_state_on_failure() -> None:
    # Partial failure during update_options must NOT corrupt previously configured options
    tts = cartesia.TTS(
        model="sonic-3", voice="initial-voice-id", speed=1.2, api_key="cartesia-test-key"
    )

    with pytest.raises(ValueError, match="speed must be a number for sonic-3"):
        tts.update_options(voice="failed-new-voice", speed="invalid-non-numeric")

    # voice must NOT have been updated to failed-new-voice; must retain initial-voice-id
    assert tts._opts.voice == "initial-voice-id"
    assert tts._opts.speed == 1.2

    # Boolean speed must also be rejected and preserve state
    with pytest.raises(ValueError, match="speed must be a number for sonic-3"):
        tts.update_options(voice="failed-new-voice-2", speed=True)

    assert tts._opts.voice == "initial-voice-id"
    assert tts._opts.speed == 1.2


def test_cartesia_tts_update_options_speed_boundaries() -> None:
    tts = cartesia.TTS(model="sonic-3", api_key="cartesia-test-key")

    tts.update_options(speed=0.6)
    assert tts._opts.speed == 0.6

    tts.update_options(speed=2.0)
    assert tts._opts.speed == 2.0


def test_cartesia_tts_update_options_api_version_invalidation() -> None:
    tts = cartesia.TTS(api_key="cartesia-test-key")
    original_version = tts._opts.api_version

    tts.update_options(api_version="2024-11-13")
    assert tts._opts.api_version == "2024-11-13"
    assert tts._opts.api_version != original_version
