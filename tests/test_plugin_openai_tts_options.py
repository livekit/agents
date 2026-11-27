from __future__ import annotations

import pytest

from livekit.plugins.openai import TTS

pytestmark = pytest.mark.unit


def test_openai_tts_partial_update_preserves_sibling_options() -> None:
    # Initialize with non-default options
    tts = TTS(
        model="tts-1-hd",
        voice="echo",
        speed=1.5,
        instructions="speak in a cheerful tone",
        response_format="flac",
        api_key="fake-test-key",
    )

    # 1. Update only voice -> preserves model, speed, instructions, response_format
    tts.update_options(voice="alloy")
    assert tts._opts.voice == "alloy"
    assert tts._opts.model == "tts-1-hd"
    assert tts._opts.speed == 1.5
    assert tts._opts.instructions == "speak in a cheerful tone"
    assert tts._opts.response_format == "flac"

    # 2. Update only speed -> preserves voice, model, instructions, response_format
    tts.update_options(speed=1.2)
    assert tts._opts.speed == 1.2
    assert tts._opts.voice == "alloy"
    assert tts._opts.model == "tts-1-hd"
    assert tts._opts.instructions == "speak in a cheerful tone"
    assert tts._opts.response_format == "flac"

    # 3. Update only model -> preserves voice, speed, instructions, response_format
    tts.update_options(model="tts-1")
    assert tts._opts.model == "tts-1"
    assert tts._opts.voice == "alloy"
    assert tts._opts.speed == 1.2
    assert tts._opts.instructions == "speak in a cheerful tone"
    assert tts._opts.response_format == "flac"


def test_openai_tts_update_options_accepts_integer_speed() -> None:
    # Calling with integer speed (e.g. speed=2 or speed=1) must convert to float
    tts = TTS(api_key="fake-test-key")
    tts.update_options(speed=2)
    assert isinstance(tts._opts.speed, float)
    assert tts._opts.speed == 2.0

    # __init__ with integer speed
    tts2 = TTS(speed=1, api_key="fake-test-key")
    assert isinstance(tts2._opts.speed, float)
    assert tts2._opts.speed == 1.0


def test_openai_tts_update_options_speed_boundary_validation() -> None:
    tts = TTS(api_key="fake-test-key")

    # Invalid speed values (< 0.25, > 4.0, non-numeric) must raise ValueError
    with pytest.raises(ValueError, match="speed must be between 0.25 and 4.0"):
        tts.update_options(speed=0.1)

    with pytest.raises(ValueError, match="speed must be between 0.25 and 4.0"):
        tts.update_options(speed=5.0)

    with pytest.raises(ValueError, match="speed must be between 0.25 and 4.0"):
        tts.update_options(speed=0.0)

    # Valid boundaries (0.25 and 4.0) must succeed
    tts.update_options(speed=0.25)
    assert tts._opts.speed == 0.25

    tts.update_options(speed=4.0)
    assert tts._opts.speed == 4.0


def test_openai_tts_update_options_atomic_preserves_state_on_failure() -> None:
    tts = TTS(voice="ash", speed=1.0, api_key="fake-test-key")

    # Attempting to update voice with an invalid speed must fail and preserve original voice
    with pytest.raises(ValueError):
        tts.update_options(voice="alloy", speed=10.0)

    # Original options must be untouched (not partially mutated)
    assert tts._opts.voice == "ash"
    assert tts._opts.speed == 1.0


def test_openai_tts_update_options_supports_response_format() -> None:
    tts = TTS(response_format="mp3", api_key="fake-test-key")
    assert tts._opts.response_format == "mp3"

    tts.update_options(response_format="wav")
    assert tts._opts.response_format == "wav"
    assert tts._opts.model == "gpt-4o-mini-tts"
    assert tts._opts.voice == "ash"


def test_openai_tts_update_options_can_clear_instructions() -> None:
    tts = TTS(instructions="speak softly", api_key="fake-test-key")
    assert tts._opts.instructions == "speak softly"

    tts.update_options(instructions=None)
    assert tts._opts.instructions is None
    assert tts._opts.voice == "ash"
