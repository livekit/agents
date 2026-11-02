import os

import pytest

from livekit.plugins import cartesia

os.environ.setdefault("CARTESIA_API_KEY", "fake-cartesia-api-key")

pytestmark = pytest.mark.unit


def test_cartesia_tts_integer_speed():
    tts = cartesia.TTS(speed=1)
    assert tts._opts.speed == 1.0


def test_cartesia_tts_update_options_speed():
    tts = cartesia.TTS(speed=1.0)
    tts.update_options(speed=2)
    assert tts._opts.speed == 2.0


def test_cartesia_tts_update_options_atomic_failure():
    tts = cartesia.TTS(speed=1.0)
    with pytest.raises(ValueError):
        tts.update_options(speed="fast")
    assert tts._opts.speed == 1.0


def test_cartesia_tts_update_options_boolean_speed_failure():
    tts = cartesia.TTS(speed=1.0)
    with pytest.raises(ValueError):
        tts.update_options(speed=False)
    assert tts._opts.speed == 1.0
