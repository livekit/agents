from __future__ import annotations

import pytest

from livekit.plugins.hume import TTS, VoiceByName

pytestmark = pytest.mark.unit


def test_update_options_rejects_instant_mode_without_a_voice():
    tts = TTS(api_key="test-key", voice=None)
    with pytest.raises(ValueError, match="instant_mode cannot be enabled without"):
        tts.update_options(instant_mode=True)
    assert tts._opts.instant_mode is False

    tts = TTS(api_key="test-key")
    voice = tts._opts.voice
    with pytest.raises(ValueError, match="instant_mode cannot be enabled without"):
        tts.update_options(voice=None, speed=1.5)
    assert tts._opts.voice == voice
    assert tts._opts.speed is None


def test_update_options_accepts_a_consistent_voice_and_instant_mode():
    tts = TTS(api_key="test-key")
    tts.update_options(voice=None, instant_mode=False)
    assert tts._opts.voice is None
    assert tts._opts.instant_mode is False

    voice = VoiceByName(name="Ava Song")
    tts.update_options(voice=voice, instant_mode=True)
    assert tts._opts.voice == voice
    assert tts._opts.instant_mode is True
