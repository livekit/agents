from __future__ import annotations

import pytest

from livekit.plugins.hume import TTS, VoiceByName

pytestmark = pytest.mark.unit

VOICE = VoiceByName(name="Ava Song")
MESSAGE = "instant_mode cannot be enabled without specifying a voice"


def _tts(**kwargs) -> TTS:  # type: ignore[no-untyped-def]
    return TTS(api_key="test-key", **kwargs)


def test_update_options_rejects_what_the_constructor_rejects() -> None:
    with pytest.raises(ValueError, match=MESSAGE):
        _tts(voice=None, instant_mode=True)

    # turning instant mode on without a voice
    tts = _tts(voice=None)
    assert tts._opts.instant_mode is False
    with pytest.raises(ValueError, match=MESSAGE):
        tts.update_options(instant_mode=True)
    assert tts._opts.instant_mode is False

    # clearing the voice while instant mode stays on, alone or in the same call
    tts = _tts(voice=VOICE)
    assert tts._opts.instant_mode is True
    with pytest.raises(ValueError, match=MESSAGE):
        tts.update_options(voice=None)
    with pytest.raises(ValueError, match=MESSAGE):
        tts.update_options(voice=None, instant_mode=True, speed=1.5)
    # a rejected update leaves every option unchanged
    assert tts._opts.voice == VOICE
    assert tts._opts.instant_mode is True
    assert tts._opts.speed is None


def test_update_options_accepts_a_consistent_voice_and_instant_mode() -> None:
    tts = _tts(voice=VOICE)
    tts.update_options(voice=None, instant_mode=False)
    assert tts._opts.voice is None
    assert tts._opts.instant_mode is False

    tts.update_options(voice=VOICE, instant_mode=True)
    assert tts._opts.voice == VOICE
    assert tts._opts.instant_mode is True

    # a voice can change on its own with instant mode on
    other = VoiceByName(name="Colton Rivers")
    tts.update_options(voice=other)
    assert tts._opts.voice == other
    assert tts._opts.instant_mode is True
