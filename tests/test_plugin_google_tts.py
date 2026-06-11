import pytest

pytestmark = pytest.mark.plugin("google")


async def test_pitch_accepts_float():
    from livekit.plugins.google import TTS

    tts = TTS(pitch=1.5)
    assert tts._opts.pitch == 1.5


async def test_pitch_accepts_negative_float():
    from livekit.plugins.google import TTS

    tts = TTS(pitch=-5.5)
    assert tts._opts.pitch == -5.5


async def test_pitch_default_is_zero():
    from livekit.plugins.google import TTS

    tts = TTS()
    assert tts._opts.pitch == 0
