from __future__ import annotations

import pytest

pytestmark = pytest.mark.plugin("google")


def _tts(**kwargs):
    from livekit.plugins.google import TTS

    return TTS(use_streaming=False, **kwargs)


async def test_update_voice_name_keeps_language_and_model():
    tts = _tts(language="fr-FR", voice_name="fr-FR-Chirp3-HD-Aoede")
    tts.update_options(voice_name="fr-FR-Chirp3-HD-Puck")

    assert tts._opts.voice.name == "fr-FR-Chirp3-HD-Puck"
    assert tts._opts.voice.language_code == "fr-FR"
    assert tts._opts.voice.model_name == ""  # unset for Chirp 3
    await tts.aclose()


async def test_update_language_keeps_voice_name():
    tts = _tts(language="fr-FR", voice_name="Puck", model_name="gemini-2.5-flash-tts")
    tts.update_options(language="de-DE")

    assert tts._opts.voice.language_code == "de-DE"
    assert tts._opts.voice.name == "Puck"
    assert tts._opts.voice.model_name == "gemini-2.5-flash-tts"
    await tts.aclose()


async def test_update_model_name_to_chirp_3_clears_voice_model_name():
    tts = _tts(voice_name="Puck", model_name="gemini-2.5-flash-tts")
    tts.update_options(model_name="chirp_3")

    assert tts._opts.model_name == "chirp_3"
    assert tts._opts.voice.model_name == ""
    assert tts._opts.voice.name == "Puck"
    await tts.aclose()
