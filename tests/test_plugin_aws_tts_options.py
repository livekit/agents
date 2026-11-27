import pytest

from livekit.plugins import aws

pytestmark = pytest.mark.unit


def test_aws_tts_update_options_preserves_language_when_updating_voice():
    tts = aws.TTS(api_key="fake", api_secret="fake", voice="Aditi", language="hi-IN")
    assert tts._opts.voice == "Aditi"
    assert tts._opts.language is not None
    assert tts._opts.language.language == "hi"

    # Calling update_options with voice alone must preserve already configured language
    tts.update_options(voice="Matthew")
    assert tts._opts.voice == "Matthew"
    assert tts._opts.language is not None
    assert tts._opts.language.language == "hi"

    # Explicitly passing language="" resets it to None
    tts.update_options(language="")
    assert tts._opts.language is None
    assert tts._opts.voice == "Matthew"


def test_aws_tts_update_options_none_language():
    tts = aws.TTS(api_key="fake", api_secret="fake", language="en-IN")
    tts.update_options(language=None)
    assert tts._opts.language is None
