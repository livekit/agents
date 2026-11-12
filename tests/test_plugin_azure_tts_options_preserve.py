import pytest

from livekit.agents import LanguageCode
from livekit.plugins.azure.tts import (
    NOT_GIVEN,
    TTS,
    ChunkedStream,
    ProsodyConfig,
    StyleConfig,
)

pytestmark = pytest.mark.plugin("azure")


@pytest.fixture(autouse=True)
def _clear_azure_env(monkeypatch):
    for var in ("AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION", "AZURE_SPEECH_ENDPOINT"):
        monkeypatch.delenv(var, raising=False)


def test_azure_tts_update_options_preserves_sibling_fields():
    prosody = ProsodyConfig(rate=1.2)
    style = StyleConfig(style="cheerful")
    tts = TTS(
        speech_key="k",
        speech_region="eastus",
        voice="en-US-JennyNeural",
        language="en-US",
        prosody=prosody,
        style=style,
        lexicon_uri="https://example.com/lex.xml",
    )

    # Updating only voice should preserve language, prosody, style, lexicon_uri
    tts.update_options(voice="en-US-GuyNeural")
    assert tts._opts.voice == "en-US-GuyNeural"
    assert tts._opts.language == LanguageCode("en-US")
    assert tts._opts.prosody == prosody
    assert tts._opts.style == style
    assert tts._opts.lexicon_uri == "https://example.com/lex.xml"


def test_azure_tts_update_options_clear_language():
    tts = TTS(speech_key="k", speech_region="eastus", language="en-US")
    assert tts._opts.language == LanguageCode("en-US")

    # Passing language=None should clear language to None without crashing
    tts.update_options(language=None)
    assert tts._opts.language is None

    # Setting language again
    tts.update_options(language="fr-FR")
    assert tts._opts.language == LanguageCode("fr-FR")

    # Passing empty string should also clear language to None
    tts.update_options(language="")
    assert tts._opts.language is None


def test_azure_tts_update_options_atomic_on_style_validation_failure():
    tts = TTS(speech_key="k", speech_region="eastus", voice="original-voice")
    bad_style = StyleConfig(style="cheerful")
    bad_style.degree = 10.0  # Invalid degree (> 2.0)

    with pytest.raises(ValueError, match="Style degree must be between 0.1 and 2.0"):
        tts.update_options(voice="new-voice", style=bad_style)

    # voice must NOT have been mutated
    assert tts._opts.voice == "original-voice"
    assert tts._opts.style is NOT_GIVEN


def test_azure_tts_update_options_atomic_on_prosody_validation_failure():
    tts = TTS(speech_key="k", speech_region="eastus", voice="original-voice")
    bad_prosody = ProsodyConfig(rate=1.0)
    bad_prosody.rate = 10.0  # Invalid rate (> 2.0)

    with pytest.raises(ValueError, match="Prosody rate must be between 0.5 and 2"):
        tts.update_options(voice="new-voice", prosody=bad_prosody)

    # voice must NOT have been mutated
    assert tts._opts.voice == "original-voice"
    assert tts._opts.prosody is NOT_GIVEN


def test_azure_tts_update_options_clear_optional_configs():
    tts = TTS(
        speech_key="k",
        speech_region="eastus",
        prosody=ProsodyConfig(rate=1.1),
        style=StyleConfig(style="empathetic"),
        lexicon_uri="https://example.com/lex.xml",
    )
    assert tts._opts.prosody is not None
    assert tts._opts.style is not None
    assert tts._opts.lexicon_uri is not None

    # Clearing configs with None should not crash on .validate()
    tts.update_options(prosody=None, style=None, lexicon_uri=None)
    assert tts._opts.prosody is None
    assert tts._opts.style is None
    assert tts._opts.lexicon_uri is None


def test_azure_tts_init_with_none_configs():
    tts = TTS(
        speech_key="k",
        speech_region="eastus",
        language=None,
        prosody=None,
        style=None,
        lexicon_uri=None,
    )
    assert tts._opts.language is None
    assert tts._opts.prosody is None
    assert tts._opts.style is None
    assert tts._opts.lexicon_uri is None


def test_azure_tts_build_ssml_with_cleared_configs():
    tts = TTS(
        speech_key="k",
        speech_region="eastus",
        prosody=ProsodyConfig(rate=1.2),
        style=StyleConfig(style="cheerful"),
        lexicon_uri="https://example.com/lex.xml",
    )
    tts.update_options(prosody=None, style=None, lexicon_uri=None)

    stream = ChunkedStream.__new__(ChunkedStream)
    stream._tts = tts
    stream._input_text = "test message"
    stream._opts = tts._opts

    ssml = stream._build_ssml()
    assert "<lexicon" not in ssml
    assert "<mstts:express-as" not in ssml
    assert "<prosody" not in ssml
    assert "test message" in ssml
