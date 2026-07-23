from dataclasses import replace

from livekit.plugins import azure
from livekit.plugins.azure import tts as azure_tts


def _build_ssml(*, voice: str, language: str | None = None, text: str = "Merhaba") -> str:
    tts = azure.TTS(
        voice=voice,
        language=language,
        speech_key="test-key",
        speech_region="westus",
    )
    stream = object.__new__(azure_tts.ChunkedStream)
    stream._opts = replace(tts._opts)
    stream._input_text = text
    return stream._build_ssml()


def test_azure_tts_wraps_cross_locale_multilingual_voice_with_lang_tag():
    ssml = _build_ssml(
        voice="en-US-JennyMultilingualNeural",
        language="tr-TR",
    )

    assert '<speak version="1.0"' in ssml
    assert 'xml:lang="en-US"' in ssml
    assert '<voice name="en-US-JennyMultilingualNeural">' in ssml
    assert '<lang xml:lang="tr-TR">Merhaba</lang>' in ssml


def test_azure_tts_does_not_wrap_same_locale_multilingual_voice():
    ssml = _build_ssml(
        voice="en-US-JennyMultilingualNeural",
        language="en-US",
    )

    assert '<speak version="1.0"' in ssml
    assert 'xml:lang="en-US"' in ssml
    assert '<voice name="en-US-JennyMultilingualNeural">Merhaba</voice>' in ssml
    assert "<lang xml:lang=" not in ssml


def test_azure_tts_does_not_wrap_non_multilingual_voice():
    ssml = _build_ssml(
        voice="tr-TR-EmelNeural",
        language="tr-TR",
    )

    assert '<speak version="1.0"' in ssml
    assert 'xml:lang="tr-TR"' in ssml
    assert '<voice name="tr-TR-EmelNeural">Merhaba</voice>' in ssml
    assert "<lang xml:lang=" not in ssml


def test_azure_tts_keeps_default_language_without_explicit_language():
    ssml = _build_ssml(
        voice="tr-TR-EmelNeural",
        language=None,
    )

    assert '<speak version="1.0"' in ssml
    assert 'xml:lang="en-US"' in ssml
    assert '<voice name="tr-TR-EmelNeural">Merhaba</voice>' in ssml
    assert "<lang xml:lang=" not in ssml


def test_azure_tts_does_not_wrap_multilingual_voice_without_explicit_language():
    ssml = _build_ssml(
        voice="fr-FR-VivienneMultilingualNeural",
        language=None,
    )

    assert '<speak version="1.0"' in ssml
    assert 'xml:lang="en-US"' in ssml
    assert '<voice name="fr-FR-VivienneMultilingualNeural">Merhaba</voice>' in ssml
    assert "<lang xml:lang=" not in ssml
