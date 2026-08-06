"""Unit tests for Cartesia TTS aligned-transcript capability narrowing (#6493)."""

from __future__ import annotations

import logging

import pytest

from livekit.agents import LanguageCode
from livekit.plugins.cartesia import tts as cartesia_tts

pytestmark = pytest.mark.plugin("cartesia")


def test_aligned_transcript_enabled_for_supported_language() -> None:
    tts = cartesia_tts.TTS(api_key="test-key", language="en", word_timestamps=True)
    assert tts.capabilities.aligned_transcript is True


def test_aligned_transcript_disabled_for_unsupported_language() -> None:
    tts = cartesia_tts.TTS(api_key="test-key", language="ja", word_timestamps=True)
    assert tts.capabilities.aligned_transcript is False


def test_aligned_transcript_disabled_when_word_timestamps_off() -> None:
    tts = cartesia_tts.TTS(api_key="test-key", language="en", word_timestamps=False)
    assert tts.capabilities.aligned_transcript is False


def test_aligned_transcript_allowed_for_preview_model_any_language() -> None:
    tts = cartesia_tts.TTS(
        api_key="test-key",
        model="sonic-preview",
        language="ja",
        word_timestamps=True,
    )
    assert tts.capabilities.aligned_transcript is True


def test_unsupported_config_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="livekit.plugins.cartesia"):
        tts = cartesia_tts.TTS(api_key="test-key", language="ja", word_timestamps=True)

    assert tts.capabilities.aligned_transcript is False
    assert any(
        "does not support aligned transcript" in record.message for record in caplog.records
    )


def test_update_options_narrows_capability_when_language_becomes_unsupported() -> None:
    tts = cartesia_tts.TTS(api_key="test-key", language="en", word_timestamps=True)
    assert tts.capabilities.aligned_transcript is True

    tts.update_options(language="ja")
    assert tts.capabilities.aligned_transcript is False


def test_update_options_restores_capability_when_language_becomes_supported() -> None:
    tts = cartesia_tts.TTS(api_key="test-key", language="ja", word_timestamps=True)
    assert tts.capabilities.aligned_transcript is False

    tts.update_options(language="fr")
    assert tts.capabilities.aligned_transcript is True


def test_update_options_to_preview_model_enables_capability() -> None:
    tts = cartesia_tts.TTS(
        api_key="test-key",
        model="sonic-3",
        language="ja",
        word_timestamps=True,
    )
    assert tts.capabilities.aligned_transcript is False

    tts.update_options(model="sonic-preview")
    assert tts.capabilities.aligned_transcript is True


def test_add_timestamps_omitted_when_unsupported() -> None:
    tts = cartesia_tts.TTS(api_key="test-key", language="ja", word_timestamps=True)
    options = cartesia_tts._to_cartesia_options(tts._opts, streaming=True)
    assert options["add_timestamps"] is False


def test_add_timestamps_requested_when_supported() -> None:
    tts = cartesia_tts.TTS(api_key="test-key", language="en", word_timestamps=True)
    options = cartesia_tts._to_cartesia_options(tts._opts, streaming=True)
    assert options["add_timestamps"] is True


def test_supports_word_timestamps_helper() -> None:
    assert cartesia_tts._supports_word_timestamps("sonic-3", LanguageCode("en")) is True
    assert cartesia_tts._supports_word_timestamps("sonic-3", LanguageCode("ja")) is False
    assert cartesia_tts._supports_word_timestamps("sonic-preview", LanguageCode("ja")) is True
    assert cartesia_tts._supports_word_timestamps("sonic-3", None) is True
