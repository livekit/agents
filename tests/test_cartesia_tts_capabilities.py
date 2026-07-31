"""Hermetic tests for cartesia's aligned_transcript capability narrowing (#6493).

``word_timestamps=True`` (the default) used to set
``capabilities.aligned_transcript=True`` unconditionally, even for
model/language combinations Cartesia documents as unable to return timing
data — the plugin warned about the combination but left the capability set,
so every turn downstream selected the TTS-aligned transcript path and logged
"no agent transcript was returned from tts".
"""

import logging

import pytest

from livekit.plugins.cartesia import TTS
from livekit.plugins.cartesia.tts import _supports_word_timestamps

pytestmark = pytest.mark.unit


class TestSupportsWordTimestamps:
    def test_supported_languages_on_sonic(self) -> None:
        for lang in ("en", "de", "es", "fr"):
            assert _supports_word_timestamps("sonic-3", lang) is True

    def test_unsupported_language_on_sonic(self) -> None:
        assert _supports_word_timestamps("sonic-3", "hi") is False
        assert _supports_word_timestamps("sonic-3", "tr") is False

    def test_preview_models_support_all_languages(self) -> None:
        assert _supports_word_timestamps("sonic-preview", "hi") is True

    def test_language_variants_normalize(self) -> None:
        # locale-tagged languages narrow to their base language code
        assert _supports_word_timestamps("sonic-3", "en-US") is True

    def test_no_language_is_supported(self) -> None:
        assert _supports_word_timestamps("sonic-3", None) is True


class TestCapabilityNarrowing:
    def test_capability_set_for_supported_combo(self) -> None:
        t = TTS(api_key="test-key", language="en")
        assert t.capabilities.aligned_transcript is True

    def test_capability_cleared_for_unsupported_combo(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="livekit.plugins.cartesia"):
            t = TTS(api_key="test-key", language="hi")
        assert t.capabilities.aligned_transcript is False
        # the request must not ask for timestamps the combo can't deliver
        assert t._opts.word_timestamps is False
        assert any("word_timestamps disabled" in r.message for r in caplog.records)

    def test_capability_kept_for_preview_model_any_language(self) -> None:
        t = TTS(api_key="test-key", model="sonic-preview", language="hi")
        assert t.capabilities.aligned_transcript is True

    def test_explicit_opt_out_unchanged(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="livekit.plugins.cartesia"):
            t = TTS(api_key="test-key", language="hi", word_timestamps=False)
        assert t.capabilities.aligned_transcript is False
        assert not any("word_timestamps disabled" in r.message for r in caplog.records)
