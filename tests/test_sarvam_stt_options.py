"""Hermetic tests for Sarvam STT model capabilities (no network, no credentials).

Regression tests for issue #6606: ``saaras:v3`` connects to
``wss://api.sarvam.ai/speech-to-text/ws``, whose schema documents no
``config``/``prompt`` message — only the legacy translate endpoint used by
``saaras:v2.5`` does. The plugin used to mark ``saaras:v3`` as
``supports_prompt=True`` and send a config message the endpoint silently
drops, so callers built hotword biasing on a parameter that was never wired
to anything.
"""

import logging

import pytest

from livekit.plugins.sarvam.stt import (
    SARVAM_STT_STREAMING_URL,
    SARVAM_STT_TRANSLATE_STREAMING_URL,
    SarvamSTTOptions,
    _get_urls_for_model,
    _model_supports_prompt,
)

pytestmark = pytest.mark.unit


class TestModelSupportsPrompt:
    def test_saaras_v3_does_not_support_prompt(self) -> None:
        # /speech-to-text/ws has no config/prompt message in its schema
        assert _model_supports_prompt("saaras:v3") is False

    def test_saaras_v25_supports_prompt(self) -> None:
        # the legacy translate endpoint documents the config/prompt message
        assert _model_supports_prompt("saaras:v2.5") is True

    def test_saarika_does_not_support_prompt(self) -> None:
        assert _model_supports_prompt("saarika:v2.5") is False

    def test_unknown_model_does_not_support_prompt(self) -> None:
        # unknown models are routed to the non-translate endpoint, which has
        # no config/prompt message — regardless of the model-family name
        assert _model_supports_prompt("saaras:v99") is False
        assert _model_supports_prompt("some-future-model") is False

    def test_prompt_support_matches_translate_endpoint(self) -> None:
        # the config/prompt message only exists on the translate endpoint;
        # any model claiming prompt support must connect there
        from livekit.plugins.sarvam.stt import MODEL_CONFIGS

        for model, config in MODEL_CONFIGS.items():
            if config.supports_prompt:
                _, streaming_url = _get_urls_for_model(model)
                assert streaming_url == SARVAM_STT_TRANSLATE_STREAMING_URL, (
                    f"{model} claims prompt support but connects to {streaming_url}, "
                    "which has no config/prompt message"
                )


class TestEndpointRouting:
    def test_saaras_v3_uses_speech_to_text_endpoint(self) -> None:
        _, streaming_url = _get_urls_for_model("saaras:v3")
        assert streaming_url == SARVAM_STT_STREAMING_URL

    def test_saaras_v25_uses_translate_endpoint(self) -> None:
        _, streaming_url = _get_urls_for_model("saaras:v2.5")
        assert streaming_url == SARVAM_STT_TRANSLATE_STREAMING_URL


class TestPromptWarning:
    def test_warns_when_prompt_set_for_unsupported_model(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="livekit.plugins.sarvam"):
            SarvamSTTOptions(
                language="en-IN",
                api_key="test-key",
                model="saaras:v3",
                prompt="guests, one, two, three",
            )
        assert any("prompt is ignored" in r.message for r in caplog.records)

    def test_no_warning_for_supported_model(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="livekit.plugins.sarvam"):
            SarvamSTTOptions(
                language="en-IN",
                api_key="test-key",
                model="saaras:v2.5",
                prompt="guests, one, two, three",
            )
        assert not any("prompt is ignored" in r.message for r in caplog.records)

    def test_no_warning_without_prompt(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="livekit.plugins.sarvam"):
            SarvamSTTOptions(language="en-IN", api_key="test-key", model="saaras:v3")
        assert not any("prompt is ignored" in r.message for r in caplog.records)
