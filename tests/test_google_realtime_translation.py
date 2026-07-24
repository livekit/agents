"""Hermetic tests for the Gemini Realtime Translation plugin.

The translate model reuses the conversational ``RealtimeSession``'s turn-based
lifecycle wholesale (connection, receive, generation, transcription, metrics); only
the connect config differs. These tests therefore cover that delta — the connect
config and capabilities — without touching the network. The turn handling itself is
the conversational realtime model's responsibility.
"""

from __future__ import annotations

import pytest
from google.genai import types

from livekit.plugins.google.realtime.translation_model import (
    DEFAULT_TRANSLATION_MODEL,
    RealtimeTranslationModel,
    RealtimeTranslationSession,
)

pytestmark = pytest.mark.plugin("google")


def test_model_capabilities_and_properties() -> None:
    model = RealtimeTranslationModel(target_language="de", api_key="test-api-key", vertexai=False)
    assert model.target_language == "de"
    assert model.echo_target_language is True
    assert model.model == DEFAULT_TRANSLATION_MODEL
    assert model.provider == "Gemini"

    caps = model.capabilities
    assert caps.turn_detection is True
    assert caps.audio_output is True
    assert caps.user_transcription is True
    assert caps.message_truncation is False
    # a translator carries no mutable instructions / chat context / tools
    assert caps.mutable_chat_context is False
    assert caps.mutable_instructions is False
    assert caps.auto_tool_reply_generation is False


def test_connect_config_is_minimal_translation() -> None:
    model = RealtimeTranslationModel(
        target_language="es", echo_target_language=False, api_key="test-api-key", vertexai=False
    )
    # __new__ avoids starting the network task; _build_connect_config only reads the model
    sess = RealtimeTranslationSession.__new__(RealtimeTranslationSession)
    sess._realtime_model = model

    conf = sess._build_connect_config()
    assert conf.translation_config is not None
    assert conf.translation_config.target_language_code == "es"
    assert conf.translation_config.echo_target_language is False
    assert conf.response_modalities == [types.Modality.AUDIO]
    assert conf.input_audio_transcription is not None
    assert conf.output_audio_transcription is not None
    # conversational config is never sent for translation
    assert conf.system_instruction is None
    assert conf.tools is None
    assert conf.speech_config is None
    assert conf.session_resumption is None


def test_update_target_language() -> None:
    model = RealtimeTranslationModel(target_language="es", api_key="test-api-key", vertexai=False)
    model.update_options(target_language="fr")
    assert model.target_language == "fr"

    sess = RealtimeTranslationSession.__new__(RealtimeTranslationSession)
    sess._realtime_model = model
    conf = sess._build_connect_config()
    assert conf.translation_config is not None
    assert conf.translation_config.target_language_code == "fr"
