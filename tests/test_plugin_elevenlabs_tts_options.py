from __future__ import annotations

import pytest

from livekit.agents.types import NOT_GIVEN
from livekit.agents.utils import is_given
from livekit.plugins.elevenlabs import TTS, Voice, VoiceSettings

pytestmark = pytest.mark.unit


def test_elevenlabs_tts_update_options_preserves_voice_settings_attributes() -> None:
    # Initialize with full voice settings
    tts = TTS(
        api_key="fake-key",
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.75,
            style=0.4,
            speed=1.1,
            use_speaker_boost=True,
        ),
    )

    # Partial update to only stability and similarity_boost
    tts.update_options(
        voice_settings=VoiceSettings(
            stability=0.8,
            similarity_boost=0.9,
        )
    )

    assert is_given(tts._opts.voice_settings)
    assert tts._opts.voice_settings.stability == 0.8
    assert tts._opts.voice_settings.similarity_boost == 0.9
    # Sibling attributes must be preserved rather than wiped to NOT_GIVEN
    assert tts._opts.voice_settings.style == 0.4
    assert tts._opts.voice_settings.speed == 1.1
    assert tts._opts.voice_settings.use_speaker_boost is True


def test_elevenlabs_tts_update_options_preserves_sibling_options() -> None:
    tts = TTS(
        api_key="fake-key",
        model="eleven_multilingual_v2",
        voice_id="custom-voice-id",
        language="en",
        voice_settings=VoiceSettings(stability=0.7, similarity_boost=0.8),
    )

    # 1. Update only voice_id -> preserves model, language, voice_settings
    tts.update_options(voice_id="new-voice-id")
    assert tts._opts.voice_id == "new-voice-id"
    assert tts._opts.model == "eleven_multilingual_v2"
    assert is_given(tts._opts.language) and tts._opts.language.language == "en"
    assert is_given(tts._opts.voice_settings)
    assert tts._opts.voice_settings.stability == 0.7
    assert tts._opts.voice_settings.similarity_boost == 0.8

    # 2. Update only model -> preserves voice_id, language, voice_settings
    tts.update_options(model="eleven_turbo_v2")
    assert tts._opts.model == "eleven_turbo_v2"
    assert tts._opts.voice_id == "new-voice-id"
    assert is_given(tts._opts.language) and tts._opts.language.language == "en"
    assert is_given(tts._opts.voice_settings)
    assert tts._opts.voice_settings.stability == 0.7


def test_elevenlabs_tts_update_options_atomic_on_failure() -> None:
    tts = TTS(
        api_key="fake-key",
        model="eleven_turbo_v2_5",
        voice_id="original-voice",
    )

    # 1. Invalid voice settings stability must fail and leave voice_id untouched
    with pytest.raises(ValueError, match="stability must be between 0.0 and 1.0"):
        tts.update_options(
            voice_id="changed-voice",
            voice_settings=VoiceSettings(stability=5.0, similarity_boost=0.5),
        )
    assert tts._opts.voice_id == "original-voice"

    # 2. Invalid non-string language must fail and leave voice_id untouched
    with pytest.raises(ValueError, match="language must be a string"):
        tts.update_options(voice_id="changed-voice", language=12345)  # type: ignore[arg-type]
    assert tts._opts.voice_id == "original-voice"


def test_elevenlabs_tts_update_options_accepts_voice_param_and_properties() -> None:
    tts = TTS(api_key="fake-key", voice_id="initial-voice")

    assert tts.voice == "initial-voice"
    assert tts.voice_id == "initial-voice"

    # Update using string voice
    tts.update_options(voice="new-voice-str")
    assert tts.voice == "new-voice-str"
    assert tts.voice_id == "new-voice-str"
    assert tts._opts.voice_id == "new-voice-str"

    # Update using Voice object
    tts.update_options(voice=Voice(id="voice-obj-id", name="TestVoice", category="premade"))
    assert tts.voice == "voice-obj-id"
    assert tts.voice_id == "voice-obj-id"
    assert tts._opts.voice_id == "voice-obj-id"


def test_elevenlabs_tts_update_options_dict_voice_settings() -> None:
    tts = TTS(
        api_key="fake-key",
        voice_settings=VoiceSettings(
            stability=0.6,
            similarity_boost=0.7,
            speed=1.0,
        ),
    )

    # Pass dict voice_settings updating only speed
    tts.update_options(voice_settings={"speed": 1.2})
    assert is_given(tts._opts.voice_settings)
    assert tts._opts.voice_settings.speed == 1.2
    assert tts._opts.voice_settings.stability == 0.6
    assert tts._opts.voice_settings.similarity_boost == 0.7


def test_elevenlabs_tts_update_options_clear_language_and_voice_settings() -> None:
    tts = TTS(
        api_key="fake-key",
        language="es",
        voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.5),
    )

    # Clearing language with None
    tts.update_options(language=None)
    assert tts._opts.language is NOT_GIVEN or tts._opts.language is None

    # Clearing voice_settings with None
    tts.update_options(voice_settings=None)
    assert tts._opts.voice_settings is NOT_GIVEN or tts._opts.voice_settings is None


def test_elevenlabs_tts_update_options_when_initial_voice_settings_none() -> None:
    tts = TTS(api_key="fake-key", voice_settings=NOT_GIVEN)
    tts._opts.voice_settings = None  # simulate None instead of NOT_GIVEN

    # Updating voice_settings when initial is None should not crash with replace(None)
    tts.update_options(voice_settings=VoiceSettings(stability=0.8))
    assert tts._opts.voice_settings is not None
    assert tts._opts.voice_settings.stability == 0.8
