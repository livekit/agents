from __future__ import annotations

import socket
from unittest.mock import patch

import pytest
from google.cloud import texttospeech

from livekit.plugins.google.tts import TTS

pytestmark = pytest.mark.plugin("google")


def test_default_initialization() -> None:
    tts = TTS()
    assert tts._opts.voice.language_code == "en-US"
    assert tts._opts.voice.name == "Charon"
    assert tts._opts.voice.ssml_gender == texttospeech.SsmlVoiceGender.NEUTRAL
    assert tts._opts.voice.model_name == "gemini-2.5-flash-tts"
    assert "model_name" in tts._opts.voice
    assert tts.model == "gemini-2.5-flash-tts"
    assert tts.provider == "Google Cloud Platform"
    assert tts._opts.speaking_rate == 1.0
    assert tts._opts.volume_gain_db == 0.0
    assert tts._opts.sample_rate == 24000


def test_update_options_preserves_attributes_on_language_change() -> None:
    # Scenario A: Customized initial parameters
    tts = TTS(voice_name="Puck", gender="male", model_name="gemini-2.5-flash-tts")
    tts.update_options(language="es-ES")
    assert tts._opts.voice.language_code == "es-ES"
    assert tts._opts.voice.name == "Puck"
    assert tts._opts.voice.ssml_gender == texttospeech.SsmlVoiceGender.MALE
    assert tts._opts.voice.model_name == "gemini-2.5-flash-tts"
    assert tts.model == "gemini-2.5-flash-tts"

    # Scenario B: Default initial parameters
    tts_default = TTS()
    tts_default.update_options(language="es-ES")
    assert tts_default._opts.voice.language_code == "es-ES"
    assert tts_default._opts.voice.name == "Charon"
    assert tts_default._opts.voice.ssml_gender == texttospeech.SsmlVoiceGender.NEUTRAL
    assert tts_default._opts.voice.model_name == "gemini-2.5-flash-tts"
    assert tts_default.model == "gemini-2.5-flash-tts"


def test_update_options_preserves_attributes_on_voice_name_change() -> None:
    tts = TTS(language="fr-FR", gender="female")
    tts.update_options(voice_name="Charon")
    assert tts._opts.voice.name == "Charon"
    assert tts._opts.voice.language_code == "fr-FR"
    assert tts._opts.voice.ssml_gender == texttospeech.SsmlVoiceGender.FEMALE
    assert tts._opts.voice.model_name == "gemini-2.5-flash-tts"
    assert tts.model == "gemini-2.5-flash-tts"

    # Update to another voice name
    tts.update_options(voice_name="Aoede")
    assert tts._opts.voice.name == "Aoede"
    assert tts._opts.voice.language_code == "fr-FR"
    assert tts._opts.voice.ssml_gender == texttospeech.SsmlVoiceGender.FEMALE


def test_update_options_preserves_attributes_on_gender_change() -> None:
    tts = TTS(voice_name="Charon", language="en-US")
    tts.update_options(gender="female")
    assert tts._opts.voice.ssml_gender == texttospeech.SsmlVoiceGender.FEMALE
    assert tts._opts.voice.name == "Charon"
    assert tts._opts.voice.language_code == "en-US"
    assert tts._opts.voice.model_name == "gemini-2.5-flash-tts"


def test_update_options_chirp_3_omits_model_name_from_voice_params() -> None:
    tts = TTS(model_name="gemini-2.5-flash-tts")
    assert "model_name" in tts._opts.voice
    assert tts._opts.voice.model_name == "gemini-2.5-flash-tts"

    tts.update_options(model_name="chirp_3")
    assert tts.model == "chirp_3"
    assert tts._opts.model_name == "chirp_3"
    assert "model_name" not in tts._opts.voice
    assert tts._opts.voice.model_name == ""
    assert tts._opts.voice.name == "en-US-Chirp3-HD-Charon"
    assert tts._opts.voice.language_code == "en-US"


def test_update_options_switch_from_chirp_3_to_gemini_restores_model_name() -> None:
    tts = TTS(model_name="chirp_3")
    assert "model_name" not in tts._opts.voice
    assert tts.model == "chirp_3"

    tts.update_options(model_name="gemini-2.5-flash-tts")
    assert tts.model == "gemini-2.5-flash-tts"
    assert tts._opts.model_name == "gemini-2.5-flash-tts"
    assert "model_name" in tts._opts.voice
    assert tts._opts.voice.model_name == "gemini-2.5-flash-tts"


def test_update_options_chirp_voice_name_infers_chirp3_model() -> None:
    tts = TTS(model_name="gemini-2.5-flash-tts")
    assert "model_name" in tts._opts.voice

    tts.update_options(voice_name="en-US-Chirp3-HD-Charon")
    assert tts.model == "chirp_3"
    assert tts._opts.model_name == "chirp_3"
    assert "model_name" not in tts._opts.voice
    assert tts._opts.voice.name == "en-US-Chirp3-HD-Charon"


def test_update_options_preserves_voice_clone() -> None:
    tts = TTS(voice_cloning_key="test_clone_key")
    assert tts._opts.voice.voice_clone.voice_cloning_key == "test_clone_key"

    # Updating language retains voice_clone
    tts.update_options(language="de-DE")
    assert tts._opts.voice.voice_clone.voice_cloning_key == "test_clone_key"
    assert tts._opts.voice.language_code == "de-DE"

    # Updating voice_name clears voice_clone to prevent conflict
    tts.update_options(voice_name="Puck")
    assert "voice_clone" not in tts._opts.voice
    assert tts._opts.voice.name == "Puck"


def test_speaking_rate_boundary_validation() -> None:
    # __init__ validation: invalid values (< 0.25, > 4.0)
    with pytest.raises(ValueError, match="speaking_rate"):
        TTS(speaking_rate=0.24)
    with pytest.raises(ValueError, match="speaking_rate"):
        TTS(speaking_rate=0.0)
    with pytest.raises(ValueError, match="speaking_rate"):
        TTS(speaking_rate=-1.0)
    with pytest.raises(ValueError, match="speaking_rate"):
        TTS(speaking_rate=4.01)
    with pytest.raises(ValueError, match="speaking_rate"):
        TTS(speaking_rate=10.0)

    # __init__ validation: valid boundaries (0.25, 4.0)
    tts_low = TTS(speaking_rate=0.25)
    assert tts_low._opts.speaking_rate == 0.25
    tts_high = TTS(speaking_rate=4.0)
    assert tts_high._opts.speaking_rate == 4.0

    # update_options validation
    tts = TTS()
    with pytest.raises(ValueError, match="speaking_rate"):
        tts.update_options(speaking_rate=0.24)
    with pytest.raises(ValueError, match="speaking_rate"):
        tts.update_options(speaking_rate=4.01)

    # Valid boundaries in update_options
    tts.update_options(speaking_rate=0.25)
    assert tts._opts.speaking_rate == 0.25
    tts.update_options(speaking_rate=4.0)
    assert tts._opts.speaking_rate == 4.0


def test_volume_gain_db_boundary_validation() -> None:
    # __init__ validation: invalid values (< -96.0, > 16.0)
    with pytest.raises(ValueError, match="volume_gain_db"):
        TTS(volume_gain_db=-96.1)
    with pytest.raises(ValueError, match="volume_gain_db"):
        TTS(volume_gain_db=-100.0)
    with pytest.raises(ValueError, match="volume_gain_db"):
        TTS(volume_gain_db=16.1)
    with pytest.raises(ValueError, match="volume_gain_db"):
        TTS(volume_gain_db=50.0)

    # __init__ validation: valid boundaries (-96.0, 16.0)
    tts_low = TTS(volume_gain_db=-96.0)
    assert tts_low._opts.volume_gain_db == -96.0
    tts_high = TTS(volume_gain_db=16.0)
    assert tts_high._opts.volume_gain_db == 16.0

    # update_options validation
    tts = TTS()
    with pytest.raises(ValueError, match="volume_gain_db"):
        tts.update_options(volume_gain_db=-96.1)
    with pytest.raises(ValueError, match="volume_gain_db"):
        tts.update_options(volume_gain_db=16.1)

    # Valid boundaries in update_options
    tts.update_options(volume_gain_db=-96.0)
    assert tts._opts.volume_gain_db == -96.0
    tts.update_options(volume_gain_db=16.0)
    assert tts._opts.volume_gain_db == 16.0


def test_transactional_atomicity_on_validation_failure() -> None:
    tts = TTS(language="en-US", voice_name="Charon")
    with pytest.raises(ValueError, match="speaking_rate"):
        tts.update_options(language="es-ES", speaking_rate=999.0)

    # Ensure voice was NOT modified when validation failed
    assert tts._opts.voice.language_code == "en-US"
    assert tts._opts.voice.name == "Charon"


def test_update_options_only_audio_or_prompt_leaves_voice_untouched() -> None:
    tts = TTS(voice_name="Aoede", language="fr-FR", gender="female")
    initial_voice = tts._opts.voice

    tts.update_options(speaking_rate=1.5)
    assert tts._opts.voice is initial_voice
    assert tts._opts.speaking_rate == 1.5
    assert tts._opts.voice.name == "Aoede"

    tts.update_options(prompt="friendly tone")
    assert tts._opts.voice is initial_voice
    assert tts._opts.prompt == "friendly tone"


def test_update_options_prompt_preserves_voice() -> None:
    tts = TTS(voice_name="Charon", language="en-US")
    tts.update_options(prompt="Speak with an enthusiastic tone")
    assert tts._opts.prompt == "Speak with an enthusiastic tone"
    assert tts._opts.voice.name == "Charon"
    assert tts._opts.voice.language_code == "en-US"


def test_sequential_partial_updates_pipeline() -> None:
    tts = TTS()
    assert tts._opts.voice.name == "Charon"
    assert tts._opts.voice.language_code == "en-US"

    tts.update_options(language="es-ES")
    assert tts._opts.voice.language_code == "es-ES"
    assert tts._opts.voice.name == "Charon"

    tts.update_options(voice_name="Aoede")
    assert tts._opts.voice.name == "Aoede"
    assert tts._opts.voice.language_code == "es-ES"

    tts.update_options(gender="female")
    assert tts._opts.voice.ssml_gender == texttospeech.SsmlVoiceGender.FEMALE
    assert tts._opts.voice.name == "Aoede"
    assert tts._opts.voice.language_code == "es-ES"

    tts.update_options(model_name="chirp_3")
    assert tts.model == "chirp_3"
    assert "model_name" not in tts._opts.voice
    assert tts._opts.voice.name == "Aoede"
    assert tts._opts.voice.language_code == "es-ES"
    assert tts._opts.voice.ssml_gender == texttospeech.SsmlVoiceGender.FEMALE

    tts.update_options(speaking_rate=1.25, volume_gain_db=-2.0)
    assert tts._opts.speaking_rate == 1.25
    assert tts._opts.volume_gain_db == -2.0
    assert tts._opts.voice.name == "Aoede"
    assert tts._opts.voice.language_code == "es-ES"


def test_offline_hermetic_guarantee(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    with patch("google.auth.default", side_effect=RuntimeError("Auth call attempted!")):
        with patch.object(
            socket.socket, "connect", side_effect=RuntimeError("Network call attempted!")
        ):
            tts = TTS()
            assert tts._opts.voice.name == "Charon"
            tts.update_options(language="fr-FR")
            assert tts._opts.voice.language_code == "fr-FR"


def test_gemini_voice_change_from_chirp_updates_model_and_wire_name() -> None:
    # Devin review point 1: Chirp voice updated to Gemini voice transitions model and wire name
    tts = TTS(voice_name="en-US-Chirp3-HD-Charon")
    assert tts.model == "chirp_3"
    assert "model_name" not in tts._opts.voice

    tts.update_options(voice_name="Puck")
    assert tts.model == "gemini-2.5-flash-tts"
    assert tts._opts.voice.name == "Puck"
    assert tts._opts.voice.model_name == "gemini-2.5-flash-tts"


def test_model_switch_without_explicit_voice_updates_compatible_defaults() -> None:
    # Devin review point 2: Switching model without explicit voice updates default voice appropriately
    tts = TTS()
    assert tts.model == "gemini-2.5-flash-tts"
    assert tts._opts.voice.name == "Charon"

    # Switch to Chirp 3 without explicit voice_name -> switches to Chirp default voice
    tts.update_options(model_name="chirp_3")
    assert tts.model == "chirp_3"
    assert tts._opts.voice.name == "en-US-Chirp3-HD-Charon"
    assert "model_name" not in tts._opts.voice

    # Switch back to Gemini without explicit voice_name -> switches back to Gemini default voice
    tts.update_options(model_name="gemini-2.5-flash-tts")
    assert tts.model == "gemini-2.5-flash-tts"
    assert tts._opts.voice.name == "Charon"
    assert tts._opts.voice.model_name == "gemini-2.5-flash-tts"


def test_gemini_model_switch_clears_chirp_voice_clone() -> None:
    # Devin review point 3: Switching to Gemini clears Chirp-only voice_clone
    tts = TTS(voice_cloning_key="custom-clone-123")
    assert tts.model == "chirp_3"
    assert tts._opts.voice.voice_clone.voice_cloning_key == "custom-clone-123"

    tts.update_options(model_name="gemini-2.5-flash-tts")
    assert tts.model == "gemini-2.5-flash-tts"
    assert "voice_clone" not in tts._opts.voice
    assert tts._opts.voice.name == "Charon"
    assert tts._opts.voice.model_name == "gemini-2.5-flash-tts"


def test_partial_update_preserves_explicitly_empty_model() -> None:
    # Devin review: Distinguish None from "" and preserve explicitly empty model
    tts = TTS(model_name="", voice_name="en-US-Standard-A")
    assert tts._opts.model_name == ""
    assert tts._opts.voice.name == "en-US-Standard-A"
    assert tts._opts.voice.model_name == ""

    tts.update_options(language="en-US")
    assert tts._opts.model_name == ""
    assert tts._opts.voice.model_name == ""
    assert tts._opts.voice.name == "en-US-Standard-A"
