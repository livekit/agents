from __future__ import annotations

import pytest

from livekit.plugins.sarvam.tts import (
    ALLOWED_SAMPLE_RATES,
    MODEL_SPEAKER_COMPATIBILITY,
    STREAMING_SAMPLE_RATES,
    TTS,
    compatible_speakers,
    pace_range,
    validate_model_speaker_compatibility,
)

pytestmark = pytest.mark.plugin("sarvam")

API_KEY = "test-key-not-used"

# Verified against the live Sarvam API and the bulbul docs.
V2_SPEAKERS = {"anushka", "abhilash", "manisha", "vidya", "arya", "karun", "hitesh"}
V3_ONLY_SPEAKERS = {"anand", "tarun", "sunny", "mani", "gokul", "vijay", "mohit", "rehan", "soham"}
NOT_REAL_SPEAKERS = {"amelia", "sophia", "niharika"}


def test_v2_speaker_set_is_exact() -> None:
    assert MODEL_SPEAKER_COMPATIBILITY["bulbul:v2"] == V2_SPEAKERS


def test_v3_and_v3_beta_share_a_speaker_set() -> None:
    assert MODEL_SPEAKER_COMPATIBILITY["bulbul:v3"] == MODEL_SPEAKER_COMPATIBILITY["bulbul:v3-beta"]
    assert len(MODEL_SPEAKER_COMPATIBILITY["bulbul:v3"]) == 37


@pytest.mark.parametrize("speaker", sorted(V3_ONLY_SPEAKERS))
def test_v3_speakers_are_accepted(speaker: str) -> None:
    assert validate_model_speaker_compatibility("bulbul:v3", speaker)
    TTS(model="bulbul:v3", speaker=speaker, api_key=API_KEY)


@pytest.mark.parametrize("speaker", sorted(NOT_REAL_SPEAKERS))
def test_speakers_the_api_rejects_are_not_offered(speaker: str) -> None:
    for model in MODEL_SPEAKER_COMPATIBILITY:
        assert speaker not in MODEL_SPEAKER_COMPATIBILITY[model]


def test_v2_speaker_rejected_on_v3() -> None:
    with pytest.raises(ValueError, match="not compatible"):
        TTS(model="bulbul:v3", speaker="anushka", api_key=API_KEY)


def test_compatible_speakers_is_sorted_and_empty_for_unknown_model() -> None:
    assert compatible_speakers("bulbul:v2") == sorted(V2_SPEAKERS)
    assert compatible_speakers("bulbul:v9") == []


def test_streaming_sample_rates_are_a_documented_subset() -> None:
    assert STREAMING_SAMPLE_RATES == {8000, 16000, 22050, 24000}
    assert STREAMING_SAMPLE_RATES < ALLOWED_SAMPLE_RATES
    assert ALLOWED_SAMPLE_RATES - STREAMING_SAMPLE_RATES == {32000, 44100, 48000}


@pytest.mark.parametrize("rate", [32000, 44100, 48000])
def test_high_rates_construct_but_reject_streaming(rate: int) -> None:
    tts = TTS(model="bulbul:v3", speech_sample_rate=rate, api_key=API_KEY)
    assert tts.sample_rate == rate  # synthesize() still works at this rate
    assert rate not in STREAMING_SAMPLE_RATES


@pytest.mark.parametrize("rate", [32000, 44100, 48000])
def test_rest_only_rates_construct_on_v2_too(rate: int) -> None:
    # REST-only rates are valid for synthesize() on every model; stream() rejects them.
    tts = TTS(model="bulbul:v2", speaker="anushka", speech_sample_rate=rate, api_key=API_KEY)
    assert tts.sample_rate == rate
    assert rate not in STREAMING_SAMPLE_RATES


def test_pace_range_is_per_model() -> None:
    assert pace_range("bulbul:v3") == (0.5, 2.0)
    assert pace_range("bulbul:v3-beta") == (0.5, 2.0)
    assert pace_range("bulbul:v2") == (0.3, 3.0)


@pytest.mark.parametrize("pace", [0.4, 2.5])
def test_v2_only_pace_rejected_on_v3(pace: float) -> None:
    with pytest.raises(ValueError, match="Pace must be between 0.5 and 2.0"):
        TTS(model="bulbul:v3", pace=pace, api_key=API_KEY)


@pytest.mark.parametrize("pace", [0.3, 1.0, 3.0])
def test_v2_accepts_its_wider_pace_range(pace: float) -> None:
    TTS(model="bulbul:v2", speaker="anushka", pace=pace, api_key=API_KEY)


@pytest.mark.parametrize("pace", [0.5, 1.0, 2.0])
def test_v3_accepts_its_own_range(pace: float) -> None:
    TTS(model="bulbul:v3", pace=pace, api_key=API_KEY)


def test_update_options_validates_pace_against_current_model() -> None:
    tts = TTS(model="bulbul:v3", api_key=API_KEY)
    with pytest.raises(ValueError, match="Pace must be between 0.5 and 2.0"):
        tts.update_options(pace=2.5)


def test_switching_to_v3_rechecks_the_pace_already_set() -> None:
    tts = TTS(model="bulbul:v2", speaker="anushka", pace=3.0, api_key=API_KEY)
    with pytest.raises(ValueError, match="Pace must be between 0.5 and 2.0"):
        tts.update_options(model="bulbul:v3", speaker="shubh")


def test_switching_to_v2_widens_the_allowed_pace() -> None:
    tts = TTS(model="bulbul:v3", pace=2.0, api_key=API_KEY)
    tts.update_options(model="bulbul:v2", speaker="anushka", pace=3.0)
    assert tts._opts.pace == 3.0


def test_failed_update_leaves_model_and_speaker_untouched() -> None:
    tts = TTS(model="bulbul:v2", speaker="anushka", api_key=API_KEY)
    with pytest.raises(ValueError, match="incompatible"):
        tts.update_options(model="bulbul:v3")  # anushka is not a v3 speaker
    assert tts._opts.model == "bulbul:v2"
    assert tts._opts.speaker == "anushka"


def test_failed_pace_update_leaves_model_untouched() -> None:
    tts = TTS(model="bulbul:v2", speaker="anushka", pace=3.0, api_key=API_KEY)
    with pytest.raises(ValueError, match="Pace must be between 0.5 and 2.0"):
        tts.update_options(model="bulbul:v3", speaker="shubh")
    assert tts._opts.model == "bulbul:v2"
    assert tts._opts.speaker == "anushka"
    assert tts._opts.pace == 3.0


def test_failed_update_does_not_apply_earlier_valid_fields() -> None:
    tts = TTS(model="bulbul:v3", speaker="shubh", api_key=API_KEY)
    with pytest.raises(ValueError, match="Loudness"):
        tts.update_options(target_language_code="hi-IN", pace=1.5, loudness=9.0)
    assert tts._opts.target_language_code == "en-IN"
    assert tts._opts.pace == 1.0


def test_model_and_speaker_change_together_is_accepted() -> None:
    tts = TTS(model="bulbul:v2", speaker="anushka", api_key=API_KEY)
    tts.update_options(model="bulbul:v3", speaker="shubh")
    assert (tts._opts.model, tts._opts.speaker) == ("bulbul:v3", "shubh")


def test_unsupported_sample_rate_still_rejected() -> None:
    with pytest.raises(ValueError, match="Sample rate must be one of"):
        TTS(speech_sample_rate=12345, api_key=API_KEY)
