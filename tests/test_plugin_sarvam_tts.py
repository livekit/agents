from __future__ import annotations

import pytest

from livekit.plugins.sarvam.tts import (
    ALLOWED_SAMPLE_RATES,
    MODEL_SPEAKER_COMPATIBILITY,
    STREAMING_SAMPLE_RATES,
    TTS,
    compatible_speakers,
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


def test_unsupported_sample_rate_still_rejected() -> None:
    with pytest.raises(ValueError, match="Sample rate must be one of"):
        TTS(speech_sample_rate=12345, api_key=API_KEY)
