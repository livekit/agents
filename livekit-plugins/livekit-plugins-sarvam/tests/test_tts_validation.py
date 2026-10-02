from __future__ import annotations

import pytest

from livekit.plugins.sarvam.tts import (
    MODEL_SPEAKER_COMPATIBILITY,
    _STREAMING_SAMPLE_RATES,
    _pace_bounds,
    validate_model_speaker_compatibility,
)

pytestmark = pytest.mark.unit

NEW_V3_SPEAKERS = [
    "anand",
    "tarun",
    "sunny",
    "mani",
    "gokul",
    "vijay",
    "mohit",
    "rehan",
    "soham",
]


def test_pace_bounds_follow_documented_limits() -> None:
    # Sarvam docs: pace 0.5–2.0 for bulbul:v3, 0.3–3.0 for bulbul:v2.
    # bulbul:v3-beta is not documented, so it keeps the previous bound.
    assert _pace_bounds("bulbul:v3") == (0.5, 2.0)
    assert _pace_bounds("bulbul:v2") == (0.3, 3.0)
    assert _pace_bounds("bulbul:v3-beta") == (0.3, 3.0)


def test_v3_speaker_list_matches_documented_speakers() -> None:
    all_v3 = MODEL_SPEAKER_COMPATIBILITY["bulbul:v3"]["all"]
    assert len(all_v3) == 37
    assert len(set(all_v3)) == 37
    for speaker in NEW_V3_SPEAKERS:
        assert speaker in all_v3
    # rejected by the live API with "not recognized", must not validate
    for speaker in ("amelia", "sophia"):
        assert speaker not in all_v3


def test_new_v3_speakers_have_no_unverified_gender_subgroup() -> None:
    # The Sarvam docs list speakers without gender information.
    v3 = MODEL_SPEAKER_COMPATIBILITY["bulbul:v3"]
    for speaker in NEW_V3_SPEAKERS:
        assert speaker not in v3["male"]
        assert speaker not in v3["female"]


def test_validate_model_speaker_compatibility() -> None:
    assert validate_model_speaker_compatibility("bulbul:v3", "anand")
    assert validate_model_speaker_compatibility("bulbul:v3", "Shubh")
    assert not validate_model_speaker_compatibility("bulbul:v3", "amelia")
    assert not validate_model_speaker_compatibility("bulbul:v3", "not-a-speaker")


def test_streaming_sample_rates_exclude_rest_only() -> None:
    # Sarvam docs: 32000/44100/48000 Hz are REST-only; streaming caps at 24 kHz.
    assert _STREAMING_SAMPLE_RATES == frozenset({8000, 16000, 22050, 24000})
    for rate in (32000, 44100, 48000):
        assert rate not in _STREAMING_SAMPLE_RATES
