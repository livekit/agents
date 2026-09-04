"""Tests for the Sarvam TTS input validation against the live Bulbul API.

Covers the three divergences reported in livekit/agents#6774:

1. bulbul:v3 ships 37 documented speakers (nine were missing, two were
   offered that the API rejects).
2. Sample rates above 24 kHz are REST-only; streaming must fail locally
   instead of mid-session on the server.
3. pace is validated per model (0.5-2.0 for v3/v3-beta, 0.3-3.0 for v2).
"""

from __future__ import annotations

import pytest

from livekit.plugins.sarvam import tts

pytestmark = pytest.mark.unit

API_KEY = "test-key"

# Exactly the speaker list published in the Sarvam Bulbul docs for bulbul:v3.
V3_SPEAKERS = [
    "shubh",
    "aditya",
    "ritu",
    "priya",
    "neha",
    "rahul",
    "pooja",
    "rohan",
    "simran",
    "kavya",
    "amit",
    "dev",
    "ishita",
    "shreya",
    "ratan",
    "varun",
    "manan",
    "sumit",
    "roopa",
    "kabir",
    "aayan",
    "ashutosh",
    "advait",
    "anand",
    "tanya",
    "tarun",
    "sunny",
    "mani",
    "gokul",
    "vijay",
    "shruti",
    "suhani",
    "mohit",
    "kavitha",
    "rehan",
    "soham",
    "rupali",
]


# ---------------------------------------------------------------------------
# Speaker / model compatibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("speaker", V3_SPEAKERS)
def test_v3_accepts_all_documented_speakers(speaker: str) -> None:
    assert tts.validate_model_speaker_compatibility("bulbul:v3", speaker)


def test_v3_speaker_list_matches_docs() -> None:
    assert set(tts.MODEL_SPEAKER_COMPATIBILITY["bulbul:v3"]["all"]) == set(V3_SPEAKERS)


@pytest.mark.parametrize("speaker", ["amelia", "sophia"])
def test_v3_rejects_speakers_not_in_docs(speaker: str) -> None:
    assert not tts.validate_model_speaker_compatibility("bulbul:v3", speaker)


def test_v2_accepts_legacy_speakers() -> None:
    for speaker in ["anushka", "manisha", "vidya", "arya", "abhilash", "karun", "hitesh"]:
        assert tts.validate_model_speaker_compatibility("bulbul:v2", speaker)


def test_v2_rejects_v3_only_speakers() -> None:
    assert not tts.validate_model_speaker_compatibility("bulbul:v2", "shubh")
    assert not tts.validate_model_speaker_compatibility("bulbul:v2", "anand")


def test_v3_beta_keeps_international_voices() -> None:
    # v3-beta is not covered by the current public docs; keep the voices the
    # plugin already shipped for it rather than guessing at the live list.
    assert tts.validate_model_speaker_compatibility("bulbul:v3-beta", "amelia")
    assert tts.validate_model_speaker_compatibility("bulbul:v3-beta", "sophia")


# ---------------------------------------------------------------------------
# Per-model pace validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["bulbul:v3", "bulbul:v3-beta"])
@pytest.mark.parametrize("pace", [0.4, 2.5, 3.0])
def test_pace_out_of_range_for_v3_models_raises(model: str, pace: float) -> None:
    with pytest.raises(ValueError, match="Pace must be between"):
        tts.TTS(model=model, pace=pace, api_key=API_KEY)


@pytest.mark.parametrize("pace", [0.5, 1.0, 2.0])
def test_pace_boundaries_accepted_for_v3(pace: float) -> None:
    tts.TTS(model="bulbul:v3", pace=pace, api_key=API_KEY)


@pytest.mark.parametrize("pace", [0.3, 1.0, 3.0])
def test_pace_range_accepted_for_v2(pace: float) -> None:
    tts.TTS(model="bulbul:v2", pace=pace, api_key=API_KEY)


def test_pace_out_of_range_for_v2_raises() -> None:
    with pytest.raises(ValueError, match="Pace must be between"):
        tts.TTS(model="bulbul:v2", pace=3.1, api_key=API_KEY)


def test_pace_2_5_still_valid_on_v2() -> None:
    # 2.5 is out of range for v3 but fine on the legacy model.
    tts.TTS(model="bulbul:v2", pace=2.5, api_key=API_KEY)


def test_update_options_validates_pace_per_model() -> None:
    instance = tts.TTS(model="bulbul:v3", api_key=API_KEY)
    with pytest.raises(ValueError, match="Pace must be between"):
        instance.update_options(pace=2.5)
    instance.update_options(pace=1.5)
    assert instance._opts.pace == 1.5


def test_update_options_model_switch_revalidates_pace() -> None:
    instance = tts.TTS(model="bulbul:v2", pace=2.5, api_key=API_KEY)
    with pytest.raises(ValueError, match="Pace must be between"):
        instance.update_options(model="bulbul:v3", speaker="shubh", pace=2.5)


def test_update_options_model_switch_clamps_stale_pace() -> None:
    # 2.5 is legal on v2 but not on v3; switching models without passing a new
    # pace must clamp the stored value instead of sending it to the API.
    instance = tts.TTS(model="bulbul:v2", pace=2.5, api_key=API_KEY)
    instance.update_options(model="bulbul:v3", speaker="shubh")
    assert instance._opts.pace == 2.0


def test_update_options_model_switch_keeps_in_range_pace() -> None:
    instance = tts.TTS(model="bulbul:v2", pace=1.5, api_key=API_KEY)
    instance.update_options(model="bulbul:v3", speaker="shubh")
    assert instance._opts.pace == 1.5


# ---------------------------------------------------------------------------
# Sample rate gating: streaming vs REST
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sample_rate", [32000, 44100, 48000])
@pytest.mark.parametrize("model", ["bulbul:v3", "bulbul:v2"])
def test_stream_rejects_rest_only_sample_rates(model: str, sample_rate: int) -> None:
    instance = tts.TTS(model=model, speech_sample_rate=sample_rate, api_key=API_KEY)
    with pytest.raises(ValueError, match="streaming supports sample rates"):
        instance.stream()


@pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 24000])
async def test_stream_accepts_streaming_sample_rates(sample_rate: int) -> None:
    instance = tts.TTS(model="bulbul:v3", speech_sample_rate=sample_rate, api_key=API_KEY)
    stream = instance.stream()
    assert stream is not None
    await stream.aclose()


async def test_synthesize_accepts_rest_only_rates_on_v3() -> None:
    instance = tts.TTS(model="bulbul:v3", speech_sample_rate=48000, api_key=API_KEY)
    stream = instance.synthesize("hello")
    assert stream is not None
    await stream.aclose()


@pytest.mark.parametrize("model", ["bulbul:v2", "bulbul:v3-beta"])
def test_synthesize_rejects_rest_only_rates_on_non_v3(model: str) -> None:
    instance = tts.TTS(model=model, speech_sample_rate=48000, api_key=API_KEY)
    with pytest.raises(ValueError, match="only available for bulbul:v3"):
        instance.synthesize("hello")
