from __future__ import annotations

from livekit.agents import APIStatusError
from livekit.plugins.slng.connection import (
    CandidateState,
    bridge_endpoint,
    bridge_model,
)
from livekit.plugins.slng.gateway_adapter import (
    build_stt_init_payload,
    build_tts_init_payload,
    is_payload_too_large,
    normalize_region_override,
)


def test_bridge_endpoint_and_model_round_trip() -> None:
    endpoint = bridge_endpoint("api.slng.ai", "stt", "deepgram/nova:3")
    assert endpoint == ("wss://api.slng.ai/v1/bridges/unmute/stt/deepgram/nova:3")
    assert bridge_model(endpoint, "stt") == "deepgram/nova:3"


def test_tts_payload_forwards_all_options_and_exact_language() -> None:
    pronunciation = {"mode": "rewrite", "name": "products"}
    payload = build_tts_init_payload(
        model="sarvam/bulbul:v3",
        voice="shubh",
        language="en-IN",
        sample_rate=24000,
        encoding="linear16",
        speed=1.0,
        model_options={"future_option": 7, "pronunciation": pronunciation},
    )
    assert payload["language"] == "en-IN"
    assert payload["config"]["future_option"] == 7
    assert payload["config"]["pronunciation"] == pronunciation


def test_stt_payload_forwards_options_and_exact_language() -> None:
    payload = build_stt_init_payload(
        model="sarvam/saaras:v3",
        language="hi-IN",
        sample_rate=16000,
        encoding="pcm_s16le",
        vad_threshold=0.5,
        vad_min_silence_duration_ms=300,
        vad_speech_pad_ms=30,
        enable_diarization=False,
        enable_partial_transcripts=True,
        model_options={"future_option": True},
    )
    assert payload["config"]["language"] == "hi-IN"
    assert payload["config"]["future_option"] is True


def test_candidate_state_recovers_primary_after_cooldown() -> None:
    state = CandidateState(2, recovery_cooldown_s=0)
    assert state.advance(0) == 1
    assert state.start() == 0


def test_region_override_normalization() -> None:
    assert normalize_region_override(["EU-WEST-1", " us-east-1 "]) == ("eu-west-1, us-east-1")


def test_is_payload_too_large_matches_only_413() -> None:
    assert is_payload_too_large(APIStatusError("too large", status_code=413))
    assert not is_payload_too_large(APIStatusError("bad request", status_code=400))
    assert not is_payload_too_large(None)
