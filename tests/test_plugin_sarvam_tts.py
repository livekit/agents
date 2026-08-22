"""Unit tests for Sarvam TTS plugin validation."""

from __future__ import annotations

import asyncio

import pytest

from livekit.plugins.sarvam import tts as sarvam_tts

pytestmark = pytest.mark.unit


def _tts(**kwargs: object) -> sarvam_tts.TTS:
    return sarvam_tts.TTS(api_key="test-key", **kwargs)


def test_v3_accepts_extended_speakers() -> None:
    # Speakers documented by Bulbul v3 but previously rejected by the plugin.
    for speaker in ("anand", "tarun", "sunny", "mani", "gokul", "vijay", "mohit", "rehan", "soham"):
        assert _tts(model="bulbul:v3", speaker=speaker)._opts.speaker == speaker


def test_v3_rejects_speakers_the_api_does_not_recognize() -> None:
    with pytest.raises(ValueError, match="amelia"):
        _tts(model="bulbul:v3", speaker="amelia")
    with pytest.raises(ValueError, match="sophia"):
        _tts(model="bulbul:v3", speaker="sophia")


def test_v3_beta_still_accepts_international_speakers() -> None:
    assert _tts(model="bulbul:v3-beta", speaker="amelia")._opts.speaker == "amelia"


def test_pace_range_for_v2() -> None:
    # 0.3-3.0 is valid on bulbul:v2
    assert _tts(model="bulbul:v2", pace=0.3)._opts.pace == 0.3
    assert _tts(model="bulbul:v2", pace=3.0)._opts.pace == 3.0


def test_pace_range_for_v3() -> None:
    # 0.5-2.0 on v3: values valid for v2 must be rejected before hitting the API
    with pytest.raises(ValueError, match="between 0.5 and 2.0"):
        _tts(model="bulbul:v3", pace=2.5)
    with pytest.raises(ValueError, match="between 0.5 and 2.0"):
        _tts(model="bulbul:v3-beta", pace=0.4)

    assert _tts(model="bulbul:v3", pace=0.5)._opts.pace == 0.5
    assert _tts(model="bulbul:v3", pace=2.0)._opts.pace == 2.0


def test_update_options_validates_pace_per_model() -> None:
    instance = _tts(model="bulbul:v3")
    with pytest.raises(ValueError, match="between 0.5 and 2.0"):
        instance.update_options(pace=2.5)


def test_stream_rejects_rest_only_sample_rates() -> None:
    # 32/44.1/48 kHz are REST-only; streaming must fail locally with a clear
    # error instead of an opaque server error frame mid-session.
    instance = _tts(model="bulbul:v3", speech_sample_rate=48000)
    with pytest.raises(ValueError, match="REST API"):
        instance.stream()


def test_stream_allows_streaming_sample_rates() -> None:
    async def _create_and_close(rate: int) -> None:
        instance = _tts(model="bulbul:v3", speech_sample_rate=rate)
        instance.stream()
        await instance.aclose()

    for rate in (8000, 16000, 22050, 24000):
        asyncio.run(_create_and_close(rate))
