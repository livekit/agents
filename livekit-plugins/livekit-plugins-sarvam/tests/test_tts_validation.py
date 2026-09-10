from __future__ import annotations

from typing import cast
from unittest.mock import Mock

import aiohttp
import pytest

from livekit.agents import APIConnectionError
from livekit.plugins.sarvam.tts import TTS, validate_model_speaker_compatibility

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "speaker",
    ["anand", "tarun", "sunny", "mani", "gokul", "vijay", "mohit", "rehan", "soham"],
)
def test_bulbul_v3_accepts_documented_speakers(speaker: str) -> None:
    assert validate_model_speaker_compatibility("bulbul:v3", speaker)


async def test_tts_initialization_accepts_a_documented_bulbul_v3_speaker() -> None:
    tts = TTS(api_key="test-key", model="bulbul:v3", speaker="anand")
    try:
        assert tts._opts.speaker == "anand"
    finally:
        await tts.aclose()


@pytest.mark.parametrize("speaker", ["amelia", "sophia"])
def test_bulbul_v3_rejects_speakers_not_supported_by_the_model(speaker: str) -> None:
    assert not validate_model_speaker_compatibility("bulbul:v3", speaker)


@pytest.mark.parametrize("speaker", ["amelia", "sophia"])
def test_tts_rejects_bulbul_v3_speakers_not_supported_by_the_model(speaker: str) -> None:
    with pytest.raises(ValueError, match="not compatible with model 'bulbul:v3'"):
        TTS(api_key="test-key", model="bulbul:v3", speaker=speaker)


def test_bulbul_v3_beta_keeps_international_speakers() -> None:
    assert validate_model_speaker_compatibility("bulbul:v3-beta", "amelia")
    assert validate_model_speaker_compatibility("bulbul:v3-beta", "sophia")


@pytest.mark.parametrize(
    ("model", "pace"),
    [
        ("bulbul:v2", 0.3),
        ("bulbul:v2", 3.0),
        ("bulbul:v3-beta", 0.5),
        ("bulbul:v3-beta", 2.0),
        ("bulbul:v3", 0.5),
        ("bulbul:v3", 2.0),
    ],
)
async def test_pace_accepts_the_bounds_for_each_model(model: str, pace: float) -> None:
    tts = TTS(api_key="test-key", model=model, pace=pace)
    try:
        assert tts._opts.pace == pace
    finally:
        await tts.aclose()


@pytest.mark.parametrize(
    ("model", "pace"),
    [
        ("bulbul:v3-beta", 0.49),
        ("bulbul:v3-beta", 2.01),
        ("bulbul:v3", 0.49),
        ("bulbul:v3", 2.01),
    ],
)
def test_pace_rejects_values_outside_v3_model_range(model: str, pace: float) -> None:
    with pytest.raises(ValueError, match=rf"Pace for {model} must be between 0.5 and 2.0"):
        TTS(api_key="test-key", model=model, pace=pace)


async def test_update_options_uses_the_selected_model_pace_range() -> None:
    tts = TTS(api_key="test-key", model="bulbul:v3", pace=1.0)
    try:
        with pytest.raises(ValueError, match="Pace for bulbul:v3 must be between 0.5 and 2.0"):
            tts.update_options(pace=2.5)
    finally:
        await tts.aclose()


async def test_update_options_rejects_a_model_switch_with_an_invalid_existing_pace() -> None:
    tts = TTS(api_key="test-key", model="bulbul:v2", pace=3.0)
    try:
        with pytest.raises(ValueError, match="Pace for bulbul:v3 must be between 0.5 and 2.0"):
            tts.update_options(model="bulbul:v3")
    finally:
        await tts.aclose()


async def test_update_options_is_atomic_when_the_new_model_rejects_the_existing_speaker() -> None:
    tts = TTS(api_key="test-key", model="bulbul:v3-beta", speaker="amelia", pace=1.0)
    try:
        with pytest.raises(ValueError, match="incompatible with bulbul:v3"):
            tts.update_options(model="bulbul:v3")
        assert tts._opts.model == "bulbul:v3-beta"
        assert tts._opts.speaker == "amelia"
        assert tts._opts.pace == 1.0
    finally:
        await tts.aclose()


async def test_update_options_invalidates_connections_when_url_options_change() -> None:
    tts = TTS(api_key="test-key", model="bulbul:v2")
    tts._pool.invalidate = Mock()
    try:
        tts.update_options(model="bulbul:v3", speaker="anand")
        tts._pool.invalidate.assert_called_once_with()
    finally:
        await tts.aclose()


@pytest.mark.parametrize("sample_rate", [32000, 44100, 48000])
async def test_rest_only_sample_rates_are_rejected_before_streaming(sample_rate: int) -> None:
    tts = TTS(api_key="test-key", speech_sample_rate=sample_rate)
    try:
        assert not tts.capabilities.streaming
        with pytest.raises(ValueError, match="streaming TTS"):
            tts.stream()
    finally:
        await tts.aclose()


@pytest.mark.parametrize("sample_rate", [32000, 44100, 48000])
async def test_rest_only_sample_rates_remain_valid_for_synthesis(sample_rate: int) -> None:
    tts = TTS(api_key="test-key", speech_sample_rate=sample_rate)
    try:
        assert tts.sample_rate == sample_rate
    finally:
        await tts.aclose()


async def test_rest_only_sample_rates_skip_websocket_prewarm() -> None:
    tts = TTS(api_key="test-key", speech_sample_rate=48000)
    tts._pool.prewarm = Mock()
    try:
        tts.prewarm()
        tts._pool.prewarm.assert_not_called()
    finally:
        await tts.aclose()


async def test_websocket_connection_errors_do_not_expose_transport_details(caplog) -> None:
    class FailingSession:
        async def ws_connect(self, *args, **kwargs):
            raise RuntimeError("api-key=secret")

    tts = TTS(
        api_key="secret",
        http_session=cast(aiohttp.ClientSession, FailingSession()),
    )
    try:
        with pytest.raises(APIConnectionError, match="RuntimeError") as exc_info:
            await tts._connect_ws(1.0)
        assert "api-key=secret" not in str(exc_info.value)
        assert exc_info.value.__cause__ is None
        assert all("api-key=secret" not in record.getMessage() for record in caplog.records)
    finally:
        await tts.aclose()
