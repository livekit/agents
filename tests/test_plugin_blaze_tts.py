"""Unit tests for Blaze TTS plugin helpers and configuration."""

from __future__ import annotations

import asyncio
import struct
import time
from unittest.mock import AsyncMock

import pytest

from livekit.agents import APITimeoutError
from livekit.plugins.blaze._config import BlazeConfig
from livekit.plugins.blaze.tts import (
    TTS,
    _apply_pcm16_fade,
    _find_batch_split,
    _generate_silence,
    _normalize_batch_text,
    _WSStreamGuard,
)

pytestmark = pytest.mark.unit


def test_find_batch_split_first_batch_uses_word_count() -> None:
    text = "One two three four. This is a longer second sentence."
    split = _find_batch_split(text, is_first_batch=True)
    assert split == len("One two three four. ")


def test_find_batch_split_hard_limit_avoids_tiny_chunk() -> None:
    text = "Hi. " + ("word " * 80)
    split = _find_batch_split(text, max_chars=20, min_chars=8, is_first_batch=False)
    assert split is not None
    assert split >= 8


def test_find_batch_split_subsequent_batch_prefers_later_punctuation() -> None:
    text = "A" * 120 + ". " + "B" * 120 + ". " + "C" * 50
    split = _find_batch_split(
        text,
        min_chars=100,
        target_chars=200,
        max_chars=350,
        is_first_batch=False,
    )
    assert split is not None
    assert split >= 100


def test_find_batch_split_force_emits_remaining_text() -> None:
    text = "Short reply."
    split = _find_batch_split(text, force=True, is_first_batch=False)
    assert split == len(text)


def test_find_batch_split_returns_none_for_short_unforced_buffer() -> None:
    assert _find_batch_split("Hi.", is_first_batch=False) is None


def test_normalize_batch_text_collapses_whitespace() -> None:
    assert _normalize_batch_text("Hello   world\n\n\nagain") == "Hello world\nagain"


def test_generate_silence_pcm16_length() -> None:
    silence = _generate_silence(24000, 150)
    assert len(silence) == int(24000 * 150 / 1000) * 2
    assert silence == b"\x00\x00" * int(24000 * 150 / 1000)


def test_apply_pcm16_fade_noop_without_flags() -> None:
    pcm = b"\x10\x00" * 8
    assert _apply_pcm16_fade(pcm, fade_samples=4) == pcm


def test_apply_pcm16_fade_in_reduces_leading_samples() -> None:
    pcm = b"\xff\x7f" * 8
    faded = _apply_pcm16_fade(pcm, fade_samples=4, fade_in=True)
    samples = struct.unpack("<8h", faded)

    assert samples[0] == 0
    assert abs(samples[3]) < abs(struct.unpack("<8h", pcm)[3])


def test_apply_pcm16_fade_out_reduces_trailing_samples() -> None:
    pcm = b"\xff\x7f" * 8
    faded = _apply_pcm16_fade(pcm, fade_samples=4, fade_out=True)
    original = struct.unpack("<8h", pcm)
    result = struct.unpack("<8h", faded)

    assert abs(result[-1]) < abs(original[-1])
    assert abs(result[-4]) == abs(original[-4])


@pytest.mark.asyncio
async def test_ws_stream_guard_idle_timeout() -> None:
    async def slow_recv() -> None:
        await asyncio.sleep(0.2)

    ws = AsyncMock()
    ws.recv = slow_recv

    guard = _WSStreamGuard(
        idle_timeout=0.05,
        session_deadline=time.monotonic() + 5.0,
        request_id="test",
    )

    with pytest.raises(APITimeoutError, match="idle timeout"):
        await guard.recv(ws)


@pytest.mark.asyncio
async def test_ws_stream_guard_session_deadline() -> None:
    ws = AsyncMock()
    guard = _WSStreamGuard(
        idle_timeout=5.0,
        session_deadline=time.monotonic() - 0.01,
        request_id="test",
    )

    with pytest.raises(APITimeoutError, match="max session duration"):
        await guard.recv(ws)


def test_tts_default_model_is_2_0_realtime() -> None:
    tts = TTS(config=BlazeConfig(api_url="http://localhost"))
    assert tts.model == "2.0-realtime"


def test_tts_provider_model_and_ws_url() -> None:
    config = BlazeConfig(api_url="https://api.example.com", api_token="tok")
    tts = TTS(config=config, speaker_id="voice-1", model="v2_pro")

    assert tts.provider == "Blaze"
    assert tts.model == "v2_pro"
    assert tts._ws_url == "wss://api.example.com/v1/tts/realtime"
    assert tts._speaker_id == "voice-1"


def test_tts_invalid_audio_format_falls_back_to_pcm() -> None:
    tts = TTS(config=BlazeConfig(api_url="http://localhost"), audio_format="ogg")
    assert tts._audio_format == "pcm"


def test_tts_update_options() -> None:
    tts = TTS(config=BlazeConfig(api_url="http://localhost"))
    tts.update_options(
        speaker_id="new-speaker",
        model="v1_5_pro",
        audio_format="wav",
        audio_quality=64,
        language="en",
        auth_token="new-token",
        normalization_rules={"API": "A P I"},
    )

    assert tts._speaker_id == "new-speaker"
    assert tts._model == "v1_5_pro"
    assert tts._audio_format == "wav"
    assert tts._audio_quality == 64
    assert tts._language == "en"
    assert tts._auth_token == "new-token"
    assert tts._normalization_rules == {"API": "A P I"}


def test_tts_speech_start_params_include_optional_fields() -> None:
    from livekit.plugins.blaze.tts import _TTSSynthesizeStream

    config = BlazeConfig(api_url="http://localhost")
    tts = TTS(
        config=config,
        speaker_id="spk",
        model="v1_5_pro",
        audio_speed="1.2",
        audio_quality=48,
        language="vi",
    )
    stream = object.__new__(_TTSSynthesizeStream)
    stream._blaze_tts = tts
    params = stream._speech_start_params()

    assert params["event"] == "speech-start"
    assert params["speaker_id"] == "spk"
    assert params["model"] == "v1_5_pro"
    assert params["audio_speed"] == "1.2"
    assert params["audio_quality"] == 48
    assert params["language"] == "vi"
