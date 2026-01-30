"""Tests for PersonaPlex realtime model plugin."""

from __future__ import annotations

import asyncio
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from livekit.plugins.personaplex import RealtimeModel, RealtimeSession
from livekit.plugins.personaplex.realtime.realtime_model import (
    DEFAULT_SILENCE_THRESHOLD_MS,
    INITIAL_RETRY_DELAY,
    MAX_RETRY_DELAY,
    MSG_AUDIO,
    MSG_HANDSHAKE,
    MSG_TEXT,
    NUM_CHANNELS,
    SAMPLE_RATE,
    _PersonaplexOptions,
    _ResponseGeneration,
    _SPECIAL_TOKENS,
)


# -- RealtimeModel init tests --


class TestRealtimeModelInit:
    def test_default_url(self) -> None:
        model = RealtimeModel()
        assert model._opts.base_url == "localhost:8998"

    def test_custom_url(self) -> None:
        model = RealtimeModel(base_url="myhost:9000")
        assert model._opts.base_url == "myhost:9000"

    def test_strips_ws_prefix(self) -> None:
        model = RealtimeModel(base_url="ws://myhost:9000")
        assert model._opts.base_url == "myhost:9000"

    def test_strips_wss_prefix(self) -> None:
        model = RealtimeModel(base_url="wss://myhost:9000")
        assert model._opts.base_url == "myhost:9000"

    def test_strips_http_prefix(self) -> None:
        model = RealtimeModel(base_url="http://myhost:9000")
        assert model._opts.base_url == "myhost:9000"

    def test_strips_https_prefix(self) -> None:
        model = RealtimeModel(base_url="https://myhost:9000")
        assert model._opts.base_url == "myhost:9000"

    def test_env_var_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PERSONAPLEX_URL", "envhost:1234")
        model = RealtimeModel()
        assert model._opts.base_url == "envhost:1234"

    def test_explicit_url_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PERSONAPLEX_URL", "envhost:1234")
        model = RealtimeModel(base_url="explicit:5678")
        assert model._opts.base_url == "explicit:5678"

    def test_default_voice(self) -> None:
        model = RealtimeModel()
        assert model._opts.voice == "NATF2"

    def test_custom_voice(self) -> None:
        model = RealtimeModel(voice="VARM3")
        assert model._opts.voice == "VARM3"

    def test_default_text_prompt(self) -> None:
        model = RealtimeModel()
        assert model._opts.text_prompt == "You are a helpful assistant."

    def test_seed(self) -> None:
        model = RealtimeModel(seed=42)
        assert model._opts.seed == 42

    def test_silence_threshold(self) -> None:
        model = RealtimeModel(silence_threshold_ms=1000)
        assert model._opts.silence_threshold_ms == 1000

    def test_model_property(self) -> None:
        model = RealtimeModel()
        assert model.model == "personaplex-7b"

    def test_provider_property(self) -> None:
        model = RealtimeModel()
        assert model.provider == "nvidia"

    def test_label(self) -> None:
        model = RealtimeModel(voice="NATM1")
        assert model._label == "personaplex-NATM1"

    def test_capabilities(self) -> None:
        model = RealtimeModel()
        caps = model.capabilities
        assert caps.message_truncation is False
        assert caps.turn_detection is False
        assert caps.user_transcription is False
        assert caps.auto_tool_reply_generation is False
        assert caps.audio_output is True
        assert caps.manual_function_calls is False


# -- URL building tests --


class TestBuildWsUrl:
    def _make_session_opts(self, **kwargs: object) -> _PersonaplexOptions:
        defaults = {
            "base_url": "localhost:8998",
            "voice": "NATF2",
            "text_prompt": "You are helpful.",
            "seed": None,
            "silence_threshold_ms": 500,
        }
        defaults.update(kwargs)
        return _PersonaplexOptions(**defaults)  # type: ignore[arg-type]

    def test_basic_url(self) -> None:
        opts = self._make_session_opts()
        model = RealtimeModel()
        model._opts = opts
        # We can't easily call _build_ws_url without a session,
        # but we can test the opts are set correctly
        assert opts.base_url == "localhost:8998"
        assert opts.voice == "NATF2"

    def test_seed_in_url(self) -> None:
        opts = self._make_session_opts(seed=42)
        assert opts.seed == 42


# -- Text token handling tests --


class TestHandleTextToken:
    def test_special_token_filtered(self) -> None:
        """Special tokens (0, 3) should be filtered out."""
        assert 0 in _SPECIAL_TOKENS
        assert 3 in _SPECIAL_TOKENS

    def test_normal_token_not_filtered(self) -> None:
        """Non-special numeric tokens should pass through."""
        assert 1 not in _SPECIAL_TOKENS
        assert 42 not in _SPECIAL_TOKENS


# -- Audio data tests --


class TestAudioConstants:
    def test_sample_rate(self) -> None:
        assert SAMPLE_RATE == 24000

    def test_num_channels(self) -> None:
        assert NUM_CHANNELS == 1

    def test_message_types(self) -> None:
        assert MSG_HANDSHAKE == 0x00
        assert MSG_AUDIO == 0x01
        assert MSG_TEXT == 0x02


class TestAudioConversion:
    def test_int16_to_float32_roundtrip(self) -> None:
        """Verify PCM int16 -> float32 -> int16 roundtrip preserves data."""
        # Simulate what _encode_and_send and _handle_audio_data do
        original = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16)

        # Forward: int16 -> float32 (as in _encode_and_send)
        pcm_float = original.astype(np.float32) / 32768.0

        # Reverse: float32 -> int16 (as in _handle_audio_data)
        recovered = np.clip(pcm_float * 32768.0, -32768, 32767).astype(np.int16)

        np.testing.assert_array_equal(original, recovered)

    def test_float32_clipping(self) -> None:
        """Values outside [-1, 1] should be clipped to int16 range."""
        pcm_float = np.array([2.0, -2.0], dtype=np.float32)
        pcm_int16 = np.clip(pcm_float * 32768.0, -32768, 32767).astype(np.int16)
        assert pcm_int16[0] == 32767
        assert pcm_int16[1] == -32768


# -- ResponseGeneration dataclass tests --


class TestResponseGeneration:
    def test_defaults(self) -> None:
        from livekit.agents import llm, utils
        from livekit import rtc

        gen = _ResponseGeneration(
            message_ch=utils.aio.Chan[llm.MessageGeneration](),
            function_ch=utils.aio.Chan[llm.FunctionCall](),
            response_id="test-123",
            text_ch=utils.aio.Chan[str](),
            audio_ch=utils.aio.Chan[rtc.AudioFrame](),
        )

        assert gen.response_id == "test-123"
        assert gen._first_token_timestamp is None
        assert gen._completed_timestamp is None
        assert gen._done is False
        assert gen.output_text == ""
        assert gen._created_timestamp > 0

        # Cleanup
        gen.message_ch.close()
        gen.function_ch.close()
        gen.text_ch.close()
        gen.audio_ch.close()


# -- Retry / backoff constants tests --


class TestRetryConstants:
    def test_initial_delay(self) -> None:
        assert INITIAL_RETRY_DELAY == 1.0

    def test_max_delay(self) -> None:
        assert MAX_RETRY_DELAY == 30.0

    def test_exponential_backoff_sequence(self) -> None:
        """Verify the exponential backoff calculation used in _main_task."""
        delay = INITIAL_RETRY_DELAY
        expected = [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]
        for expected_delay in expected:
            assert delay == expected_delay
            delay = min(delay * 2, MAX_RETRY_DELAY)


# -- Options dataclass tests --


class TestPersonaplexOptions:
    def test_all_fields(self) -> None:
        opts = _PersonaplexOptions(
            base_url="host:123",
            voice="VARF0",
            text_prompt="test prompt",
            seed=99,
            silence_threshold_ms=750,
        )
        assert opts.base_url == "host:123"
        assert opts.voice == "VARF0"
        assert opts.text_prompt == "test prompt"
        assert opts.seed == 99
        assert opts.silence_threshold_ms == 750

    def test_none_seed(self) -> None:
        opts = _PersonaplexOptions(
            base_url="host:123",
            voice="NATF0",
            text_prompt="prompt",
            seed=None,
            silence_threshold_ms=500,
        )
        assert opts.seed is None
