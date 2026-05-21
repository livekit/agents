"""Unit tests for SmallestAI TTS plugin — Lightning v3.1 / v3.1 Pro."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.plugins.smallestai import TTS
from livekit.plugins.smallestai.tts import ChunkedStream, _to_smallest_options


def _make_mock_post(captured: dict):
    """Return a sync callable that acts as aiohttp session.post async context manager."""
    async def iter_chunks():
        yield b"\x00\x01", False

    def post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["body"] = json

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.content = MagicMock()
        mock_resp.content.iter_chunks = iter_chunks

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    return post


# ---------------------------------------------------------------------------
# Construction and defaults
# ---------------------------------------------------------------------------


def test_default_model_is_v31():
    tts = TTS(api_key="test-key")
    assert tts._opts.model == "lightning_v3.1"


def test_pro_model_accepted():
    tts = TTS(api_key="test-key", model="lightning_v3.1_pro")
    assert tts._opts.model == "lightning_v3.1_pro"


def test_lightning_v2_not_a_default():
    tts = TTS(api_key="test-key")
    assert "v2" not in tts._opts.model


def test_no_consistency_similarity_enhancement_attrs():
    tts = TTS(api_key="test-key")
    assert not hasattr(tts._opts, "consistency")
    assert not hasattr(tts._opts, "similarity")
    assert not hasattr(tts._opts, "enhancement")


def test_constructor_rejects_unknown_kwargs():
    with pytest.raises(TypeError):
        TTS(api_key="test-key", consistency=0.5)  # type: ignore[call-arg]


def test_missing_api_key_raises():
    with patch.dict("os.environ", {}, clear=True):
        import os
        os.environ.pop("SMALLEST_API_KEY", None)
        with pytest.raises(ValueError, match="API key"):
            TTS()


def test_api_key_from_env():
    with patch.dict("os.environ", {"SMALLEST_API_KEY": "env-key"}):
        tts = TTS()
        assert tts._opts.api_key == "env-key"


# ---------------------------------------------------------------------------
# _to_smallest_options — request body shape
# ---------------------------------------------------------------------------


def test_request_body_includes_model():
    tts = TTS(api_key="test-key", model="lightning_v3.1_pro")
    body = _to_smallest_options(tts._opts)
    assert body["model"] == "lightning_v3.1_pro"


def test_request_body_standard_fields():
    tts = TTS(api_key="test-key", voice_id="meher", sample_rate=44100, speed=1.2, language="hi")
    body = _to_smallest_options(tts._opts)
    assert body["voice_id"] == "meher"
    assert body["sample_rate"] == 44100
    assert body["speed"] == 1.2
    assert body["language"] == "hi"
    assert body["output_format"] == "pcm"


def test_request_body_no_v2_only_fields():
    tts = TTS(api_key="test-key")
    body = _to_smallest_options(tts._opts)
    assert "consistency" not in body
    assert "similarity" not in body
    assert "enhancement" not in body


def test_language_code_serialized_as_string():
    tts = TTS(api_key="test-key", language="en")
    body = _to_smallest_options(tts._opts)
    assert isinstance(body["language"], str)
    assert body["language"] == "en"


# ---------------------------------------------------------------------------
# update_options
# ---------------------------------------------------------------------------


def test_update_options_model():
    tts = TTS(api_key="test-key")
    tts.update_options(model="lightning_v3.1_pro")
    assert tts._opts.model == "lightning_v3.1_pro"


def test_update_options_voice_speed_language():
    tts = TTS(api_key="test-key")
    tts.update_options(voice_id="cressida", speed=1.5, language="hi")
    assert tts._opts.voice_id == "cressida"
    assert tts._opts.speed == 1.5
    assert tts._opts.language.language == "hi"


def test_update_options_no_consistency_param():
    tts = TTS(api_key="test-key")
    with pytest.raises(TypeError):
        tts.update_options(consistency=0.8)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Endpoint URL — must be /tts, not /{model}/get_speech
# ---------------------------------------------------------------------------


async def test_synthesize_hits_unified_tts_endpoint():
    """The HTTP request must go to /tts, with model in the JSON body."""
    tts_instance = TTS(api_key="test-key", model="lightning_v3.1_pro", voice_id="meher")

    captured: dict = {}
    mock_session = MagicMock()
    mock_session.post = _make_mock_post(captured)
    tts_instance._session = mock_session

    stream = ChunkedStream(
        tts=tts_instance, input_text="Hello", conn_options=DEFAULT_API_CONNECT_OPTIONS
    )
    mock_emitter = MagicMock()
    mock_emitter.initialize = MagicMock()
    mock_emitter.push = MagicMock()
    mock_emitter.flush = MagicMock()

    try:
        await stream._run(mock_emitter)
    finally:
        await stream.aclose()

    assert captured["url"].endswith("/tts"), f"Expected /tts endpoint, got: {captured['url']}"
    assert "get_speech" not in captured["url"], "Old /{model}/get_speech path must not be used"
    assert captured["body"]["model"] == "lightning_v3.1_pro"
    assert captured["body"]["text"] == "Hello"
    assert captured["body"]["voice_id"] == "meher"


async def test_synthesize_standard_model_also_uses_unified_endpoint():
    tts_instance = TTS(api_key="test-key", model="lightning_v3.1")

    captured: dict = {}
    mock_session = MagicMock()
    mock_session.post = _make_mock_post(captured)
    tts_instance._session = mock_session

    stream = ChunkedStream(
        tts=tts_instance, input_text="Test", conn_options=DEFAULT_API_CONNECT_OPTIONS
    )
    mock_emitter = MagicMock()
    mock_emitter.initialize = MagicMock()
    mock_emitter.push = MagicMock()
    mock_emitter.flush = MagicMock()

    try:
        await stream._run(mock_emitter)
    finally:
        await stream.aclose()

    assert captured["url"].endswith("/tts")
