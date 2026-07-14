"""Tests for Gnani Vachana TTS plugin configuration and behavior."""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.plugin("gnani")


def test_tts_requires_api_key():
    """TTS constructor raises when no API key is provided."""
    from livekit.plugins.gnani import TTS

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API key"):
            TTS(api_key=None)


def test_tts_accepts_api_key_directly():
    """TTS constructor accepts api_key argument."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.api_key == "test-key"


def test_tts_accepts_api_key_from_env():
    """TTS constructor reads GNANI_API_KEY from environment."""
    from livekit.plugins.gnani import TTS

    with patch.dict("os.environ", {"GNANI_API_KEY": "env-key"}):
        tts = TTS()
        assert tts._opts.api_key == "env-key"


def test_tts_default_voice():
    """TTS defaults to 'Pranav' voice."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.voice == "Pranav"


def test_tts_custom_voice():
    """TTS accepts custom voice."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", voice="Deepak")
    assert tts._opts.voice == "Deepak"


def test_tts_all_voices_accepted():
    """TTS accepts all documented voices."""
    from livekit.plugins.gnani import TTS

    voices = [
        "Pranav",
        "Kaveri",
        "Shubhra",
        "Deepak",
    ]
    for voice in voices:
        tts = TTS(api_key="test-key", voice=voice)
        assert tts._opts.voice == voice


def test_tts_rejects_invalid_voice():
    """TTS rejects unsupported voices."""
    from livekit.plugins.gnani import TTS

    with pytest.raises(ValueError, match="not supported"):
        TTS(api_key="test-key", voice="nonexistent")


def test_tts_default_model():
    """TTS defaults to vachana-voice-v3 for the default v3 voice."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.model == "vachana-voice-v3"


def test_tts_model_v3():
    """TTS uses vachana-voice-v3 model."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", voice="Kaveri")
    assert tts._opts.model == "vachana-voice-v3"
    assert tts.model == "vachana-voice-v3"


def test_tts_model_explicit_override():
    """Explicit model parameter overrides default."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", voice="Pranav", model="custom-model")
    assert tts._opts.model == "custom-model"


def test_tts_model_property():
    """TTS model property returns the configured model."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    assert tts.model == "vachana-voice-v3"


def test_tts_provider_property():
    """TTS provider property returns 'Gnani'."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    assert tts.provider == "Gnani"


def test_tts_capabilities():
    """TTS reports streaming=True."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    assert tts.capabilities.streaming is True


def test_tts_default_sample_rate():
    """TTS defaults to 16000 Hz sample rate."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    assert tts.sample_rate == 16000


def test_tts_custom_sample_rate():
    """TTS accepts custom sample rate."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", sample_rate=44100)
    assert tts.sample_rate == 44100


def test_tts_default_encoding():
    """TTS defaults to linear_pcm encoding."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.encoding == "linear_pcm"


def test_tts_default_container():
    """TTS defaults to wav container."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.container == "wav"


def test_tts_custom_audio_config():
    """TTS accepts custom encoding and container."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", encoding="oggopus", container="ogg")
    assert tts._opts.encoding == "oggopus"
    assert tts._opts.container == "ogg"


def test_tts_update_options_voice():
    """update_options can change voice."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", voice="Pranav")
    tts.update_options(voice="Kaveri")
    assert tts._opts.voice == "Kaveri"


def test_tts_update_options_voice_and_model():
    """update_options with both voice and model uses the explicit model."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", voice="Pranav")
    tts.update_options(voice="Shubhra", model="custom-model")
    assert tts._opts.voice == "Shubhra"
    assert tts._opts.model == "custom-model"


def test_tts_update_options_rejects_invalid_voice():
    """update_options rejects unsupported voices."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    with pytest.raises(ValueError, match="not supported"):
        tts.update_options(voice="nonexistent")


def test_tts_update_options_model():
    """update_options can change model."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    tts.update_options(model="custom-model")
    assert tts._opts.model == "custom-model"


def test_tts_default_synthesize_method():
    """TTS defaults to 'rest' synthesize method."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.synthesize_method == "rest"


def test_tts_sse_synthesize_method():
    """TTS with synthesize_method='sse' stores the option correctly."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", synthesize_method="sse")
    assert tts._opts.synthesize_method == "sse"


def test_tts_rest_synthesize_method():
    """TTS with synthesize_method='rest' stores the option correctly."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", synthesize_method="rest")
    assert tts._opts.synthesize_method == "rest"


def test_tts_websocket_synthesize_method():
    """TTS with synthesize_method='websocket' stores the option correctly."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", synthesize_method="websocket")
    assert tts._opts.synthesize_method == "websocket"


def test_synthesize_routes_rest_by_default():
    """synthesize() returns RESTChunkedStream when method is 'rest'."""
    from unittest.mock import MagicMock, patch

    from livekit.plugins.gnani import TTS
    from livekit.plugins.gnani.tts import RESTChunkedStream

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    tts = TTS(api_key="test-key", synthesize_method="rest")
    with patch("livekit.agents.tts.tts.asyncio.create_task", side_effect=_fake_create_task):
        stream = tts.synthesize("hello")
    assert isinstance(stream, RESTChunkedStream)


def test_synthesize_routes_sse():
    """synthesize() returns SSEChunkedStream when method is 'sse'."""
    from unittest.mock import MagicMock, patch

    from livekit.plugins.gnani import TTS
    from livekit.plugins.gnani.tts import SSEChunkedStream

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    tts = TTS(api_key="test-key", synthesize_method="sse")
    with patch("livekit.agents.tts.tts.asyncio.create_task", side_effect=_fake_create_task):
        stream = tts.synthesize("hello")
    assert isinstance(stream, SSEChunkedStream)


def test_synthesize_routes_websocket():
    """synthesize() returns WebSocketChunkedStream when method is 'websocket'."""
    from unittest.mock import MagicMock, patch

    from livekit.plugins.gnani import TTS
    from livekit.plugins.gnani.tts import WebSocketChunkedStream

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    tts = TTS(api_key="test-key", synthesize_method="websocket")
    with patch("livekit.agents.tts.tts.asyncio.create_task", side_effect=_fake_create_task):
        stream = tts.synthesize("hello")
    assert isinstance(stream, WebSocketChunkedStream)


def test_tts_base_url_default():
    """TTS defaults to Vachana API base URL."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.base_url == "https://api.vachana.ai"


def test_tts_ws_url_from_https():
    """SynthesizeStream builds wss:// from https:// base."""
    from unittest.mock import MagicMock, patch

    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
    from livekit.plugins.gnani.tts import TTS, SynthesizeStream

    tts = TTS(api_key="test-key")

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    with patch("livekit.agents.tts.tts.asyncio.create_task", side_effect=_fake_create_task):
        stream = SynthesizeStream(tts=tts, conn_options=DEFAULT_API_CONNECT_OPTIONS)

    assert stream._build_ws_url() == "wss://api.vachana.ai/api/v1/tts"


def test_tts_ws_url_from_http():
    """SynthesizeStream builds ws:// from http:// base."""
    from unittest.mock import MagicMock, patch

    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
    from livekit.plugins.gnani.tts import TTS, SynthesizeStream

    tts = TTS(api_key="test-key", base_url="http://localhost:9090")

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    with patch("livekit.agents.tts.tts.asyncio.create_task", side_effect=_fake_create_task):
        stream = SynthesizeStream(tts=tts, conn_options=DEFAULT_API_CONNECT_OPTIONS)

    assert stream._build_ws_url() == "ws://localhost:9090/api/v1/tts"


def test_tts_default_num_channels():
    """TTS defaults to 1 channel."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.num_channels == 1
    assert tts.num_channels == 1


def test_tts_custom_num_channels():
    """TTS accepts custom num_channels."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", num_channels=2)
    assert tts._opts.num_channels == 2


def test_tts_default_bitrate():
    """TTS defaults bitrate to None."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.bitrate is None


def test_tts_custom_bitrate():
    """TTS accepts custom bitrate."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", bitrate="128k")
    assert tts._opts.bitrate == "128k"


def test_tts_rejects_invalid_sample_rate():
    """TTS rejects unsupported sample rates."""
    from livekit.plugins.gnani import TTS

    with pytest.raises(ValueError, match="sample_rate"):
        TTS(api_key="test-key", sample_rate=48000)


def test_tts_all_sample_rates_accepted():
    """TTS accepts all documented sample rates."""
    from livekit.plugins.gnani import TTS

    for rate in (8000, 16000, 22050, 44100):
        tts = TTS(api_key="test-key", sample_rate=rate)
        assert tts.sample_rate == rate


def test_tts_stream_returns_synthesize_stream():
    """TTS.stream() returns a SynthesizeStream instance."""
    from unittest.mock import MagicMock, patch

    from livekit.plugins.gnani import TTS
    from livekit.plugins.gnani.tts import SynthesizeStream

    tts = TTS(api_key="test-key")

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    with patch("livekit.agents.tts.tts.asyncio.create_task", side_effect=_fake_create_task):
        stream = tts.stream()
    assert isinstance(stream, SynthesizeStream)


def test_tts_update_options_preserves_other_fields():
    """update_options only changes specified fields."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", voice="Pranav")
    tts.update_options(voice="Deepak")
    assert tts._opts.voice == "Deepak"
    assert tts._opts.model == "vachana-voice-v3"
    assert tts._opts.encoding == "linear_pcm"


def test_tts_websocket_chunked_stream_ws_url():
    """WebSocketChunkedStream builds correct WS URL."""
    from unittest.mock import MagicMock, patch

    from livekit.plugins.gnani import TTS
    from livekit.plugins.gnani.tts import WebSocketChunkedStream

    tts = TTS(api_key="test-key", synthesize_method="websocket")

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    with patch("livekit.agents.tts.tts.asyncio.create_task", side_effect=_fake_create_task):
        stream = tts.synthesize("hello")
    assert isinstance(stream, WebSocketChunkedStream)
    assert stream._build_ws_url() == "wss://api.vachana.ai/api/v1/tts"
