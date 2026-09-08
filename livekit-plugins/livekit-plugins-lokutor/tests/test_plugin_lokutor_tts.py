from __future__ import annotations

from unittest.mock import patch

import pytest


def test_tts_requires_api_key():
    from livekit.plugins.lokutor import TTS

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API key"):
            TTS(api_key=None)


def test_tts_accepts_api_key_directly():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.api_key == "test-key"


def test_tts_accepts_api_key_from_env():
    from livekit.plugins.lokutor import TTS

    with patch.dict("os.environ", {"LOKUTOR_API_KEY": "env-key"}):
        tts = TTS()
        assert tts._opts.api_key == "env-key"


def test_tts_default_voice():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.voice == "F1"


def test_tts_custom_voice():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key", voice="M3")
    assert tts._opts.voice == "M3"


def test_tts_all_voices_accepted():
    from livekit.plugins.lokutor import TTS

    voices = ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]
    for voice in voices:
        tts = TTS(api_key="test-key", voice=voice)
        assert tts._opts.voice == voice


def test_tts_default_language():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.language == "en"


def test_tts_custom_language():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key", language="es")
    assert tts._opts.language == "es"


def test_tts_none_language():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key", language=None)
    assert tts._opts.language is None


def test_tts_default_speed():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.speed == 1.05


def test_tts_default_steps():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.steps == 5


def test_tts_default_visemes():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.visemes is False


def test_tts_custom_params():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key", speed=1.5, steps=8, visemes=True)
    assert tts._opts.speed == 1.5
    assert tts._opts.steps == 8
    assert tts._opts.visemes is True


def test_tts_default_sample_rate():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key")
    assert tts.sample_rate == 44100


def test_tts_custom_sample_rate():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key", sample_rate=22050)
    assert tts.sample_rate == 22050


def test_tts_model_property():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key")
    assert tts.model == "versa-1.0"


def test_tts_provider_property():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key")
    assert tts.provider == "Lokutor"


def test_tts_capabilities():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key")
    assert tts.capabilities.streaming is True
    assert tts.capabilities.aligned_transcript is False


def test_tts_num_channels():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key")
    assert tts.num_channels == 1


def test_tts_default_base_url():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.base_url == "wss://api.lokutor.com"


def test_tts_ws_url():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.get_ws_url() == "wss://api.lokutor.com/ws/tts"


def test_synthesize_returns_chunked_stream():
    from unittest.mock import MagicMock, patch

    from livekit.plugins.lokutor import TTS, ChunkedStream

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    tts = TTS(api_key="test-key")
    with patch("livekit.agents.tts.tts.asyncio.create_task", side_effect=_fake_create_task):
        stream = tts.synthesize("hello")
    assert isinstance(stream, ChunkedStream)


def test_stream_returns_synthesize_stream():
    from unittest.mock import MagicMock, patch

    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
    from livekit.plugins.lokutor import TTS
    from livekit.plugins.lokutor.tts import SynthesizeStream

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    tts = TTS(api_key="test-key")
    with patch("livekit.agents.tts.tts.asyncio.create_task", side_effect=_fake_create_task):
        stream = tts.stream(conn_options=DEFAULT_API_CONNECT_OPTIONS)
    assert isinstance(stream, SynthesizeStream)


def test_build_request_includes_language():
    from livekit.plugins.lokutor.tts import _build_request, _TTSOptions

    opts = _TTSOptions(
        api_key="key",
        voice="F1",
        language="es",
        speed=1.0,
        steps=5,
        visemes=False,
        base_url="wss://api.lokutor.com",
        sample_rate=44100,
    )
    request = _build_request(opts, "hola")
    assert request["text"] == "hola"
    assert request["voice"] == "F1"
    assert request["speed"] == 1.0
    assert request["steps"] == 5
    assert request["visemes"] is False
    assert request["lang"] == "es"


def test_build_request_without_language():
    from livekit.plugins.lokutor.tts import _build_request, _TTSOptions

    opts = _TTSOptions(
        api_key="key",
        voice="M1",
        language=None,
        speed=1.05,
        steps=5,
        visemes=False,
        base_url="wss://api.lokutor.com",
        sample_rate=44100,
    )
    request = _build_request(opts, "hello")
    assert request["text"] == "hello"
    assert request["voice"] == "M1"
    assert "lang" not in request
