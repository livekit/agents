"""Tests for Gnani Vachana STT plugin configuration and behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.plugin("gnani")


def test_stt_requires_api_key():
    """STT constructor raises when no API key is provided."""
    from livekit.plugins.gnani import STT

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API key"):
            STT(api_key=None)


def test_stt_accepts_api_key_directly():
    """STT constructor accepts api_key argument."""
    from livekit.plugins.gnani import STT

    stt = STT(api_key="test-key")
    assert stt._opts.api_key == "test-key"


def test_stt_accepts_api_key_from_env():
    """STT constructor reads GNANI_API_KEY from environment."""
    from livekit.plugins.gnani import STT

    with patch.dict("os.environ", {"GNANI_API_KEY": "env-key"}):
        stt = STT()
        assert stt._opts.api_key == "env-key"


def test_stt_default_language():
    """STT defaults to en-IN."""
    from livekit.plugins.gnani import STT

    stt = STT(api_key="test-key")
    assert stt._opts.language == "en-IN"


def test_stt_custom_language():
    """STT accepts custom language."""
    from livekit.plugins.gnani import STT

    stt = STT(api_key="test-key", language="hi-IN")
    assert stt._opts.language == "hi-IN"


def test_stt_default_sample_rate():
    """STT defaults to 16000 Hz sample rate."""
    from livekit.plugins.gnani import STT

    stt = STT(api_key="test-key")
    assert stt._opts.sample_rate == 16000


def test_stt_accepts_8k_sample_rate():
    """STT accepts 8000 Hz sample rate."""
    from livekit.plugins.gnani import STT

    stt = STT(api_key="test-key", sample_rate=8000)
    assert stt._opts.sample_rate == 8000


def test_stt_rejects_invalid_sample_rate():
    """STT rejects sample rates other than 8000 or 16000."""
    from livekit.plugins.gnani import STT

    with pytest.raises(ValueError, match="sample_rate"):
        STT(api_key="test-key", sample_rate=44100)


def test_stt_capabilities():
    """STT reports streaming=True, interim_results=False."""
    from livekit.plugins.gnani import STT

    stt = STT(api_key="test-key")
    assert stt.capabilities.streaming is True
    assert stt.capabilities.interim_results is False


def test_stt_model_property():
    """STT model property returns the correct identifier."""
    from livekit.plugins.gnani import STT

    stt = STT(api_key="test-key")
    assert stt.model == "vachana-stt-v3"


def test_stt_provider_property():
    """STT provider property returns 'Gnani'."""
    from livekit.plugins.gnani import STT

    stt = STT(api_key="test-key")
    assert stt.provider == "Gnani"


def test_stt_base_url_default():
    """STT defaults to Vachana API base URL."""
    from livekit.plugins.gnani import STT

    stt = STT(api_key="test-key")
    assert stt._opts.base_url == "https://api.vachana.ai"


def test_stt_custom_base_url():
    """STT accepts custom base URL."""
    from livekit.plugins.gnani import STT

    stt = STT(api_key="test-key", base_url="https://custom.api.com")
    assert stt._opts.base_url == "https://custom.api.com"


def test_stt_only_api_key_auth():
    """STT options only contain api_key for authentication (no organization_id or user_id)."""
    from livekit.plugins.gnani import STT

    stt = STT(api_key="test-key")
    assert not hasattr(stt._opts, "organization_id")
    assert not hasattr(stt._opts, "user_id")


def test_speech_stream_ws_url_https():
    """SpeechStream builds wss:// URL from https:// base."""
    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
    from livekit.plugins.gnani.stt import STT, GnaniSTTOptions, SpeechStream

    stt = STT(api_key="test-key")
    opts = GnaniSTTOptions(
        api_key="test-key",
        language="en-IN",
        base_url="https://api.vachana.ai",
    )

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        task = MagicMock()
        return task

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        stream = SpeechStream(
            stt=stt,
            opts=opts,
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
        )

    assert stream._build_ws_url() == "wss://api.vachana.ai/stt/v3/stream"


def test_speech_stream_ws_url_http():
    """SpeechStream builds ws:// URL from http:// base."""
    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
    from livekit.plugins.gnani.stt import STT, GnaniSTTOptions, SpeechStream

    stt = STT(api_key="test-key")
    opts = GnaniSTTOptions(
        api_key="test-key",
        language="en-IN",
        base_url="http://localhost:8080",
    )

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        task = MagicMock()
        return task

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        stream = SpeechStream(
            stt=stt,
            opts=opts,
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
        )

    assert stream._build_ws_url() == "ws://localhost:8080/stt/v3/stream"
