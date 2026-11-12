from __future__ import annotations

import pytest

from livekit.plugins.sarvam.stt import SpeechToTextLanguage, _validate_language_for_model
from livekit.plugins.sarvam.stt_streaming import (
    RealtimeSpeechStream,
    RealtimeSTTOptions,
    STTRealtime,
    _build_realtime_ws_url,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "language, expected",
    [
        ("od-IN", "or-IN"),
        ("od-in", "or-IN"),
        ("OD-IN", "or-IN"),
        ("or-IN", "or-IN"),
        ("hi-IN", "hi-IN"),
    ],
)
def test_realtime_language_normalization(language: str, expected: str) -> None:
    opts = RealtimeSTTOptions(language=language, api_key="test")
    assert opts.language == expected


def test_realtime_rejects_unsupported_language() -> None:
    with pytest.raises(ValueError, match="not supported"):
        RealtimeSTTOptions(language="xx-YY", api_key="test")


def test_realtime_url_uses_standard_odia_code() -> None:
    opts = RealtimeSTTOptions(language="od-IN", api_key="test")
    url = _build_realtime_ws_url(opts.base_url, opts)
    assert "language_code=or-IN" in url
    assert "od-IN" not in url


@pytest.mark.parametrize("language", ["od-IN", "od-in", "OD-IN", "or-IN"])
def test_realtime_constructor_accepts_odia(language: str) -> None:
    instance = STTRealtime(language=language, api_key="test")
    assert instance._opts.language == "or-IN"


def test_realtime_update_options_normalizes_odia() -> None:
    instance = STTRealtime(language="en-IN", api_key="test")
    instance.update_options(language="od-IN")
    assert instance._opts.language == "or-IN"


def test_stream_update_queues_standard_odia_code() -> None:
    # Bypass initialization to avoid starting tasks or opening a WebSocket.
    stream = RealtimeSpeechStream.__new__(RealtimeSpeechStream)
    stream._opts = RealtimeSTTOptions(language="en-IN", api_key="test")
    stream._pending_config_update = None

    stream.update_options(language="od-IN")

    assert stream._opts.language == "or-IN"
    assert stream._pending_config_update == {
        "event": "config.update",
        "language_code": "or-IN",
    }


def test_batch_odia_code_is_unchanged() -> None:
    assert SpeechToTextLanguage.OD_IN.value == "od-IN"
    assert _validate_language_for_model("saaras:v3", "od-IN") == "od-IN"
    with pytest.raises(ValueError, match="not supported"):
        _validate_language_for_model("saaras:v3", "or-IN")
