"""Tests for Reson8 STT plugin configuration and payload parsing."""

import math

import pytest

from livekit.agents import stt as agents_stt

pytestmark = pytest.mark.unit


async def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("RESON8_API_KEY", raising=False)
    from livekit.plugins.reson8 import STT

    with pytest.raises(ValueError):
        STT()


async def test_capabilities_and_metadata():
    from livekit.plugins.reson8 import STT

    stt = STT(api_key="test-key")
    assert stt.capabilities.streaming
    assert stt.capabilities.interim_results
    assert stt.provider == "reson8"
    assert stt.model == "default"


async def test_custom_model_id_reported_as_model():
    from livekit.plugins.reson8 import STT

    stt = STT(api_key="test-key", custom_model_id="my-model")
    assert stt.model == "my-model"


async def test_language_omitted_when_auto_detect():
    from livekit.plugins.reson8 import STT

    stt = STT(api_key="test-key")
    assert "language" not in stt._opts.query_params(streaming=True)


async def test_query_params_streaming_vs_batch():
    from livekit.plugins.reson8 import STT

    stt = STT(api_key="test-key", language="en", include_confidence=True, include_language=True)

    streaming = stt._opts.query_params(streaming=True)
    assert streaming["language"] == "en"
    assert streaming["include_language"] == "true"
    assert "include_confidence" not in streaming  # confidence is batch-only

    batch = stt._opts.query_params(streaming=False)
    assert batch["include_confidence"] == "true"
    assert "include_language" not in batch  # language reporting is streaming-only


async def test_update_options():
    from livekit.plugins.reson8 import STT

    stt = STT(api_key="test-key")
    stt.update_options(language="nl", custom_model_id="m2")
    assert stt._opts.language == "nl"
    assert stt._opts.custom_model_id == "m2"


async def test_build_speech_data_parses_turn_payload():
    from livekit.plugins.reson8._utils import build_speech_data

    payload = {
        "text": "hello world",
        "language": "en",
        "start_ms": 1000,
        "duration_ms": 500,
        "words": [
            {"text": "hello", "start_ms": 1000, "duration_ms": 200, "confidence": math.log(0.9)},
            {"text": "world", "start_ms": 1200, "duration_ms": 300, "confidence": math.log(0.8)},
        ],
    }
    data = build_speech_data(payload, language=None, start_time_offset=2.0)

    assert isinstance(data, agents_stt.SpeechData)
    assert data.text == "hello world"
    assert data.language == "en"
    assert data.start_time == pytest.approx(3.0)  # 2.0 offset + 1000ms
    assert data.end_time == pytest.approx(3.5)
    # log-probabilities are converted back to (0, 1] probabilities
    assert data.confidence == pytest.approx((0.9 + 0.8) / 2, rel=1e-6)
    assert len(data.words) == 2
    assert data.words[0].start_time == pytest.approx(3.0)
