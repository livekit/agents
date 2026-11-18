from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import pytest

from livekit import rtc
from livekit.agents import stt, utils
from livekit.plugins.gladia import stt as gladia_stt

pytestmark = pytest.mark.plugin("gladia")


async def test_update_options_stores_region_for_new_streams():
    from livekit.plugins.gladia import STT

    stt = STT(api_key="test-key", region="eu-west")
    stt.update_options(region="us-west")

    assert stt._opts.region == "us-west"


@pytest.mark.parametrize("translate", [False, True])
@pytest.mark.parametrize("end_time", [0.4, 0.0])
def test_endpoint_preserves_source_timing(translate, end_time):
    stream = object.__new__(gladia_stt.SpeechStream)
    stream._opts = gladia_stt.STT(api_key="test-key")._opts
    stream._opts.translation_config.enabled = translate
    stream._speaking = False
    stream._input_ch = SimpleNamespace(start_time_offset=7.0)
    stream._request_id = "test"
    events = []
    stream._event_ch = SimpleNamespace(send_nowait=events.append)
    utterance = {"text": "hello", "language": "en", "start": 0.1, "end": end_time}
    stream._process_gladia_message(
        {"type": "transcript", "data": {"is_final": True, "utterance": utterance}}
    )
    if translate:
        stream._process_gladia_message(
            {
                "type": "translation",
                "data": {
                    "utterance": utterance,
                    "translated_utterance": {"text": "bonjour", "language": "fr", "end": 9.0},
                },
            }
        )
    endpoint = events[-1]
    assert endpoint.type == stt.SpeechEventType.END_OF_SPEECH
    if end_time:
        assert endpoint.alternatives[0].end_time == pytest.approx(7.4)
    else:
        assert endpoint.alternatives == []


@pytest.mark.no_concurrent
async def test_internal_reconnect_reanchors_audio_clock(monkeypatch):
    from unittest.mock import AsyncMock

    from livekit.agents import APIConnectOptions
    from livekit.plugins.gladia.stt import STT, SpeechStream

    from .stt_reconnect import ReconnectClock

    clock = ReconnectClock(monkeypatch)
    monkeypatch.setattr(
        SpeechStream,
        "_init_live_session",
        AsyncMock(return_value={"url": "wss://test.invalid", "id": "test"}),
    )
    provider = STT(api_key="test-key", http_session=clock)
    stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
    try:
        await clock.check(stream, lambda: stream.update_options(languages=["fr"]))
    finally:
        await stream.aclose()
        await provider.aclose()


async def test_update_options_rejects_translation_without_target_languages():
    from livekit.plugins.gladia import STT

    stt = STT(api_key="test-key")
    with pytest.raises(ValueError, match="translation_target_languages is required"):
        stt.update_options(translation_enabled=True, region="us-west")
    assert stt._opts.translation_config.enabled is False
    assert stt._opts.region == "eu-west"

    stt = STT(api_key="test-key", translation_enabled=True, translation_target_languages=["fr"])
    with pytest.raises(ValueError, match="translation_target_languages is required"):
        stt.update_options(translation_target_languages=[])
    assert stt._opts.translation_config.target_languages == ["fr"]

    stt.update_options(translation_enabled=False, translation_target_languages=[])
    assert stt._opts.translation_config.enabled is False


async def test_send_audio_forwards_silence_and_flushes_partial_chunk():
    sample_rate = 16000

    def _frame(sample: int, duration_ms: int) -> rtc.AudioFrame:
        num_samples = sample_rate * duration_ms // 1000
        return rtc.AudioFrame(
            data=sample.to_bytes(2, "little", signed=True) * num_samples,
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=num_samples,
        )

    messages: list[dict] = []

    async def _send_str(payload: str) -> None:
        messages.append(json.loads(payload))

    usage: list[float] = []
    stream = object.__new__(gladia_stt.SpeechStream)
    stream._opts = gladia_stt.STT(api_key="test-key", sample_rate=sample_rate)._opts
    stream._ws = SimpleNamespace(send_str=_send_str)
    stream._session = SimpleNamespace(closed=False)
    stream._input_ch = utils.aio.Chan()
    stream._audio_duration_collector = gladia_stt.PeriodicCollector(
        callback=usage.append, duration=5.0
    )

    # speech and silence share 100 ms chunks, and 30 ms of trailing silence stays buffered
    pushed = [_frame(1000, 50), _frame(0, 100), _frame(1000, 50), _frame(0, 30)]
    for frame in pushed:
        stream._input_ch.send_nowait(frame)
    stream._input_ch.send_nowait(gladia_stt.SpeechStream._FlushSentinel())
    stream._input_ch.close()

    await stream._send_audio_task()

    assert [m["type"] for m in messages] == [
        "audio_chunk",
        "audio_chunk",
        "audio_chunk",
        "stop_recording",  # explicit flush
        "stop_recording",  # end of input
    ]
    chunks = [base64.b64decode(m["data"]["chunk"]) for m in messages[:3]]
    assert [len(chunk) for chunk in chunks] == [3200, 3200, 960]
    assert b"".join(chunks) == b"".join(frame.data.tobytes() for frame in pushed)
    assert sum(usage) == pytest.approx(0.23)
