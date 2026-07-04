"""Tests for Tencent Cloud ASR STT plugin."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import urllib.parse
from unittest.mock import MagicMock, patch

import pytest

from livekit.agents import APIStatusError
from livekit.agents.stt import SpeechEventType

pytestmark = pytest.mark.unit


def _make_stream_for_unit_test():
    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
    from livekit.plugins.tencent import STT
    from livekit.plugins.tencent.stt import SpeechStream

    stt = STT(app_id="app", secret_id="sid", secret_key="skey")

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        stream = SpeechStream(
            stt=stt,
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
            language="zh-CN",
        )
    return stream


def test_stt_requires_credentials(monkeypatch):
    from livekit.plugins.tencent import STT

    monkeypatch.delenv("TENCENT_ASR_APP_ID", raising=False)
    monkeypatch.delenv("TENCENT_ASR_SECRET_ID", raising=False)
    monkeypatch.delenv("TENCENT_ASR_SECRET_KEY", raising=False)

    with pytest.raises(ValueError, match="Tencent ASR credentials"):
        STT()


def test_build_url_sorts_params_and_signs():
    stream = _make_stream_for_unit_test()

    url = stream._build_url(voice_id="voice-1", now=1_700_000_000)
    unsigned = url.removeprefix("wss://").split("&signature=", 1)[0]
    query = unsigned.split("?", 1)[1]
    keys = [item.split("=", 1)[0] for item in query.split("&")]

    assert unsigned.startswith("asr.cloud.tencent.com/asr/v2/app?")
    assert keys == sorted(keys)
    assert "engine_model_type=16k_zh_en" in query
    assert "voice_id=voice-1" in query
    assert "vad_silence_time=500" in query

    expected_signature = base64.b64encode(
        hmac.new(b"skey", unsigned.encode("utf-8"), hashlib.sha1).digest()
    ).decode("utf-8")
    assert url.endswith(f"&signature={urllib.parse.quote(expected_signature, safe='')}")


def test_process_stream_event_maps_tencent_events():
    stream = _make_stream_for_unit_test()
    stream._request_id = "voice-1"

    stream._process_stream_event(
        {
            "code": 0,
            "voice_id": "voice-1",
            "result": {
                "slice_type": 0,
                "index": 0,
                "start_time": 0,
                "end_time": 120,
                "voice_text_str": "ni",
            },
        }
    )
    stream._process_stream_event(
        {
            "code": 0,
            "voice_id": "voice-1",
            "result": {
                "slice_type": 1,
                "index": 0,
                "start_time": 0,
                "end_time": 200,
                "voice_text_str": "ni hao",
            },
        }
    )
    stream._process_stream_event(
        {
            "code": 0,
            "voice_id": "voice-1",
            "result": {
                "slice_type": 2,
                "index": 0,
                "start_time": 0,
                "end_time": 300,
                "voice_text_str": "ni hao.",
                "word_list": [{"word": "ni hao", "start_time": 0, "end_time": 250}],
            },
        }
    )

    events = [stream._event_ch.recv_nowait() for _ in range(5)]
    assert [event.type for event in events] == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.INTERIM_TRANSCRIPT,
        SpeechEventType.INTERIM_TRANSCRIPT,
        SpeechEventType.FINAL_TRANSCRIPT,
        SpeechEventType.END_OF_SPEECH,
    ]
    assert events[1].alternatives[0].text == "ni"
    assert events[3].alternatives[0].text == "ni hao."
    assert events[3].alternatives[0].language == "zh-CN"
    assert str(events[3].alternatives[0].words[0]) == "ni hao"


def test_process_stream_event_nonzero_code_raises_status_error():
    stream = _make_stream_for_unit_test()
    stream._request_id = "voice-1"

    with pytest.raises(APIStatusError) as exc_info:
        stream._process_stream_event(
            {
                "code": 4001,
                "message": "auth failed",
                "voice_id": "voice-1",
            }
        )

    assert exc_info.value.status_code == 4001
    assert exc_info.value.request_id == "voice-1"
    assert exc_info.value.body["message"] == "auth failed"


async def test_send_audio_batches_pcm_and_sends_end():
    from livekit import rtc

    stream = _make_stream_for_unit_test()
    sent_bytes: list[bytes] = []
    sent_text: list[str] = []

    fake_ws = MagicMock()

    async def _send_bytes(data: bytes):
        sent_bytes.append(data)

    async def _send_str(data: str):
        sent_text.append(data)

    fake_ws.send_bytes.side_effect = _send_bytes
    fake_ws.send_str.side_effect = _send_str
    stream._ws = fake_ws

    samples = b"\x01\x00" * 1600
    frame = rtc.AudioFrame(
        data=samples,
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=1600,
    )

    task = asyncio.create_task(stream._send_audio_task())
    stream._input_ch.send_nowait(frame)
    stream._input_ch.send_nowait(frame)
    stream._input_ch.close()
    await task

    assert sent_bytes == [samples + samples]
    assert sent_text == ['{"type":"end"}']
