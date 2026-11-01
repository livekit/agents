from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from mistralai.client.models import TranscriptionResponse

from livekit import rtc
from livekit.agents import APIConnectOptions, stt
from livekit.plugins.mistralai import STT

pytestmark = pytest.mark.unit


async def test_transcription_without_segment_timestamps_keeps_text() -> None:
    response = TranscriptionResponse.model_validate(
        {
            "model": "voxtral-mini-latest",
            "text": "hello",
            "language": "en",
            "usage": {},
            "segments": [{"text": "hello", "start": None, "end": None}],
        }
    )
    client = Mock()
    client.audio.transcriptions.complete_async = AsyncMock(return_value=response)
    model = STT(client=client)
    try:
        result = await model.recognize(
            rtc.AudioFrame.create(sample_rate=16000, num_channels=1, samples_per_channel=160),
            conn_options=APIConnectOptions(max_retry=0),
        )
        assert result.type == stt.SpeechEventType.FINAL_TRANSCRIPT
        assert result.alternatives[0].text == "hello"
        assert result.alternatives[0].words is not None
        assert result.alternatives[0].words[0].start_time is None
        assert result.alternatives[0].words[0].end_time is None
    finally:
        await model.aclose()
