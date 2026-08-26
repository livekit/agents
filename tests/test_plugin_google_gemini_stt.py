from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit import rtc
from livekit.agents import stt
from livekit.plugins.google.beta.gemini_stt import STT

pytestmark = pytest.mark.plugin("google")


@pytest.mark.asyncio
@patch("livekit.plugins.google.beta.gemini_stt.Client")
async def test_gemini_stt_stream(mock_genai_client_class) -> None:
    mock_client = MagicMock()
    mock_genai_client_class.return_value = mock_client

    mock_session = AsyncMock()
    mock_client.aio.live.connect.return_value.__aenter__.return_value = mock_session

    class MockInputTranscription:
        def __init__(self, text: str):
            self.text = text

    class MockServerContent:
        def __init__(self, text: str, is_final: bool):
            if is_final:
                self.input_transcription = MockInputTranscription(text)
                self.interim_input_transcription = None
            else:
                self.input_transcription = None
                self.interim_input_transcription = MockInputTranscription(text)

    class MockMessage:
        def __init__(self, text: str, is_final: bool):
            self.server_content = MockServerContent(text, is_final)

    async def mock_receive():
        yield MockMessage("hello", is_final=False)
        yield MockMessage("hello world", is_final=True)

    mock_session.receive = MagicMock(side_effect=mock_receive)

    google_stt = STT(api_key="test-key")
    stream = google_stt.stream()

    frame = rtc.AudioFrame(
        data=b"\x00" * 320,
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=160,
    )
    stream.push_frame(frame)
    stream.end_input()

    events = []
    async for event in stream:
        events.append(event)

    assert len(events) >= 2
    assert events[0].type == stt.SpeechEventType.INTERIM_TRANSCRIPT
    assert events[0].alternatives[0].text == "hello"
    assert events[1].type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert events[1].alternatives[0].text == "hello world"
