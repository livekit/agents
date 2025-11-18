import pytest
from unittest.mock import AsyncMock
from events.transcript_handler import TranscriptHandler


class DummyEvent:
    def __init__(self, text, confidence):
        self.text = text
        self.transcript_confidence = confidence
        self.role = "user"


@pytest.mark.asyncio
async def test_event_pass_through():
    mock_sm = AsyncMock()
    handler = TranscriptHandler(mock_sm)

    event = DummyEvent("stop", 0.92)
    await handler.handle(event)

    mock_sm.handle_user_transcript.assert_called_once_with("stop", 0.92)
