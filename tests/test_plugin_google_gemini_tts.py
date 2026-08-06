from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit.agents import tts
from livekit.plugins.google.beta.gemini_tts import TTS

pytestmark = pytest.mark.plugin("google")


@pytest.mark.asyncio
@patch("livekit.plugins.google.beta.gemini_tts.Client")
async def test_gemini_tts_success(mock_genai_client_class) -> None:
    # Setup mocks for GenAI Client
    mock_client = MagicMock()
    mock_genai_client_class.return_value = mock_client

    mock_stream = AsyncMock()
    mock_client.aio.models.generate_content_stream = mock_stream

    # Mock chunk response candidates
    class MockInlineData:
        def __init__(self, data: bytes):
            self.data = data
            self.mime_type = "audio/pcm"

    class MockPart:
        def __init__(self, data: bytes):
            self.inline_data = MockInlineData(data)

    class MockContent:
        def __init__(self, data: bytes):
            self.parts = [MockPart(data)]

    class MockCandidate:
        def __init__(self, data: bytes):
            self.content = MockContent(data)

    class MockChunk:
        def __init__(self, data: bytes):
            self.candidates = [MockCandidate(data)]

    async def mock_generator(*args, **kwargs):
        yield MockChunk(b"\x00" * 4800)
        yield MockChunk(b"\x01" * 4800)

    mock_stream.side_effect = mock_generator

    # Initialize TTS
    google_tts = TTS(api_key="test-api-key")

    # Create output emitter mock
    mock_emitter = MagicMock(spec=tts.AudioEmitter)

    # Run ChunkedStream
    stream = google_tts.synthesize("Hello world")
    await stream._run(mock_emitter)

    # Assertions
    mock_stream.assert_called_once()
    mock_emitter.initialize.assert_called_once()
    assert mock_emitter.push.call_count == 2
