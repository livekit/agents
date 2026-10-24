from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectionError, APIError
from livekit.agents.tts import AudioEmitter
from livekit.plugins import resemble

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_resemble_chunked_stream_preserves_api_error():
    tts = resemble.TTS(api_key="fake-key", voice_uuid="fake-voice")
    stream = resemble.tts.ChunkedStream(
        tts=tts, input_text="hello world", conn_options=DEFAULT_API_CONNECT_OPTIONS
    )

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = AsyncMock(return_value={"success": False, "issues": ["Invalid voice UUID"]})

    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_post_ctx.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post.return_value = mock_post_ctx

    emitter = MagicMock(spec=AudioEmitter)

    try:
        with patch.object(tts, "_ensure_session", return_value=mock_session):
            with pytest.raises(APIError) as exc_info:
                await stream._run(emitter)
            assert not isinstance(exc_info.value, APIConnectionError)
            assert "Invalid voice UUID" in str(exc_info.value)
    finally:
        await stream.aclose()
