from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectionError, APIStatusError
from livekit.agents.tts import AudioEmitter
from livekit.plugins import baseten

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_baseten_chunked_stream_preserves_api_status_error():
    tts = baseten.TTS(api_key="fake-key", model_endpoint="https://fake.baseten.co")
    stream = baseten.tts.ChunkedStream(
        tts=tts, input_text="hello world", conn_options=DEFAULT_API_CONNECT_OPTIONS
    )

    mock_resp = MagicMock()
    # Simulate an error that causes a direct APIStatusError
    mock_resp.raise_for_status.side_effect = APIStatusError(
        "Unauthorized", status_code=401, request_id=None, body=None
    )

    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_post_ctx.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post.return_value = mock_post_ctx

    emitter = MagicMock(spec=AudioEmitter)

    try:
        with patch.object(tts, "_ensure_session", return_value=mock_session):
            with pytest.raises(APIStatusError) as exc_info:
                await stream._run(emitter)
            assert not isinstance(exc_info.value, APIConnectionError)
            assert exc_info.value.status_code == 401
    finally:
        await stream.aclose()
