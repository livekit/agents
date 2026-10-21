from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit import rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIStatusError
from livekit.plugins import clova

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_clova_stt_recognize_raises_api_status_error_on_service_error():
    stt_instance = clova.STT(secret="fake-secret")

    # Mock audio buffer with 16kHz audio frame
    buffer = rtc.AudioFrame(
        data=b"\x00" * 3200, sample_rate=16000, num_channels=1, samples_per_channel=1600
    )

    mock_resp = MagicMock()
    mock_resp.status = 400
    mock_resp.json = AsyncMock(return_value={"error": "Invalid API key"})

    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_post_ctx.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post.return_value = mock_post_ctx

    with patch.object(stt_instance, "_ensure_session", return_value=mock_session):
        # On unpatched code, raises unhandled ValueError instead of APIStatusError
        with pytest.raises(APIStatusError) as exc_info:
            await stt_instance._recognize_impl(buffer, conn_options=DEFAULT_API_CONNECT_OPTIONS)
        assert exc_info.value.status_code == 400
        assert "Invalid API key" in exc_info.value.message
