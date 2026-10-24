from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
)
from livekit.agents.tts import AudioEmitter
from livekit.plugins import azure

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_azure_chunked_stream_preserves_api_status_error():
    tts = azure.TTS(speech_key="fake-key", speech_region="eastus")
    stream = azure.tts.ChunkedStream(
        tts=tts, input_text="hello world", conn_options=DEFAULT_API_CONNECT_OPTIONS
    )

    mock_resp = MagicMock()
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
            assert exc_info.value.retryable is False
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_azure_chunked_stream_preserves_api_error():
    tts = azure.TTS(speech_key="fake-key", speech_region="eastus")
    stream = azure.tts.ChunkedStream(
        tts=tts, input_text="hello world", conn_options=DEFAULT_API_CONNECT_OPTIONS
    )

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = APIError("Azure custom API error", retryable=False)

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
            assert exc_info.value.retryable is False
            assert "Azure custom API error" in str(exc_info.value)
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_azure_chunked_stream_preserves_api_timeout_error():
    tts = azure.TTS(speech_key="fake-key", speech_region="eastus")
    stream = azure.tts.ChunkedStream(
        tts=tts, input_text="hello world", conn_options=DEFAULT_API_CONNECT_OPTIONS
    )

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = APITimeoutError()

    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_post_ctx.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post.return_value = mock_post_ctx

    emitter = MagicMock(spec=AudioEmitter)

    try:
        with patch.object(tts, "_ensure_session", return_value=mock_session):
            with pytest.raises(APITimeoutError) as exc_info:
                await stream._run(emitter)
            assert type(exc_info.value) is APITimeoutError
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_azure_chunked_stream_maps_client_response_error():
    tts = azure.TTS(speech_key="fake-key", speech_region="eastus")
    stream = azure.tts.ChunkedStream(
        tts=tts, input_text="hello world", conn_options=DEFAULT_API_CONNECT_OPTIONS
    )

    req_info = MagicMock(spec=aiohttp.RequestInfo)
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = aiohttp.ClientResponseError(
        request_info=req_info,
        history=(),
        status=403,
        message="Forbidden",
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
            assert exc_info.value.status_code == 403
            assert exc_info.value.retryable is False
    finally:
        await stream.aclose()
