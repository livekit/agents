from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIError,
    APIStatusError,
)
from livekit.agents.tts import AudioEmitter
from livekit.plugins import groq

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_groq_chunked_stream_preserves_api_error_on_non_audio_response():
    tts = groq.TTS(api_key="fake-key")
    stream = groq.tts.ChunkedStream(
        tts=tts, input_text="hello world", conn_options=DEFAULT_API_CONNECT_OPTIONS
    )

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.content_type = "application/json"
    mock_resp.text = AsyncMock(return_value='{"error": "model_not_found"}')

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
            assert "Groq returned non-audio data" in str(exc_info.value)
            assert exc_info.value.body == '{"error": "model_not_found"}'
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_groq_chunked_stream_preserves_direct_api_status_error():
    tts = groq.TTS(api_key="fake-key")
    stream = groq.tts.ChunkedStream(
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
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_groq_chunked_stream_preserves_direct_api_error():
    tts = groq.TTS(api_key="fake-key")
    stream = groq.tts.ChunkedStream(
        tts=tts, input_text="hello world", conn_options=DEFAULT_API_CONNECT_OPTIONS
    )

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = APIError("Groq custom upstream error")

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
            assert "Groq custom upstream error" in str(exc_info.value)
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_groq_chunked_stream_client_response_error_mapped_to_api_status_error():
    tts = groq.TTS(api_key="fake-key")
    stream = groq.tts.ChunkedStream(
        tts=tts, input_text="hello world", conn_options=DEFAULT_API_CONNECT_OPTIONS
    )

    req_info = MagicMock(spec=aiohttp.RequestInfo)
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = aiohttp.ClientResponseError(
        request_info=req_info,
        history=(),
        status=429,
        message="Too Many Requests",
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
            assert exc_info.value.status_code == 429
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_groq_chunked_stream_generic_exception_becomes_api_connection_error():
    tts = groq.TTS(api_key="fake-key")
    stream = groq.tts.ChunkedStream(
        tts=tts, input_text="hello world", conn_options=DEFAULT_API_CONNECT_OPTIONS
    )

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = RuntimeError("Socket failed unexpectedly")

    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_post_ctx.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post.return_value = mock_post_ctx

    emitter = MagicMock(spec=AudioEmitter)

    try:
        with patch.object(tts, "_ensure_session", return_value=mock_session):
            with pytest.raises(APIConnectionError) as exc_info:
                await stream._run(emitter)
            assert isinstance(exc_info.value, APIConnectionError)
    finally:
        await stream.aclose()
