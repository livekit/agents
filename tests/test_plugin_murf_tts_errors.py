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
from livekit.plugins import murf

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_murf_chunked_stream_preserves_api_status_error():
    tts = murf.TTS(api_key="fake-key")
    stream = murf.tts.ChunkedStream(
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
async def test_murf_chunked_stream_preserves_api_error():
    tts = murf.TTS(api_key="fake-key")
    stream = murf.tts.ChunkedStream(
        tts=tts, input_text="hello world", conn_options=DEFAULT_API_CONNECT_OPTIONS
    )

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = APIError("Murf custom API error", retryable=False)

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
            assert "Murf custom API error" in str(exc_info.value)
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_murf_chunked_stream_preserves_api_timeout_error():
    tts = murf.TTS(api_key="fake-key")
    stream = murf.tts.ChunkedStream(
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
async def test_murf_chunked_stream_handles_asyncio_timeout():
    tts = murf.TTS(api_key="fake-key")
    stream = murf.tts.ChunkedStream(
        tts=tts, input_text="hello world", conn_options=DEFAULT_API_CONNECT_OPTIONS
    )

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = TimeoutError()

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
async def test_murf_chunked_stream_maps_client_response_error():
    tts = murf.TTS(api_key="fake-key")
    stream = murf.tts.ChunkedStream(
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


@pytest.mark.asyncio
async def test_murf_synthesize_stream_preserves_api_status_error():
    tts = murf.TTS(api_key="fake-key")
    stream = murf.tts.SynthesizeStream(tts=tts, conn_options=DEFAULT_API_CONNECT_OPTIONS)

    mock_pool_ctx = MagicMock()
    mock_pool_ctx.__aenter__ = AsyncMock(
        side_effect=APIStatusError("Unauthorized", status_code=401, request_id=None, body=None)
    )
    mock_pool_ctx.__aexit__ = AsyncMock(return_value=None)

    emitter = MagicMock(spec=AudioEmitter)

    try:
        with patch.object(tts._pool, "connection", return_value=mock_pool_ctx):
            with pytest.raises(APIStatusError) as exc_info:
                await stream._run(emitter)
            assert not isinstance(exc_info.value, APIConnectionError)
            assert exc_info.value.status_code == 401
            assert exc_info.value.retryable is False
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_murf_synthesize_stream_preserves_api_error():
    tts = murf.TTS(api_key="fake-key")
    stream = murf.tts.SynthesizeStream(tts=tts, conn_options=DEFAULT_API_CONNECT_OPTIONS)

    mock_pool_ctx = MagicMock()
    mock_pool_ctx.__aenter__ = AsyncMock(
        side_effect=APIError("Murf custom stream error", retryable=False)
    )
    mock_pool_ctx.__aexit__ = AsyncMock(return_value=None)

    emitter = MagicMock(spec=AudioEmitter)

    try:
        with patch.object(tts._pool, "connection", return_value=mock_pool_ctx):
            with pytest.raises(APIError) as exc_info:
                await stream._run(emitter)
            assert not isinstance(exc_info.value, APIConnectionError)
            assert exc_info.value.retryable is False
            assert "Murf custom stream error" in str(exc_info.value)
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_murf_synthesize_stream_preserves_unexpected_ws_closure_status_error():
    tts = murf.TTS(api_key="fake-key")
    stream = murf.tts.SynthesizeStream(tts=tts, conn_options=DEFAULT_API_CONNECT_OPTIONS)

    mock_ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
    mock_ws.send_str = AsyncMock()
    mock_close_msg = MagicMock()
    mock_close_msg.type = aiohttp.WSMsgType.CLOSED
    mock_ws.receive = AsyncMock(return_value=mock_close_msg)

    mock_pool_ctx = MagicMock()
    mock_pool_ctx.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_pool_ctx.__aexit__ = AsyncMock(return_value=None)

    emitter = MagicMock(spec=AudioEmitter)

    try:
        with patch.object(tts._pool, "connection", return_value=mock_pool_ctx):
            stream.push_text("hello")
            stream.end_input()
            with pytest.raises(APIStatusError) as exc_info:
                await stream._run(emitter)
            assert not isinstance(exc_info.value, APIConnectionError)
            assert "Murf AI connection closed unexpectedly" in str(exc_info.value)
    finally:
        await stream.aclose()
