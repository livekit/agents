import asyncio
import logging
from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError, APITimeoutError
from livekit.plugins import giggy
from livekit.plugins.giggy import tts as giggy_tts

pytestmark = pytest.mark.unit

KEY = "offline-test-key"
VOICE = "00000000-0000-0000-0000-000000000000"
PCM = b"\x01\x02" * 12000


@pytest.fixture(autouse=True)
def clear_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GIGGY_API_KEY", raising=False)
    monkeypatch.delenv("GIGGY_VOICE_ID", raising=False)


@pytest.mark.parametrize("kwargs", [{"voice": VOICE}, {"api_key": KEY}])
def test_missing_credentials(kwargs: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        giggy.TTS(**kwargs)


@pytest.mark.parametrize("speed", [0, 0.24, 4.01, float("nan"), float("inf")])
def test_invalid_speed(speed: float) -> None:
    with pytest.raises(ValueError, match="speed"):
        giggy.TTS(api_key=KEY, voice=VOICE, speed=speed)


@pytest.mark.parametrize("speed", [0.25, 1, 4])
def test_constructor(speed: float) -> None:
    provider = giggy.TTS(api_key=KEY, voice=VOICE, speed=speed)
    assert provider.sample_rate == 24000
    assert provider.num_channels == 1
    assert provider.capabilities.streaming is False
    assert provider.capabilities.aligned_transcript is False
    assert provider.provider == "Giggy"
    assert provider.model == "giggyspeech"


def test_environment_and_explicit_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIGGY_API_KEY", "environment-key")
    monkeypatch.setenv("GIGGY_VOICE_ID", "environment-voice")
    assert giggy.TTS()._api_key == "environment-key"
    provider = giggy.TTS(api_key=KEY, voice=VOICE)
    assert provider._api_key == KEY
    assert provider._voice == VOICE


@pytest.mark.parametrize("text", ["", " \n\t"])
def test_empty_text(text: str) -> None:
    with pytest.raises(ValueError, match="empty"):
        giggy.TTS(api_key=KEY, voice=VOICE).synthesize(text)


class Response:
    """Deterministic HTTP mock preserving individual transport chunk boundaries."""

    def __init__(
        self,
        chunks: list[bytes],
        *,
        status: int = 200,
        content_type: str = "application/octet-stream",
        error: Exception | None = None,
        hold: bool = False,
    ) -> None:
        self.status = status
        self.headers = {"Content-Type": content_type, "x-request-id": "offline-request"}
        self.content = self
        self.chunks = chunks
        self.error = error
        self.hold = hold
        self.started = asyncio.Event()
        self.closed = False

    async def __aenter__(self) -> "Response":
        return self

    async def __aexit__(self, *args: object) -> None:
        self.closed = True

    async def iter_chunked(self, size: int) -> AsyncIterator[bytes]:
        assert size == 4096
        self.started.set()
        for chunk in self.chunks:
            yield chunk
        if self.error:
            raise self.error
        if self.hold:
            await asyncio.Event().wait()


async def test_pcm_and_request() -> None:
    response = Response([PCM[:4095], b"", PCM[4095:8192], PCM[8192:]])
    session = MagicMock(spec=aiohttp.ClientSession)
    session.post.return_value = response
    provider = giggy.TTS(api_key=KEY, voice=VOICE, speed=1.25, http_session=session)
    options = APIConnectOptions(max_retry=5, timeout=7, retry_interval=0.01)
    async with provider.synthesize("Complete text.", conn_options=options) as stream:
        events = [event async for event in stream]
        assert stream._conn_options.max_retry == 0
        assert stream._conn_options.timeout == 7
        assert options.max_retry == 5
    assert events
    assert all(event.frame.sample_rate == 24000 for event in events)
    assert all(event.frame.num_channels == 1 for event in events)
    assert all(event.request_id == "offline-request" for event in events)
    assert b"".join(bytes(event.frame.data) for event in events) == PCM
    session.post.assert_called_once()
    args, kwargs = session.post.call_args
    assert args == ("https://giggy.ai/v1/audio/speech",)
    assert kwargs["json"] == {
        "model": "giggyspeech",
        "input": "Complete text.",
        "voice": VOICE,
        "response_format": "pcm",
        "sample_rate": 24000,
        "speed": 1.25,
    }
    assert kwargs["headers"] == {
        "Authorization": "Bearer " + KEY,
        "Content-Type": "application/json",
        "Accept": "application/octet-stream",
    }
    assert kwargs["allow_redirects"] is False
    assert kwargs["timeout"].total is None
    assert kwargs["timeout"].sock_connect == 7
    assert kwargs["timeout"].sock_read == 90
    assert response.closed
    await provider.aclose()
    session.close.assert_not_called()


@pytest.mark.parametrize("content_type", ["audio/pcm", "Application/Octet-Stream; charset=binary"])
async def test_accepted_content_type(content_type: str) -> None:
    session = MagicMock(spec=aiohttp.ClientSession)
    session.post.return_value = Response([PCM], content_type=content_type)
    async with giggy.TTS(api_key=KEY, voice=VOICE, http_session=session).synthesize("Hi") as stream:
        assert (await stream.collect()).sample_rate == 24000


@pytest.mark.parametrize("status", [301, 401, 402, 429, 503])
async def test_http_error_one_post(status: int, caplog: pytest.LogCaptureFixture) -> None:
    response = Response([], status=status)
    session = MagicMock(spec=aiohttp.ClientSession)
    session.post.return_value = response
    provider = giggy.TTS(api_key=KEY, voice=VOICE, http_session=session)
    errors = []
    provider.on("error", errors.append)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(APIStatusError) as caught:
            async with provider.synthesize(
                "Hi", conn_options=APIConnectOptions(max_retry=5, retry_interval=0)
            ) as stream:
                await stream.collect()
    assert caught.value.status_code == status
    assert caught.value.request_id == "offline-request"
    assert caught.value.body is None
    assert caught.value.retryable is False
    assert errors and all(not event.recoverable for event in errors)
    session.post.assert_called_once()
    assert response.closed
    assert KEY not in caplog.text + str(caught.value) + repr(caught.value)


@pytest.mark.parametrize(
    ("chunks", "content_type", "message"),
    [
        ([], "application/octet-stream", "no audio"),
        ([b"\x01\x02\x03"], "audio/pcm", "incomplete PCM"),
        ([PCM], "application/json", "unexpected audio format"),
        ([PCM], "", "unexpected audio format"),
    ],
)
async def test_invalid_audio(chunks: list[bytes], content_type: str, message: str) -> None:
    response = Response(chunks, content_type=content_type)
    session = MagicMock(spec=aiohttp.ClientSession)
    session.post.return_value = response
    with pytest.raises(APIConnectionError, match=message):
        async with giggy.TTS(api_key=KEY, voice=VOICE, http_session=session).synthesize(
            "Hi"
        ) as stream:
            await stream.collect()
    assert response.closed
    session.post.assert_called_once()


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("error_type", [asyncio.TimeoutError, aiohttp.ClientPayloadError])
async def test_transport_error_no_retry_or_secret(
    partial: bool, error_type: type[Exception], caplog: pytest.LogCaptureFixture
) -> None:
    response = Response([PCM] if partial else [], error=error_type(KEY))
    session = MagicMock(spec=aiohttp.ClientSession)
    session.post.return_value = response
    expected = APITimeoutError if error_type is asyncio.TimeoutError else APIConnectionError
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(expected) as caught:
            async with giggy.TTS(api_key=KEY, voice=VOICE, http_session=session).synthesize(
                "Hi", conn_options=APIConnectOptions(max_retry=5, retry_interval=0)
            ) as stream:
                await stream.collect()
    assert caught.value.__cause__ is None
    assert KEY not in caplog.text + str(caught.value) + repr(caught.value)
    assert response.closed
    session.post.assert_called_once()


async def test_cancel_closes_response() -> None:
    response = Response([PCM], hold=True)
    session = MagicMock(spec=aiohttp.ClientSession)
    session.post.return_value = response
    provider = giggy.TTS(api_key=KEY, voice=VOICE, http_session=session)
    stream = provider.synthesize("Hi")
    try:
        await asyncio.wait_for(response.started.wait(), 2)
        event = await asyncio.wait_for(anext(stream), 2)
        assert event.frame.samples_per_channel > 0  # emitted before response EOF
    finally:
        await stream.aclose()
    assert response.closed
    assert stream._synthesize_task.cancelled()
    session.post.assert_called_once()
    await provider.aclose()
    session.close.assert_not_called()


async def test_shared_session_ownership(monkeypatch: pytest.MonkeyPatch) -> None:
    session = MagicMock(spec=aiohttp.ClientSession)
    shared = MagicMock(return_value=session)
    monkeypatch.setattr(giggy_tts.utils.http_context, "http_session", shared)
    provider = giggy.TTS(api_key=KEY, voice=VOICE)
    assert provider._ensure_session() is session
    await provider.aclose()
    session.close.assert_not_called()


async def test_real_http_error_is_single_post(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise actual aiohttp transport against an offline loopback server."""
    requests: list[dict[str, object]] = []

    async def handler(request: web.Request) -> web.Response:
        requests.append({"headers": dict(request.headers), "body": await request.json()})
        return web.Response(status=503, text=KEY)

    app = web.Application()
    app.router.add_post("/v1/audio/speech", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        monkeypatch.setattr(giggy_tts, "GIGGY_URL", f"http://127.0.0.1:{port}/v1/audio/speech")
        async with aiohttp.ClientSession() as session:
            provider = giggy.TTS(api_key=KEY, voice=VOICE, http_session=session)
            with pytest.raises(APIStatusError) as caught:
                async with provider.synthesize(
                    "Hi", conn_options=APIConnectOptions(max_retry=5, retry_interval=0)
                ) as stream:
                    await stream.collect()
            assert KEY not in str(caught.value)
            await provider.aclose()
            assert not session.closed
        assert len(requests) == 1
        assert requests[0]["headers"]["Authorization"] == "Bearer " + KEY
        assert "Idempotency-Key" not in requests[0]["headers"]
    finally:
        await runner.cleanup()
