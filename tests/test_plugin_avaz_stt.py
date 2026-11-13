from __future__ import annotations

import asyncio
import io
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import numpy as np
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
)
from livekit.agents.stt import SpeechEventType
from livekit.plugins import avaz
from livekit.plugins.avaz import stt as avaz_stt

pytestmark = pytest.mark.unit

API_KEY = "test-api-key"
CONN_OPTIONS = APIConnectOptions(max_retry=2, retry_interval=0.01, timeout=2.0)


def _transcript(*texts: str, segments: list[tuple[float, float]] | None = None) -> dict[str, Any]:
    """An Avaz `/stt/transcribe` response: one text per channel, timed VAD segments."""
    segments = segments if segments is not None else [(0.1, 0.9)] * bool(texts)
    return {
        "success": True,
        "status_code": 200,
        "data": {
            "transcribed": True,
            "error": None,
            "language_used": "tr",
            "text": list(texts),
            "segments": [
                {"start_time": start, "end_time": end, "text": " ".join(texts), "channel": 0}
                for start, end in segments
            ],
            "total_segments": len(segments),
        },
    }


@dataclass
class Upload:
    headers: dict[str, str]
    filename: str | None
    content_type: str | None
    channels: int
    sample_width: int
    sample_rate: int
    num_frames: int


@dataclass
class FakeAvaz:
    """Mimics the Avaz `/stt/transcribe` endpoint."""

    responses: list[web.Response] = field(default_factory=list)
    hang: bool = False
    redirect_to: str | None = None
    uploads: list[Upload] = field(default_factory=list)

    async def handler(self, request: web.Request) -> web.StreamResponse:
        if self.redirect_to is not None:
            raise web.HTTPTemporaryRedirect(self.redirect_to)

        part = (await request.post())["file"]
        assert isinstance(part, web.FileField)
        with wave.open(io.BytesIO(part.file.read()), "rb") as wav:
            self.uploads.append(
                Upload(
                    headers=dict(request.headers),
                    filename=part.filename,
                    content_type=part.content_type,
                    channels=wav.getnchannels(),
                    sample_width=wav.getsampwidth(),
                    sample_rate=wav.getframerate(),
                    num_frames=wav.getnframes(),
                )
            )

        if self.hang:
            await asyncio.sleep(10)
        if self.responses:
            return self.responses.pop(0)
        return web.json_response(_transcript("merhaba size nasıl yardımcı olabilirim"))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AVAZ_API_KEY", raising=False)
    monkeypatch.delenv("AVAZ_BASE_URL", raising=False)


async def _start_server(fake: FakeAvaz) -> TestServer:
    app = web.Application()
    app.router.add_post("/api/stt/transcribe", fake.handler)
    test_server = TestServer(app)
    await test_server.start_server()
    return test_server


@pytest.fixture
async def server() -> AsyncIterator[tuple[FakeAvaz, str]]:
    fake = FakeAvaz()
    test_server = await _start_server(fake)
    try:
        yield fake, str(test_server.make_url("/api"))
    finally:
        await test_server.close()


@pytest.fixture
async def make_stt(server: tuple[FakeAvaz, str]) -> AsyncIterator[Any]:
    _, base_url = server
    async with aiohttp.ClientSession() as session:

        def _make(**kwargs: Any) -> avaz.STT:
            kwargs.setdefault("base_url", base_url)
            return avaz.STT(api_key=API_KEY, http_session=session, **kwargs)

        yield _make


def _speech(seconds: float, *, sample_rate: int = 48000, num_channels: int = 1) -> rtc.AudioFrame:
    """A 440 Hz tone, cut into 10 ms frames like microphone input."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    mono = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
    samples = np.repeat(mono[:, None], num_channels, axis=1)
    return rtc.AudioFrame(samples.tobytes(), sample_rate, num_channels, len(mono))


def _frames(frame: rtc.AudioFrame) -> list[rtc.AudioFrame]:
    step = frame.sample_rate // 100
    data = np.frombuffer(frame.data, dtype=np.int16).reshape(-1, frame.num_channels)
    return [
        rtc.AudioFrame(chunk.tobytes(), frame.sample_rate, frame.num_channels, len(chunk))
        for chunk in (data[i : i + step] for i in range(0, len(data), step))
    ]


def test_requires_api_key_for_hosted_api() -> None:
    with pytest.raises(ValueError, match="AVAZ_API_KEY"):
        avaz.STT()


def test_reads_api_key_and_base_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AVAZ_API_KEY", API_KEY)
    monkeypatch.setenv("AVAZ_BASE_URL", "https://avaz.example.com/api/")
    stt = avaz.STT()
    assert stt._opts.api_key == API_KEY
    assert stt._opts.url == "https://avaz.example.com/api/stt/transcribe"


def test_api_key_is_only_sent_over_tls_or_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AVAZ_API_KEY", API_KEY)
    monkeypatch.setenv("AVAZ_BASE_URL", "http://10.0.0.5:8080")
    with pytest.raises(ValueError, match="https"):
        avaz.STT()
    with pytest.raises(ValueError, match="https"):
        avaz.STT(base_url="ws://10.0.0.5:8080", api_key=API_KEY)

    stt = avaz.STT(api_key="")
    assert stt._opts.api_key is None
    assert stt._opts.url == "http://10.0.0.5:8080/stt/transcribe"

    # the same base_url as avaz.TTS works, a wss:// URL becomes https://
    assert avaz.STT(base_url="wss://avaz.example.com/api", api_key=API_KEY)._opts.url == (
        "https://avaz.example.com/api/stt/transcribe"
    )
    assert avaz.STT(base_url="http://127.0.0.1:8080", api_key=API_KEY)._opts.api_key == API_KEY
    assert avaz.STT(base_url="ws://localhost:8080", api_key=API_KEY)._opts.api_key == API_KEY


def test_env_api_key_is_not_sent_to_other_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AVAZ_API_KEY", API_KEY)
    assert avaz.STT(base_url="https://avaz.example.com/api")._opts.api_key is None
    assert avaz.STT(base_url="http://10.0.0.5:8080")._opts.api_key is None
    assert avaz.STT(base_url="https://dashboard.mankyazilim.com.tr/api/")._opts.api_key == API_KEY


@pytest.mark.parametrize("base_url", ["10.0.0.5:8080", "ftp://avaz.example.com", "https://"])
def test_rejects_invalid_base_url(base_url: str) -> None:
    with pytest.raises(ValueError, match="base_url"):
        avaz.STT(base_url=base_url, api_key=API_KEY)


def test_options() -> None:
    stt = avaz.STT(api_key=API_KEY)
    assert (stt.model, stt.provider) == ("avaz-stt-v1", "Avaz")
    assert not stt.capabilities.streaming
    assert not stt.capabilities.interim_results


async def test_recognize(server: tuple[FakeAvaz, str], make_stt: Any) -> None:
    fake, _ = server
    fake.responses.append(
        web.json_response(_transcript(" merhaba ", "", segments=[(0.2, 1.1), (1.3, 2.4)]))
    )

    event = await make_stt().recognize(_frames(_speech(2.5)), conn_options=CONN_OPTIONS)

    assert event.type == SpeechEventType.FINAL_TRANSCRIPT
    [alt] = event.alternatives
    assert (alt.text, alt.language, alt.start_time, alt.end_time) == ("merhaba", "tr", 0.2, 2.4)

    [upload] = fake.uploads
    assert upload.headers["Authorization"] == f"Bearer {API_KEY}"
    assert upload.headers["User-Agent"].startswith("livekit-plugins-avaz/")
    assert (upload.filename, upload.content_type) == ("audio.wav", "audio/wav")


@pytest.mark.parametrize(("sample_rate", "num_channels"), [(48000, 2), (24000, 1), (16000, 1)])
async def test_uploads_16khz_mono(
    server: tuple[FakeAvaz, str], make_stt: Any, sample_rate: int, num_channels: int
) -> None:
    fake, _ = server
    frames = _frames(_speech(1.0, sample_rate=sample_rate, num_channels=num_channels))

    await make_stt().recognize(frames, conn_options=CONN_OPTIONS)

    [upload] = fake.uploads
    assert (upload.channels, upload.sample_width, upload.sample_rate) == (1, 2, 16000)
    assert upload.num_frames == pytest.approx(16000, abs=160)


async def test_silence_gives_an_empty_transcript(
    server: tuple[FakeAvaz, str], make_stt: Any
) -> None:
    fake, _ = server
    fake.responses.append(web.json_response(_transcript()))

    event = await make_stt().recognize([_speech(0.5)], conn_options=CONN_OPTIONS)

    [alt] = event.alternatives
    assert (alt.text, alt.start_time, alt.end_time) == ("", 0.0, 0.0)


async def test_rejects_other_languages(server: tuple[FakeAvaz, str], make_stt: Any) -> None:
    fake, _ = server
    stt = make_stt()

    with pytest.raises(ValueError, match="Turkish"):
        await stt.recognize([_speech(0.5)], language="en", conn_options=CONN_OPTIONS)
    assert fake.uploads == []

    event = await stt.recognize([_speech(0.5)], language="tr-TR", conn_options=CONN_OPTIONS)
    assert event.alternatives[0].text


async def test_rejected_request_is_not_retried(server: tuple[FakeAvaz, str], make_stt: Any) -> None:
    fake, _ = server
    fake.responses.append(
        web.json_response({"success": False, "message": "Unauthorized"}, status=401)
    )

    with pytest.raises(APIStatusError) as exc_info:
        await make_stt().recognize([_speech(0.5)], conn_options=CONN_OPTIONS)

    assert (exc_info.value.status_code, exc_info.value.body) == (401, None)
    assert "Unauthorized" not in str(exc_info.value)
    assert len(fake.uploads) == 1


async def test_server_error_is_retried(server: tuple[FakeAvaz, str], make_stt: Any) -> None:
    fake, _ = server
    fake.responses.append(web.json_response({"success": False, "message": "boom"}, status=500))

    event = await make_stt().recognize([_speech(0.5)], conn_options=CONN_OPTIONS)

    assert event.alternatives[0].text == "merhaba size nasıl yardımcı olabilirim"
    assert len(fake.uploads) == 2


async def test_persistent_server_error_fails(server: tuple[FakeAvaz, str], make_stt: Any) -> None:
    fake, _ = server
    fake.responses.extend(web.Response(status=502, text="bad gateway") for _ in range(3))

    with pytest.raises(APIConnectionError, match="3 attempts"):
        await make_stt().recognize([_speech(0.5)], conn_options=CONN_OPTIONS)
    assert len(fake.uploads) == 3


@pytest.mark.parametrize(
    "response",
    [
        web.Response(text="<html>not json</html>"),
        web.json_response({"success": False, "data": None}),
        web.json_response({"success": True, "data": {"text": "not a list"}}),
        web.json_response(_transcript("merhaba", segments=[(0.1, None)])),  # type: ignore[list-item]
    ],
)
async def test_unexpected_response_fails(
    server: tuple[FakeAvaz, str], make_stt: Any, response: web.Response
) -> None:
    fake, _ = server
    fake.responses.append(response)

    with pytest.raises(APIError):
        await make_stt().recognize(
            [_speech(0.5)], conn_options=APIConnectOptions(max_retry=0, timeout=2.0)
        )


async def test_slow_response_times_out(
    server: tuple[FakeAvaz, str], make_stt: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, _ = server
    fake.hang = True
    monkeypatch.setattr(avaz_stt, "REQUEST_TIMEOUT", 0.2)

    with pytest.raises(APITimeoutError):
        await make_stt().recognize(
            [_speech(0.5)], conn_options=APIConnectOptions(max_retry=0, timeout=2.0)
        )


async def test_unreachable_server_fails(make_stt: Any) -> None:
    with pytest.raises(APIConnectionError):
        await make_stt(base_url="http://127.0.0.1:9").recognize(
            [_speech(0.5)], conn_options=APIConnectOptions(max_retry=0, timeout=2.0)
        )


async def test_api_key_is_not_sent_to_a_redirected_origin(make_stt: Any) -> None:
    target = FakeAvaz()
    target_server = await _start_server(target)
    redirect = FakeAvaz(redirect_to=str(target_server.make_url("/api/stt/transcribe")))
    redirect_server = await _start_server(redirect)
    try:
        stt = make_stt(base_url=str(redirect_server.make_url("/api")))
        await stt.recognize([_speech(0.5)], conn_options=CONN_OPTIONS)
    finally:
        await redirect_server.close()
        await target_server.close()

    [upload] = target.uploads
    assert "Authorization" not in upload.headers
