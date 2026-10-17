from __future__ import annotations

import asyncio
import base64
import io
import json
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import APIConnectOptions, APIError, APIStatusError, APITimeoutError
from livekit.plugins import avaz

pytestmark = pytest.mark.unit

API_KEY = "test-api-key"
CONN_OPTIONS = APIConnectOptions(max_retry=2, retry_interval=0.01, timeout=2.0)
SERVER_ERROR = "provider error text"


def _wav_b64(text: str, sample_rate: int, *, truncate: bool = False) -> str:
    """Silence of 10 ms per character, so tests can check which text was spoken."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * (sample_rate // 100) * len(text))
    data = buf.getvalue()
    return base64.b64encode(data[:-10] if truncate else data).decode()


@dataclass
class FakeAvaz:
    """Mimics the Avaz `/tts/stream-input` WebSocket."""

    sample_rate: int = 48000
    init_error: str | None = None
    reject_status: int | None = None
    drop_connection_attempts: int = 0
    skip_lines: set[int] = field(default_factory=set)
    error_lines: set[int] = field(default_factory=set)
    truncate_audio: bool = False
    audio_override: str | None = None
    hang_after_flush: bool = False
    connections: list[list[dict[str, Any]]] = field(default_factory=list)
    headers: list[dict[str, str]] = field(default_factory=list)

    async def handler(self, request: web.Request) -> web.StreamResponse:
        self.headers.append(dict(request.headers))
        if self.reject_status is not None:
            return web.json_response({"error": "unauthorized"}, status=self.reject_status)

        ws = web.WebSocketResponse()
        await ws.prepare(request)
        received: list[dict[str, Any]] = []
        self.connections.append(received)
        buffer, line_idx = "", 0
        async for msg in ws:
            data = json.loads(msg.data)
            received.append(data)
            if "model_settings" in data:
                if self.init_error:
                    await ws.send_json({"error": self.init_error})
                else:
                    await ws.send_json({"status": "initialized"})
                continue

            if len(self.connections) <= self.drop_connection_attempts:
                break

            if data.get("flush"):
                if self.hang_after_flush:
                    # keep pinging so only the client's own timeout can end the wait
                    pinger = asyncio.create_task(self._ping_forever(ws))
                    await ws.receive()  # until the client gives up
                    pinger.cancel()
                    await asyncio.gather(pinger, return_exceptions=True)
                    break
                await ws.send_json({"status": "closed", "chunks_generated": line_idx})
                break

            buffer += data.get("text", "")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if line_idx in self.error_lines:
                    await ws.send_json({"error": SERVER_ERROR})
                elif line_idx not in self.skip_lines:
                    audio = self.audio_override or _wav_b64(
                        line, self.sample_rate, truncate=self.truncate_audio
                    )
                    await ws.send_json({"audio": audio, "chunk_index": line_idx})
                line_idx += 1

        await ws.close()
        return ws

    @staticmethod
    async def _ping_forever(ws: web.WebSocketResponse) -> None:
        while True:
            await ws.ping()
            await asyncio.sleep(0.05)

    @property
    def texts(self) -> list[str]:
        return [m["text"] for m in self.connections[-1] if "text" in m]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AVAZ_API_KEY", raising=False)
    monkeypatch.delenv("AVAZ_BASE_URL", raising=False)


@pytest.fixture
async def server() -> AsyncIterator[tuple[FakeAvaz, str]]:
    fake = FakeAvaz()
    app = web.Application()
    app.router.add_get("/api/tts/stream-input", fake.handler)
    test_server = TestServer(app)
    await test_server.start_server()
    try:
        yield fake, str(test_server.make_url("/api"))
    finally:
        await test_server.close()


@pytest.fixture
async def make_tts(server: tuple[FakeAvaz, str]) -> AsyncIterator[Any]:
    _, base_url = server
    instances: list[avaz.TTS] = []
    async with aiohttp.ClientSession() as session:

        def _make(**kwargs: Any) -> avaz.TTS:
            tts = avaz.TTS(base_url=base_url, api_key=API_KEY, http_session=session, **kwargs)
            instances.append(tts)
            return tts

        yield _make
        for tts in instances:
            await tts.aclose()


async def _synthesize_stream(
    tts: avaz.TTS, chunks: list[str], conn_options: APIConnectOptions = CONN_OPTIONS
) -> list[rtc.AudioFrame]:
    async with tts.stream(conn_options=conn_options) as stream:
        for chunk in chunks:
            stream.push_text(chunk)
        stream.end_input()
        return [ev.frame async for ev in stream]


def _duration(frames: list[rtc.AudioFrame]) -> float:
    return sum(frame.duration for frame in frames)


def _approx_duration(seconds: float) -> Any:
    # the framework can end a segment with a 10 ms silent marker frame
    return pytest.approx(seconds, abs=0.0101)


def test_requires_api_key_for_hosted_api() -> None:
    with pytest.raises(ValueError, match="AVAZ_API_KEY"):
        avaz.TTS()


def test_reads_api_key_and_base_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AVAZ_API_KEY", API_KEY)
    monkeypatch.setenv("AVAZ_BASE_URL", "https://avaz.example.com/api/")
    tts = avaz.TTS()
    assert tts._opts.api_key == API_KEY
    assert tts._opts.ws_url == "wss://avaz.example.com/api/tts/stream-input"


def test_api_key_is_only_sent_over_tls_or_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AVAZ_API_KEY", API_KEY)
    monkeypatch.setenv("AVAZ_BASE_URL", "http://10.0.0.5:8080")
    with pytest.raises(ValueError, match="https"):
        avaz.TTS()
    with pytest.raises(ValueError, match="https"):
        avaz.TTS(base_url="http://10.0.0.5:8080", api_key=API_KEY)

    tts = avaz.TTS(api_key="")
    assert tts._opts.api_key is None
    assert tts._opts.ws_url == "ws://10.0.0.5:8080/tts/stream-input"

    assert avaz.TTS(base_url="http://127.0.0.1:8080", api_key=API_KEY)._opts.api_key == API_KEY
    assert avaz.TTS(base_url="ws://localhost:8080", api_key=API_KEY)._opts.api_key == API_KEY


def test_env_api_key_is_not_sent_to_other_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AVAZ_API_KEY", API_KEY)
    assert avaz.TTS(base_url="https://avaz.example.com/api")._opts.api_key is None
    assert avaz.TTS(base_url="http://10.0.0.5:8080")._opts.api_key is None
    assert avaz.TTS(base_url="https://dashboard.mankyazilim.com.tr/api/")._opts.api_key == API_KEY


@pytest.mark.parametrize("base_url", ["10.0.0.5:8080", "ftp://avaz.example.com", "https://"])
def test_rejects_invalid_base_url(base_url: str) -> None:
    with pytest.raises(ValueError, match="base_url"):
        avaz.TTS(base_url=base_url, api_key=API_KEY)


def test_options() -> None:
    tts = avaz.TTS(api_key=API_KEY)
    assert (tts.model, tts.provider, tts.sample_rate, tts.num_channels) == (
        "avaz3",
        "Avaz",
        48000,
        1,
    )
    assert avaz.TTS(model="avaz2", api_key=API_KEY).sample_rate == 24000

    tts.update_options(speaker_id=3)
    assert tts._opts.speaker_id == 3


@pytest.mark.parametrize("speaker_id", [-1, True, "5"])
def test_rejects_invalid_speaker_id(speaker_id: Any) -> None:
    with pytest.raises(ValueError, match="speaker_id"):
        avaz.TTS(api_key=API_KEY, speaker_id=speaker_id)

    tts = avaz.TTS(api_key=API_KEY, speaker_id=1)
    with pytest.raises(ValueError, match="speaker_id"):
        tts.update_options(speaker_id=speaker_id)
    assert tts._opts.speaker_id == 1


def test_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="model"):
        avaz.TTS(model="avaz9", api_key=API_KEY)  # type: ignore[arg-type]


async def test_stream_sends_one_sentence_per_line(
    server: tuple[FakeAvaz, str], make_tts: Any
) -> None:
    fake, _ = server
    tts = make_tts(speaker_id=5)
    reply = "Merhaba, size nasıl yardımcı olabilirim? Toplam borcunuz 1.500 TL\ngörünüyor."
    frames = await _synthesize_stream(tts, [word + " " for word in reply.split(" ")])

    init, *texts, flush = fake.connections[0]
    assert init == {
        "model_settings": {"model_id": "avaz3", "speaker_id": 5},
        "voice_settings": {"chunk_notation": "\n"},
    }
    assert flush == {"flush": True, "force_flush": True}
    assert [m["text"] for m in texts] == [
        "Merhaba, size nasıl yardımcı olabilirim?\n",
        "Toplam borcunuz 1.500 TL görünüyor.\n",
    ]
    assert fake.headers[0]["X-API-Key"] == API_KEY
    assert fake.headers[0]["User-Agent"].startswith("livekit-plugins-avaz/")

    assert {frame.sample_rate for frame in frames} == {48000}
    assert _duration(frames) == _approx_duration(sum(len(t) - 1 for t in fake.texts) * 0.01)


async def test_synthesize(server: tuple[FakeAvaz, str], make_tts: Any) -> None:
    fake, _ = server
    fake.sample_rate = 24000
    tts = make_tts(model="avaz2")
    async with tts.synthesize("Kısa bir deneme cümlesi.", conn_options=CONN_OPTIONS) as stream:
        frames = [ev.frame async for ev in stream]

    assert fake.connections[0][0]["model_settings"] == {"model_id": "avaz2"}
    assert fake.texts == ["Kısa bir deneme cümlesi.\n"]
    assert {frame.sample_rate for frame in frames} == {24000}
    assert _duration(frames) == _approx_duration(0.24)


async def test_text_pause_longer_than_timeout(server: tuple[FakeAvaz, str], make_tts: Any) -> None:
    fake, _ = server
    tts = make_tts()
    async with tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.2)) as stream:
        stream.push_text("İlk cümle hemen geliyor ve okunuyor. Sonraki cümle ")
        await asyncio.sleep(0.5)
        stream.push_text("ise yarım saniye sonra geliyor.")
        stream.end_input()
        frames = [ev.frame async for ev in stream]

    assert len(fake.texts) == 2
    assert _duration(frames) == _approx_duration(sum(len(t) - 1 for t in fake.texts) * 0.01)


async def test_empty_input_opens_no_connection(server: tuple[FakeAvaz, str], make_tts: Any) -> None:
    fake, _ = server
    assert await _synthesize_stream(make_tts(), ["  ", "\n"]) == []
    assert fake.headers == []


async def test_stream_keeps_options_from_creation(
    server: tuple[FakeAvaz, str], make_tts: Any
) -> None:
    fake, _ = server
    tts = make_tts(speaker_id=1)
    async with tts.stream(conn_options=CONN_OPTIONS) as stream:
        tts.update_options(speaker_id=2)
        stream.push_text("Ses ayarı değişse de bu cümle ilk sesle okunur.")
        stream.end_input()
        assert [ev.frame async for ev in stream]

    assert fake.connections[0][0]["model_settings"]["speaker_id"] == 1


async def test_rejected_voice_settings_are_not_retried(
    server: tuple[FakeAvaz, str], make_tts: Any
) -> None:
    fake, _ = server
    fake.init_error = SERVER_ERROR
    with pytest.raises(APIError, match="speaker_id=999") as exc_info:
        await _synthesize_stream(make_tts(speaker_id=999), ["Merhaba."])

    assert not exc_info.value.retryable
    assert SERVER_ERROR not in str(exc_info.value)
    assert len(fake.connections) == 1


async def test_rejected_connection_is_not_retried(
    server: tuple[FakeAvaz, str], make_tts: Any
) -> None:
    fake, _ = server
    fake.reject_status = 401
    with pytest.raises(APIStatusError) as exc_info:
        await _synthesize_stream(make_tts(), ["Merhaba."])

    assert exc_info.value.status_code == 401
    assert not exc_info.value.retryable
    assert exc_info.value.__cause__ is None
    assert API_KEY not in str(exc_info.value)
    assert len(fake.headers) == 1


async def test_dropped_connection_is_retried_with_the_same_text(
    server: tuple[FakeAvaz, str], make_tts: Any
) -> None:
    fake, _ = server
    fake.drop_connection_attempts = 1
    frames = await _synthesize_stream(
        make_tts(), ["Bağlantı koptuğunda ", "aynı metin tekrar gönderilir."]
    )

    assert len(fake.connections) == 2
    assert fake.texts == ["Bağlantı koptuğunda aynı metin tekrar gönderilir.\n"]
    assert _duration(frames) > 0


async def test_server_error_does_not_leak_provider_text(
    server: tuple[FakeAvaz, str], make_tts: Any
) -> None:
    fake, _ = server
    fake.error_lines = {1}
    with pytest.raises(APIError) as exc_info:
        await _synthesize_stream(
            make_tts(), ["İlk cümle sorunsuz okunuyor. İkinci cümlede sunucu hata veriyor."]
        )

    assert SERVER_ERROR not in str(exc_info.value)
    assert len(fake.connections) == 1  # audio was already played, so no retry


async def test_missing_sentence_audio_fails(server: tuple[FakeAvaz, str], make_tts: Any) -> None:
    fake, _ = server
    fake.skip_lines = {1}
    with pytest.raises(APIError, match="1 of 2 sentences"):
        await _synthesize_stream(
            make_tts(), ["İlk cümle burada bitiyor. İkinci cümle sunucuda kayboluyor."]
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"sample_rate": 16000}, "sample rate"),
        ({"truncate_audio": True}, "truncated"),
        ({"audio_override": "çöş"}, "not a valid WAV"),
    ],
)
async def test_invalid_audio_fails(
    server: tuple[FakeAvaz, str], make_tts: Any, change: dict[str, Any], message: str
) -> None:
    fake, _ = server
    for name, value in change.items():
        setattr(fake, name, value)

    with pytest.raises(APIError, match=message) as exc_info:
        await _synthesize_stream(make_tts(), ["Merhaba, bugün nasılsınız?"])

    assert not exc_info.value.retryable
    assert len(fake.connections) == 1


async def test_no_response_after_flush_times_out(
    server: tuple[FakeAvaz, str], make_tts: Any
) -> None:
    fake, _ = server
    fake.hang_after_flush = True
    stream = _synthesize_stream(
        make_tts(),
        ["Sunucu bu cümleden sonra yanıt vermiyor."],
        conn_options=APIConnectOptions(max_retry=0, timeout=0.2),
    )
    with pytest.raises(APITimeoutError):
        await asyncio.wait_for(stream, 5)
