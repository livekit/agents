"""Giggy STT and TTS smoke tests against an in-memory fake of the API (no sockets)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError, stt

pytestmark = pytest.mark.unit

BASE = "https://giggy.test/v1"
GENERATION = "0b6f8f0e-2d1c-4a8e-9a43-6f1f8c1f0a01"


def _frame(ms: int = 100, rate: int = 16000) -> rtc.AudioFrame:
    n = rate * ms // 1000
    return rtc.AudioFrame(b"\x10\x00" * n, sample_rate=rate, num_channels=1, samples_per_channel=n)


class FakeWS:
    """A WebSocket whose peer is a callback reacting to each message the client sends."""

    def __init__(self, on_message: Callable[[FakeWS, str | bytes], None]) -> None:
        self._on_message = on_message
        self._incoming: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()
        self.sent: list[str | bytes] = []
        self.closed = False

    def push(self, kind: str, **fields: Any) -> None:
        data = json.dumps({"type": kind, "generation_uuid": GENERATION, **fields})
        self._incoming.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, data, None))

    def close_from_server(self, code: int) -> None:
        self._incoming.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.CLOSE, code, ""))

    async def send_str(self, data: str) -> None:
        self.sent.append(data)
        self._on_message(self, data)

    async def send_bytes(self, data: bytes) -> None:
        self.sent.append(data)
        self._on_message(self, data)

    async def receive(self) -> aiohttp.WSMessage:
        return await self._incoming.get()

    async def close(self) -> bool:
        self.closed = True
        return True


class FakeResponse:
    def __init__(
        self, status: int = 200, chunks: list[bytes] | None = None, body: str = ""
    ) -> None:
        self.status = status
        self.reason = {200: "OK", 401: "Unauthorized"}.get(status)
        self.headers = {"x-request-id": "req-1"}
        self._chunks = chunks or []
        self._body = body
        self.content = self

    async def text(self) -> str:
        return self._body

    async def iter_chunks(self) -> AsyncIterator[tuple[bytes, bool]]:
        for chunk in self._chunks:
            yield chunk, False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for line in b"".join(self._chunks).splitlines(keepends=True):
            yield line

    async def __aenter__(self) -> FakeResponse:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


class FakeSession:
    def __init__(
        self,
        *,
        on_ws: Callable[[FakeWS, str | bytes], None] | None = None,
        on_post: Callable[[str, dict[str, Any]], FakeResponse] | None = None,
    ) -> None:
        self._on_ws = on_ws
        self._on_post = on_post
        self.sockets: list[FakeWS] = []
        self.ws_kwargs: dict[str, Any] = {}

    async def ws_connect(self, url: str, **kwargs: Any) -> FakeWS:
        assert url == "wss://giggy.test/v1/audio/transcriptions" and self._on_ws
        self.ws_kwargs = kwargs
        ws = FakeWS(self._on_ws)
        self.sockets.append(ws)
        return ws

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        assert self._on_post
        return self._on_post(url, kwargs)


def test_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.plugins.giggy import STT, TTS

    monkeypatch.delenv("GIGGY_API_KEY", raising=False)
    monkeypatch.delenv("GIGGY_VOICE_ID", raising=False)
    with pytest.raises(ValueError, match="API key"):
        STT()
    with pytest.raises(ValueError, match="voice"):
        TTS(api_key="key")
    with pytest.raises(ValueError, match="language"):
        STT(api_key="key", language="xx")


async def test_stt_stream_uses_server_endpointing() -> None:
    from livekit.plugins.giggy import STT

    def server(ws: FakeWS, data: str | bytes) -> None:
        if isinstance(data, bytes):
            if sum(isinstance(m, bytes) for m in ws.sent) == 1:
                ws.push("transcription", is_final=False, alternatives=[{"transcript": "hello"}])
            return
        message = json.loads(data)
        if message["type"] == "start":
            ws.push("generation_created")
            ws.push("ready", attempt=1, sampleRateHz=16000)
        elif message["type"] == "stop":
            ws.push("input_complete", sequence=10)
            ws.push("transcription", is_final=True, alternatives=[{"transcript": "hello there"}])
            ws.push("generation_updated", status="completed")
            ws.push("some_future_event")
            ws.close_from_server(1000)

    session = FakeSession(on_ws=server)
    stream = STT(
        api_key="key",
        language="en-US",
        base_url=BASE,
        http_session=session,  # type: ignore[arg-type]
    ).stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
    # Giggy rejects messages over 1 s of PCM, so a long frame must be split.
    stream.push_frame(_frame(1500))
    stream.push_frame(_frame(25))
    stream.end_input()
    events = [ev async for ev in stream]
    await stream.aclose()

    assert [e.type for e in events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
        stt.SpeechEventType.RECOGNITION_USAGE,
    ]
    assert events[2].alternatives[0].text == "hello there"
    assert events[2].alternatives[0].language == "en"
    assert events[2].request_id == GENERATION
    usage = events[4].recognition_usage
    assert usage is not None and usage.audio_duration == pytest.approx(1.525, abs=0.01)

    (ws,) = session.sockets
    assert session.ws_kwargs["headers"] == {"Authorization": "Bearer key"}
    assert json.loads(ws.sent[0]) == {
        "type": "start",
        "format": "raw",
        "encoding": "LINEAR16",
        "sampleRateHz": 16000,
        "interimResults": True,
        "language": "en",
    }
    audio = [m for m in ws.sent if isinstance(m, bytes)]
    assert sum(len(m) for m in audio) == 48800
    assert max(len(m) for m in audio) <= 32000
    assert json.loads(ws.sent[-1]) == {"type": "stop"}
    assert ws.closed


@pytest.mark.virtual_time  # the first retry waits 0.1 s
async def test_stt_stream_respects_max_retry() -> None:
    from livekit.plugins.giggy import STT

    def server(ws: FakeWS, data: str | bytes) -> None:
        if isinstance(data, str) and json.loads(data)["type"] == "start":
            ws.push("error", error="STT generation failed")
            ws.close_from_server(1011)

    session = FakeSession(on_ws=server)
    stream = STT(api_key="key", base_url=BASE, http_session=session).stream(  # type: ignore[arg-type]
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0, timeout=5)
    )
    stream.push_frame(_frame())
    with pytest.raises(APIConnectionError):
        async for _ in stream:
            pass
    await stream.aclose()

    assert len(session.sockets) == 2


async def test_stt_recognize() -> None:
    from livekit.plugins.giggy import STT

    seen: dict[str, Any] = {}

    def server(url: str, kwargs: dict[str, Any]) -> FakeResponse:
        seen.update(kwargs, url=url)
        events = [
            ("generation_created", {"generation_uuid": GENERATION}),
            ("transcript_chunk", {"generation_uuid": GENERATION, "text": "hi", "complete": False}),
            (
                "transcript_chunk",
                {"generation_uuid": GENERATION, "text": "hi all", "complete": True},
            ),
        ]
        return FakeResponse(
            chunks=[
                f"event: {kind}\ndata: {json.dumps(data)}\n\n".encode() for kind, data in events
            ]
        )

    session = FakeSession(on_post=server)
    result = await STT(api_key="key", base_url=BASE, http_session=session).recognize(  # type: ignore[arg-type]
        [_frame(500, 48000)]
    )

    assert result.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert result.request_id == GENERATION
    assert result.alternatives[0].text == "hi all"
    assert seen["url"] == f"{BASE}/audio/transcriptions"
    assert seen["headers"]["xi-api-key"] == "key"
    assert seen["params"]["mode"] == "streaming"
    assert seen["data"][:4] == b"RIFF"


async def test_tts_synthesize() -> None:
    from livekit.plugins.giggy import TTS

    seen: dict[str, Any] = {}
    error = '{"detail":{"code":"invalid_api_key"}}'

    def server(url: str, kwargs: dict[str, Any]) -> FakeResponse:
        seen.update(kwargs, url=url)
        if kwargs["json"]["input"] == "fail":
            return FakeResponse(status=401, body=error)
        pcm = b"\x01\x00" * 2400  # 100 ms at 24 kHz, sent in odd-sized chunks
        return FakeResponse(chunks=[pcm[i : i + 333] for i in range(0, len(pcm), 333)])

    tts = TTS(
        api_key="key", voice="voice-id", base_url=BASE, http_session=FakeSession(on_post=server)
    )  # type: ignore[arg-type]
    frames = [ev.frame async for ev in tts.synthesize("Hello there.")]

    # Every sample arrives despite odd chunk boundaries (the emitter may pad trailing silence).
    assert sum(f.samples_per_channel for f in frames) >= 2400
    assert all(f.sample_rate == 24000 for f in frames)
    assert seen["url"] == f"{BASE}/audio/speech"
    assert seen["headers"] == {"Authorization": "Bearer key"}
    assert seen["json"]["voice"] == "voice-id"

    with pytest.raises(APIStatusError) as failure:
        async for _ in tts.synthesize(
            "fail", conn_options=APIConnectOptions(max_retry=0, timeout=5)
        ):
            pass
    # Giggy's error body and request ID reach the caller.
    assert failure.value.status_code == 401
    assert failure.value.request_id == "req-1"
    assert failure.value.body == error
