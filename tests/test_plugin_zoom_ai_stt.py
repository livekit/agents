"""Hermetic tests for the Zoom AI (Zoom Scribe) STT plugin.

A local aiohttp server stands in for Zoom Scribe:

- ``/live`` speaks the Scribe Live WebSocket protocol (``live-asr`` subprotocol):
  ``session.update`` -> ``session.updated``, binary PCM16 audio, ``speech_started``,
  ``transcription.completed``, ``session.close`` -> ``session.closed``.
- ``/transcribe`` accepts the Scribe Fast multipart upload (``file`` + ``config``).

Docs: https://developers.zoom.us/docs/ai-services/scribe/live-mode/ and
https://developers.zoom.us/docs/ai-services/scribe/fast-mode/
"""

from __future__ import annotations

import asyncio
import gc
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError
from livekit.agents.stt import SpeechEvent, SpeechEventType
from livekit.plugins import zoom_ai
from livekit.plugins.zoom_ai import stt as zoom_stt

# The fake server and HTTP session are fixtures bound to a single task.
pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

API_KEY = "test-key"
FAST_RETRY = APIConnectOptions(max_retry=1, retry_interval=0.01, timeout=5)
NO_RETRY = APIConnectOptions(max_retry=0, retry_interval=0.01, timeout=5)


# ---------------------------------------------------------------------------
# Fake Zoom Scribe server
# ---------------------------------------------------------------------------


@dataclass
class LiveSession:
    headers: dict[str, str]
    protocol: str | None
    updates: list[dict[str, Any]] = field(default_factory=list)
    controls: list[dict[str, Any]] = field(default_factory=list)
    audio: bytearray = field(default_factory=bytearray)


LiveScript = Callable[[web.WebSocketResponse, LiveSession], Awaitable[None]]


class FakeScribe:
    """Records what the plugin sends and replays scripted Scribe behavior."""

    def __init__(self) -> None:
        self.sessions: list[LiveSession] = []
        self.fast_requests: list[dict[str, Any]] = []
        # Called after each binary frame; lets a test emit events mid-stream.
        self.on_audio: LiveScript | None = None
        # Called right after session.updated; lets a test end the session early.
        self.on_start: LiveScript | None = None
        self.transcript = "Hi there, what is the capital of France?"
        self.fast_status = 200
        self.fast_body: dict[str, Any] = {
            "request_id": "req_1",
            "result": {
                "text_display": "Hi there! What is the capital of France?",
                "segments": [
                    {"start": 0.2, "end": 2.7, "speaker": "speaker_1", "text": "Hi there!"}
                ],
            },
        }

    async def live(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(protocols=("live-asr",))
        await ws.prepare(request)
        session = LiveSession(headers=dict(request.headers), protocol=ws.ws_protocol)
        self.sessions.append(session)
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                session.audio.extend(msg.data)
                if self.on_audio:
                    await self.on_audio(ws, session)
                continue
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            event = json.loads(msg.data)
            if event["type"] == "session.update":
                session.updates.append(event)
                await ws.send_json({"type": "session.updated"})
                if self.on_start:
                    await self.on_start(ws, session)
            elif event["type"] == "session.close":
                session.controls.append(event)
                await ws.send_json({"type": "session.closed", "reason": "client_close"})
                await ws.close()
            else:
                session.controls.append(event)
        return ws

    async def transcribe(self, request: web.Request) -> web.Response:
        form = await request.post()
        upload = form["file"]
        assert isinstance(upload, web.FileField)
        self.fast_requests.append(
            {
                "headers": dict(request.headers),
                "config": json.loads(str(form["config"])),
                "filename": upload.filename,
                "content_type": upload.content_type,
                "audio": upload.file.read(),
            }
        )
        return web.json_response(
            self.fast_body, status=self.fast_status, headers={"x-zm-trackingid": "ZOAP-test"}
        )


async def speak_once(ws: web.WebSocketResponse, session: LiveSession, transcript: str) -> None:
    """Scribe's VAD: speech_started after the first audio, a final once 1 s of audio arrived."""
    if len(session.audio) >= 3200 and not getattr(session, "_started", False):
        session._started = True  # type: ignore[attr-defined]
        await ws.send_json(
            {
                "type": "input_audio_buffer.speech_started",
                "item_id": "item_1",
                "audio_start_ms": 100,
            }
        )
    if len(session.audio) >= 32000 and not getattr(session, "_done", False):
        session._done = True  # type: ignore[attr-defined]
        await ws.send_json(
            {"type": "input_audio_buffer.speech_stopped", "item_id": "item_1", "audio_end_ms": 900}
        )
        await ws.send_json(
            {
                "type": "transcription.completed",
                "item_id": "item_1",
                "transcript": transcript,
                "audio_start_ms": 100,
                "audio_end_ms": 900,
                "transcription_latency_ms": 180,
            }
        )


@pytest.fixture
async def scribe() -> AsyncIterator[tuple[FakeScribe, str, str]]:
    fake = FakeScribe()
    app = web.Application()
    app.router.add_get("/live", fake.live)
    app.router.add_post("/transcribe", fake.transcribe)
    server = TestServer(app)
    await server.start_server()
    base = str(server.make_url(""))
    try:
        yield fake, base.replace("http://", "ws://") + "/live", base + "/transcribe"
    finally:
        await server.close()


@pytest.fixture
async def http() -> AsyncIterator[aiohttp.ClientSession]:
    session = aiohttp.ClientSession()
    try:
        yield session
    finally:
        await session.close()


def make_stt(http: aiohttp.ClientSession, urls: tuple[str, str], **kwargs: Any) -> zoom_ai.STT:
    live_url, fast_url = urls
    return zoom_ai.STT(
        api_key=API_KEY, live_url=live_url, fast_url=fast_url, http_session=http, **kwargs
    )


def frames(seconds: float, sample_rate: int = 16000, frame_ms: int = 20) -> list[rtc.AudioFrame]:
    """Silence at ``sample_rate`` (the fake server doesn't decode audio)."""
    per_frame = sample_rate * frame_ms // 1000
    count = int(seconds * 1000 / frame_ms)
    return [rtc.AudioFrame(bytes(per_frame * 2), sample_rate, 1, per_frame) for _ in range(count)]


async def wait_until(condition: Callable[[], bool], what: str, timeout: float = 3.0) -> None:
    async def poll() -> None:
        while not condition():
            await asyncio.sleep(0.01)

    try:
        await asyncio.wait_for(poll(), timeout)
    except asyncio.TimeoutError:
        pytest.fail(f"timed out waiting for {what}")


async def run_stream(
    stream: zoom_stt.SpeechStream, audio: list[rtc.AudioFrame], *, end: bool = True
) -> list[SpeechEvent]:
    events: list[SpeechEvent] = []

    async def collect() -> None:
        async for ev in stream:
            events.append(ev)

    collector = asyncio.create_task(collect())
    for f in audio:
        stream.push_frame(f)
    if end:
        stream.end_input()
    try:
        await asyncio.wait_for(collector, 5)
    finally:
        await stream.aclose()
    return events


# ---------------------------------------------------------------------------
# Construction and options
# ---------------------------------------------------------------------------


def test_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZOOM_SCRIBE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ZOOM_SCRIBE_API_KEY"):
        zoom_ai.STT()


def test_reads_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZOOM_SCRIBE_API_KEY", "env-key")
    assert zoom_ai.STT()._api_key == "env-key"


def test_capabilities_and_labels() -> None:
    live = zoom_ai.STT(api_key=API_KEY)
    fast = zoom_ai.STT(api_key=API_KEY, mode="fast", diarization=True)
    assert (live.provider, live.model, fast.model) == ("Zoom", "scribe-live", "scribe-fast")
    assert live.capabilities.streaming and not live.capabilities.interim_results
    assert not fast.capabilities.streaming and fast.capabilities.diarization
    assert not zoom_ai.STT(api_key=API_KEY, diarization=True).capabilities.diarization


@pytest.mark.parametrize(
    ("given", "expected"),
    [("en", "en-US"), ("en-us", "en-US"), ("en_US", "en-US"), ("ja", "ja-JP"), ("pt-PT", "pt-PT")],
)
def test_language_mapping(given: str, expected: str) -> None:
    assert zoom_ai.STT(api_key=API_KEY, language=given)._opts.language == expected


def test_fast_mode_is_not_streaming() -> None:
    with pytest.raises(NotImplementedError):
        zoom_ai.STT(api_key=API_KEY, mode="fast").stream()


# ---------------------------------------------------------------------------
# Scribe Live
# ---------------------------------------------------------------------------


async def test_live_handshake_matches_docs(scribe, http) -> None:
    fake, live_url, fast_url = scribe
    stt = make_stt(http, (live_url, fast_url))
    await run_stream(stt.stream(), frames(0.2))

    [session] = fake.sessions
    assert session.protocol == "live-asr"
    assert session.headers["Authorization"] == f"Bearer {API_KEY}"
    assert session.headers["User-Agent"].startswith("livekit-plugins-zoom-ai/")
    # Exactly the documented session.update, nothing else.
    assert session.updates == [
        {"type": "session.update", "language": "en-US", "audio": {"format": "pcm16"}}
    ]
    assert session.controls == [{"type": "session.close"}]


async def test_live_resamples_to_16k_pcm16(scribe, http) -> None:
    fake, live_url, fast_url = scribe
    stt = make_stt(http, (live_url, fast_url))
    await run_stream(stt.stream(), frames(1.0, sample_rate=48000))

    # 1 s of mono PCM16 at 16 kHz is 32000 bytes (allow for resampler latency).
    assert 30000 <= len(fake.sessions[0].audio) <= 32000


async def test_live_events_follow_livekit_order(scribe, http) -> None:
    fake, live_url, fast_url = scribe
    fake.on_audio = lambda ws, s: speak_once(ws, s, "[Speaker 1] " + fake.transcript)
    stt = make_stt(http, (live_url, fast_url))
    events = await run_stream(stt.stream(), frames(1.5))

    types = [e.type for e in events]
    assert types[:4] == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.FINAL_TRANSCRIPT,
        SpeechEventType.END_OF_SPEECH,
        SpeechEventType.RECOGNITION_USAGE,
    ]
    # Audio sent after the final is reported when the session closes.
    assert set(types[4:]) <= {SpeechEventType.RECOGNITION_USAGE}
    final = events[1]
    assert final.request_id == "item_1"
    assert final.alternatives[0].text == fake.transcript  # speaker tag removed
    assert final.alternatives[0].language == "en-US"
    assert final.alternatives[0].start_time == pytest.approx(0.1, abs=0.01)
    assert final.alternatives[0].end_time == pytest.approx(0.9, abs=0.01)
    usage = [e.recognition_usage.audio_duration for e in events if e.recognition_usage]
    assert sum(usage) == pytest.approx(1.5, abs=0.01)  # every second sent is accounted for


async def test_live_stream_language_override(scribe, http) -> None:
    fake, live_url, fast_url = scribe
    stt = make_stt(http, (live_url, fast_url))
    await run_stream(stt.stream(language="de"), frames(0.1))
    assert fake.sessions[0].updates[0]["language"] == "de-DE"


async def test_live_update_options_reconnects(scribe, http) -> None:
    fake, live_url, fast_url = scribe
    stt = make_stt(http, (live_url, fast_url))
    stream = stt.stream()
    for f in frames(0.2):  # more than one 50 ms send chunk
        stream.push_frame(f)
    await wait_until(lambda: bool(fake.sessions and fake.sessions[0].audio), "audio")
    stt.update_options(language="ja-JP")
    await wait_until(
        lambda: len(fake.sessions) == 2 and bool(fake.sessions[1].updates), "reconnect"
    )
    await run_stream(stream, frames(0.2))

    assert [s.updates[0]["language"] for s in fake.sessions] == ["en-US", "ja-JP"]
    assert fake.sessions[1].audio  # streaming continued on the new session
    assert stream.start_time_offset > 0  # timestamps keep moving forward


async def test_live_reconnects_when_scribe_ends_the_session(scribe, http) -> None:
    fake, live_url, fast_url = scribe

    async def end_first_session(ws: web.WebSocketResponse, session: LiveSession) -> None:
        if len(fake.sessions) == 1:  # e.g. idle timeout or session time limit
            await ws.send_json({"type": "session.closed", "reason": "idle_timeout"})
            await ws.close()

    fake.on_start = end_first_session
    stt = make_stt(http, (live_url, fast_url))
    stream = stt.stream(conn_options=FAST_RETRY)
    await wait_until(lambda: len(fake.sessions) == 2 and bool(fake.sessions[1].updates), "retry")
    await run_stream(stream, frames(0.1))

    assert len(fake.sessions) == 2
    assert fake.sessions[1].controls == [{"type": "session.close"}]


async def test_live_fatal_error_is_raised(scribe, http) -> None:
    fake, live_url, fast_url = scribe

    async def reject(ws: web.WebSocketResponse, session: LiveSession) -> None:
        error = {"code": "invalid_config", "message": "bad language", "fatal": True}
        await ws.send_json({"type": "error", "error": error})

    fake.on_start = reject
    stt = make_stt(http, (live_url, fast_url))
    with pytest.raises(APIStatusError, match="bad language") as exc:
        await run_stream(stt.stream(conn_options=FAST_RETRY), [], end=False)
    assert exc.value.retryable is False
    assert len(fake.sessions) == 1  # invalid config is not retried


async def test_live_non_fatal_error_keeps_the_session(scribe, http) -> None:
    fake, live_url, fast_url = scribe

    async def warn(ws: web.WebSocketResponse, session: LiveSession) -> None:
        error = {"code": "audio_gap", "message": "late audio", "fatal": False}
        await ws.send_json({"type": "error", "error": error})

    fake.on_start = warn
    fake.on_audio = lambda ws, s: speak_once(ws, s, fake.transcript)
    stt = make_stt(http, (live_url, fast_url))
    events = await run_stream(stt.stream(), frames(1.2))

    assert any(e.type == SpeechEventType.FINAL_TRANSCRIPT for e in events)
    assert len(fake.sessions) == 1


async def test_live_keepalive_sends_silence_when_input_is_idle(
    scribe, http, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, live_url, fast_url = scribe
    monkeypatch.setattr(zoom_stt, "KEEPALIVE_INTERVAL_S", 0.05)
    stt = make_stt(http, (live_url, fast_url))
    stream = stt.stream()
    await wait_until(
        lambda: bool(fake.sessions) and len(fake.sessions[0].audio) >= 3 * 3200, "keepalive audio"
    )
    await run_stream(stream, [])

    # No frames were pushed: everything received is 100 ms keepalive silence.
    audio = fake.sessions[0].audio
    assert len(audio) >= 3 * 3200 and len(audio) % 3200 == 0
    assert not any(audio)


async def test_live_reconnect_mid_speech_closes_the_utterance(scribe, http) -> None:
    """A session that drops mid-utterance must not leave the turn open forever."""
    fake, live_url, fast_url = scribe

    async def drop_after_speech(ws: web.WebSocketResponse, session: LiveSession) -> None:
        if len(fake.sessions) == 1 and len(session.audio) >= 3200:
            await ws.send_json(
                {"type": "input_audio_buffer.speech_started", "item_id": "a", "audio_start_ms": 0}
            )
            await ws.send_json({"type": "session.closed", "reason": "server_shutdown"})
            await ws.close()

    fake.on_audio = drop_after_speech
    stt = make_stt(http, (live_url, fast_url))
    stream = stt.stream(conn_options=FAST_RETRY)
    events: list[SpeechEvent] = []

    async def collect() -> None:
        async for ev in stream:
            events.append(ev)

    collector = asyncio.create_task(collect())
    for f in frames(0.3):
        stream.push_frame(f)
    await wait_until(lambda: len(fake.sessions) == 2 and bool(fake.sessions[1].updates), "retry")
    stream.end_input()
    await asyncio.wait_for(collector, 5)
    await stream.aclose()

    types = [e.type for e in events if e.type != SpeechEventType.RECOGNITION_USAGE]
    assert types == [SpeechEventType.START_OF_SPEECH, SpeechEventType.END_OF_SPEECH]
    assert len(fake.sessions) == 2


async def test_live_aclose_mid_session_is_clean(scribe, http) -> None:
    """Closing a stream while audio is flowing leaves no unretrieved errors behind."""
    fake, live_url, fast_url = scribe
    stt = make_stt(http, (live_url, fast_url))
    loop = asyncio.get_running_loop()
    errors: list[dict[str, Any]] = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: errors.append(context))
    try:
        stream = stt.stream()
        for f in frames(0.3):
            stream.push_frame(f)
        await wait_until(lambda: bool(fake.sessions and fake.sessions[0].audio), "audio")
        await stream.aclose()
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous)
    assert errors == []


# ---------------------------------------------------------------------------
# Scribe Fast
# ---------------------------------------------------------------------------


async def test_fast_uploads_wav_and_config(scribe, http) -> None:
    fake, live_url, fast_url = scribe
    stt = make_stt(http, (live_url, fast_url), mode="fast")
    event = await stt.recognize(frames(0.5), conn_options=NO_RETRY)

    [request] = fake.fast_requests
    assert request["headers"]["Authorization"] == f"Bearer {API_KEY}"
    assert request["headers"]["User-Agent"].startswith("livekit-plugins-zoom-ai/")
    assert request["config"] == {"language": "en-US"}
    assert request["content_type"] == "audio/wav"
    assert request["audio"][:4] == b"RIFF"

    assert event.type == SpeechEventType.FINAL_TRANSCRIPT
    assert event.request_id == "req_1"
    data = event.alternatives[0]
    assert data.text == "Hi there! What is the capital of France?"
    assert (data.start_time, data.end_time) == (0.2, 2.7)
    assert data.speaker_id is None  # diarization off


async def test_fast_diarization_and_language(scribe, http) -> None:
    fake, live_url, fast_url = scribe
    stt = make_stt(http, (live_url, fast_url), mode="fast", diarization=True)
    event = await stt.recognize(frames(0.5), language="fr", conn_options=NO_RETRY)

    assert fake.fast_requests[0]["config"] == {"language": "fr-FR", "diarization": True}
    assert event.alternatives[0].speaker_id == "speaker_1"
    assert event.alternatives[0].language == "fr-FR"


async def test_fast_http_error_raises_status_error(scribe, http) -> None:
    fake, live_url, fast_url = scribe
    fake.fast_status = 401
    fake.fast_body = {"code": 124, "message": "Invalid access token"}
    stt = make_stt(http, (live_url, fast_url), mode="fast")

    with pytest.raises(APIStatusError) as exc:
        await stt.recognize(frames(0.2), conn_options=NO_RETRY)
    assert exc.value.status_code == 401
    assert exc.value.request_id == "ZOAP-test"
    assert exc.value.retryable is False


async def test_fast_non_json_response_raises_status_error(scribe, http) -> None:
    fake, live_url, fast_url = scribe

    async def garbage(request: web.Request) -> web.Response:
        return web.Response(text="<html>gateway error</html>", status=200)

    stt = make_stt(http, (live_url, fast_url), mode="fast")
    fake.transcribe = garbage  # type: ignore[method-assign]
    app = web.Application()
    app.router.add_post("/transcribe", garbage)
    server = TestServer(app)
    await server.start_server()
    try:
        stt = make_stt(http, (live_url, str(server.make_url("/transcribe"))), mode="fast")
        with pytest.raises(APIStatusError, match="non-JSON"):
            await stt.recognize(frames(0.2), conn_options=NO_RETRY)
    finally:
        await server.close()
