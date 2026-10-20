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
import io
import json
import logging
import time
import wave
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError
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
                "outputs": json.loads(str(form["outputs"])) if "outputs" in form else None,
                "fields": {k: json.loads(str(v)) for k, v in form.items() if k != "file"},
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
    assert not live.capabilities.diarization  # off by default
    assert zoom_ai.STT(api_key=API_KEY, diarization=True).capabilities.diarization
    # A chosen Scribe model is what metrics report.
    assert zoom_ai.STT(api_key=API_KEY, model="zoom-scribe-en-pro").model == "zoom-scribe-en-pro"


@pytest.mark.parametrize(
    ("given", "expected"),
    [("en", "en-US"), ("en-us", "en-US"), ("en_US", "en-US"), ("ja", "ja-JP"), ("pt-PT", "pt-PT")],
)
def test_language_mapping(given: str, expected: str) -> None:
    assert zoom_ai.STT(api_key=API_KEY, language=given)._opts.language == expected


@pytest.mark.parametrize("language", ["en-GB", "ko-KR", "xx", "xx-YY"])
def test_unsupported_language_points_to_the_docs(language: str) -> None:
    with pytest.raises(ValueError, match="supported-languages"):
        zoom_ai.STT(api_key=API_KEY, language=language)
    with pytest.raises(ValueError, match="supported-languages"):
        zoom_ai.STT(api_key=API_KEY).update_options(language=language)


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
    assert session.headers["x-zm-trackingid"].startswith("zoom_ai_community_integration_livekit_")
    # The documented session.update, plus the 350 ms end-of-turn silence.
    assert session.updates == [
        {
            "type": "session.update",
            "language": "en-US",
            "audio": {"format": "pcm16"},
            "turn_detection": {"silence_duration_ms": 350},
        }
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
    # END_OF_SPEECH carries Scribe's end of speech, so "stt" turn detection counts its
    # endpointing delay from when the user stopped, not from when the final arrived.
    end_of_speech = events[2]
    assert end_of_speech.speech_end_time == pytest.approx(final.speech_end_time)
    assert end_of_speech.speech_end_time is not None
    usage = [e.recognition_usage.audio_duration for e in events if e.recognition_usage]
    assert sum(usage) == pytest.approx(1.5, abs=0.01)  # every second sent is accounted for


def _completed(**fields: Any) -> dict[str, Any]:
    """A transcription.completed as production Scribe Live sends it."""
    return {
        "type": "transcription.completed",
        "item_id": "item_1",
        "transcript": "Hello there.",
        "audio_start_ms": 0,
        "audio_end_ms": 4200,
        **fields,
    }


async def _live_final(scribe, http, completed: dict[str, Any], **stt_kwargs: Any):
    fake, live_url, fast_url = scribe

    async def send_final(ws: web.WebSocketResponse, session: LiveSession) -> None:
        if len(session.audio) >= 3200 and not getattr(session, "_done", False):
            session._done = True  # type: ignore[attr-defined]
            await ws.send_json(completed)

    fake.on_audio = send_final
    stt = make_stt(http, (live_url, fast_url), **stt_kwargs)
    events = await run_stream(stt.stream(), frames(0.5))
    [final] = [e for e in events if e.type == SpeechEventType.FINAL_TRANSCRIPT]
    return fake.sessions[0].updates[0], final.alternatives[0]


@pytest.mark.parametrize("diarization", [False, True])
async def test_live_diarization(scribe, http, diarization: bool) -> None:
    update, alt = await _live_final(
        scribe,
        http,
        _completed(transcript="[Speaker 1] Hello there.", speaker="Speaker 1"),
        diarization=diarization,
    )

    assert update.get("config") == ({"diarization": True} if diarization else None)
    assert alt.text == "Hello there."
    assert alt.speaker_id == ("Speaker 1" if diarization else None)
    assert alt.metadata is None  # one speaker: no speaker_segments


async def test_live_recognition_options_are_requested_only_when_set(scribe, http) -> None:
    update, _ = await _live_final(
        scribe,
        http,
        _completed(),
        model="zoom-scribe-multilingual",
        language_detection="continuous",
        vocabulary=["Pipecat", "LiveKit"],
        text_polishing=True,
        extra_kwargs={"config": {"profanity_filter": True}},
    )

    assert update["config"] == {
        "model": "zoom-scribe-multilingual",
        "language_detection": {"mode": "continuous"},
        "profanity_filter": True,
        "vocabulary": {"phrases": ["Pipecat", "LiveKit"]},
    }
    assert update["outputs"] == {"text_polishing": {"enabled": True, "include_transcript": True}}


async def test_live_extra_kwargs_are_merged_into_the_session_update(scribe, http) -> None:
    extra = {
        "config": {"vocabulary_id": "acct_vocab_1", "diarization": False},
        "turn_detection": {"max_segment_duration_ms": 15000},
        "outputs": {"future_output": {"enabled": True}},
        "audio": {"format": "opus"},  # the plugin sends PCM16, so this is kept
    }
    update, _ = await _live_final(scribe, http, _completed(), diarization=True, extra_kwargs=extra)

    assert update["config"] == {"diarization": False, "vocabulary_id": "acct_vocab_1"}
    assert update["turn_detection"] == {
        "silence_duration_ms": 350,
        "max_segment_duration_ms": 15000,
    }
    assert update["outputs"] == {"future_output": {"enabled": True}}
    assert update["audio"] == {"format": "pcm16"}
    assert extra["audio"] == {"format": "opus"}  # the caller's dict isn't modified


async def test_live_acts_on_options_set_through_extra_kwargs(scribe, http) -> None:
    update, alt = await _live_final(
        scribe,
        http,
        _completed(
            transcript="[Speaker 1] Um, hello hello there.",
            speaker="Speaker 1",
            outputs={"text_polishing": {"text": "[Speaker 1] Hello there."}},
        ),
        extra_kwargs={
            "config": {"diarization": True},
            "outputs": {"text_polishing": {"enabled": True}},
        },
    )

    # The transcript is the fallback when polishing fails, so it is always requested.
    assert update["outputs"] == {"text_polishing": {"enabled": True, "include_transcript": True}}
    assert alt.text == "Hello there."
    assert alt.speaker_id == "Speaker 1"
    assert alt.metadata == {"transcript": "Um, hello hello there."}


def test_extra_kwargs_must_be_a_dict() -> None:
    with pytest.raises(ValueError, match="extra_kwargs"):
        zoom_ai.STT(api_key=API_KEY, extra_kwargs=[("config", {})])  # type: ignore[arg-type]
    stt = zoom_ai.STT(api_key=API_KEY, extra_kwargs={"config": {"diarization": True}})
    assert stt.capabilities.diarization


async def test_live_language_is_empty_when_detection_reports_none(scribe, http) -> None:
    _, alt = await _live_final(scribe, http, _completed(), language_detection="continuous")

    assert alt.language == ""


async def test_live_reports_the_detected_language(scribe, http) -> None:
    _, alt = await _live_final(
        scribe,
        http,
        _completed(transcript="Buenos días.", language="es"),
        language_detection="continuous",
    )

    assert alt.language == "es"


async def test_live_text_polishing_reports_the_polished_text(scribe, http) -> None:
    _, alt = await _live_final(
        scribe,
        http,
        _completed(
            transcript="Um, hello hello there.",
            outputs={"text_polishing": {"text": "Hello there."}},
        ),
        text_polishing=True,
    )

    assert alt.text == "Hello there."
    assert alt.metadata == {"transcript": "Um, hello hello there."}


async def test_live_failed_polishing_falls_back_to_the_transcript(scribe, http) -> None:
    _, alt = await _live_final(
        scribe,
        http,
        _completed(
            transcript="Um, hello there.",
            outputs={"text_polishing": {"text": "[polishing_failed: timeout]"}},
        ),
        text_polishing=True,
    )

    assert alt.text == "Um, hello there."
    assert alt.metadata is None


async def test_live_polished_text_loses_speaker_tags(scribe, http) -> None:
    _, alt = await _live_final(
        scribe,
        http,
        _completed(
            transcript="[Speaker 1] um, hi [Speaker 2] uh, hello",
            outputs={"text_polishing": {"text": "[Speaker 1] Hi. [Speaker 2] Hello."}},
        ),
        text_polishing=True,
    )

    assert alt.text == "Hi. Hello."
    assert alt.metadata == {"transcript": "um, hi uh, hello"}


async def test_live_mixed_speakers_have_segments_and_no_speaker_id(scribe, http) -> None:
    spans = [
        {"speaker": "Speaker 1", "transcript": "Hi", "audio_start_ms": 0, "audio_end_ms": 500},
        {
            "speaker": "Speaker 2",
            "transcript": "I am fine, thanks for asking",
            "audio_start_ms": 500,
            "audio_end_ms": 3500,
        },
        {
            "speaker": "Speaker 1",
            "transcript": "Great",
            "audio_start_ms": 3500,
            "audio_end_ms": 4200,
        },
    ]
    _, alt = await _live_final(
        scribe, http, _completed(speaker="Speaker 1", speaker_segments=spans), diarization=True
    )

    # Several speakers: no single speaker_id; the spans carry who said what.
    assert alt.speaker_id is None
    assert alt.metadata == {"speaker_segments": spans}


async def test_live_warns_when_scribe_cannot_diarize(
    scribe, http, caplog: pytest.LogCaptureFixture
) -> None:
    fake, live_url, fast_url = scribe

    async def cannot_diarize(ws: web.WebSocketResponse, session: LiveSession) -> None:
        await ws.send_json({"type": "session.updated", "diarization_enabled": False})

    fake.on_start = cannot_diarize
    stt = make_stt(http, (live_url, fast_url), diarization=True)
    with caplog.at_level(logging.WARNING):
        await run_stream(stt.stream(), frames(0.1))
    assert "can't label speakers" in caplog.text


async def test_live_final_without_speech_started_opens_the_turn(scribe, http) -> None:
    fake, live_url, fast_url = scribe

    async def final_only(ws: web.WebSocketResponse, session: LiveSession) -> None:
        if len(session.audio) >= 3200 and not getattr(session, "_done", False):
            session._done = True  # type: ignore[attr-defined]
            await ws.send_json(_completed())

    fake.on_audio = final_only
    stt = make_stt(http, (live_url, fast_url))
    events = await run_stream(stt.stream(), frames(0.5))

    assert [e.type for e in events][:3] == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.FINAL_TRANSCRIPT,
        SpeechEventType.END_OF_SPEECH,
    ]


async def test_live_handshake_rejection_is_a_status_error_without_the_key(http) -> None:
    async def reject(request: web.Request) -> web.Response:
        return web.Response(status=401, text="Invalid access token")

    app = web.Application()
    app.router.add_get("/live", reject)
    server = TestServer(app)
    await server.start_server()
    try:
        stt = make_stt(http, (str(server.make_url("/live")), "http://127.0.0.1:9/unused"))
        with pytest.raises(APIStatusError) as exc:
            await run_stream(stt.stream(conn_options=NO_RETRY), frames(0.1))
    finally:
        await server.close()

    assert exc.value.status_code == 401
    assert exc.value.__cause__ is None and API_KEY not in repr(exc.value)


async def test_fast_connection_error_raises_connection_error(http) -> None:
    stt = make_stt(http, ("ws://127.0.0.1:9/unused", "http://127.0.0.1:9/transcribe"), mode="fast")
    with pytest.raises(APIConnectionError):
        await stt.recognize(frames(0.1), conn_options=NO_RETRY)


@pytest.mark.parametrize(
    ("silence_duration_ms", "expected"),
    [
        (500, {"silence_duration_ms": 500}),
        (None, None),  # Scribe's server default: exactly the documented session.update
    ],
)
async def test_live_silence_duration(
    scribe, http, silence_duration_ms: int | None, expected: Any
) -> None:
    fake, live_url, fast_url = scribe
    stt = make_stt(http, (live_url, fast_url), silence_duration_ms=silence_duration_ms)
    await run_stream(stt.stream(), frames(0.1))
    assert fake.sessions[0].updates[0].get("turn_detection") == expected


async def test_live_update_options_changes_the_silence_duration(scribe, http) -> None:
    fake, live_url, fast_url = scribe
    stt = make_stt(http, (live_url, fast_url))
    stream = stt.stream()
    for f in frames(0.2):
        stream.push_frame(f)
    await wait_until(lambda: bool(fake.sessions and fake.sessions[0].audio), "audio")
    stt.update_options(silence_duration_ms=500)
    await wait_until(
        lambda: len(fake.sessions) == 2 and bool(fake.sessions[1].updates), "reconnect"
    )
    await run_stream(stream, frames(0.1))

    silences = [s.updates[0]["turn_detection"]["silence_duration_ms"] for s in fake.sessions]
    assert silences == [350, 500]


def test_urls_derive_from_base_url() -> None:
    default = zoom_ai.STT(api_key=API_KEY)
    assert default._live_url == "wss://api.zoom.us/v2/aiservices/scribe/live"
    assert default._fast_url == "https://api.zoom.us/v2/aiservices/scribe/transcribe"
    custom = zoom_ai.STT(api_key=API_KEY, base_url="http://localhost:8080/scribe/")
    assert custom._live_url == "ws://localhost:8080/scribe/live"
    assert custom._fast_url == "http://localhost:8080/scribe/transcribe"


@pytest.mark.parametrize(("silence_duration_ms", "expected"), [(350, 0.55), (None, 0.9)])
async def test_live_speech_end_excludes_scribes_silence_window(
    http, silence_duration_ms: int | None, expected: float
) -> None:
    # Scribe ends a segment after the silence window we set, so audio_end_ms (900 ms here)
    # is that much later than the end of speech. With None the window is Scribe's own, so
    # audio_end_ms is used as is.
    stt = make_stt(
        http,
        ("ws://127.0.0.1:9/unused", "http://127.0.0.1:9/unused"),
        silence_duration_ms=silence_duration_ms,
    )
    stream = stt.stream()
    try:
        assert stream._speech_end_time(900) == pytest.approx(stream.start_time + expected)
        assert stream._speech_end_time(0) is None
    finally:
        await stream.aclose()


def test_rejects_an_unknown_mode_or_url() -> None:
    with pytest.raises(ValueError, match="mode"):
        zoom_ai.STT(api_key=API_KEY, mode="Live")  # type: ignore[arg-type]
    for bad in ("wss://api.zoom.us/x", "api.zoom.us/x", "https"):
        with pytest.raises(ValueError, match="base_url"):
            zoom_ai.STT(api_key=API_KEY, base_url=bad)
    assert zoom_ai.STT(api_key=API_KEY, base_url="HTTPS://h/x")._live_url == "wss://h/x/live"


def test_plaintext_urls_warn_except_for_localhost(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        zoom_ai.STT(api_key=API_KEY, base_url="http://scribe.example.com/x")
    warned = [r.getMessage() for r in caplog.records if "not encrypted" in r.getMessage()]
    assert len(warned) == 3  # base_url, and the live_url and fast_url derived from it
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        zoom_ai.STT(api_key=API_KEY, base_url="http://localhost:8080/x")
        zoom_ai.STT(api_key=API_KEY)
    assert not [r for r in caplog.records if "not encrypted" in r.getMessage()]


async def test_live_empty_final_closes_the_turn_without_a_transcript(scribe, http) -> None:
    fake, live_url, fast_url = scribe

    async def speech_then_nothing(ws: web.WebSocketResponse, session: LiveSession) -> None:
        if len(session.audio) >= 3200 and not getattr(session, "_done", False):
            session._done = True  # type: ignore[attr-defined]
            await ws.send_json(
                {"type": "input_audio_buffer.speech_started", "item_id": "i", "audio_start_ms": 0}
            )
            await ws.send_json(_completed(transcript="", audio_end_ms=500))

    fake.on_audio = speech_then_nothing
    stt = make_stt(http, (live_url, fast_url))
    events = await run_stream(stt.stream(), frames(0.5))

    types = [e.type for e in events if e.type != SpeechEventType.RECOGNITION_USAGE]
    assert types == [SpeechEventType.START_OF_SPEECH, SpeechEventType.END_OF_SPEECH]


async def test_live_reconnects_after_a_dropped_connection(scribe, http) -> None:
    fake, live_url, fast_url = scribe

    async def drop_first(ws: web.WebSocketResponse, session: LiveSession) -> None:
        if len(fake.sessions) == 1:  # close the socket with no session.closed
            await ws.close()

    fake.on_start = drop_first
    stt = make_stt(http, (live_url, fast_url))
    stream = stt.stream(conn_options=FAST_RETRY)
    await wait_until(lambda: len(fake.sessions) == 2 and bool(fake.sessions[1].updates), "retry")
    await run_stream(stream, frames(0.1))

    assert len(fake.sessions) == 2


async def test_live_stream_language_override(scribe, http) -> None:
    fake, live_url, fast_url = scribe
    stt = make_stt(http, (live_url, fast_url))
    await run_stream(stt.stream(language="de-DE"), frames(0.1))
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
    assert stream._session_offset > 0  # timestamps keep moving forward


async def test_live_update_options_without_changes_keeps_the_session(scribe, http) -> None:
    fake, live_url, fast_url = scribe
    stt = make_stt(http, (live_url, fast_url))
    stream = stt.stream(language="de-DE")
    for f in frames(0.2):
        stream.push_frame(f)
    await wait_until(lambda: bool(fake.sessions and fake.sessions[0].audio), "audio")
    stt.update_options()  # nothing given: no reconnect, and the override stays
    await asyncio.sleep(0.2)
    await run_stream(stream, frames(0.1))

    assert [s.updates[0]["language"] for s in fake.sessions] == ["de-DE"]


async def test_live_timestamps_survive_a_reconnect_then_a_retry(scribe, http) -> None:
    fake, live_url, fast_url = scribe

    async def on_start(ws: web.WebSocketResponse, session: LiveSession) -> None:
        if fake.sessions.index(session) == 1:  # the session opened by update_options
            await asyncio.sleep(0.3)
            await ws.send_json({"type": "session.closed", "reason": "idle_timeout"})
            await ws.close()

    async def on_audio(ws: web.WebSocketResponse, session: LiveSession) -> None:
        if fake.sessions.index(session) == 2 and not getattr(session, "_done", False):
            session._done = True  # type: ignore[attr-defined]
            await ws.send_json(
                {
                    "type": "transcription.completed",
                    "item_id": "item_1",
                    "transcript": "Hello.",
                    "audio_start_ms": 0,
                    "audio_end_ms": 100,
                }
            )

    fake.on_start, fake.on_audio = on_start, on_audio
    stt = make_stt(http, (live_url, fast_url))
    started = time.time()
    stream = stt.stream(conn_options=FAST_RETRY)
    finals: list[SpeechEvent] = []

    async def collect() -> None:
        async for ev in stream:
            if ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                finals.append(ev)

    collector = asyncio.create_task(collect())
    for f in frames(0.2):
        stream.push_frame(f)
    await wait_until(lambda: bool(fake.sessions and fake.sessions[0].audio), "audio")
    await asyncio.sleep(0.3)
    stt.update_options(language="ja-JP")
    await wait_until(lambda: len(fake.sessions) == 3 and bool(fake.sessions[2].updates), "retry")
    for f in frames(0.2):
        stream.push_frame(f)
    await wait_until(lambda: bool(finals), "final")
    elapsed = time.time() - started
    stream.end_input()
    await asyncio.wait_for(collector, 5)
    await stream.aclose()

    # The final's audio starts at 0 on the third session, so its stream-relative start
    # is the time before that session opened: never more than the time that has passed.
    start = finals[0].alternatives[0].start_time
    assert 0.5 < start <= elapsed


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


@pytest.mark.parametrize(
    ("code", "fatal"),
    [
        ("invalid_config", True),
        ("UNSUPPORTED_LANGUAGE", True),
        # Scribe sends this one as non-fatal, but it didn't apply the session.update.
        ("invalid_config", False),
    ],
)
async def test_live_fatal_error_is_raised(scribe, http, code: str, fatal: bool) -> None:
    fake, live_url, fast_url = scribe

    async def reject(ws: web.WebSocketResponse, session: LiveSession) -> None:
        error = {"code": code, "message": "bad language", "fatal": fatal}
        await ws.send_json({"type": "error", "error": error})

    fake.on_start = reject
    stt = make_stt(http, (live_url, fast_url))
    with pytest.raises(APIStatusError, match="bad language") as exc:
        await run_stream(stt.stream(conn_options=FAST_RETRY), [], end=False)
    assert exc.value.retryable is False
    assert len(fake.sessions) == 1  # a config error is not retried


async def test_live_non_fatal_error_keeps_the_session(
    scribe, http, caplog: pytest.LogCaptureFixture
) -> None:
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
    # Scribe's message stays out of the log text, under the lk.pii key telemetry filters.
    [record] = [r for r in caplog.records if "warning (audio_gap)" in r.getMessage()]
    assert "late audio" not in record.getMessage()
    assert record.__dict__["lk.pii.message"] == "late audio"


async def test_live_input_gap_mid_turn_sends_silence_to_end_it(
    scribe, http, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Input stopping mid-turn (e.g. a mute) gets enough silence for Scribe to end the turn."""
    fake, live_url, fast_url = scribe
    monkeypatch.setattr(zoom_stt, "INPUT_GAP_S", 0.1)

    async def start_speech(ws: web.WebSocketResponse, session: LiveSession) -> None:
        if not getattr(session, "_started", False):
            session._started = True  # type: ignore[attr-defined]
            await ws.send_json({"type": "input_audio_buffer.speech_started", "audio_start_ms": 0})

    fake.on_audio = start_speech
    stt = make_stt(http, (live_url, fast_url))
    stream = stt.stream()
    for f in frames(0.2):  # 6400 bytes, then the input stops
        stream.push_frame(f)
    window = 16000 * 2 * (350 + 100) // 1000  # silence_duration_ms plus 100 ms
    await wait_until(
        lambda: bool(fake.sessions) and len(fake.sessions[0].audio) >= 6400 + window, "gap silence"
    )
    await asyncio.sleep(0.3)
    assert len(fake.sessions[0].audio) == 6400 + window  # sent once per gap
    await stream.aclose()


async def test_stt_aclose_closes_open_streams(scribe, http) -> None:
    fake, live_url, fast_url = scribe
    stt = make_stt(http, (live_url, fast_url))
    stream = stt.stream()
    for f in frames(0.2):
        stream.push_frame(f)
    await wait_until(lambda: bool(fake.sessions and fake.sessions[0].audio), "audio")

    await stt.aclose()

    assert stream._task.done()
    assert not stt._streams


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
    assert request["headers"]["x-zm-trackingid"].startswith(
        "zoom_ai_community_integration_livekit_"
    )
    assert request["config"] == {"language": "en-US"}
    assert request["outputs"] is None
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
    event = await stt.recognize(frames(0.5), language="fr-FR", conn_options=NO_RETRY)

    assert fake.fast_requests[0]["config"] == {"language": "fr-FR", "diarization": True}
    assert event.alternatives[0].speaker_id == "speaker_1"
    assert event.alternatives[0].language == "fr-FR"


async def test_fast_mixed_speakers_have_segments_and_no_speaker_id(scribe, http) -> None:
    fake, live_url, fast_url = scribe
    segments = [  # as production returns them
        {"start": 0.0, "end": 0.6, "speaker": "Speaker 1", "text": "Hi."},
        {"start": 0.6, "end": 4.8, "speaker": "Speaker 2", "text": "We built it in seven steps."},
    ]
    fake.fast_body = {
        "request_id": "req_2",
        "result": {
            "text_display": "[Speaker 1] Hi. [Speaker 2] We built it in seven steps.",
            "segments": segments,
        },
    }
    stt = make_stt(http, (live_url, fast_url), mode="fast", diarization=True)
    alt = (await stt.recognize(frames(0.5), conn_options=NO_RETRY)).alternatives[0]

    assert alt.text == "Hi. We built it in seven steps."
    assert alt.speaker_id is None  # several speakers: see speaker_segments
    assert alt.metadata == {
        "speaker_segments": [
            {"speaker": s["speaker"], "text": s["text"], "start": s["start"], "end": s["end"]}
            for s in segments
        ]
    }


async def test_fast_recognition_options_and_text_polishing(scribe, http) -> None:
    fake, live_url, fast_url = scribe
    fake.fast_body = {
        "request_id": "req_5",
        "result": {"text_display": "Um, hello hello there.", "segments": []},
        "outputs": {"text_polishing": {"text": "Hello there."}},
    }
    stt = make_stt(
        http,
        (live_url, fast_url),
        mode="fast",
        model="zoom-scribe-en-pro",
        language_detection="once",
        vocabulary=["Pipecat"],
        text_polishing=True,
        extra_kwargs={"config": {"profanity_filter": True}},
    )
    alt = (await stt.recognize(frames(0.5), conn_options=NO_RETRY)).alternatives[0]

    [request] = fake.fast_requests
    assert request["config"] == {
        "language": "en-US",
        "model": "zoom-scribe-en-pro",
        "language_detection": {"mode": "once"},
        "profanity_filter": True,
        "vocabulary": {"phrases": ["Pipecat"]},
    }
    # Text polishing is its own multipart field, as `outputs` sits beside `config`.
    assert request["outputs"] == {"text_polishing": {"enabled": True}}
    assert alt.text == "Hello there."
    assert alt.metadata == {"transcript": "Um, hello hello there."}


async def test_fast_extra_kwargs_become_request_fields(scribe, http) -> None:
    fake, live_url, fast_url = scribe
    stt = make_stt(
        http,
        (live_url, fast_url),
        mode="fast",
        extra_kwargs={
            "config": {"segmentation_mode": "none"},
            "outputs": {"future_output": {"enabled": True}},
            "reference_id": "req-42",
            "file": "ignored",  # the plugin uploads the audio
        },
    )
    await stt.recognize(frames(0.5), conn_options=NO_RETRY)

    [request] = fake.fast_requests
    assert request["fields"] == {
        "config": {"language": "en-US", "segmentation_mode": "none"},
        "outputs": {"future_output": {"enabled": True}},
        "reference_id": "req-42",
    }
    assert request["audio"][:4] == b"RIFF"


async def test_fast_polished_text_has_no_word_timings(scribe, http) -> None:
    """Word timings belong to the recognized text, so they aren't paired with polished text."""
    fake, live_url, fast_url = scribe
    words = [{"word": "Um", "start": 0.0, "end": 0.2}, {"word": "hello", "start": 0.3, "end": 0.6}]
    fake.fast_body = {
        "request_id": "req_6",
        "result": {
            "text_display": "Um, hello.",
            "segments": [{"text": "Um, hello.", "words": words}],
        },
        "outputs": {"text_polishing": {"text": "Hello."}},
    }
    stt = make_stt(
        http, (live_url, fast_url), mode="fast", word_time_offsets=True, text_polishing=True
    )
    alt = (await stt.recognize(frames(0.5), conn_options=NO_RETRY)).alternatives[0]

    assert alt.text == "Hello."
    assert alt.words is None


@pytest.mark.parametrize(
    ("result", "detection", "expected"),
    [
        ({"language": "es-ES", "segments": []}, "once", "es-ES"),  # one-time detection
        ({"segments": [{"text": "Hola.", "language": "es"}]}, "continuous", "es"),  # per segment
        ({"segments": [{"text": "Hola."}]}, "continuous", ""),  # detecting, none reported
        ({"segments": [{"text": "Hola."}]}, None, "en-US"),  # not detecting: the configured one
    ],
)
async def test_fast_reports_the_detected_language(
    scribe, http, result: dict[str, Any], detection: str | None, expected: str
) -> None:
    fake, live_url, fast_url = scribe
    fake.fast_body = {"request_id": "req_7", "result": {"text_display": "Hola.", **result}}
    stt = make_stt(http, (live_url, fast_url), mode="fast", language_detection=detection)
    alt = (await stt.recognize(frames(0.5), conn_options=NO_RETRY)).alternatives[0]

    assert alt.language == expected


@pytest.mark.parametrize("channel_separation", [False, True])
async def test_fast_channel_separation(scribe, http, channel_separation: bool) -> None:
    fake, live_url, fast_url = scribe
    fake.fast_body = {
        "request_id": "req_ch",
        "result": {
            "text_display": "Hi there. Hello.",
            "segments": [
                {"channel": 0, "start": 0.1, "end": 0.9, "speaker": "", "text": "Hi there."},
                {"channel": 1, "start": 1.2, "end": 1.8, "speaker": "", "text": "Hello."},
            ],
        },
    }
    stt = make_stt(http, (live_url, fast_url), mode="fast", channel_separation=channel_separation)
    stereo = [
        rtc.AudioFrame(
            b"\x00\x00" * 320 * 2, sample_rate=16000, num_channels=2, samples_per_channel=320
        )
        for _ in range(25)
    ]
    alt = (await stt.recognize(stereo, conn_options=NO_RETRY)).alternatives[0]

    assert ("channel_separation" in fake.fast_requests[0]["config"]) is channel_separation
    with wave.open(io.BytesIO(fake.fast_requests[0]["audio"])) as wav:
        assert wav.getnchannels() == 2  # the upload keeps the input's channels
    if channel_separation:
        assert alt.metadata == {
            "channel_segments": [
                {"channel": 0, "text": "Hi there.", "start": 0.1, "end": 0.9},
                {"channel": 1, "text": "Hello.", "start": 1.2, "end": 1.8},
            ]
        }
    else:
        assert alt.metadata is None
    assert alt.text == "Hi there. Hello."


@pytest.mark.parametrize("word_time_offsets", [False, True])
async def test_fast_word_time_offsets(scribe, http, word_time_offsets: bool) -> None:
    fake, live_url, fast_url = scribe
    fake.fast_body = {
        "request_id": "req_4",
        "result": {
            "text_display": "Hi there.",
            "segments": [
                {
                    "start": 0.1,
                    "end": 0.9,
                    "speaker": "",
                    "text": "Hi there.",
                    "words": [
                        {"word": "Hi", "start": 0.1, "end": 0.3},
                        {"word": "there.", "start": 0.4, "end": 0.9},
                    ],
                }
            ],
        },
    }
    stt = make_stt(http, (live_url, fast_url), mode="fast", word_time_offsets=word_time_offsets)
    alt = (await stt.recognize(frames(0.5), conn_options=NO_RETRY)).alternatives[0]

    assert ("word_time_offsets" in fake.fast_requests[0]["config"]) is word_time_offsets
    if word_time_offsets:
        assert alt.words is not None
        assert [(str(w), w.start_time, w.end_time) for w in alt.words] == [
            ("Hi", 0.1, 0.3),
            ("there.", 0.4, 0.9),
        ]
    else:
        assert alt.words is None


async def test_fast_without_diarization_reports_no_speaker(scribe, http) -> None:
    fake, live_url, fast_url = scribe
    fake.fast_body = {  # production sends speaker "" when diarization is off
        "request_id": "req_3",
        "result": {"text_display": "Hi.", "segments": [{"start": 0, "end": 1, "speaker": ""}]},
    }
    stt = make_stt(http, (live_url, fast_url), mode="fast")
    alt = (await stt.recognize(frames(0.5), conn_options=NO_RETRY)).alternatives[0]

    assert alt.speaker_id is None and alt.metadata is None


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
    _, live_url, _ = scribe

    async def garbage(request: web.Request) -> web.Response:
        return web.Response(text="<html>gateway error</html>", status=200)

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
