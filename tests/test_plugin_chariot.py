from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError, stt
from livekit.agents.stt import SpeechEventType
from livekit.plugins import chariot

from .stt_reconnect import wait_until

pytestmark = pytest.mark.unit

PCM = b"\x01\x00" * 441  # 10 ms at 44.1 kHz
NO_RETRY = APIConnectOptions(max_retry=0, timeout=2.0)


@dataclass
class FakeChariot:
    """A minimal Chariot API: multi-context TTS and streaming ASR WebSockets, and the HTTP
    endpoints behind `synthesize()` and `recognize()`."""

    chunks: int = 3  # audio chunks per generated segment
    pcm: bytes = PCM  # each chunk's audio
    chunk_delay: float = 0.0
    refuse: dict[str, Any] | None = None  # sent instead of the session's first message
    context_error: dict[str, Any] | None = None  # ends the next context with this error
    session_error: dict[str, Any] | None = None  # ends the whole connection on the next text
    heartbeat: float | None = None  # ping every this often; close a connection that misses one
    start_delay: float = 0.0  # delay before the first ASR session starts
    connections: int = 0
    queries: list[dict[str, str]] = field(default_factory=list)
    received: list[Any] = field(default_factory=list)
    flushed: list[str] = field(default_factory=list)  # each generated segment's text
    commits: int = 0
    uploads: list[dict[str, Any]] = field(default_factory=list)

    async def _accept(self, request: web.Request) -> web.WebSocketResponse | None:
        ws = web.WebSocketResponse(heartbeat=self.heartbeat)
        await ws.prepare(request)
        self.connections += 1
        self.queries.append(dict(request.query))
        assert request.headers["chariotai-api-key"] == "key"
        if self.refuse:
            await ws.send_json({"type": "error", **self.refuse})
            await ws.close(code=self.refuse["code"], message=self.refuse["message"].encode())
            return None
        return ws

    async def tts_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = await self._accept(request)
        if ws is None:
            return web.WebSocketResponse()
        await ws.send_json({"type": "session.ready", "sample_rate": 44100})
        buffers: dict[str, str] = {}
        tasks: dict[str, asyncio.Task[None]] = {}

        async def generate(cid: str, text: str, *, close: bool) -> None:
            if text:
                self.flushed.append(text)
                await ws.send_json({"type": "audio.start", "context_id": cid, "segment": 0})
                for _ in range(self.chunks):
                    audio = base64.b64encode(self.pcm).decode()
                    await ws.send_json({"type": "audio.chunk", "context_id": cid, "audio": audio})
                    await asyncio.sleep(self.chunk_delay)
                await ws.send_json({"type": "audio.done", "context_id": cid, "error": False})
            if close:
                await ws.send_json(
                    {"type": "context.done", "context_id": cid, "reason": "completed"}
                )

        async def run_after(previous: asyncio.Task[None] | None, coro: Any) -> None:
            if previous is not None:  # a context's segments are generated in order
                with contextlib.suppress(asyncio.CancelledError):
                    await previous
            await coro

        async for msg in ws:
            event = json.loads(msg.data)
            self.received.append(event)
            kind, cid = event["type"], event.get("context_id", "")
            if kind == "ping":
                await ws.send_json({"type": "pong"})
            elif kind == "input.text":
                if event["text"] and self.session_error:
                    error, self.session_error = self.session_error, None
                    await ws.send_json({"type": "error", **error})
                    await ws.send_json(
                        {"type": "context.done", "context_id": cid, "reason": "session_closed"}
                    )
                    await ws.close(code=4402)
                    break
                if cid not in buffers and self.context_error:
                    error, self.context_error = self.context_error, None
                    await ws.send_json(
                        {"type": "context.done", "context_id": cid, "reason": "error", **error}
                    )
                    continue
                buffers[cid] = buffers.get(cid, "") + event["text"]
            elif kind in ("input.flush", "context.close"):
                text, buffers[cid] = buffers.get(cid, ""), ""
                tasks[cid] = asyncio.create_task(
                    run_after(tasks.get(cid), generate(cid, text, close=kind == "context.close"))
                )
            elif kind == "context.cancel" and cid in buffers:
                del buffers[cid]
                if (task := tasks.pop(cid, None)) is not None:
                    task.cancel()
                await ws.send_json(
                    {"type": "context.done", "context_id": cid, "reason": "cancelled"}
                )
        for task in tasks.values():
            task.cancel()
        return ws

    async def tts_http(self, request: web.Request) -> web.StreamResponse:
        self.received.append(await request.json())
        if self.refuse:
            return web.json_response(
                {"statusCode": 402, "message": self.refuse["message"]}, status=402
            )
        resp = web.StreamResponse(headers={"Content-Type": "audio/L16", "X-inference-id": "i-1"})
        await resp.prepare(request)
        for _ in range(self.chunks):
            await resp.write(PCM)
        return resp

    async def asr_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = await self._accept(request)
        if ws is None:
            return web.WebSocketResponse()
        if self.start_delay:
            delay, self.start_delay = self.start_delay, 0.0
            await asyncio.sleep(delay)
        await ws.send_json({"type": "session.started", "session_id": "s"})
        heard = 0  # audio bytes since the last commit
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                self.received.append(len(msg.data))
                if heard == 0:
                    if request.query.get("commit_strategy") == "vad":
                        await ws.send_json({"type": "input_audio_buffer.speech_started"})
                    await ws.send_json({"type": "transcript.delta", "text": "namaste"})
                heard += len(msg.data)
                continue
            event = json.loads(msg.data)
            self.received.append(event)
            if event["type"] == "input.commit":
                self.commits += 1
                text = f"segment {self.commits}"
                words = [{"text": text, "start": 0.1, "end": 0.5}]
                await ws.send_json(
                    {"type": "transcript.committed", "text": text, "start": 0.0, "end": 0.6}
                    | {"words": words}
                )
                heard = 0
            elif event["type"] == "input.done":
                await ws.send_json({"type": "transcript.done", "text": "the tail" if heard else ""})
                await ws.close()
        return ws

    async def asr_http(self, request: web.Request) -> web.Response:
        upload: dict[str, Any] = {}
        async for part in await request.multipart():
            if part.name == "input_audio_file":
                with wave.open(io.BytesIO(await part.read())) as wav:
                    upload["wav"] = (wav.getframerate(), wav.getnchannels(), wav.getnframes())
            else:
                upload[part.name] = await part.text()
        self.uploads.append(upload)
        words = [{"text": "namaste", "start": 0.2, "end": 0.7}]
        return web.json_response(
            {"request_id": "r-1", "text": "namaste", "duration": 1.0, "words": words}
        )


@contextlib.asynccontextmanager
async def serve(fake: FakeChariot) -> AsyncIterator[tuple[str, aiohttp.ClientSession]]:
    app = web.Application()
    app.router.add_get("/v1/tts/ws-multi", fake.tts_ws)
    app.router.add_post("/v1/tts/stream", fake.tts_http)
    app.router.add_get("/v1/asr/ws", fake.asr_ws)
    app.router.add_post("/v1/asr/transcribe", fake.asr_http)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    async with aiohttp.ClientSession() as session:
        try:
            yield f"http://127.0.0.1:{port}", session
        finally:
            await runner.cleanup()


@contextlib.asynccontextmanager
async def tts_server(fake: FakeChariot, **kwargs: Any) -> AsyncIterator[chariot.TTS]:
    async with serve(fake) as (url, session):
        tts = chariot.TTS(api_key="key", base_url=url, http_session=session, **kwargs)
        try:
            yield tts
        finally:
            await tts.aclose()


@contextlib.asynccontextmanager
async def stt_server(fake: FakeChariot, **kwargs: Any) -> AsyncIterator[chariot.STT]:
    async with serve(fake) as (url, session):
        yield chariot.STT(api_key="key", base_url=url, http_session=session, **kwargs)


async def _speak_frames(tts: chariot.TTS, text: str) -> list[rtc.AudioFrame]:
    stream = tts.stream(conn_options=NO_RETRY)
    stream.push_text(text)
    stream.end_input()
    async with stream:
        return [ev.frame async for ev in stream]


async def _speak(tts: chariot.TTS, text: str) -> float:
    return sum(frame.duration for frame in await _speak_frames(tts, text))


def _without_usage(events: list[stt.SpeechEvent]) -> list[stt.SpeechEvent]:
    return [ev for ev in events if ev.type != SpeechEventType.RECOGNITION_USAGE]


def _frames(ms: int, rate: int = 16000) -> list[rtc.AudioFrame]:
    samples = rate // 100
    return [rtc.AudioFrame(b"\x00\x00" * samples, rate, 1, samples) for _ in range(ms // 10)]


async def test_tts_replies_share_one_connection_and_flush_each_sentence() -> None:
    fake = FakeChariot()
    async with tts_server(fake, voice="v-1", speed=1.25) as tts:
        assert await _speak(tts, "Hello there, good to talk again. How are you doing today?") > 0
        assert await _speak(tts, "नमस्ते, आज मौसम बहुत अच्छा है। आप आज कैसे हैं, मेरे दोस्त?") > 0
    assert fake.connections == 1
    assert fake.queries[0] == {"model_type": "chariot-multilingual-v1", "idle_timeout": "300"}
    opens = [m for m in fake.received if "voice_id" in m]
    assert [(m["voice_id"], m["speed"], m["text"]) for m in opens] == [("v-1", 1.25, "")] * 2
    # every sentence is generated as soon as it is complete, including after a danda
    assert fake.flushed == [
        "Hello there, good to talk again.",
        "How are you doing today?",
        "नमस्ते, आज मौसम बहुत अच्छा है।",
        "आप आज कैसे हैं, मेरे दोस्त?",
    ]
    assert sum(m["type"] == "context.close" for m in fake.received) == 2


async def test_tts_interruption_cancels_the_reply_and_keeps_the_connection() -> None:
    fake = FakeChariot(chunks=200, chunk_delay=0.001)
    async with tts_server(fake) as tts:
        stream = tts.stream(conn_options=NO_RETRY)
        stream.push_text("A long reply that the user interrupts.")
        stream.end_input()
        await stream.__anext__()
        await stream.aclose()
        await asyncio.gather(*tts._tasks)
        fake.chunks, fake.chunk_delay, fake.pcm = 3, 0.0, b"\x02\x00" * 441
        frames = await _speak_frames(tts, "The next reply.")
    # audio the cancelled reply had already sent is dropped, not played in the next one
    samples = {s for frame in frames for s in frame.data}
    assert samples <= {0, 2}
    assert sum(frame.duration for frame in frames) == pytest.approx(0.03, abs=0.02)
    cancels = [m for m in fake.received if m["type"] == "context.cancel"]
    assert len(cancels) == 1
    assert fake.connections == 1


async def test_tts_refused_reply_keeps_the_connection() -> None:
    error = {"code": "voice_unavailable", "message": "voice is not available on this model"}
    fake = FakeChariot(context_error=error)
    async with tts_server(fake) as tts:
        with pytest.raises(APIStatusError) as exc:
            await _speak(tts, "Hello there.")
        assert await _speak(tts, "Hello again.") > 0
    assert (exc.value.status_code, exc.value.retryable) == (422, False)
    assert fake.connections == 1


@pytest.mark.parametrize(
    ("refuse", "status"),
    [
        ({"code": 1008, "message": "unauthorized: missing, invalid, or expired API key"}, 401),
        ({"code": 4402, "message": "Workspace has 0 credits"}, 402),
    ],
)
async def test_tts_refused_connection_is_not_retried(refuse: dict[str, Any], status: int) -> None:
    fake = FakeChariot(refuse=refuse)
    async with tts_server(fake) as tts:
        with pytest.raises(APIStatusError) as exc:
            await _speak(tts, "Hello there.")
    assert (exc.value.status_code, exc.value.retryable) == (status, False)


async def test_tts_idle_connection_answers_the_server_pings() -> None:
    # the server closes a connection that misses a ping; one waiting in the pool between
    # replies must still answer, or the next reply would find it dead
    fake = FakeChariot(heartbeat=0.02)
    async with tts_server(fake) as tts:
        assert await _speak(tts, "Hello there.") > 0
        await asyncio.sleep(0.06)  # three missed pings would close it
        assert await _speak(tts, "Hello again.") > 0
    assert fake.connections == 1


async def test_tts_session_ending_error_is_reported_and_the_connection_dropped() -> None:
    error = {"code": "insufficient_credits", "message": "Workspace has 0 credits"}
    fake = FakeChariot(session_error=error)
    async with tts_server(fake) as tts:
        with pytest.raises(APIStatusError) as exc:
            await _speak(tts, "Hello there.")
        assert await _speak(tts, "Hello again.") > 0
    assert (exc.value.status_code, exc.value.retryable) == (402, False)
    assert fake.connections == 2


async def test_tts_keepalive_while_the_llm_is_slow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chariot.tts, "KEEPALIVE_INTERVAL", 0.01)
    fake = FakeChariot()
    async with tts_server(fake) as tts:
        stream = tts.stream(conn_options=NO_RETRY)
        stream.push_text("Let me check that")
        await wait_until(lambda: sum(m.get("text") == "" for m in fake.received) >= 3)
        stream.push_text(" for you.")
        stream.end_input()
        async with stream:
            assert sum([ev.frame.duration async for ev in stream]) > 0
    assert fake.flushed == ["Let me check that for you."]


async def test_tts_options_reach_the_connection_and_long_sentences_are_split() -> None:
    fake = FakeChariot()
    sentence = "word " * 150  # longer than one input.text may carry
    async with tts_server(fake, enable_logging=False, optimize_streaming_latency=2) as tts:
        await _speak(tts, sentence)
    assert fake.queries[0]["enable_logging"] == "false"
    assert fake.queries[0]["optimize_streaming_latency"] == "2"
    pieces = [m["text"] for m in fake.received if m["type"] == "input.text" and m["text"]]
    assert len(pieces) == 2 and max(map(len, pieces)) <= chariot.tts.MAX_FRAGMENT_CHARS
    assert "".join(pieces) == sentence.strip()


async def test_tts_synthesize_streams_pcm_over_http() -> None:
    fake = FakeChariot(chunks=50)
    async with tts_server(fake, voice="v-1", speed=0.75) as tts:
        frame = await tts.synthesize("Hello there.", conn_options=NO_RETRY).collect()
        fake.refuse = {"message": "Workspace has 0 credits"}
        with pytest.raises(APIStatusError) as exc:
            await tts.synthesize("Hello there.", conn_options=NO_RETRY).collect()
    assert frame.duration == pytest.approx(0.5, abs=0.02)
    assert fake.received[0] == {
        "voice_id": "v-1",
        "text": "Hello there.",
        "model_type": "chariot-multilingual-v1",
        "speed": 0.75,
    }
    assert (exc.value.status_code, exc.value.message) == (402, "Workspace has 0 credits")


async def test_stt_flush_commits_and_returns_the_final_transcript() -> None:
    fake = FakeChariot()
    async with stt_server(fake, language="hi-IN") as stt:
        assert stt.capabilities.manual_flush
        stream = stt.stream(conn_options=NO_RETRY)
        events: list[stt.SpeechEvent] = []

        async def read() -> None:
            async for ev in stream:
                events.append(ev)

        reader = asyncio.create_task(read())
        for frame in _frames(250):
            stream.push_frame(frame)
        stream.flush()
        stream.flush()  # nothing new while the commit is pending: its answer covers both
        finals = lambda: [  # noqa: E731
            ev.alternatives[0].text for ev in events if ev.type == SpeechEventType.FINAL_TRANSCRIPT
        ]
        await wait_until(lambda: finals() == ["segment 1"])
        stream.flush()  # nothing new and nothing pending: answered at once, without a commit
        stream.end_input()  # flushes as it closes; input.done collects that audio instead
        await reader
    assert finals() == ["segment 1", ""]
    assert [ev.type for ev in _without_usage(events)][:2] == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.INTERIM_TRANSCRIPT,
    ]
    assert fake.commits == 1
    assert fake.queries[0] == {
        "model_type": "chariot-1-asr-realtime",
        "audio_format": "pcm_16000",
        "mode": "balanced",
        "style": "codemix",
        "commit_strategy": "manual",
        "language": "hi",
    }
    # audio goes out in 100 ms chunks, the last one shorter
    sizes = [m for m in fake.received if isinstance(m, int)]
    assert sizes == [3200, 3200, 1600]
    assert fake.received[-1] == {"type": "input.done"}


async def test_stt_server_vad_lets_the_server_split_utterances() -> None:
    fake = FakeChariot()
    vad: chariot.VADOptions = {"silence_duration_ms": 600}
    async with stt_server(fake, server_vad=vad) as stt:
        assert not stt.capabilities.manual_flush
        stream = stt.stream(conn_options=NO_RETRY)
        for frame in _frames(100):
            stream.push_frame(frame)
        stream.flush()  # no commit: the server decides where utterances end
        stream.end_input()
        events = _without_usage([ev async for ev in stream])
    assert fake.commits == 0
    assert events[0].type == SpeechEventType.START_OF_SPEECH
    finals = [
        ev.alternatives[0].text for ev in events if ev.type == SpeechEventType.FINAL_TRANSCRIPT
    ]
    assert finals == ["the tail"]
    assert events[-1].type == SpeechEventType.END_OF_SPEECH
    query = fake.queries[0]
    assert (query["commit_strategy"], query["silence_duration_ms"], query["vad_events"]) == (
        "vad",
        "600",
        "true",
    )
    assert "language" not in query  # detected


async def test_stt_slow_session_start_is_retried() -> None:
    fake = FakeChariot(start_delay=0.06)
    async with stt_server(fake) as stt:
        options = APIConnectOptions(max_retry=1, retry_interval=0, timeout=0.03)
        stream = stt.stream(conn_options=options)
        for frame in _frames(100):
            stream.push_frame(frame)
        stream.end_input()
        events = [ev async for ev in stream]
    finals = [
        ev.alternatives[0].text for ev in events if ev.type == SpeechEventType.FINAL_TRANSCRIPT
    ]
    assert finals == ["the tail"]
    assert fake.connections == 2


async def test_stt_refused_session_is_not_retried() -> None:
    fake = FakeChariot(refuse={"code": 4402, "message": "insufficient credits"})
    async with stt_server(fake) as stt:
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=2, retry_interval=0))
        stream.end_input()
        with pytest.raises(APIStatusError) as exc:
            async for _ in stream:
                pass
    assert (exc.value.status_code, exc.value.retryable) == (402, False)
    assert fake.connections == 1


async def test_stt_recognize_uploads_the_frames_as_wav(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeChariot()
    async with stt_server(fake, include_timestamps=True) as stt:
        event = await stt.recognize(_frames(1000, rate=48000), conn_options=NO_RETRY)
        monkeypatch.setattr(chariot.stt, "MAX_RECOGNIZE_SECONDS", 0.5)
        with pytest.raises(APIStatusError) as exc:
            await stt.recognize(_frames(1000), conn_options=NO_RETRY)
    alt = event.alternatives[0]
    assert (alt.text, alt.start_time, alt.end_time) == ("namaste", 0.2, 0.7)
    assert [str(w) for w in alt.words or []] == ["namaste"]
    assert fake.uploads == [
        {
            "wav": (48000, 1, 48000),  # sent at the frames' own rate
            "model_type": "chariot-1-asr-realtime",
            "mode": "high_accuracy",
            "style": "codemix",
            "language": "none",
            "include_timestamps": "true",
        }
    ]
    assert (exc.value.status_code, exc.value.retryable) == (413, False)


def test_stt_rejects_languages_it_does_not_transcribe() -> None:
    with pytest.raises(ValueError, match="does not support"):
        chariot.STT(api_key="key", language="fr")
    assert chariot.STT(api_key="key", language="ta-IN")._opts.language == "ta"
