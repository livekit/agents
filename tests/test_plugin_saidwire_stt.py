"""Unit tests for livekit-plugins-saidwire.

The plugin runs against a fake Saidwire server on 127.0.0.1 (``FakeSaidwire`` below), which
speaks the ``/v1/listen`` protocol (https://saidwire.com/docs/api). There is no model: each
tone burst in the audio counts as one word, so transcripts and timings are deterministic. No
network calls leave the machine and no API key is needed.
"""

from __future__ import annotations

import asyncio
import io
import json
import time
import uuid
import wave
from dataclasses import dataclass, field

import aiohttp
import numpy as np
import pytest
from aiohttp import WSMsgType, web

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError, stt as lkstt
from livekit.plugins import saidwire

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------- fake server

WORDS = [
    "hello",
    "this",
    "is",
    "a",
    "test",
    "can",
    "you",
    "hear",
    "me",
    "yes",
    "i",
    "can",
    "stop",
    "now",
]
FRAME_S = 0.01  # energy frame
GAP_END_S = 0.06  # silence that ends a word (burst)
THR_DB = -30.0  # energy threshold (dBFS of a float signal in [-1, 1])
MODEL_INFO = {"name": "saidwire-en", "version": "fake", "arch": "streaming"}


@dataclass
class Conn:
    """What one stream did, for test assertions."""

    query: dict
    auth: str
    request_id: str
    texts: list = field(default_factory=list)  # text messages received (parsed)
    audio_bytes: int = 0
    sent: list = field(default_factory=list)  # messages sent
    close_code: int | None = None


class Recognizer:
    """Turns PCM into words, one per tone burst, with an endpointer on audio time."""

    def __init__(
        self,
        sr: int,
        endpointing_s: float | None,
        utterance_end_s: float | None,
        word_offset: int = 0,
    ):
        self.sr = sr
        self.frame = max(1, int(sr * FRAME_S))
        self.buf = np.zeros(0, dtype=np.float32)
        self.t = 0.0  # audio time of the next energy frame
        self.endpointing_s = (
            endpointing_s  # None = endpointing=false (finals at 1.0 s, speech_final false)
        )
        self.utterance_end_s = utterance_end_s
        self.burst_start: float | None = None
        self.last_loud: float | None = None
        self.utt: list[dict] = []  # committed words of the open utterance
        self.utt_start: float | None = None
        self.last_final_end = 0.0
        self.last_word_end: float | None = None
        self.utterance_end_sent = True
        self.n = word_offset

    def _word(self, s: float, e: float) -> dict:
        w = WORDS[self.n % len(WORDS)]
        self.n += 1
        return {
            "word": w,
            "start": round(s, 3),
            "end": round(e, 3),
            "confidence": 0.95,
            "punctuated_word": w.capitalize() if not self.utt else w,
        }

    def feed(self, pcm: bytes):
        """Yields events: ("speech_started", t), ("word", w), ("endpoint", speech_final), ("utterance_end", t)."""
        x = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
        self.buf = np.concatenate([self.buf, x])
        events = []
        while len(self.buf) >= self.frame:
            fr, self.buf = self.buf[: self.frame], self.buf[self.frame :]
            db = 10 * np.log10(float((fr * fr).mean()) + 1e-12)
            t0, self.t = self.t, self.t + FRAME_S
            if db > THR_DB:
                if self.burst_start is None:
                    self.burst_start = t0
                    if not self.utt:
                        self.utt_start = t0
                        events.append(("speech_started", t0))
                self.last_loud = self.t
            elif self.burst_start is not None and self.t - self.last_loud >= GAP_END_S:
                w = self._word(self.burst_start, self.last_loud)
                self.utt.append(w)
                self.last_word_end = self.last_loud
                self.utterance_end_sent = False
                self.burst_start = None
                events.append(("word", (w, list(self.utt), self.utt_start or 0.0, self.t)))
            if self.burst_start is None and self.utt and self.last_word_end is not None:
                pause = self.t - self.last_word_end
                ep = self.endpointing_s if self.endpointing_s is not None else 1.0
                if pause >= ep:
                    s, e, words = self.cut(False)
                    events.append(("final", (s, e, words, self.endpointing_s is not None)))
            if (
                self.utterance_end_s
                and not self.utterance_end_sent
                and not self.utt
                and self.burst_start is None
                and self.last_word_end is not None
                and self.t - self.last_word_end >= self.utterance_end_s
            ):
                self.utterance_end_sent = True
                events.append(("utterance_end", self.last_word_end))
        return events

    def cut(self, include_open_burst: bool) -> tuple[float, float, list]:
        """Close the open utterance; returns (start, end, words)."""
        if include_open_burst and self.burst_start is not None:
            w = self._word(self.burst_start, self.last_loud or self.t)
            self.utt.append(w)
            self.last_word_end = w["end"]
            self.utterance_end_sent = False
            self.burst_start = None
        start = self.last_final_end
        end = max(self.t, start)
        words, self.utt = self.utt, []
        self.last_final_end = end
        return start, end, words


def _bool(q, k, default=False):
    v = q.get(k)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes")


class FakeSaidwire:
    """aiohttp server; start() -> self.url (ws://127.0.0.1:PORT/v1/listen)."""

    def __init__(
        self,
        *,
        api_key: str = "sk_test_fake.secretsecretsecret",
        idle_timeout_s: float = 10.0,
        handshake_status: int | None = None,
        handshake_err_code: str = "SERVICE_OFFLINE",
        error_after_s: float | None = None,
        error_code: str = "ENGINE_UNAVAILABLE",
        error_close: int = 1011,
        final_delay_s: float = 0.0,
        error_on_first_n_streams: int | None = None,
        warming_refusals: int = 0,
        retry_after_s: int = 1,
    ):
        self.api_key = api_key
        self.idle_timeout_s = idle_timeout_s
        self.handshake_status = handshake_status
        self.handshake_err_code = handshake_err_code
        self.error_after_s = error_after_s
        self.error_code = error_code
        self.error_close = error_close
        self.error_on_first_n_streams = error_on_first_n_streams
        self.final_delay_s = final_delay_s
        self.warming_refusals = warming_refusals  # first N handshakes: 503 SERVICE_WARMING
        self.retry_after_s = retry_after_s
        self.conns: list[Conn] = []
        self.handshakes = 0
        self.batch_requests: list[dict] = []
        self._runner: web.AppRunner | None = None
        self.port = 0

    # ------------------------------------------------------------------ lifecycle
    async def start(self):
        app = web.Application(client_max_size=50 * 1024 * 1024)
        app.router.add_get("/v1/listen", self._ws)
        app.router.add_post("/v1/listen", self._batch)
        app.router.add_get("/health", self._health)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        return self

    @staticmethod
    async def _health(_req: web.Request):
        return web.json_response({"status": "ok"})

    async def stop(self):
        if self._runner:
            await self._runner.cleanup()

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/v1/listen"

    @property
    def http_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1/listen"

    # ------------------------------------------------------------------ helpers
    def _auth(self, req: web.Request) -> tuple[str, str | None]:
        h = req.headers.get("Authorization", "")
        for p in ("Token ", "Bearer ", "token ", "bearer "):
            if h.startswith(p):
                return h[len(p) :].strip(), None
        if req.query.get("token"):
            return req.query["token"], None
        proto = [
            s.strip() for s in req.headers.get("Sec-WebSocket-Protocol", "").split(",") if s.strip()
        ]
        if len(proto) == 2 and proto[0] == "token":
            return proto[1], "token"
        return "", None

    def _err(self, status: int, code: str, msg: str, rid: str):
        return web.json_response(
            {"err_code": code, "err_msg": msg, "request_id": rid},
            status=status,
            headers={"x-request-id": rid, "Retry-After": str(self.retry_after_s)},
        )

    # ------------------------------------------------------------------ streaming
    async def _ws(self, req: web.Request):
        rid = str(uuid.uuid4())
        self.handshakes += 1
        if self.handshakes <= self.warming_refusals:
            return self._err(503, "SERVICE_WARMING", "The service is starting up.", rid)
        if self.handshake_status:
            return self._err(self.handshake_status, self.handshake_err_code, "fake refusal", rid)
        key, proto = self._auth(req)
        if key != self.api_key:
            return self._err(401, "INVALID_AUTH", "Invalid credentials.", rid)
        q = dict(req.query)
        lang = q.get("language", "en").lower()
        if not lang.startswith("en"):
            return self._err(400, "UNSUPPORTED_LANGUAGE", "English only.", rid)
        enc = q.get("encoding", "linear16")
        if enc != "linear16":
            return self._err(400, "UNSUPPORTED_AUDIO", "fake supports linear16 only", rid)
        sr = int(q.get("sample_rate", "16000"))
        ep_raw = q.get("endpointing", "800")
        endpointing_s = None if ep_raw.lower() == "false" else max(500, int(float(ep_raw))) / 1000.0
        ue = q.get("utterance_end_ms")
        utterance_end_s = int(ue) / 1000.0 if ue else None
        interim = _bool(q, "interim_results")
        early = _bool(q, "early_words")
        vad_events = _bool(q, "vad_events")
        punct = _bool(q, "punctuate") or _bool(q, "smart_format")

        ws = web.WebSocketResponse(protocols=("token",) if proto else (), heartbeat=None)
        ws.headers["x-request-id"] = rid
        await ws.prepare(req)
        conn = Conn(query=q, auth=key, request_id=rid)
        self.conns.append(conn)
        rec = Recognizer(sr, endpointing_s, utterance_end_s)
        last_rx = time.monotonic()
        t_open = time.monotonic()
        audio_s = 0.0
        stream_no = len(self.conns)

        async def send(m):
            conn.sent.append(m)
            await ws.send_str(json.dumps(m))

        def words_out(words):
            return [
                dict(w, punctuated_word=w["punctuated_word"] if punct else w["word"]) for w in words
            ]

        def transcript(words):
            return " ".join((w["punctuated_word"] if punct else w["word"]) for w in words)

        async def results(words, start, end, is_final, speech_final=False, from_finalize=False):
            if self.final_delay_s and is_final:
                await asyncio.sleep(self.final_delay_s)
            await send(
                {
                    "type": "Results",
                    "channel_index": [0, 1],
                    "start": round(start, 3),
                    "duration": round(max(0.0, end - start), 3),
                    "is_final": is_final,
                    "speech_final": speech_final,
                    "from_finalize": from_finalize,
                    "channel": {
                        "alternatives": [
                            {
                                "transcript": transcript(words),
                                "confidence": 0.95,
                                "words": words_out(words),
                            }
                        ]
                    },
                    "metadata": {"request_id": rid, "model_info": MODEL_INFO, "model_uuid": "fake"},
                }
            )

        async def handle_events(events):
            for kind, val in events:
                if kind == "speech_started" and vad_events:
                    await send(
                        {"type": "SpeechStarted", "channel": [0, 1], "timestamp": round(val, 3)}
                    )
                elif kind == "word":
                    w, utt, u0, t_now = val
                    if early:
                        await send(
                            {
                                "type": "EarlyWord",
                                "channel_index": [0, 1],
                                "word": w["word"],
                                "punctuated_word": w["punctuated_word"] if punct else w["word"],
                                "start": w["start"],
                                "end": w["end"],
                                "confidence": 0.95,
                                "audio_end": round(t_now, 3),
                                "committed_by": "engine",
                            }
                        )
                    if interim:
                        await results(utt, u0, t_now, False)
                elif kind == "final":
                    s, e, words, speech_final = val
                    await results(words, s, e, True, speech_final=speech_final)
                elif kind == "utterance_end":
                    await send(
                        {"type": "UtteranceEnd", "channel": [0, 1], "last_word_end": round(val, 3)}
                    )

        async def idle_watch():
            while not ws.closed:
                await asyncio.sleep(0.05)
                if time.monotonic() - last_rx > self.idle_timeout_s:
                    await send(
                        {
                            "type": "Error",
                            "err_code": "IDLE_TIMEOUT",
                            "err_msg": "No audio or KeepAlive received within the timeout window.",
                            "description": "",
                            "request_id": rid,
                        }
                    )
                    conn.close_code = 1008
                    await ws.close(code=1008)
                    return
                if (
                    self.error_after_s is not None
                    and time.monotonic() - t_open > self.error_after_s
                    and (
                        self.error_on_first_n_streams is None
                        or stream_no <= self.error_on_first_n_streams
                    )
                ):
                    await send(
                        {
                            "type": "Error",
                            "err_code": self.error_code,
                            "err_msg": "fake error",
                            "description": "",
                            "request_id": rid,
                        }
                    )
                    conn.close_code = self.error_close
                    await ws.close(code=self.error_close)
                    return

        watcher = asyncio.create_task(idle_watch())

        async def close_stream():
            s, e, words = rec.cut(True)
            if words:
                await results(words, s, e, True, speech_final=False)
            await send(
                {
                    "type": "Metadata",
                    "request_id": rid,
                    "created": "2026-10-28T00:00:00+00:00",
                    "duration": round(audio_s, 3),
                    "channels": 1,
                    "models": ["fake"],
                    "model_info": {"fake": MODEL_INFO},
                }
            )
            conn.close_code = 1000
            await ws.close(code=1000)

        try:
            async for msg in ws:
                last_rx = time.monotonic()
                if msg.type == WSMsgType.BINARY:
                    if not msg.data:
                        await close_stream()
                        break
                    conn.audio_bytes += len(msg.data)
                    audio_s += len(msg.data) / 2 / sr
                    await handle_events(rec.feed(msg.data))
                elif msg.type == WSMsgType.TEXT:
                    try:
                        m = json.loads(msg.data)
                    except ValueError:
                        m = {"type": "?"}
                    conn.texts.append(m)
                    typ = m.get("type")
                    if typ == "KeepAlive":
                        pass
                    elif typ == "Finalize":
                        s, e, words = rec.cut(True)
                        await results(words, s, e, True, speech_final=False, from_finalize=True)
                    elif typ == "CloseStream":
                        await close_stream()
                        break
                    else:
                        await send(
                            {
                                "type": "Error",
                                "err_code": "BAD_MESSAGE",
                                "err_msg": f"unknown {typ}",
                                "description": "",
                                "request_id": rid,
                            }
                        )
        finally:
            watcher.cancel()
        return ws

    # ------------------------------------------------------------------ pre-recorded
    async def _batch(self, req: web.Request):
        rid = str(uuid.uuid4())
        key, _ = self._auth(req)
        if key != self.api_key:
            return self._err(401, "INVALID_AUTH", "Invalid credentials.", rid)
        body = await req.read()
        self.batch_requests.append(
            {
                "query": dict(req.query),
                "bytes": len(body),
                "content_type": req.headers.get("Content-Type", ""),
            }
        )
        if body[:4] == b"RIFF":
            with wave.open(io.BytesIO(body)) as w:
                sr, pcm = w.getframerate(), w.readframes(w.getnframes())
        else:
            sr, pcm = int(req.query.get("sample_rate", "16000")), body
        rec = Recognizer(sr, 0.5, None)
        words = []
        for kind, val in rec.feed(pcm):
            if kind == "final":
                words += val[2]
        words += rec.cut(True)[2]
        punct = _bool(req.query, "punctuate") or _bool(req.query, "smart_format")
        text = " ".join((w["punctuated_word"] if punct else w["word"]) for w in words)
        dur = len(pcm) / 2 / sr
        return web.json_response(
            {
                "metadata": {
                    "request_id": rid,
                    "duration": round(dur, 3),
                    "channels": 1,
                    "models": ["fake"],
                    "model_info": {"fake": dict(MODEL_INFO, arch="batch")},
                },
                "results": {
                    "channels": [
                        {"alternatives": [{"transcript": text, "confidence": 0.95, "words": words}]}
                    ]
                },
            },
            headers={"x-request-id": rid},
        )


# ---------------------------------------------------------------------- audio helpers for tests
def tone_pcm(
    sr: int,
    n_words: int,
    word_s: float = 0.25,
    gap_s: float = 0.15,
    lead_s: float = 0.2,
    tail_s: float = 0.0,
    amp: float = 0.3,
) -> bytes:
    """n_words tone bursts (one fake word each), with leading silence and optional trailing silence."""
    parts = [np.zeros(int(sr * lead_s), dtype=np.float32)]
    t = np.arange(int(sr * word_s)) / sr
    burst = (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    for i in range(n_words):
        parts.append(burst)
        if i < n_words - 1:
            parts.append(np.zeros(int(sr * gap_s), dtype=np.float32))
    parts.append(np.zeros(int(sr * tail_s), dtype=np.float32))
    x = np.concatenate(parts)
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def silence_pcm(sr: int, seconds: float) -> bytes:
    return bytes(int(sr * seconds) * 2)


def wav_bytes(pcm: bytes, sr: int) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return b.getvalue()


# ---------------------------------------------------------------------- tests

KEY = "sk_test_fake.secretsecretsecret"
E = lkstt.SpeechEventType
FAST_RETRY = APIConnectOptions(max_retry=3, retry_interval=0.1, timeout=5.0)


def frames(pcm: bytes, sr: int = 16000, ms: int = 20) -> list[rtc.AudioFrame]:
    n = sr * ms // 1000
    out = []
    for i in range(0, len(pcm) // 2, n):
        chunk = pcm[i * 2 : (i + n) * 2]
        out.append(
            rtc.AudioFrame(
                data=chunk, sample_rate=sr, num_channels=1, samples_per_channel=len(chunk) // 2
            )
        )
    return out


@pytest.fixture
async def server():
    s = await FakeSaidwire().start()
    yield s
    await s.stop()


@pytest.fixture
async def session():
    async with aiohttp.ClientSession() as s:
        yield s


async def run_stream(stream, pcm, sr=16000, before_end=None):
    async def feed():
        for f in frames(pcm, sr):
            stream.push_frame(f)
        if before_end:
            await before_end(stream)
        stream.end_input()

    t = asyncio.create_task(feed())
    events = [ev async for ev in stream]
    await t
    return events


def finals(events):
    """Non-empty finals (end_input() flushes, and an empty flush answer is an empty final)."""
    return [
        e.alternatives[0].text
        for e in events
        if e.type == E.FINAL_TRANSCRIPT and e.alternatives[0].text
    ]


def kinds(events):
    return [
        e.type
        for e in events
        if e.type != E.RECOGNITION_USAGE
        and not (e.type == E.FINAL_TRANSCRIPT and not e.alternatives[0].text)
    ]


async def test_interim_final_endpointing_and_speech_events(server, session):
    s = saidwire.STT(api_key=KEY, base_url=server.url, http_session=session, endpointing_ms=500)
    ev = await run_stream(s.stream(), tone_pcm(16000, 3) + silence_pcm(16000, 0.8))
    assert ev[-1].type == E.FINAL_TRANSCRIPT and ev[-1].alternatives[0].text == ""
    assert kinds(ev) == [
        E.START_OF_SPEECH,
        E.INTERIM_TRANSCRIPT,
        E.INTERIM_TRANSCRIPT,
        E.INTERIM_TRANSCRIPT,
        E.FINAL_TRANSCRIPT,
        E.END_OF_SPEECH,
    ]
    assert finals(ev) == ["Hello this is"]
    fin = next(e for e in ev if e.type == E.FINAL_TRANSCRIPT and e.alternatives[0].text)
    alt = fin.alternatives[0]
    assert list(alt.words or []) == ["Hello", "this", "is"]
    assert alt.start_time == pytest.approx(0.2, abs=0.05)
    assert alt.end_time == pytest.approx(1.25, abs=0.05)
    usage = sum(e.recognition_usage.audio_duration for e in ev if e.type == E.RECOGNITION_USAGE)
    assert usage == pytest.approx(0.2 + 0.25 * 3 + 0.15 * 2 + 0.8, abs=0.06)
    q = server.conns[0].query
    assert q["encoding"] == "linear16" and q["sample_rate"] == "16000" and q["channels"] == "1"
    assert q["endpointing"] == "500" and q["model"] == saidwire.DEFAULT_MODEL
    assert q["language"] == "en" and q["interim_results"] == "true" and q["punctuate"] == "true"
    assert server.conns[0].auth == KEY
    assert server.conns[0].texts[-1] == {"type": "CloseStream"}
    assert server.conns[0].close_code == 1000


async def test_default_query(server, session):
    s = saidwire.STT(api_key=KEY, base_url=server.url, http_session=session)
    await run_stream(s.stream(), silence_pcm(16000, 0.1))
    q = server.conns[0].query
    assert "endpointing" not in q  # server default
    assert "early_words" not in q and "vad_events" not in q and "utterance_end_ms" not in q
    assert saidwire.DEFAULT_URL == "wss://api.saidwire.com/v1/listen"


async def test_flush_sends_finalize_and_returns_final_at_once(server, session):
    s = saidwire.STT(api_key=KEY, base_url=server.url, http_session=session, endpointing_ms=0)

    async def flush(st):
        st.flush()
        await asyncio.sleep(0.3)

    ev = await run_stream(
        s.stream(), tone_pcm(16000, 2) + silence_pcm(16000, 0.2), before_end=flush
    )
    assert finals(ev) == ["Hello this"]
    assert server.conns[0].query["endpointing"] == "false"
    assert {"type": "Finalize"} in server.conns[0].texts
    fin = [m for m in server.conns[0].sent if m.get("is_final")]
    assert fin[0]["from_finalize"] is True


async def test_empty_flush_still_emits_empty_final(server, session):
    s = saidwire.STT(api_key=KEY, base_url=server.url, http_session=session)

    async def flush(st):
        st.flush()
        await asyncio.sleep(0.3)

    ev = await run_stream(s.stream(), silence_pcm(16000, 0.4), before_end=flush)
    fin = [e for e in ev if e.type == E.FINAL_TRANSCRIPT]
    assert len(fin) == 2 and all(f.alternatives[0].text == "" for f in fin)
    assert not [e for e in ev if e.type == E.START_OF_SPEECH]


async def test_early_words_vad_events_utterance_end(server, session):
    s = saidwire.STT(
        api_key=KEY,
        base_url=server.url,
        http_session=session,
        early_words=True,
        vad_events=True,
        utterance_end_ms=1000,
        interim_results=False,
        endpointing_ms=500,
    )
    early = []
    s.on("early_word", lambda w: early.append(w["word"]))
    ev = await run_stream(s.stream(), tone_pcm(16000, 4) + silence_pcm(16000, 1.3))
    assert early == ["hello", "this", "is", "a"]
    assert kinds(ev) == [E.START_OF_SPEECH, E.FINAL_TRANSCRIPT, E.END_OF_SPEECH]
    assert not s.capabilities.interim_results
    q = server.conns[0].query
    assert q["early_words"] == "true" and q["vad_events"] == "true"
    assert q["utterance_end_ms"] == "1000"


async def test_bad_key_fails_fast_without_retries(server, session):
    s = saidwire.STT(api_key="sk_live_wrong.key", base_url=server.url, http_session=session)
    errors = []
    s.on("error", lambda e: errors.append(e))
    with pytest.raises(APIStatusError) as ei:
        await run_stream(s.stream(conn_options=FAST_RETRY), tone_pcm(16000, 1))
    assert ei.value.status_code == 401 and ei.value.retryable is False
    assert "invalid API key" in str(ei.value) and "sk_live_wrong" not in str(ei.value)
    assert server.handshakes == 1
    assert errors and errors[0].recoverable is False


async def test_waits_while_service_is_starting_up(session):
    s_ = await FakeSaidwire(warming_refusals=2, retry_after_s=1).start()
    try:
        s = saidwire.STT(api_key=KEY, base_url=s_.url, http_session=session, endpointing_ms=500)
        stream = s.stream(conn_options=FAST_RETRY)

        async def feed():
            # audio during the wait is dropped, not sent late in a burst
            for f in frames(tone_pcm(16000, 3)):
                stream.push_frame(f)
            await asyncio.sleep(2.5)
            for f in frames(tone_pcm(16000, 2) + silence_pcm(16000, 0.8)):
                stream.push_frame(f)
            stream.end_input()

        t = asyncio.create_task(feed())
        ev = [e async for e in stream]
        await t
        assert s_.handshakes == 3 and len(s_.conns) == 1
        assert finals(ev) == ["Hello this"]
    finally:
        await s_.stop()


async def test_gives_up_after_warmup_timeout(session):
    s_ = await FakeSaidwire(warming_refusals=100, retry_after_s=1).start()
    try:
        s = saidwire.STT(api_key=KEY, base_url=s_.url, http_session=session, warmup_timeout=1.5)
        stream = s.stream(conn_options=FAST_RETRY)

        async def feed():
            await asyncio.sleep(3.0)
            stream.end_input()

        t = asyncio.create_task(feed())
        with pytest.raises(APIStatusError) as ei:
            async for _ in stream:
                pass
        t.cancel()
        assert ei.value.status_code == 503 and ei.value.retryable is False
        assert "not ready" in str(ei.value)
        assert s_.handshakes == 2
        await stream.aclose()
    finally:
        await s_.stop()


async def test_server_error_on_handshake_is_retried(session):
    s_ = await FakeSaidwire(handshake_status=500, handshake_err_code="INTERNAL").start()
    try:
        s = saidwire.STT(api_key=KEY, base_url=s_.url, http_session=session)
        with pytest.raises(APIConnectionError):
            await run_stream(s.stream(conn_options=FAST_RETRY), tone_pcm(16000, 1))
        assert s_.handshakes == 4  # 1 + max_retry
    finally:
        await s_.stop()


async def test_midstream_server_error_reconnects(session):
    s_ = await FakeSaidwire(
        error_after_s=0.3,
        error_code="ENGINE_UNAVAILABLE",
        error_close=1011,
        error_on_first_n_streams=1,
    ).start()
    try:
        s = saidwire.STT(api_key=KEY, base_url=s_.url, http_session=session, endpointing_ms=500)
        stream = s.stream(conn_options=FAST_RETRY)
        errors = []
        s.on("error", lambda e: errors.append(e))

        async def feed():
            await asyncio.sleep(1.0)  # the first connection dies at 0.3 s
            for f in frames(tone_pcm(16000, 2) + silence_pcm(16000, 0.8)):
                stream.push_frame(f)
            stream.end_input()

        t = asyncio.create_task(feed())
        ev = [e async for e in stream]
        await t
        assert len(s_.conns) == 2 and s_.conns[0].close_code == 1011
        assert finals(ev) == ["Hello this"]
        assert errors and errors[0].recoverable is True
    finally:
        await s_.stop()


async def test_quota_exceeded_midstream_is_permanent(session):
    s_ = await FakeSaidwire(
        error_after_s=0.2, error_code="QUOTA_EXCEEDED", error_close=1008
    ).start()
    try:
        s = saidwire.STT(api_key=KEY, base_url=s_.url, http_session=session)
        stream = s.stream(conn_options=FAST_RETRY)
        with pytest.raises(APIStatusError) as ei:
            async for _ in stream:
                pass
        assert ei.value.status_code == 402 and ei.value.retryable is False
        assert len(s_.conns) == 1
        await stream.aclose()
    finally:
        await s_.stop()


async def test_keepalive_holds_idle_stream_open(session):
    s_ = await FakeSaidwire(idle_timeout_s=4.5).start()
    try:
        s = saidwire.STT(api_key=KEY, base_url=s_.url, http_session=session)
        stream = s.stream()

        async def idle_then_end():
            await asyncio.sleep(6.5)
            stream.end_input()

        t = asyncio.create_task(idle_then_end())
        _ = [e async for e in stream]
        await t
        c = s_.conns[0]
        assert {"type": "KeepAlive"} in c.texts and c.close_code == 1000 and len(s_.conns) == 1
        assert c.audio_bytes == 0
    finally:
        await s_.stop()


async def test_resamples_48k_input_to_16k(server, session):
    s = saidwire.STT(api_key=KEY, base_url=server.url, http_session=session, endpointing_ms=500)
    ev = await run_stream(s.stream(), tone_pcm(48000, 2) + silence_pcm(48000, 0.8), sr=48000)
    assert server.conns[0].query["sample_rate"] == "16000"
    expected = 2 * 16000 * (0.2 + 0.5 + 0.15 + 0.8)
    assert server.conns[0].audio_bytes == pytest.approx(expected, rel=0.03)
    assert finals(ev) == ["Hello this"]


async def test_telephony_8k(server, session):
    s = saidwire.STT(
        api_key=KEY, base_url=server.url, http_session=session, sample_rate=8000, endpointing_ms=500
    )
    ev = await run_stream(s.stream(), tone_pcm(8000, 2) + silence_pcm(8000, 0.8), sr=8000)
    assert server.conns[0].query["sample_rate"] == "8000"
    assert finals(ev) == ["Hello this"]


async def test_update_options_reconnects(server, session):
    s = saidwire.STT(api_key=KEY, base_url=server.url, http_session=session)
    stream = s.stream()

    async def feed():
        await asyncio.sleep(0.2)
        s.update_options(endpointing_ms=650)
        await asyncio.sleep(0.3)
        for f in frames(tone_pcm(16000, 1) + silence_pcm(16000, 0.9)):
            stream.push_frame(f)
        stream.end_input()

    t = asyncio.create_task(feed())
    ev = [e async for e in stream]
    await t
    assert len(server.conns) == 2 and server.conns[1].query["endpointing"] == "650"
    assert finals(ev) == ["Hello"]


async def test_prerecorded_recognize(server, session):
    s = saidwire.STT(api_key=KEY, base_url=server.url, http_session=session)
    ev = await s.recognize(frames(tone_pcm(16000, 3) + silence_pcm(16000, 0.3)))
    assert ev.type == E.FINAL_TRANSCRIPT and ev.alternatives[0].text == "Hello this is"
    req = server.batch_requests[0]
    assert req["content_type"] == "audio/wav" and req["query"]["punctuate"] == "true"
    assert req["query"]["model"] == saidwire.DEFAULT_MODEL


def test_capabilities_and_validation(monkeypatch):
    monkeypatch.delenv("SAIDWIRE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        saidwire.STT()
    with pytest.raises(ValueError):
        saidwire.STT(api_key=KEY, language="de")
    monkeypatch.setenv("SAIDWIRE_API_KEY", KEY)
    s = saidwire.STT()
    c = s.capabilities
    assert c.streaming and c.interim_results and c.manual_flush and c.offline_recognize
    assert c.aligned_transcript == "word"
    assert s.provider == "Saidwire" and s.model == "saidwire-en-2026-10-30"
    assert saidwire.STT(base_url="https://api.saidwire.com/v1/listen")._opts.base_url == (
        "wss://api.saidwire.com/v1/listen"
    )
