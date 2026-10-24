"""Tests for the Paxa TTS plugin, run against a local stand-in for the Paxa API."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import struct
import time
import traceback
from collections.abc import AsyncIterator, Callable
from typing import Any
from unittest.mock import patch

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
)
from livekit.plugins import paxa
from livekit.plugins.paxa._utils import (
    check_pcm_content_type,
    http_status_error,
    live_close_error,
    live_event_error,
    normalize_base_url,
    take_frame,
)
from livekit.plugins.paxa.tts import _FrameGate

pytestmark = pytest.mark.unit

API_KEY = "pxa_test_secret_key"
CLIENT = f"livekit-plugins-paxa/{paxa.__version__}"
SAMPLE_RATE = 24000
_SECRET = "customer-secret-marker"
# Paxa's live server starts a leg when its buffer ends a sentence or a frame carries a
# newline, or once the buffer reaches 400 characters.
_SENTENCE_END = re.compile(r"[.!?…。！？][\"'”’」』)）]*\s*$")
# Labels for a POST /v1/tts reply that is not the mono pcm a 16 kHz request asked for.
_MISLABELED = {
    "rate_ignored": "audio/pcm;rate=24000;channels=1",
    "stereo": "audio/pcm;rate={rate};channels=2",
    "no_rate": "audio/pcm;channels=1",
    "wav": "audio/wav",
}


def _pcm(samples: int, seed: int = 0) -> bytes:
    """Distinct little-endian int16 PCM, so a lost, extra or reordered byte shows up."""
    return struct.pack(f"<{samples}h", *((((i + seed) * 97) % 2000) - 1000 for i in range(samples)))


def _odd_chunks(data: bytes) -> list[bytes]:
    """Cut audio at odd byte offsets, so chunks split samples as a network can."""
    return [data[:1001], data[1001:3001], data[3001:]]


class _LiveSession:
    """One ``/v1/tts/live`` connection, handled as Paxa does: frames run one at a time,
    and each leg sends ``charged``, its pcm audio, then ``flushed``."""

    def __init__(self, script: dict[str, Any]) -> None:
        self.script = script
        self.frames: list[dict[str, Any]] = []
        self.legs: list[str] = []
        self.pcm: list[bytes] = []
        self.max_pending = 0
        self.leg_in_flight = False
        self.closed_during_leg = False
        self.closed_at: float | None = None
        self._pending = 0
        self._buffer = ""
        self._seq = 0
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.transport: asyncio.Transport | None = None

    async def serve(self, ws: web.WebSocketResponse) -> None:
        worker = asyncio.create_task(self._work(ws))
        try:
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                frame = json.loads(msg.data)
                self.frames.append(frame)
                if self._pending >= 32:
                    # Paxa's limit on frames queued behind a leg.
                    await ws.close(code=4413)
                    break
                self._pending += 1
                self.max_pending = max(self.max_pending, self._pending)
                self._queue.put_nowait(frame)
        finally:
            self.closed_at = time.monotonic()
            self.closed_during_leg = self.leg_in_flight
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError, ConnectionError):
                await worker

    async def _work(self, ws: web.WebSocketResponse) -> None:
        while True:
            frame = await self._queue.get()
            try:
                if not await self._handle(ws, frame):
                    return
            finally:
                self._pending -= 1

    async def _handle(self, ws: web.WebSocketResponse, frame: dict[str, Any]) -> bool:
        if self.script.get("silent"):
            return True
        kind = frame["type"]
        if kind == "start":
            if code := self.script.get("start_error"):
                await ws.send_json({"type": "error", "code": code})
                await ws.close(code=4400)
                return False
            await ws.send_json(
                {
                    "type": "started",
                    "connection": "conn_paxa_test",
                    "model": frame["model"],
                    "voice": frame["voice"],
                }
            )
            return True
        if kind == "text":
            self._buffer += frame["text"]
            if (
                "\n" in frame["text"]
                or _SENTENCE_END.search(self._buffer)
                or len(self._buffer) >= 400
            ):
                return await self._leg(ws)
            return True
        if kind == "end":
            if self._buffer and not await self._leg(ws):
                return False
            if not self.script.get("skip_done"):
                await ws.send_json({"type": "done", "total_chars": 1, "total_credits": 0.1})
            await ws.close(code=1000)
            return False
        return True

    async def _leg(self, ws: web.WebSocketResponse) -> bool:
        self._seq += 1
        seq = self._seq
        text, self._buffer = self._buffer, ""
        self.legs.append(text)
        self.leg_in_flight = True
        try:
            await ws.send_json(
                {
                    "type": "charged",
                    "seq": seq,
                    "chars": len(text),
                    "credits": 0.1,
                    "total_credits": 0.1 * seq,
                }
            )
            if close_code := self.script.get("close_on_leg", {}).get(seq):
                await ws.close(code=close_code)
                return False
            if close_code := self.script.get("close_and_drop_on_leg", {}).get(seq):
                # As Paxa does: the close frame, then the connection drops at once.
                assert ws._writer is not None and self.transport is not None
                await ws._writer.close(close_code)
                self.transport.abort()
                return False
            if code := self.script.get("error_on_leg", {}).get(seq):
                await ws.send_json({"type": "error", "code": code, "seq": seq})
                return True
            await asyncio.sleep(self.script.get("leg_delay", 0.0))

            pcm = _pcm(2400, seed=seq * 1000)
            self.pcm.append(pcm)
            chunks = _odd_chunks(pcm)
            for chunk in chunks[:-2]:
                await ws.send_bytes(chunk)
            if code := self.script.get("error_after_audio_on_leg", {}).get(seq):
                # Let the audio reach the listener before the leg fails, as it does live.
                await asyncio.sleep(0.05)
                await ws.send_json({"type": "error", "code": code, "seq": seq})
                return True
            if self.script.get("stall_mid_leg"):
                await asyncio.sleep(30)
            for chunk in chunks[-2:]:
                await ws.send_bytes(chunk)
            await ws.send_json({"type": "flushed", "seq": seq})
            return True
        finally:
            self.leg_in_flight = False


class _FakePaxa:
    """Local stand-in for ``POST /v1/tts`` and ``/v1/tts/live`` on api.paxalabs.com."""

    def __init__(self) -> None:
        self.http_requests: list[tuple[dict[str, str], dict[str, Any]]] = []
        # Queued replies for POST /v1/tts: "ok", "truncate", "json", a key of
        # _MISLABELED, or a (status, title) problem. "ok" once the queue is empty.
        self.http_replies: list[Any] = []
        self.http_pcm = _pcm(4800)
        self.upgrades: list[dict[str, str]] = []
        self.refuse_upgrade: int | None = None
        # One script per live connection, in order; an empty script once they run out.
        self.live_scripts: list[dict[str, Any]] = []
        self.sessions: list[_LiveSession] = []

    async def start(self) -> None:
        app = web.Application()
        app.router.add_post("/v1/tts", self._tts)
        app.router.add_get("/v1/tts/live", self._live)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = self._runner.addresses[0][1]
        self.base_url = f"http://127.0.0.1:{port}"
        self.session = aiohttp.ClientSession()

    async def stop(self) -> None:
        await self.session.close()
        await self._runner.cleanup()

    def tts(self, **kwargs: Any) -> paxa.TTS:
        return paxa.TTS(
            api_key=API_KEY, base_url=self.base_url, http_session=self.session, **kwargs
        )

    async def _tts(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.http_requests.append((dict(request.headers), body))
        reply = self.http_replies.pop(0) if self.http_replies else "ok"
        if isinstance(reply, tuple):
            status, title = reply
            return web.json_response(
                {"title": title, "status": status},
                status=status,
                content_type="application/problem+json",
                headers={"x-request-id": "req_failed"},
            )
        if reply == "json":
            return web.json_response({"title": "not audio"})

        # Paxa labels pcm with the rate the request chose, 24 kHz by default.
        rate = body.get("sample_rate", SAMPLE_RATE)
        content_type = _MISLABELED.get(reply, "audio/pcm;rate={rate};channels=1").format(rate=rate)
        resp = web.StreamResponse(headers={"Content-Type": content_type, "x-request-id": "req_ok"})
        resp.enable_chunked_encoding()
        await resp.prepare(request)
        if reply == "truncate":
            await resp.write(self.http_pcm[:2001])
            assert request.transport is not None
            request.transport.close()
            return resp
        for chunk in _odd_chunks(self.http_pcm):
            await resp.write(chunk)
        await resp.write_eof()
        return resp

    async def _live(self, request: web.Request) -> web.StreamResponse:
        self.upgrades.append(dict(request.headers))
        if self.refuse_upgrade is not None:
            return web.json_response(
                {"title": "concurrency_limited", "status": self.refuse_upgrade},
                status=self.refuse_upgrade,
                content_type="application/problem+json",
            )
        session = _LiveSession(self.live_scripts.pop(0) if self.live_scripts else {})
        self.sessions.append(session)
        session.transport = request.transport
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await session.serve(ws)
        return ws


@pytest.fixture
async def paxa_api() -> AsyncIterator[_FakePaxa]:
    server = _FakePaxa()
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


def _options(**kwargs: Any) -> APIConnectOptions:
    return APIConnectOptions(**{"max_retry": 0, "retry_interval": 0.0, "timeout": 5.0, **kwargs})


async def _drain(stream: Any, sample_rate: int = SAMPLE_RATE) -> tuple[bytes, set[str]]:
    audio = bytearray()
    segments: set[str] = set()
    try:
        async for ev in stream:
            assert ev.frame.sample_rate == sample_rate
            assert ev.frame.num_channels == 1
            audio.extend(ev.frame.data.tobytes())
            segments.add(ev.segment_id)
    finally:
        await stream.aclose()
    return bytes(audio), segments


def _assert_audio_is(audio: bytes, expected: bytes) -> None:
    """Every PCM byte arrives in order; the only surplus allowed is the short silent
    marker the emitter may end a segment with."""
    assert audio[: len(expected)] == expected
    assert set(audio[len(expected) :]) <= {0}


def _assert_safe(exc: BaseException) -> None:
    rendered = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    for secret in (_SECRET, API_KEY):
        assert secret not in str(exc)
        assert secret not in repr(exc)
        assert secret not in rendered


async def _eventually(condition: Callable[[], bool], timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        await asyncio.sleep(0.01)


async def _speak(tts: paxa.TTS, text: str, **conn: Any) -> tuple[bytes, set[str]]:
    """Push text the way an LLM streams it, a few characters at a time."""
    stream = tts.stream(conn_options=_options(**conn))
    for i in range(0, len(text), 4):
        stream.push_text(text[i : i + 4])
    stream.end_input()
    return await _drain(stream, tts.sample_rate)


# Configuration


def test_requires_api_key() -> None:
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="PAXA_API_KEY"):
            paxa.TTS()


def test_reads_api_key_and_base_url_from_env() -> None:
    env = {"PAXA_API_KEY": "env-key", "PAXA_BASE_URL": "https://paxa.example/v1/"}
    with patch.dict("os.environ", env, clear=True):
        tts = paxa.TTS()
    assert tts._opts.api_key == "env-key"
    assert tts._opts.base_url == "https://paxa.example"


def test_defaults() -> None:
    tts = paxa.TTS(api_key=API_KEY)
    assert tts.model == "paxa-tts-flash-v2"
    assert tts.provider == "Paxa"
    assert tts.sample_rate == SAMPLE_RATE
    assert tts.num_channels == 1
    assert tts.capabilities.streaming is True
    assert tts.capabilities.aligned_transcript is False
    assert tts._opts.voice == "foithong"
    assert tts._opts.language == "auto"
    assert tts._opts.speed == 1.0
    assert tts._opts.sample_rate == SAMPLE_RATE
    assert tts._opts.base_url == "https://api.paxalabs.com"


@pytest.mark.parametrize(
    ("options", "field"),
    [
        ({"speed": 0.45}, "speed"),
        ({"speed": 1.6}, "speed"),
        ({"language": "th-TH"}, "language"),
        ({"voice": "  "}, "voice"),
        ({"model": ""}, "model"),
        ({"sample_rate": 12000}, "sample_rate"),
        ({"base_url": "api.paxalabs.com"}, "base_url"),
    ],
)
def test_rejects_invalid_options(options: dict[str, Any], field: str) -> None:
    with pytest.raises(ValueError, match=field):
        paxa.TTS(api_key=API_KEY, **options)


def test_sample_rate_must_be_served_by_the_model() -> None:
    v2_rates = [8000, 16000, 22050, 24000, 32000, 44100, 48000]
    for rate in v2_rates:
        assert paxa.TTS(api_key=API_KEY, sample_rate=rate).sample_rate == rate

    # Paxa TTS Flash serves 24 kHz only.
    assert paxa.TTS(api_key=API_KEY, model="paxa-tts-flash-v1").sample_rate == 24000
    for rate in v2_rates:
        if rate != 24000:
            with pytest.raises(ValueError, match="one of 24000 on paxa-tts-flash-v1, got"):
                paxa.TTS(api_key=API_KEY, model="paxa-tts-flash-v1", sample_rate=rate)

    # A model the plugin doesn't list takes any rate Paxa serves; Paxa decides the rest.
    assert paxa.TTS(api_key=API_KEY, model="paxa-tts-next", sample_rate=22050).sample_rate == 22050
    with pytest.raises(ValueError, match="sample_rate"):
        paxa.TTS(api_key=API_KEY, model="paxa-tts-next", sample_rate=12000)


def test_update_options_keeps_a_sample_rate_the_model_serves() -> None:
    tts = paxa.TTS(api_key=API_KEY, sample_rate=16000)
    before = tts._opts
    with pytest.raises(ValueError, match="sample_rate"):
        tts.update_options(model="paxa-tts-flash-v1")
    assert tts._opts is before
    assert tts.sample_rate == 16000


def test_accepts_any_voice_id() -> None:
    tts = paxa.TTS(api_key=API_KEY, voice="voice_0123456789abcdef0123456789abcdef")
    assert tts._opts.voice == "voice_0123456789abcdef0123456789abcdef"


def test_update_options_applies_atomically() -> None:
    tts = paxa.TTS(api_key=API_KEY)
    tts.update_options(model="paxa-tts-flash-v1", voice="massaman", language="th", speed=1.2)
    assert tts.model == "paxa-tts-flash-v1"
    assert (tts._opts.voice, tts._opts.language, tts._opts.speed) == ("massaman", "th", 1.2)

    before = tts._opts
    with pytest.raises(ValueError, match="speed"):
        tts.update_options(voice="foithong", speed=2.0)
    assert tts._opts is before


def test_normalize_base_url_accepts_openai_style_url() -> None:
    assert normalize_base_url("https://api.paxalabs.com/v1") == "https://api.paxalabs.com"
    assert normalize_base_url(" https://api.paxalabs.com/ ") == "https://api.paxalabs.com"


# synthesize(): POST /v1/tts


async def test_synthesize_streams_pcm_into_frames(paxa_api: _FakePaxa) -> None:
    tts = paxa_api.tts(voice="massaman", language="th", speed=1.25)
    stream = tts.synthesize("สวัสดีครับ Paxa", conn_options=_options())
    audio, _ = await _drain(stream)
    await tts.aclose()

    # The fake splits samples across chunks; every byte still arrives in order.
    _assert_audio_is(audio, paxa_api.http_pcm)
    headers, body = paxa_api.http_requests[0]
    assert body == {
        "text": "สวัสดีครับ Paxa",
        "voice": "massaman",
        "model": "paxa-tts-flash-v2",
        "format": "pcm",
        "sample_rate": 24000,
        "stream": True,
        "speed": 1.25,
        "language": "th",
    }
    assert headers["Authorization"] == f"Bearer {API_KEY}"
    assert headers["x-paxa-client"] == CLIENT
    assert headers["User-Agent"] == CLIENT


@pytest.mark.parametrize("rate", [8000, 16000, 48000])
async def test_synthesize_requests_the_chosen_sample_rate(paxa_api: _FakePaxa, rate: int) -> None:
    tts = paxa_api.tts(sample_rate=rate)
    assert tts.sample_rate == rate
    audio, _ = await _drain(tts.synthesize("Hello there.", conn_options=_options()), rate)
    await tts.aclose()

    _assert_audio_is(audio, paxa_api.http_pcm)
    assert paxa_api.http_requests[0][1]["sample_rate"] == rate


@pytest.mark.parametrize("reply", list(_MISLABELED))
async def test_synthesize_refuses_audio_it_did_not_ask_for(paxa_api: _FakePaxa, reply: str) -> None:
    paxa_api.http_replies.append(reply)
    tts = paxa_api.tts(sample_rate=16000)
    with pytest.raises(APIError, match="not mono pcm at 16000 Hz") as exc_info:
        await _drain(tts.synthesize("Hello there.", conn_options=_options(max_retry=2)), 16000)
    await tts.aclose()
    assert exc_info.value.retryable is False
    assert len(paxa_api.http_requests) == 1


async def test_synthesize_uses_paxa_request_id(paxa_api: _FakePaxa) -> None:
    tts = paxa_api.tts()
    stream = tts.synthesize("Hello there.", conn_options=_options())
    request_ids = {ev.request_id async for ev in stream}
    await stream.aclose()
    await tts.aclose()
    assert request_ids == {"req_ok"}


@pytest.mark.parametrize(
    ("status", "code", "retryable"),
    [
        (400, "unknown_voice", False),
        (400, "unsupported_format", False),
        (401, "unauthorized", False),
        (402, "insufficient_credits", False),
        (422, "unspeakable_text", False),
        (429, "rate_limited", True),
        (503, "provider_unavailable", True),
    ],
)
async def test_synthesize_maps_problem_responses(
    paxa_api: _FakePaxa, status: int, code: str, retryable: bool
) -> None:
    paxa_api.http_replies.append((status, code))
    tts = paxa_api.tts()
    with pytest.raises(APIStatusError) as exc_info:
        await _drain(tts.synthesize("Hello there.", conn_options=_options()))
    await tts.aclose()

    error = exc_info.value
    assert error.status_code == status
    assert error.retryable is retryable
    assert error.body == {"code": code}
    assert error.request_id == "req_failed"
    assert code in error.message


async def test_synthesize_retries_a_retryable_failure(paxa_api: _FakePaxa) -> None:
    paxa_api.http_replies.append((503, "provider_unavailable"))
    tts = paxa_api.tts()
    audio, _ = await _drain(tts.synthesize("Hello there.", conn_options=_options(max_retry=1)))
    await tts.aclose()
    _assert_audio_is(audio, paxa_api.http_pcm)
    assert len(paxa_api.http_requests) == 2


async def test_synthesize_reports_a_truncated_stream(paxa_api: _FakePaxa) -> None:
    paxa_api.http_replies.append("truncate")
    tts = paxa_api.tts()
    with pytest.raises(APIConnectionError):
        await _drain(tts.synthesize("Hello there.", conn_options=_options()))
    await tts.aclose()


async def test_synthesize_rejects_a_non_audio_response(paxa_api: _FakePaxa) -> None:
    paxa_api.http_replies.append("json")
    tts = paxa_api.tts()
    with pytest.raises(APIStatusError, match="non-audio") as exc_info:
        await _drain(tts.synthesize("Hello there.", conn_options=_options()))
    await tts.aclose()
    assert exc_info.value.status_code == 502


async def test_synthesize_skips_blank_text(paxa_api: _FakePaxa) -> None:
    tts = paxa_api.tts()
    audio, _ = await _drain(tts.synthesize("  \n ", conn_options=_options()))
    await tts.aclose()
    assert audio == b""
    assert paxa_api.http_requests == []


async def test_synthesize_errors_keep_no_response_text(paxa_api: _FakePaxa) -> None:
    paxa_api.http_replies.append((400, f"<html>{_SECRET}</html>"))
    tts = paxa_api.tts()
    with pytest.raises(APIStatusError) as exc_info:
        await _drain(tts.synthesize(f"text {_SECRET}", conn_options=_options()))
    await tts.aclose()
    assert exc_info.value.body is None
    _assert_safe(exc_info.value)


# stream(): /v1/tts/live


async def test_stream_speaks_text_over_one_live_session(paxa_api: _FakePaxa) -> None:
    tts = paxa_api.tts(voice="massaman", language="en", speed=0.9)
    text = "Hello there, welcome to the cafe. What would you like to drink today?"
    audio, segments = await _speak(tts, text)
    await tts.aclose()

    session = paxa_api.sessions[0]
    assert session.frames[0] == {
        "type": "start",
        "model": "paxa-tts-flash-v2",
        "voice": "massaman",
        "format": "pcm",
        "sample_rate": 24000,
        "speed": 0.9,
        "language": "en",
    }
    texts = [frame["text"] for frame in session.frames if frame["type"] == "text"]
    assert all(t.endswith("\n") for t in texts)
    assert " ".join(texts).split() == text.split()
    assert session.frames[-1] == {"type": "end"}

    # The fake splits samples across frames; every byte still arrives in order.
    _assert_audio_is(audio, b"".join(session.pcm))
    assert segments == {"conn_paxa_test"}

    upgrade = paxa_api.upgrades[0]
    assert upgrade["Authorization"] == f"Bearer {API_KEY}"
    assert upgrade["x-paxa-client"] == CLIENT
    assert upgrade["User-Agent"] == CLIENT
    assert len(paxa_api.upgrades) == 1


@pytest.mark.parametrize("rate", [8000, 16000, 44100])
async def test_stream_requests_the_chosen_sample_rate(paxa_api: _FakePaxa, rate: int) -> None:
    tts = paxa_api.tts(sample_rate=rate)
    audio, _ = await _speak(tts, "Hello there, welcome. Here is a second sentence.")
    await tts.aclose()

    session = paxa_api.sessions[0]
    assert session.frames[0]["format"] == "pcm"
    assert session.frames[0]["sample_rate"] == rate
    _assert_audio_is(audio, b"".join(session.pcm))


async def test_stream_plays_the_pcm_of_every_leg_in_order(paxa_api: _FakePaxa) -> None:
    tts = paxa_api.tts()
    stream = tts.stream(conn_options=_options())
    stream.push_text("Hello there, welcome to the cafe. What would")
    # The first sentence becomes a leg while the rest of the reply is still coming.
    await _eventually(lambda: bool(paxa_api.sessions) and bool(paxa_api.sessions[0].legs))
    stream.push_text(" you like to drink today?")
    stream.end_input()
    audio, _ = await _drain(stream)
    await tts.aclose()

    session = paxa_api.sessions[0]
    assert len(session.legs) == 2
    # Each leg's pcm arrives split mid-sample and carries straight on from the last.
    _assert_audio_is(audio, b"".join(session.pcm))


async def test_stream_sends_thai_without_punctuation_as_one_leg(paxa_api: _FakePaxa) -> None:
    tts = paxa_api.tts()
    text = "สวัสดีครับ ยินดีต้อนรับสู่ร้านกาแฟของเรา วันนี้รับเครื่องดื่มอะไรดีครับ"
    audio, _ = await _speak(tts, text)
    await tts.aclose()

    session = paxa_api.sessions[0]
    assert session.legs == [text + "\n"]
    _assert_audio_is(audio, b"".join(session.pcm))


async def test_stream_batches_sentences_while_a_leg_synthesizes(paxa_api: _FakePaxa) -> None:
    paxa_api.live_scripts.append({"leg_delay": 0.3})
    tts = paxa_api.tts()
    sentences = [f"This is sentence number {i}, and it is spoken in order." for i in range(16)]
    audio, _ = await _speak(tts, " ".join(sentences))
    await tts.aclose()

    session = paxa_api.sessions[0]
    texts = [frame["text"] for frame in session.frames if frame["type"] == "text"]
    assert len(texts) < len(sentences)
    assert " ".join(texts).split() == " ".join(sentences).split()
    # Paxa closes a session with 32 frames queued; the stream never comes close.
    assert session.max_pending <= 3
    _assert_audio_is(audio, b"".join(session.pcm))


async def test_stream_without_text_opens_no_session(paxa_api: _FakePaxa) -> None:
    tts = paxa_api.tts()
    stream = tts.stream(conn_options=_options())
    stream.push_text(" \n")
    stream.end_input()
    audio, _ = await _drain(stream)
    await tts.aclose()
    assert audio == b""
    assert paxa_api.upgrades == []


@pytest.mark.parametrize("code", ["unknown_voice", "unsupported_format"])
async def test_stream_start_error_is_not_retried(paxa_api: _FakePaxa, code: str) -> None:
    paxa_api.live_scripts.append({"start_error": code})
    tts = paxa_api.tts(voice="nobody")
    with pytest.raises(APIStatusError) as exc_info:
        await _speak(tts, "Hello there, nice to meet you.", max_retry=2)
    await tts.aclose()

    error = exc_info.value
    assert error.status_code == 400
    assert error.body == {"code": code}
    assert error.retryable is False
    assert len(paxa_api.upgrades) == 1


async def test_stream_retries_a_failed_leg_before_any_audio(paxa_api: _FakePaxa) -> None:
    paxa_api.live_scripts.append({"error_on_leg": {1: "provider_error"}})
    tts = paxa_api.tts()
    audio, _ = await _speak(tts, "Hello there, nice to meet you.", max_retry=1)
    await tts.aclose()

    assert len(paxa_api.upgrades) == 2
    _assert_audio_is(audio, b"".join(paxa_api.sessions[1].pcm))


async def test_stream_does_not_retry_after_partial_audio(paxa_api: _FakePaxa) -> None:
    paxa_api.live_scripts.append({"error_after_audio_on_leg": {1: "provider_error"}})
    tts = paxa_api.tts()
    with pytest.raises(APIStatusError) as exc_info:
        await _speak(tts, "Hello there, nice to meet you.", max_retry=3)
    await tts.aclose()

    assert exc_info.value.status_code == 502
    assert exc_info.value.request_id == "conn_paxa_test"
    assert len(paxa_api.upgrades) == 1


async def test_stream_maps_a_failure_close_code(paxa_api: _FakePaxa) -> None:
    paxa_api.live_scripts.append({"close_on_leg": {1: 4402}})
    tts = paxa_api.tts()
    with pytest.raises(APIStatusError) as exc_info:
        await _speak(tts, "Hello there, nice to meet you.")
    await tts.aclose()
    assert exc_info.value.status_code == 402
    assert exc_info.value.retryable is False


async def test_stream_reads_the_close_code_sent_before_a_drop(paxa_api: _FakePaxa) -> None:
    # Paxa drops the connection right after its close frame. When the drop lands first,
    # aiohttp's close reply fails and ws.close_code reads 1006; the frame keeps the code.
    real_ws_connect = aiohttp.ClientSession._ws_connect

    async def ws_connect(self: aiohttp.ClientSession, *args: Any, **kwargs: Any) -> Any:
        ws = await real_ws_connect(self, *args, **kwargs)

        async def reply_fails(code: int = 1000, message: bytes | str = b"") -> None:
            raise ConnectionResetError("Cannot write to closing transport")

        ws._writer.close = reply_fails  # type: ignore[method-assign]
        return ws

    paxa_api.live_scripts.append({"close_and_drop_on_leg": {1: 4413}})
    tts = paxa_api.tts()
    with patch.object(aiohttp.ClientSession, "_ws_connect", ws_connect):
        with pytest.raises(APIStatusError) as exc_info:
            await _speak(tts, "Hello there, nice to meet you.")
    await tts.aclose()
    assert exc_info.value.status_code == 413
    assert exc_info.value.retryable is False


async def test_stream_requires_done(paxa_api: _FakePaxa) -> None:
    paxa_api.live_scripts.append({"skip_done": True})
    tts = paxa_api.tts()
    with pytest.raises(APIConnectionError):
        await _speak(tts, "Hello there, nice to meet you.")
    await tts.aclose()


async def test_stream_reports_a_refused_upgrade_without_the_key(paxa_api: _FakePaxa) -> None:
    paxa_api.refuse_upgrade = 429
    tts = paxa_api.tts()
    with pytest.raises(APIStatusError) as exc_info:
        await _speak(tts, "Hello there, nice to meet you.")
    await tts.aclose()
    assert exc_info.value.status_code == 429
    assert exc_info.value.retryable is True
    _assert_safe(exc_info.value)


async def test_stream_times_out_when_the_session_never_starts(paxa_api: _FakePaxa) -> None:
    paxa_api.live_scripts.append({"silent": True})
    tts = paxa_api.tts()
    with pytest.raises(APITimeoutError):
        await _speak(tts, "Hello there, nice to meet you.", timeout=0.3)
    await tts.aclose()


async def test_stream_interruption_closes_the_session(paxa_api: _FakePaxa) -> None:
    paxa_api.live_scripts.append({"stall_mid_leg": True})
    tts = paxa_api.tts()
    stream = tts.stream(conn_options=_options())
    stream.push_text("Hello there, this answer gets interrupted.")
    stream.end_input()

    first = await asyncio.wait_for(anext(stream), timeout=5)
    assert first.frame.samples_per_channel > 0
    interrupted_at = time.monotonic()
    await asyncio.wait_for(stream.aclose(), timeout=2)

    session = paxa_api.sessions[0]
    await _eventually(lambda: session.closed_at is not None)
    assert session.closed_during_leg
    assert session.closed_at is not None and session.closed_at - interrupted_at < 1.0
    await tts.aclose()


# Helpers


def test_live_errors_map_codes_to_http_statuses() -> None:
    assert live_event_error({"code": "insufficient_credits"}, connection_id="c").status_code == 402
    assert live_event_error({"code": "unsupported_format"}, connection_id="c").status_code == 400
    assert live_event_error({"code": "rate_limited"}, connection_id="c").retryable is True
    assert live_event_error({"code": "tag_invalid"}, connection_id="c").retryable is False

    unknown = live_event_error({"code": f"<{_SECRET}>"}, connection_id=None)
    assert unknown.status_code == -1
    assert unknown.body is None
    _assert_safe(unknown)


def test_live_close_codes_carry_the_http_status() -> None:
    closed = live_close_error(4401, connection_id="c")
    assert isinstance(closed, APIStatusError)
    assert closed.status_code == 401
    assert isinstance(live_close_error(1001, connection_id="c"), APIConnectionError)
    assert isinstance(live_close_error(None, connection_id=None), APIError)


def test_http_errors_tolerate_a_body_that_is_not_json() -> None:
    error = http_status_error(502, b"\xff<html>bad gateway</html>", request_id=None)
    assert error.status_code == 502
    assert error.body is None


def test_take_frame_joins_sentences_up_to_the_limit() -> None:
    pending = ["One two three.", "Four five six.", "Seven eight nine."]
    assert take_frame(pending, 30) == "One two three. Four five six."
    assert pending == ["Seven eight nine."]


def test_take_frame_cuts_a_long_sentence_at_whitespace() -> None:
    pending = ["word " * 10 + "end"]
    first = take_frame(pending, 22)
    assert first == "word word word word"
    assert len(first) <= 22
    assert pending == ["word " * 6 + "end"]


def test_take_frame_keeps_thai_marks_with_their_letter() -> None:
    thai = "กี่" * 20  # each syllable is a consonant followed by two combining marks
    pending = [thai]
    first = take_frame(pending, 10)
    assert len(first) <= 10
    assert first + pending[0] == thai
    assert pending[0][0] not in "ี่"


def test_pcm_content_type_must_match_the_request() -> None:
    check_pcm_content_type("audio/pcm;rate=16000;channels=1", sample_rate=16000)
    # Parameters in any order, case and quoting, or with others added, still match.
    check_pcm_content_type('Audio/PCM; Channels=1; rate="16000"; x=y', sample_rate=16000)

    for content_type in (
        "audio/pcm;rate=24000;channels=1",
        "audio/pcm;rate=16000;channels=2",
        "audio/pcm;rate=16000",
        "audio/pcm",
        "audio/wav",
        "",
    ):
        with pytest.raises(APIError, match="unexpected format"):
            check_pcm_content_type(content_type, sample_rate=16000)


def test_frame_gate_waits_for_each_leg_and_never_stalls() -> None:
    gate = _FrameGate()
    assert gate.can_send(0.0)
    gate.sent(0.0)
    assert not gate.can_send(0.1)
    assert gate.wait_time(0.1) == pytest.approx(0.4)
    # A frame that starts no leg, such as tags alone, does not hold the next one back.
    assert gate.can_send(0.6)

    gate.leg_started(1)
    assert not gate.can_send(10.0)
    assert gate.wait_time(10.0) is None
    assert not gate.leg_flushed(0)
    assert gate.leg_flushed(1)
    assert gate.can_send(10.0)
