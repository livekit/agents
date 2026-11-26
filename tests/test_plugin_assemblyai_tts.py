"""Tests for the AssemblyAI Streaming TTS plugin, run against a local fake of the protocol."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError
from livekit.agents.types import USERDATA_TIMED_TRANSCRIPT

pytestmark = pytest.mark.plugin("assemblyai")

AUDREY = "49c443a4-8a1d-5f7e-b70b-98bdbd9823c3"  # the default voice

SAMPLE_RATE = 24000
# 80 ms per Audio frame at 24 kHz pcm_s16le, as the service sends
FRAME_BYTES = SAMPLE_RATE * 2 * 80 // 1000


def _error(code: int, text: str, **fields: Any) -> dict[str, Any]:
    """An Error frame; servers older than the documented API send only code and text."""
    return {"type": "Error", "error_code": code, "error": text, **fields}


def _pcm(num_bytes: int, seed: int) -> bytes:
    samples = num_bytes // 2
    return struct.pack(f"<{samples}h", *(((i * 97 + seed) % 2000) - 1000 for i in range(samples)))


@dataclass
class _Session:
    params: dict[str, str]
    authorization: str | None
    received: list[dict[str, Any]] = field(default_factory=list)
    ws: web.WebSocketResponse | None = None


class _FakeServer:
    """A local stand-in for wss://tts.us.assemblyai.com/v1/ws.

    Each Flush is answered with `frames_per_request` Audio frames, a FlushDone and
    (unless disabled) a WordBoundaries frame with one entry per whitespace token. The
    remaining knobs reproduce the orderings the protocol allows and the plugin must survive.

    `flush_id_model` picks how request ids continue after a Cancel:
    - "live": what the production service does today. The next request takes the id of
      the cancelled request in flight plus one, reusing the ids of discarded requests.
    - "documented": what the API reference specifies. Every Flush and every Cancel
      advances the id, so the next request's id counts both frames sent before it.
    """

    def __init__(
        self,
        *,
        frames_per_request: int = 3,
        frame_delay: float = 0.0,
        word_boundaries: bool = True,
        drop_word_boundaries: bool | set[int] = False,
        connect_error: dict[str, Any] | None = None,
        error_after_begin: dict[str, Any] | None = None,
        tail_frames_after_cancel: int = 0,
        cancelled_delay: float = 0.0,
        late_word_boundaries: bool = False,
        hold_late_word_boundaries: bool = False,
        error_after_flush_dones: tuple[int, dict[str, Any]] | None = None,
        begin_delay: float = 0.0,
        flush_id_model: Literal["live", "documented"] = "live",
        session_seconds: float = 3600,
        word_boundaries_after_cancelled: bool = False,
    ) -> None:
        self.frames_per_request = frames_per_request
        self.frame_delay = frame_delay
        self.word_boundaries = word_boundaries
        self.drop_word_boundaries = drop_word_boundaries
        self.connect_error = connect_error
        self.error_after_begin = error_after_begin
        # Audio frames of the cancelled request still sent between Cancel and Cancelled
        self.tail_frames_after_cancel = tail_frames_after_cancel
        self.cancelled_delay = cancelled_delay
        # send WordBoundaries(N) after the first Audio of request N+1
        self.late_word_boundaries = late_word_boundaries
        # keep a late WordBoundaries for the next request even when none is queued, so it
        # can outlast the stream that owns it
        self.hold_late_word_boundaries = hold_late_word_boundaries
        # (count, Error frame): Error + close after that many FlushDone frames
        self.error_after_flush_dones = error_after_flush_dones
        self.begin_delay = begin_delay
        self.flush_id_model = flush_id_model
        self.session_seconds = session_seconds
        # echoed in Begin as if the account set it; the fake doesn't enforce it
        # hold the WordBoundaries of completed requests until the first Cancel, then send
        # them right after its Cancelled
        self.word_boundaries_after_cancelled = word_boundaries_after_cancelled
        self._held_word_boundaries: list[dict[str, Any]] = []
        self.sessions: list[_Session] = []
        self.flush_dones = 0
        # audio of every request, in order, for asserting what reached the client
        self.audio_by_request: list[bytes] = []

    async def __aenter__(self) -> _FakeServer:
        app = web.Application()
        app.router.add_get("/v1/ws", self._handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = self._runner.addresses[0][1]
        self.base_url = f"ws://127.0.0.1:{port}"
        self.http = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.http.close()
        await self._runner.cleanup()

    def frames(self, kind: str) -> list[dict[str, Any]]:
        return [m for s in self.sessions for m in s.received if m.get("type") == kind]

    async def _handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        session = _Session(
            params=dict(request.query),
            authorization=request.headers.get("Authorization"),
            ws=ws,
        )
        self.sessions.append(session)

        if self.connect_error is not None:
            await ws.send_json(self.connect_error)
            await ws.close(
                code=self.connect_error["error_code"], message=b"See Error message for details"
            )
            return ws

        if self.begin_delay:
            await asyncio.sleep(self.begin_delay)
        await ws.send_json(
            {
                "type": "Begin",
                "id": f"session-{len(self.sessions)}",
                "expires_at": int(time.time() + self.session_seconds),
                "configuration": {
                    "voice": session.params.get("voice_id", "jane"),
                    "language": session.params.get("language", "english"),
                    "sample_rate": int(session.params.get("sample_rate", SAMPLE_RATE)),
                    "encoding": session.params.get("encoding"),
                    "inactivity_timeout": None,
                    "word_boundaries": self.word_boundaries
                    and session.params.get("word_boundaries") == "true",
                    "voice_clone": False,
                    "voice_clone_preprocess": False,
                },
            }
        )

        if self.error_after_begin is not None:
            await asyncio.sleep(0.05)
            await ws.send_json(self.error_after_begin)
            await ws.close(
                code=self.error_after_begin["error_code"], message=b"See Error message for details"
            )
            return ws

        # flush_id: the request in flight ("live") or the next one to be flushed ("documented")
        state = {"flush_id": 0, "timeline_ms": 0, "in_flight": 0}
        buffer: list[str] = []
        queue: asyncio.Queue[tuple[int | None, str] | None] = asyncio.Queue()
        worker = asyncio.create_task(self._synthesize(ws, queue, state))

        try:
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                data = json.loads(msg.data)
                session.received.append(data)
                kind = data.get("type")
                if kind == "Generate":
                    buffer.append(data["text"])
                elif kind == "Flush":
                    if self.flush_id_model == "documented":
                        await queue.put((state["flush_id"], "".join(buffer)))
                        state["flush_id"] += 1
                    else:
                        await queue.put((None, "".join(buffer)))
                    buffer.clear()
                elif kind == "Cancel":
                    buffer.clear()
                    worker.cancel()
                    await asyncio.gather(worker, return_exceptions=True)
                    while not queue.empty():
                        queue.get_nowait()
                    for _ in range(self.tail_frames_after_cancel):
                        await ws.send_json(
                            {
                                "type": "Audio",
                                "audio": base64.b64encode(_pcm(FRAME_BYTES, seed=777)).decode(),
                                "flush_id": state["in_flight"],
                            }
                        )
                        state["timeline_ms"] += 80
                    state["flush_id"] += 1
                    if self.cancelled_delay:
                        await asyncio.sleep(self.cancelled_delay)
                    await ws.send_json({"type": "Cancelled", "audio_duration_seconds": 0.0})
                    if self.word_boundaries_after_cancelled:
                        self.word_boundaries_after_cancelled = False
                        for held in self._held_word_boundaries:
                            await ws.send_json(held)
                    worker = asyncio.create_task(self._synthesize(ws, queue, state))
                elif kind == "Terminate":
                    break
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        return ws

    async def _synthesize(
        self,
        ws: web.WebSocketResponse,
        queue: asyncio.Queue[tuple[int | None, str] | None],
        state: dict[str, int],
    ) -> None:
        late: dict[str, Any] | None = None
        while True:
            if late is not None and queue.empty() and not self.hold_late_word_boundaries:
                # nothing follows to trail behind
                await ws.send_json(late)
                late = None
            item = await queue.get()
            if item is None:
                return
            assigned, text = item
            flush_id = state["flush_id"] if assigned is None else assigned
            state["in_flight"] = flush_id
            audio_start_ms = state["timeline_ms"]
            frames = [_pcm(FRAME_BYTES, seed=flush_id) for _ in range(self.frames_per_request)]
            for i, frame in enumerate(frames):
                if self.frame_delay:
                    await asyncio.sleep(self.frame_delay)
                await ws.send_json(
                    {
                        "type": "Audio",
                        "audio": base64.b64encode(frame).decode(),
                        "flush_id": flush_id,
                    }
                )
                # the session timeline counts every byte sent, cancelled audio included
                state["timeline_ms"] += 80
                if i == 0 and late is not None:
                    await ws.send_json(late)
                    late = None
            self.audio_by_request.append(b"".join(frames))
            duration_ms = self.frames_per_request * 80
            if assigned is None:
                state["flush_id"] += 1
            self.flush_dones += 1
            await ws.send_json(
                {"type": "FlushDone", "flush_id": flush_id, "audio_duration_ms": duration_ms}
            )
            if self.error_after_flush_dones and self.error_after_flush_dones[0] == self.flush_dones:
                error = self.error_after_flush_dones[1]
                await ws.send_json(error)
                await ws.close(code=error["error_code"], message=b"See Error message for details")
                return
            dropped = self.drop_word_boundaries
            if self.word_boundaries and not (
                dropped is True or (isinstance(dropped, set) and flush_id in dropped)
            ):
                tokens = text.split()
                step = duration_ms // max(len(tokens), 1)
                word_boundaries = {
                    "type": "WordBoundaries",
                    "flush_id": flush_id,
                    "audio_start_ms": audio_start_ms,
                    "words": [
                        {
                            "word": token,
                            "start": audio_start_ms + i * step,
                            "end": audio_start_ms + (i + 1) * step,
                            "confidence": 0.9,
                        }
                        for i, token in enumerate(tokens)
                    ],
                }
                if self.word_boundaries_after_cancelled:
                    self._held_word_boundaries.append(word_boundaries)
                elif self.late_word_boundaries:
                    late = word_boundaries
                else:
                    await ws.send_json(word_boundaries)


def _make_tts(srv: _FakeServer, **kwargs: Any):  # noqa: ANN202
    from livekit.plugins.assemblyai import TTS

    return TTS(api_key="test-key", base_url=srv.base_url, http_session=srv.http, **kwargs)


async def _speak(tts, chunks: list[str], *, max_retry: int = 0):  # noqa: ANN001, ANN202
    """Run one stream to completion; returns (audio, timed transcripts, segment ids)."""
    audio = bytearray()
    timed: list[Any] = []
    segments: set[str] = set()
    async with tts.stream(conn_options=APIConnectOptions(max_retry=max_retry, timeout=5)) as s:
        for chunk in chunks:
            s.push_text(chunk)
        s.end_input()
        async for ev in s:
            audio += ev.frame.data.tobytes()
            timed.extend(ev.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, []))
            segments.add(ev.segment_id)
    return bytes(audio), timed, segments


def _strip_padding(audio: bytes, expected_len: int) -> bytes:
    assert set(audio[expected_len:]) <= {0}
    return audio[:expected_len]


# --- configuration ---------------------------------------------------------------------


def test_voice_and_language() -> None:
    from livekit.plugins.assemblyai import TTS

    # nothing given: Audrey, and the server's default language (English)
    tts = TTS(api_key="k")
    assert (tts._opts.voice, tts._opts.language, tts.model) == (AUDREY, None, "assemblyai-tts-1")

    # voices pass through; the language is never inferred from one
    tts = TTS(api_key="k", voice=" aai-vera ")
    assert (tts._opts.voice, tts._opts.language) == ("aai-vera", None)

    # any code or English name becomes the name the API takes
    for language in ("spanish", "Spanish", "es", "es-MX", "spa"):
        assert TTS(api_key="k", language=language)._opts.language == "spanish"
    assert TTS(api_key="k", language="pt-BR")._opts.language == "portuguese"
    # a language the service may add later passes through
    assert TTS(api_key="k", language="ja")._opts.language == "japanese"
    assert TTS(api_key="k", language="Klingon")._opts.language == "klingon"

    with pytest.raises(ValueError, match="voice"):
        TTS(api_key="k", voice=" ")
    with pytest.raises(ValueError, match="language"):
        TTS(api_key="k", language="")


def test_custom_voice_and_model() -> None:
    from livekit.plugins.assemblyai import TTS

    # a custom voice is a voice like any other; it speaks its clip's language by default
    clone = "01a142a4-66f3-7315-b15f-668b6edfb595"
    tts = TTS(api_key="k", voice=f" {clone} ")
    assert (tts._opts.voice, tts._opts.language, tts.model) == (clone, None, "assemblyai-tts-1")

    tts = TTS(api_key="k", voice="my-voice", language="es", model="latest")
    assert (tts._opts.voice, tts._opts.language, tts.model) == ("my-voice", "spanish", "latest")

    with pytest.raises(ValueError, match="model"):
        TTS(api_key="k", model=" ")


def test_update_options() -> None:
    from livekit.plugins.assemblyai import TTS

    tts = TTS(api_key="k", voice="aai-vera", language="es")
    tts.update_options(voice="clone", language="fr", model="latest")
    assert (tts._opts.voice, tts._opts.language, tts.model) == ("clone", "french", "latest")
    tts.update_options(language=None)  # back to the server's default
    assert (tts._opts.voice, tts._opts.language) == ("clone", None)
    with pytest.raises(ValueError, match="voice"):
        tts.update_options(voice=" ")


def test_invalid_sample_rate() -> None:
    from livekit.plugins.assemblyai import TTS

    with pytest.raises(ValueError, match="sample_rate"):
        TTS(api_key="k", sample_rate=12345)


def test_api_key_required(monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.plugins.assemblyai import TTS

    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        TTS()
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "from-env")
    assert TTS()._opts.api_key == "from-env"


def test_update_options_sets_each_field_on_its_own() -> None:
    from livekit.plugins.assemblyai import TTS

    tts = TTS(api_key="k", voice="aai-anna")
    tts.update_options(language="fr")  # the voice is kept: the server judges the pair
    assert (tts._opts.voice, tts._opts.language) == ("aai-anna", "french")
    tts.update_options(voice="aai-linda-3")
    assert (tts._opts.voice, tts._opts.language) == ("aai-linda-3", "french")


def test_split_request_text() -> None:
    from livekit.plugins.assemblyai.tts import _MAX_REQUEST_CHARS, _split_request_text

    assert _split_request_text("  hello there  ") == ["hello there"]
    assert _split_request_text("   ") == []

    words = " ".join(["word"] * 1000)  # 4999 chars
    pieces = _split_request_text(words)
    assert all(len(p) <= _MAX_REQUEST_CHARS for p in pieces)
    assert " ".join(pieces) == words

    # a run the server can't speak in one piece is broken up, in code points
    from livekit.plugins.assemblyai.tts import _MAX_UNBROKEN_CHARS

    blob = "é" * (2 * _MAX_UNBROKEN_CHARS + 5)
    pieces = _split_request_text(blob)
    assert all(len(p) <= _MAX_REQUEST_CHARS for p in pieces)
    assert [len(run) for p in pieces for run in p.split()] == [
        _MAX_UNBROKEN_CHARS,
        _MAX_UNBROKEN_CHARS,
        5,
    ]
    # any whitespace is a place to cut, not just a space
    lines = "\n".join(["w" * (_MAX_UNBROKEN_CHARS - 1)] * 100)
    pieces = _split_request_text(lines)
    assert len(pieces) > 1
    assert all(len(p) <= _MAX_REQUEST_CHARS for p in pieces)
    assert [len(run) for p in pieces for run in p.split()] == [_MAX_UNBROKEN_CHARS - 1] * 100


# --- streaming -------------------------------------------------------------------------


async def test_stream_sends_one_request_per_sentence() -> None:
    async with _FakeServer() as srv:
        tts = _make_tts(srv, voice="aai-anna", language="en")
        audio, timed, segments = await _speak(
            tts, ["Hi, thanks for ", "calling. How can I ", "help you today?"]
        )
        await tts.aclose()

    session = srv.sessions[0]
    assert session.authorization == "test-key"  # raw key, no "Bearer"
    assert session.params == {
        "model_id": "assemblyai-tts-1",
        "voice_id": "aai-anna",
        "language": "english",
        "sample_rate": "24000",
        "encoding": "pcm_s16le",
        "word_boundaries": "true",
    }

    generates = [f["text"] for f in srv.frames("Generate")]
    assert generates == ["Hi, thanks for calling.", "How can I help you today?"]
    assert len(srv.frames("Flush")) == 2

    expected = b"".join(srv.audio_by_request)
    assert _strip_padding(audio, len(expected)) == expected
    assert len(segments) == 1

    # word timings are moved from the session timeline onto the segment's own audio
    assert "".join(timed) == "Hi, thanks for calling. How can I help you today? "
    starts = [w.start_time for w in timed]
    assert starts == sorted(starts)
    assert starts[0] == pytest.approx(0.0)
    second_request_start = srv.frames_per_request * 0.08
    assert timed[4].start_time == pytest.approx(second_request_start)  # "How"
    assert timed[-1].end_time == pytest.approx(2 * second_request_start)


async def test_connection_is_reused_and_stale_frames_are_dropped() -> None:
    async with _FakeServer() as srv:
        tts = _make_tts(srv)
        audio1, timed1, _ = await _speak(tts, ["First reply here."])
        audio2, timed2, _ = await _speak(tts, ["Second reply here."])
        await tts.aclose()

    assert len(srv.sessions) == 1
    assert srv.frames("Cancel") == []  # nothing was outstanding
    first, second = srv.audio_by_request
    assert _strip_padding(audio1, len(first)) == first
    assert _strip_padding(audio2, len(second)) == second
    assert "".join(timed1) == "First reply here. "
    assert "".join(timed2) == "Second reply here. "
    # the second stream's timings are relative to its own audio, not the session's
    assert timed2[0].start_time == pytest.approx(0.0)


FLUSH_ID_MODELS = pytest.mark.parametrize("flush_id_model", ["live", "documented"])


@FLUSH_ID_MODELS
async def test_interrupted_stream_cancels_and_frees_the_socket(flush_id_model: str) -> None:
    async with _FakeServer(
        frames_per_request=40, frame_delay=0.01, flush_id_model=flush_id_model
    ) as srv:
        tts = _make_tts(srv)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
        stream.push_text("This reply gets interrupted by the user.")
        stream.end_input()
        async for _ in stream:
            break  # first audio arrived; barge in
        await stream.aclose()

        srv.frames_per_request = 2
        srv.frame_delay = 0.0
        audio, timed, _ = await _speak(tts, ["After the interruption."])
        await tts.aclose()

    assert len(srv.sessions) == 1, "the socket should go back to the pool after Cancel"
    assert len(srv.frames("Cancel")) == 1
    # no tail of the cancelled request may leak into the next reply
    expected = srv.audio_by_request[-1]
    assert _strip_padding(audio, len(expected)) == expected
    assert "".join(timed) == "After the interruption. "


async def test_missing_word_boundaries_fall_back_to_sentence_timing() -> None:
    async with _FakeServer(drop_word_boundaries=True) as srv:
        tts = _make_tts(srv)
        _, timed, _ = await _speak(
            tts, ["The first sentence is right here. The second one is over there."]
        )
        await tts.aclose()

    assert [str(t) for t in timed] == [
        "The first sentence is right here. ",
        "The second one is over there. ",
    ]
    assert timed[0].start_time == pytest.approx(0.0)
    assert timed[1].start_time == pytest.approx(srv.frames_per_request * 0.08)


async def test_one_missing_word_boundaries_keeps_transcript_order() -> None:
    async with _FakeServer(drop_word_boundaries={0}) as srv:
        tts = _make_tts(srv)
        _, timed, _ = await _speak(
            tts, ["The first sentence is right here. The second one is over there."]
        )
        await tts.aclose()

    assert [str(t) for t in timed] == [
        "The first sentence is right here. ",
        *(f"{w} " for w in "The second one is over there.".split()),
    ]
    starts = [w.start_time for w in timed]
    assert starts == sorted(starts)


async def test_session_without_word_boundaries_paces_per_sentence() -> None:
    async with _FakeServer(word_boundaries=False) as srv:
        tts = _make_tts(srv)
        _, timed, _ = await _speak(
            tts, ["The first sentence is right here. The second one is over there."]
        )
        await tts.aclose()

    assert [str(t) for t in timed] == [
        "The first sentence is right here. ",
        "The second one is over there. ",
    ]


async def test_word_timestamps_disabled() -> None:
    async with _FakeServer() as srv:
        tts = _make_tts(srv, word_timestamps=False)
        assert tts.capabilities.aligned_transcript is False
        _, timed, _ = await _speak(tts, ["Nothing timed here."])
        await tts.aclose()

    assert srv.sessions[0].params["word_boundaries"] == "false"
    assert timed == []


async def test_synthesize() -> None:
    async with _FakeServer() as srv:
        tts = _make_tts(srv)
        frame = await tts.synthesize("A one-shot synthesis.").collect()
        await tts.aclose()

    assert frame.sample_rate == SAMPLE_RATE
    expected = srv.audio_by_request[0]
    assert _strip_padding(frame.data.tobytes(), len(expected)) == expected


async def test_long_sentence_is_split_into_requests() -> None:
    from livekit.plugins.assemblyai.tts import _MAX_REQUEST_CHARS

    text = " ".join(["word"] * 900) + "."  # one 4500-char "sentence"
    async with _FakeServer(frames_per_request=1) as srv:
        tts = _make_tts(srv)
        await _speak(tts, [text])
        await tts.aclose()

    generates = [f["text"] for f in srv.frames("Generate")]
    assert len(generates) == 3
    assert all(len(g) <= _MAX_REQUEST_CHARS for g in generates)
    assert len(srv.frames("Flush")) == 3


async def test_update_options_opens_a_new_session() -> None:
    async with _FakeServer() as srv:
        tts = _make_tts(srv)
        await _speak(tts, ["Hello."])
        tts.update_options(voice="aai-paul", language="it")
        await _speak(tts, ["Ciao."])
        await tts.aclose()

    assert [(s.params.get("voice_id"), s.params.get("language")) for s in srv.sessions] == [
        (AUDREY, None),
        ("aai-paul", "italian"),
    ]


async def test_voice_and_model_are_sent_at_connect() -> None:
    async with _FakeServer() as srv:
        tts = _make_tts(srv, voice="clone", model="some-model")
        await _speak(tts, ["Hello."])
        tts.update_options(voice="aai-jane", model="assemblyai-tts-1")
        await _speak(tts, ["Hello again."])
        await tts.aclose()

    keys = ("voice", "voice_id", "language", "model_id")
    assert [tuple(s.params.get(k) for k in keys) for s in srv.sessions] == [
        (None, "clone", None, "some-model"),
        (None, "aai-jane", None, "assemblyai-tts-1"),
    ]


async def test_socket_closed_while_idle_is_replaced() -> None:
    async with _FakeServer() as srv:
        tts = _make_tts(srv)
        await _speak(tts, ["Hello."])
        # the server ends the idle session (inactivity timeout, expiry, network drop)
        ws = srv.sessions[0].ws
        assert ws is not None
        await ws.send_json(
            _error(
                3008,
                "Session reached its maximum duration; open a new connection to continue",
                error_type="session_expired",
                retryable=True,
            )
        )
        await ws.close(code=3008)
        await asyncio.sleep(0.05)
        audio, _, _ = await _speak(tts, ["Still here."])
        await tts.aclose()

    assert len(srv.sessions) == 2
    assert audio


@FLUSH_ID_MODELS
async def test_next_holder_waits_for_cancelled_and_drops_the_tail(flush_id_model: str) -> None:
    """A reply that starts while the previous one's Cancel is unanswered gets none of its tail."""
    async with _FakeServer(
        frames_per_request=40,
        frame_delay=0.01,
        tail_frames_after_cancel=3,
        cancelled_delay=0.2,
        flush_id_model=flush_id_model,
    ) as srv:
        tts = _make_tts(srv)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
        stream.push_text("This reply gets interrupted by the user.")
        stream.end_input()
        async for _ in stream:
            break
        await stream.aclose()

        (conn,) = tts._pool._available
        assert conn._cancels_pending == 1

        srv.frames_per_request = 2
        srv.frame_delay = 0.0
        audio, timed, _ = await _speak(tts, ["After the interruption."])
        await tts.aclose()

    assert len(srv.sessions) == 1
    expected = srv.audio_by_request[-1]
    assert _strip_padding(audio, len(expected)) == expected
    assert "".join(timed) == "After the interruption. "
    # re-anchored although the session timeline holds the cancelled audio and its tail
    assert timed[0].start_time == pytest.approx(0.0)


@FLUSH_ID_MODELS
async def test_second_interrupt_before_the_first_cancelled(flush_id_model: str) -> None:
    """A reply interrupted before it heard anything adds a Cancel; the next one waits for both."""
    async with _FakeServer(
        frames_per_request=40,
        frame_delay=0.01,
        cancelled_delay=0.3,
        flush_id_model=flush_id_model,
    ) as srv:
        tts = _make_tts(srv)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
        stream.push_text("The first reply gets interrupted.")
        stream.end_input()
        async for _ in stream:
            break
        await stream.aclose()

        # still waiting for the first Cancelled: this reply hears nothing and is cut too
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
        stream.push_text("The second reply never gets a word in.")
        stream.end_input()
        await asyncio.sleep(0.1)
        await stream.aclose()

        srv.frames_per_request = 2
        srv.frame_delay = 0.0
        srv.cancelled_delay = 0.0
        audio, timed, _ = await _speak(tts, ["The third reply is heard."])
        await tts.aclose()

    assert len(srv.sessions) == 1
    assert len(srv.frames("Cancel")) == 2
    expected = srv.audio_by_request[-1]
    assert _strip_padding(audio, len(expected)) == expected
    assert "".join(timed) == "The third reply is heard. "


@FLUSH_ID_MODELS
async def test_word_boundaries_after_cancelled_are_dropped(flush_id_model: str) -> None:
    """Timings of a completed request that trail the Cancel don't reach the next reply."""
    async with _FakeServer(
        frames_per_request=10,
        frame_delay=0.03,
        word_boundaries_after_cancelled=True,
        flush_id_model=flush_id_model,
    ) as srv:
        tts = _make_tts(srv)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
        stream.push_text("The first sentence finishes playing. The second one is cut off.")
        stream.end_input()
        received = 0
        async for ev in stream:
            received += len(ev.frame.data.tobytes())
            if received > 12 * FRAME_BYTES:  # well into the second request
                break
        await stream.aclose()

        srv.frames_per_request = 2
        srv.frame_delay = 0.0
        audio, timed, _ = await _speak(tts, ["A fresh reply."])
        await tts.aclose()

    assert len(srv.frames("Cancel")) == 1
    expected = srv.audio_by_request[-1]
    assert _strip_padding(audio, len(expected)) == expected
    assert "".join(timed) == "A fresh reply. "
    assert timed[0].start_time == pytest.approx(0.0)


@FLUSH_ID_MODELS
async def test_cancel_after_some_requests_completed(flush_id_model: str) -> None:
    """FlushDone(0) arrives, then a Cancel cuts request 1; the next reply's id is learned."""
    # request 1 runs until ~0.6s at the fake; the Cancel lands at ~0.4s
    async with _FakeServer(
        frames_per_request=10, frame_delay=0.03, flush_id_model=flush_id_model
    ) as srv:
        tts = _make_tts(srv)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
        stream.push_text("The first sentence finishes playing. The second one is cut off.")
        stream.end_input()
        received = 0
        async for ev in stream:
            received += len(ev.frame.data.tobytes())
            if received > 12 * FRAME_BYTES:  # well into the second request
                break
        await stream.aclose()

        srv.frames_per_request = 2
        srv.frame_delay = 0.0
        audio, timed, _ = await _speak(tts, ["A fresh reply."])
        await tts.aclose()

    assert len(srv.frames("Cancel")) == 1
    expected = srv.audio_by_request[-1]
    assert _strip_padding(audio, len(expected)) == expected
    assert "".join(timed) == "A fresh reply. "


async def test_word_timings_arrive_while_waiting_for_more_text() -> None:
    """A sentence's trailing WordBoundaries are read during the pause before the next one."""
    async with _FakeServer() as srv:
        tts = _make_tts(srv)
        timed: list[Any] = []
        async with tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5)) as s:
            s.push_text("The first sentence. ")
            s.flush()  # the LLM pauses here; no more text yet

            async def until_the_sentence_is_timed() -> None:
                async for ev in s:
                    timed.extend(ev.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, []))
                    if "".join(timed) == "The first sentence. ":
                        return

            await asyncio.wait_for(until_the_sentence_is_timed(), 2)
            s.end_input()
        await tts.aclose()

    assert all(isinstance(t.start_time, float) for t in timed)
    assert len(timed) == 3  # one per word: the WordBoundaries, not the untimed fallback


async def test_word_boundaries_from_the_previous_stream_are_ignored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The last request's timings outlast the grace and land in the next stream."""
    async with _FakeServer(late_word_boundaries=True, hold_late_word_boundaries=True) as srv:
        tts = _make_tts(srv)
        await _speak(tts, ["The first reply."])
        with caplog.at_level(logging.WARNING):
            _, timed, _ = await _speak(tts, ["The second reply."])
        await tts.aclose()

    assert len(srv.sessions) == 1
    assert "".join(timed) == "The second reply. "
    assert "unknown request" not in caplog.text


async def test_word_boundaries_trailing_the_next_requests_audio() -> None:
    async with _FakeServer(late_word_boundaries=True) as srv:
        tts = _make_tts(srv)
        _, timed, _ = await _speak(
            tts,
            ["The first sentence is right here. The second one is over there. And a third."],
        )
        await tts.aclose()

    assert "".join(timed) == (
        "The first sentence is right here. The second one is over there. And a third. "
    )
    starts = [w.start_time for w in timed]
    assert starts == sorted(starts)
    request_duration = srv.frames_per_request * 0.08
    assert timed[6].start_time == pytest.approx(request_duration)  # "The" of the second
    assert timed[-1].end_time == pytest.approx(3 * request_duration)


async def test_missing_word_boundaries_do_not_hold_back_audio() -> None:
    """Only the end-of-segment marker waits for timings; the audio itself is released."""
    async with _FakeServer(drop_word_boundaries=True) as srv:
        tts = _make_tts(srv)
        loop = asyncio.get_running_loop()
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
        t0 = loop.time()
        stream.push_text("A short reply that ends the turn.")
        stream.end_input()
        received = 0
        all_audio_at = None
        async for ev in stream:
            received += len(ev.frame.data.tobytes())
            if all_audio_at is None and received >= srv.frames_per_request * FRAME_BYTES:
                all_audio_at = loop.time() - t0
        await stream.aclose()
        await tts.aclose()

    from livekit.plugins.assemblyai.tts import _WORD_BOUNDARIES_GRACE

    assert all_audio_at is not None
    assert all_audio_at < _WORD_BOUNDARIES_GRACE / 2


async def test_interrupt_while_connecting_leaks_no_tasks() -> None:
    async with _FakeServer(begin_delay=0.3) as srv:
        tts = _make_tts(srv, text_pacing=True)
        for _ in range(3):
            stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
            stream.push_text("Interrupted before the session opens.")
            await asyncio.sleep(0.05)
            await stream.aclose()
        await tts.aclose()
        await asyncio.sleep(0)

    leaked = [
        t.get_name()
        for t in asyncio.all_tasks()
        if t is not asyncio.current_task()
        and not t.done()
        and ("StreamPacer" in repr(t.get_coro()) or "AssemblyAITTS" in t.get_name())
    ]
    assert leaked == []


# --- errors ----------------------------------------------------------------------------


async def test_rejected_credential_is_not_retried() -> None:
    async with _FakeServer(connect_error=_error(1008, "Unauthorized: Invalid API key")) as srv:
        tts = _make_tts(srv)
        with pytest.raises(APIStatusError) as exc_info:
            await _speak(tts, ["Hello."], max_retry=2)
        await tts.aclose()

    assert exc_info.value.status_code == 1008
    assert exc_info.value.retryable is False
    assert "Invalid API key" in str(exc_info.value)
    assert len(srv.sessions) == 1


async def test_unsupported_voice_error_after_begin() -> None:
    error = _error(
        3006, "Unsupported preset voice 'nobody'. Available on this connection: ['jane']."
    )
    async with _FakeServer(error_after_begin=error) as srv:
        tts = _make_tts(srv, voice="nobody")
        with pytest.raises(APIStatusError) as exc_info:
            await _speak(tts, ["Hello."], max_retry=2)
        await tts.aclose()

    assert exc_info.value.status_code == 3006
    assert exc_info.value.retryable is False
    assert len(srv.sessions) == 1


async def test_transient_error_is_retried_on_a_new_session() -> None:
    error = _error(3005, "The TTS service is temporarily unavailable; please reconnect and retry")
    async with _FakeServer(connect_error=error) as srv:
        tts = _make_tts(srv)
        with pytest.raises(APIStatusError) as exc_info:
            await _speak(tts, ["Hello."], max_retry=1)
        await tts.aclose()

    assert exc_info.value.retryable is True
    assert len(srv.sessions) == 2


@pytest.mark.parametrize(
    ("frame", "retryable"),
    [
        # servers that send only the code and the text
        (_error(3006, "Generate text is 2500 characters, over the 2000-character limit."), False),
        (_error(3010, "Input text rate exceeded"), False),
        (_error(3009, "Too many sessions opened for this account"), True),
        (_error(3012, "The requested voice was not found."), False),
        (_error(3014, "Voice v has no ready pairing with model m"), False),
        # the documented frame: `retryable` decides, whatever the code or text
        (_error(3006, "reworded", error_type="inactivity_timeout", retryable=True), True),
        (_error(3005, "The server is out of room", error_type="at_capacity", retryable=True), True),
        (
            _error(3011, "Audio consumption too slow", error_type="audio_backlog", retryable=False),
            False,
        ),
        (_error(4321, "A code added later", error_type="something_new", retryable=False), False),
    ],
)
def test_error_retryability(frame: dict[str, Any], retryable: bool) -> None:
    from livekit.plugins.assemblyai.tts import _error_from_frame

    err = _error_from_frame(frame, session_id="s")
    assert err.retryable is retryable
    assert err.request_id == "s"
    if "error_type" in frame:
        assert frame["error_type"] in err.message


async def test_retry_after_is_honoured_before_the_next_session() -> None:
    error = _error(
        3009,
        "Too many sessions opened for this account in the last minute (allowance 5 per minute); retry in 1 s.",
        error_type="rate_limited",
        retryable=True,
        retry_after_seconds=0.5,
    )
    async with _FakeServer(connect_error=error) as srv:
        tts = _make_tts(srv)
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        with pytest.raises(APIStatusError):
            await _speak(tts, ["Hello."], max_retry=1)
        elapsed = loop.time() - t0
        await tts.aclose()

    assert len(srv.sessions) == 2
    assert elapsed >= 0.5


async def test_retry_after_longer_than_the_timeout_ends_the_retries() -> None:
    error = _error(
        3009,
        "Too many sessions opened for this account in the last minute (allowance 5 per minute); retry in 30 s.",
        error_type="rate_limited",
        retryable=True,
        retry_after_seconds=30,
    )
    async with _FakeServer(connect_error=error) as srv:
        tts = _make_tts(srv)
        with pytest.raises(APIConnectionError) as exc_info:
            await _speak(tts, ["Hello."], max_retry=2)
        await tts.aclose()

    assert exc_info.value.retryable is False
    assert len(srv.sessions) == 1  # no session start spent on a retry that can't succeed


async def test_session_unusable_on_arrival_is_not_replaced_in_a_loop() -> None:
    async with _FakeServer(session_seconds=-1) as srv:  # already past its expires_at
        tts = _make_tts(srv)
        with pytest.raises(APIConnectionError, match="unusable as soon as it opened"):
            await _speak(tts, ["Hello."], max_retry=1)
        await tts.aclose()

    assert len(srv.sessions) == 2  # one per attempt


async def test_short_session_is_still_used() -> None:
    """A session shorter than twice the expiry margin (a capped token) is not discarded."""
    async with _FakeServer(session_seconds=60) as srv:
        tts = _make_tts(srv)
        audio, _, _ = await _speak(tts, ["Hello."])
        await tts.aclose()

    assert len(srv.sessions) == 1
    assert audio


async def test_unbroken_run_is_sent_in_speakable_pieces() -> None:
    from livekit.plugins.assemblyai.tts import _MAX_UNBROKEN_CHARS

    digest = "f" * 1900  # the service closes the session on a run this long
    async with _FakeServer(frames_per_request=1) as srv:
        tts = _make_tts(srv)
        await _speak(tts, [f"The digest is {digest}."])
        await tts.aclose()

    runs = [run for f in srv.frames("Generate") for run in f["text"].split()]
    assert max(len(r) for r in runs) <= _MAX_UNBROKEN_CHARS


async def test_error_mid_stream_is_raised_after_partial_audio() -> None:
    error = _error(3005, "The TTS service is temporarily unavailable; please reconnect and retry")
    async with _FakeServer(error_after_flush_dones=(1, error)) as srv:
        tts = _make_tts(srv)
        with pytest.raises(APIStatusError) as exc_info:
            await _speak(
                tts,
                ["The first sentence is right here. The second one is over there."],
                max_retry=2,
            )
        assert not tts._pool._available  # the failed session is not reused
        await tts.aclose()

    assert exc_info.value.status_code == 3005
    assert exc_info.value.retryable is True
    # audio already reached the user, so the framework does not retry
    assert len(srv.sessions) == 1
