"""Tests for the AssemblyAI Streaming TTS plugin, run against a local fake of the protocol."""

from __future__ import annotations

import asyncio
import base64
import json
import struct
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import APIConnectOptions, APIStatusError
from livekit.agents.types import USERDATA_TIMED_TRANSCRIPT

pytestmark = pytest.mark.plugin("assemblyai")

SAMPLE_RATE = 24000
# 80 ms per Audio frame at 24 kHz pcm_s16le, as the service sends
FRAME_BYTES = SAMPLE_RATE * 2 * 80 // 1000


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
    """A local stand-in for wss://streaming-tts.assemblyai.com/v1/ws.

    Each Flush is answered with `frames_per_request` Audio frames, a FlushDone and
    (unless disabled) a WordBoundaries frame with one entry per whitespace token.
    """

    def __init__(
        self,
        *,
        frames_per_request: int = 3,
        frame_delay: float = 0.0,
        word_boundaries: bool = True,
        drop_word_boundaries: bool = False,
        connect_error: tuple[int, str] | None = None,
        error_after_begin: tuple[int, str] | None = None,
        echo_sample_rate: int | None = None,
    ) -> None:
        self.frames_per_request = frames_per_request
        self.frame_delay = frame_delay
        self.word_boundaries = word_boundaries
        self.drop_word_boundaries = drop_word_boundaries
        self.connect_error = connect_error
        self.error_after_begin = error_after_begin
        self.echo_sample_rate = echo_sample_rate
        self.sessions: list[_Session] = []
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
            code, text = self.connect_error
            await ws.send_json({"type": "Error", "error_code": code, "error": text})
            await ws.close(code=code, message=b"See Error message for details")
            return ws

        await ws.send_json(
            {
                "type": "Begin",
                "id": f"session-{len(self.sessions)}",
                "expires_at": 4102444800,
                "configuration": {
                    "voice": session.params.get("voice"),
                    "language": session.params.get("language", "english"),
                    "sample_rate": self.echo_sample_rate
                    or int(session.params.get("sample_rate", SAMPLE_RATE)),
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
            code, text = self.error_after_begin
            await asyncio.sleep(0.05)
            await ws.send_json({"type": "Error", "error_code": code, "error": text})
            await ws.close(code=code, message=b"See Error message for details")
            return ws

        state = {"flush_id": 0, "timeline_ms": 0}
        buffer: list[str] = []
        queue: asyncio.Queue[str | None] = asyncio.Queue()
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
                    await queue.put("".join(buffer))
                    buffer.clear()
                elif kind == "Cancel":
                    buffer.clear()
                    worker.cancel()
                    await asyncio.gather(worker, return_exceptions=True)
                    while not queue.empty():
                        queue.get_nowait()
                    state["flush_id"] += 1
                    await ws.send_json({"type": "Cancelled", "audio_duration_seconds": 0.0})
                    worker = asyncio.create_task(self._synthesize(ws, queue, state))
                elif kind == "Terminate":
                    break
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        return ws

    async def _synthesize(
        self, ws: web.WebSocketResponse, queue: asyncio.Queue[str | None], state: dict[str, int]
    ) -> None:
        while True:
            text = await queue.get()
            if text is None:
                return
            flush_id = state["flush_id"]
            audio_start_ms = state["timeline_ms"]
            frames = [_pcm(FRAME_BYTES, seed=flush_id) for _ in range(self.frames_per_request)]
            for frame in frames:
                if self.frame_delay:
                    await asyncio.sleep(self.frame_delay)
                await ws.send_json(
                    {
                        "type": "Audio",
                        "audio": base64.b64encode(frame).decode(),
                        "flush_id": flush_id,
                    }
                )
            self.audio_by_request.append(b"".join(frames))
            duration_ms = self.frames_per_request * 80
            state["timeline_ms"] += duration_ms
            state["flush_id"] += 1
            await ws.send_json(
                {"type": "FlushDone", "flush_id": flush_id, "audio_duration_ms": duration_ms}
            )
            if self.word_boundaries and not self.drop_word_boundaries:
                tokens = text.split()
                step = duration_ms // max(len(tokens), 1)
                await ws.send_json(
                    {
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
                )


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


def test_voice_and_language_resolution() -> None:
    from livekit.plugins.assemblyai import TTS

    tts = TTS(api_key="k")
    assert (tts.voice, tts.language) == ("jane", "english")

    tts = TTS(api_key="k", voice="lola")
    assert (tts.voice, tts.language) == ("lola", "spanish")

    for language in ("spanish", "Spanish", "es", "es-MX"):
        tts = TTS(api_key="k", language=language)
        assert (tts.voice, tts.language) == ("lola", "spanish")

    # a voice outside the published catalog is passed through for the server to judge
    tts = TTS(api_key="k", voice="newvoice", language="german")
    assert (tts.voice, tts.language) == ("newvoice", "german")
    tts = TTS(api_key="k", voice="newvoice")
    assert (tts.voice, tts.language) == ("newvoice", None)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"voice": "lola", "language": "english"}, "speaks spanish"),
        ({"voice": "Jane"}, "case-sensitive"),
        ({"language": "klingon"}, "no default voice"),
        ({"sample_rate": 12345}, "sample_rate"),
    ],
)
def test_invalid_configuration(kwargs: dict[str, Any], match: str) -> None:
    from livekit.plugins.assemblyai import TTS

    with pytest.raises(ValueError, match=match):
        TTS(api_key="k", **kwargs)


def test_api_key_required(monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.plugins.assemblyai import TTS

    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        TTS()
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "from-env")
    assert TTS()._opts.api_key == "from-env"


def test_update_options_keeps_voice_and_language_paired() -> None:
    from livekit.plugins.assemblyai import TTS

    tts = TTS(api_key="k", voice="anna")
    tts.update_options(language="en")  # anna already speaks English
    assert (tts.voice, tts.language) == ("anna", "english")
    tts.update_options(language="fr")
    assert (tts.voice, tts.language) == ("estelle", "french")
    tts.update_options(voice="paul")
    assert (tts.voice, tts.language) == ("paul", "english")


def test_split_request_text() -> None:
    from livekit.plugins.assemblyai.tts import _MAX_REQUEST_CHARS, _split_request_text

    assert _split_request_text("  hello there  ") == ["hello there"]
    assert _split_request_text("   ") == []

    words = " ".join(["word"] * 1000)  # 4999 chars
    pieces = _split_request_text(words)
    assert all(len(p) <= _MAX_REQUEST_CHARS for p in pieces)
    assert " ".join(pieces) == words

    # no whitespace to split on: cut at the limit, counted in code points
    blob = "é" * (_MAX_REQUEST_CHARS + 5)
    assert [len(p) for p in _split_request_text(blob)] == [_MAX_REQUEST_CHARS, 5]


# --- streaming -------------------------------------------------------------------------


async def test_stream_sends_one_request_per_sentence() -> None:
    async with _FakeServer() as srv:
        tts = _make_tts(srv, voice="anna")
        audio, timed, segments = await _speak(
            tts, ["Hi, thanks for ", "calling. How can I ", "help you today?"]
        )
        await tts.aclose()

    session = srv.sessions[0]
    assert session.authorization == "test-key"  # raw key, no "Bearer"
    assert session.params == {
        "voice": "anna",
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
    first, second = srv.audio_by_request
    assert _strip_padding(audio1, len(first)) == first
    assert _strip_padding(audio2, len(second)) == second
    assert "".join(timed1) == "First reply here. "
    assert "".join(timed2) == "Second reply here. "
    # the second stream's timings are relative to its own audio, not the session's
    assert timed2[0].start_time == pytest.approx(0.0)


async def test_interrupted_stream_cancels_and_frees_the_socket() -> None:
    async with _FakeServer(frames_per_request=40, frame_delay=0.01) as srv:
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
    assert len(frame.data.tobytes()) >= len(srv.audio_by_request[0])


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
        tts.update_options(voice="giovanni")
        await _speak(tts, ["Ciao."])
        await tts.aclose()

    assert [(s.params["voice"], s.params["language"]) for s in srv.sessions] == [
        ("jane", "english"),
        ("giovanni", "italian"),
    ]


async def test_socket_closed_while_idle_is_replaced() -> None:
    async with _FakeServer() as srv:
        tts = _make_tts(srv)
        await _speak(tts, ["Hello."])
        # the server ends the idle session (inactivity timeout, expiry, network drop)
        assert srv.sessions[0].ws is not None
        await srv.sessions[0].ws.close(code=3008)
        await asyncio.sleep(0.05)
        audio, _, _ = await _speak(tts, ["Still here."])
        await tts.aclose()

    assert len(srv.sessions) == 2
    assert audio


# --- errors ----------------------------------------------------------------------------


async def test_rejected_credential_is_not_retried() -> None:
    async with _FakeServer(connect_error=(1008, "Unauthorized: Invalid API key")) as srv:
        tts = _make_tts(srv)
        with pytest.raises(APIStatusError) as exc_info:
            await _speak(tts, ["Hello."], max_retry=2)
        await tts.aclose()

    assert exc_info.value.status_code == 1008
    assert exc_info.value.retryable is False
    assert "Invalid API key" in str(exc_info.value)
    assert len(srv.sessions) == 1


async def test_unsupported_voice_error_after_begin() -> None:
    error = "Unsupported preset voice 'nobody'. Available on this connection: ['jane']."
    async with _FakeServer(error_after_begin=(3006, error)) as srv:
        tts = _make_tts(srv, voice="nobody")
        with pytest.raises(APIStatusError) as exc_info:
            await _speak(tts, ["Hello."], max_retry=2)
        await tts.aclose()

    assert exc_info.value.status_code == 3006
    assert exc_info.value.retryable is False
    assert len(srv.sessions) == 1


async def test_transient_error_is_retried_on_a_new_session() -> None:
    async with _FakeServer(connect_error=(3009, "Too many concurrent sessions")) as srv:
        tts = _make_tts(srv)
        with pytest.raises(APIStatusError) as exc_info:
            await _speak(tts, ["Hello."], max_retry=1)
        await tts.aclose()

    assert exc_info.value.retryable is True
    assert len(srv.sessions) == 2


def test_inactivity_close_is_retryable() -> None:
    from livekit.plugins.assemblyai.tts import _error_from_frame

    err = _error_from_frame(
        {"type": "Error", "error_code": 3006, "error": "No message received for 30 seconds."},
        session_id="s",
    )
    assert err.retryable is True
    assert err.request_id == "s"


async def test_sample_rate_mismatch_is_rejected() -> None:
    async with _FakeServer(echo_sample_rate=16000) as srv:
        tts = _make_tts(srv)
        with pytest.raises(Exception, match="16000 Hz"):
            await _speak(tts, ["Hello."], max_retry=2)
        await tts.aclose()

    assert len(srv.sessions) == 1
