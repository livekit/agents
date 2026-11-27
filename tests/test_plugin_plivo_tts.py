"""Tests of the Plivo TTS plugin against a local stand-in for Plivo's TTS API."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import gc
import json
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from unittest.mock import MagicMock, patch

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
)
from livekit.agents.metrics import TTSMetrics
from livekit.agents.types import USERDATA_TIMED_TRANSCRIPT
from livekit.plugins import plivo

from .stt_reconnect import FakeWebSocket, wait_until

pytestmark = pytest.mark.unit

AUTH_ID = "MA1"
AUTH_TOKEN = "token"
SAMPLES_100MS = 2400  # at 24 kHz: the fake's audio for one character
MAX_TEXT = 20000  # the API's limit for the text of one context
NO_RETRY = APIConnectOptions(max_retry=0, timeout=2.0)
ONE_RETRY = APIConnectOptions(max_retry=1, timeout=2.0)
SHORT_TIMEOUT = APIConnectOptions(max_retry=0, timeout=0.05)
OPTIONS: dict[str, Any] = {
    "model": "koyal-v2",
    "voice": "v2",
    "language": "hi",
    "speed": 1.1,
    "volume": 1.5,
    "pronunciation_dictionary_id": "dictionary-1",
    "seed": 0,
    "continuation": False,
    "text_normalization": False,
}
SENT_OPTIONS = {
    "model_id": "koyal-v2",
    "voice_id": "v2",
    "language": "hi",
    "pace": 1.1,
    "volume": 1.5,
    "pronunciation_dictionary_id": "dictionary-1",
    "seed": 0,
    "continuation": False,
    "text_normalization": False,
}
HTTP_STATUS = {
    "unauthorized": 401,
    "text_too_long": 400,
    "voice_not_found": 404,
    "insufficient_balance": 400,
    "concurrency_limit": 429,
    "internal_error": 500,
    "at_capacity": 503,
}
MESSAGES = {
    "unauthorized": "The auth ID or token is wrong.",
    "text_too_long": "The text of a context is over 20,000 characters.",
    "voice_not_found": "No voice has this ID.",
    "insufficient_balance": (
        "Insufficient credits in the account. Please add more credits to continue."
    ),
    "concurrency_limit": "Too many requests are open at once.",
    "internal_error": "Something went wrong.",
    "at_capacity": "The service is busy. Please try again later.",
}


class _FakeServer:
    """A local stand-in for Plivo's TTS API: the Realtime WebSocket and the Batch route.

    At each flush and at its finish, a context speaks the text it has, a sentence at a time:
    100 ms of audio for each character, then the sentence's timestamps event. Contexts speak at
    the same time.
    """

    def __init__(self) -> None:
        app = web.Application()
        app.router.add_get(f"/v1/Account/{AUTH_ID}/TTS/Realtime", self._realtime)
        app.router.add_post(f"/v1/Account/{AUTH_ID}/TTS", self._synthesize)
        self.server = TestServer(app)
        self.session = aiohttp.ClientSession()
        self.sockets: list[web.WebSocketResponse] = []
        self.ended_sockets = 0
        self.received: list[dict[str, Any]] = []
        self.posted: list[dict[str, Any]] = []
        self.spoke: list[str] = []  # the context of each audio message, in the order sent
        self.refusals: list[str] = []  # error codes that refuse the next first messages or POSTs
        # the answers to the next finishes: "close" (the done, then close code 1012), "fault" (an
        # error with no context first; the connection stays), "bare" (an event with no context and
        # no code or message first), "again" (an error after the done), "ping" (a ping first),
        # "stall" (no done). To the next POSTs: "stall" (the body stays open), "abort" (the
        # connection is cut), and a proxy's own 502: "bad gateway" (JSON that is not the API's),
        # "bad page" (a page that is not UTF-8)
        self.answers: list[str] = []
        self.refused_handshakes: list[str] = []  # error codes that refuse the next handshakes
        # error codes that end the next connections after their handshakes, as the API ends one for
        # an account with no credit: the error with no context, then close code 1008 ("bare": an
        # error with no fields)
        self.ended_handshakes: list[str] = []
        self.completes_handshakes = True  # off: a handshake never completes
        self.held: list[web.Request] = []  # the handshakes that do not complete
        self.pongs: list[bytes] = []  # the data of each pong from the plugin
        self.ponged = asyncio.Event()
        self.answers_closes = True  # off: no answer to the plugin's close frame
        self.audio_interval = 0.0
        credentials = base64.b64encode(f"{AUTH_ID}:{AUTH_TOKEN}".encode()).decode()
        self._authorization = f"Basic {credentials}"
        self._taken_ids: set[str] = set()
        self._open: set[str] = set()
        self._pending: dict[str, str] = {}
        self._text_chars: dict[str, int] = {}
        self._spoken_chars: dict[str, int] = {}
        self._steps: dict[str, asyncio.Task[None]] = {}
        self._instances: list[plivo.TTS] = []

    def tts(self, **options: Any) -> plivo.TTS:
        options.setdefault("http_session", self.session)
        tts = plivo.TTS(base_url=str(self.server.make_url("/")), **options)  # it ends with "/"
        self._instances.append(tts)
        return tts

    async def aclose(self) -> None:
        for tts in self._instances:
            await tts.aclose()
        for step in self._steps.values():
            step.cancel()
        await self.session.close()
        await self.server.close()

    async def _synthesize(self, request: web.Request) -> web.StreamResponse:
        self.posted.append(await request.json())
        api_id = f"{request.headers.get('client-ref', 'plivo')}-api-{len(self.posted)}"
        if request.headers["Authorization"] != self._authorization:
            return self._refuse("unauthorized", api_id)
        if self.refusals:
            return self._refuse(self.refusals.pop(0), api_id)
        answer = self.answers.pop(0) if self.answers else ""
        if answer == "bad gateway":
            return web.json_response({"message": "Bad Gateway"}, status=502)
        if answer == "bad page":
            page = b"Passerelle incorrecte \xe9"
            return web.Response(status=502, body=page, content_type="text/html")
        response = web.StreamResponse(headers={"api-id": api_id})
        await response.prepare(request)
        for char in self.posted[-1]["text"]:
            await response.write(char.encode("utf-16-le") * SAMPLES_100MS)
        if answer == "stall":
            await asyncio.Event().wait()
        if answer == "abort":
            request.transport.abort()
        return response

    def _refuse(self, code: str, api_id: str) -> web.Response:
        error = {"api_id": api_id, "error": MESSAGES[code], "code": code}
        return web.json_response(error, status=HTTP_STATUS[code], headers={"api-id": api_id})

    async def _realtime(self, request: web.Request) -> web.StreamResponse:
        # as the API: every answer names its API ID, which starts with the client-ref
        api_id = f"{request.headers.get('client-ref', 'plivo')}-api-{len(self.sockets) + 1}"
        if request.headers["Authorization"] != self._authorization:
            return self._refuse("unauthorized", api_id)
        if self.refused_handshakes:
            return self._refuse(self.refused_handshakes.pop(0), api_id)
        if not self.completes_handshakes:
            self.held.append(request)
            await asyncio.Event().wait()

        # as the API: a message over 1 MiB closes the connection with code 1009
        ws = web.WebSocketResponse(autoclose=False, autoping=False, max_msg_size=2**20)
        ws.headers["api-id"] = api_id
        await ws.prepare(request)
        self.sockets.append(ws)
        if self.ended_handshakes:
            code = self.ended_handshakes.pop(0)
            await ws.send_json({"type": "error"} if code == "bare" else _error_event(code, api_id))
            await ws.ping()  # the API pings at any time: here between the error and the close
            await ws.pong()  # and an unsolicited pong
            await ws.close(code=1008)
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.PONG:
                    self.pongs.append(msg.data)
                    self.ponged.set()
                if msg.type == web.WSMsgType.TEXT:
                    with contextlib.suppress(ConnectionResetError):  # the plugin closed
                        await self._on_message(ws, msg.json())
            if not self.answers_closes:
                await asyncio.Event().wait()
        finally:
            self.ended_sockets += 1
        return ws

    async def _on_message(self, ws: web.WebSocketResponse, message: dict[str, Any]) -> None:
        self.received.append(message)
        kind, context_id, text = message["type"], message["context_id"], message.get("text", "")
        if context_id not in self._open:
            if kind != "text" or context_id in self._taken_ids:
                return  # as the API: no answer for an ended context, or to a cancel
            self._taken_ids.add(context_id)  # a first message uses up its ID, also when refused
            if self.refusals:
                return await self._end(ws, context_id, self.refusals.pop(0))
            if len(text) > MAX_TEXT:  # as the API: an opening over the limit is refused
                return await self._end(ws, context_id, "text_too_long")
            self._open.add(context_id)
            self._pending[context_id], self._spoken_chars[context_id] = "", 0
            self._text_chars[context_id] = 0

        if kind == "cancel":
            await ws.send_bytes(_audio_message(context_id))  # audio that was on its way
            self._run_in_order(context_id, self._end, ws, context_id)
        elif kind == "flush":
            text, self._pending[context_id] = self._pending[context_id], ""
            self._run_in_order(context_id, self._send_audio, ws, context_id, text)
        elif context_id in self._pending:
            # as the API: the text that takes the context past its limit is dropped with a warning,
            # and so is every text after that one
            self._text_chars[context_id] += len(text)
            if text and self._text_chars[context_id] > MAX_TEXT:
                warning = {
                    "type": "warning",
                    "code": "text_too_long",
                    "param": "text",
                    "message": "text dropped",
                }
                await ws.send_json({**warning, "context_id": context_id})
                text = ""
            self._pending[context_id] += text
            if message.get("finish"):
                text = self._pending.pop(context_id)
                self._run_in_order(context_id, self._finish, ws, context_id, text)

    def _run_in_order(
        self, context_id: str, step: Callable[..., Awaitable[None]], *args: Any
    ) -> None:
        """Run a flush or a finish after the context's earlier one, while other contexts speak."""
        earlier = self._steps.get(context_id)

        async def run() -> None:
            if earlier:
                await earlier
            with contextlib.suppress(ConnectionResetError):
                await step(*args)

        self._steps[context_id] = asyncio.create_task(run())

    async def _send_audio(self, ws: web.WebSocketResponse, context_id: str, text: str) -> None:
        # as the API: a sentence after the first starts with the space before it
        for sentence in filter(None, re.split(r"(?<=\.)(?= )", text)):
            spoken = self._spoken_chars[context_id]
            self._spoken_chars[context_id] += len(sentence)
            for char in sentence:
                await asyncio.sleep(self.audio_interval)
                await ws.send_bytes(_audio_message(context_id, char))
                self.spoke.append(context_id)
            chars = list(sentence)
            start_ms = [100 * n for n in range(spoken, spoken + len(chars))]
            timing = {"chars": chars, "start_ms": start_ms, "duration_ms": [100] * len(chars)}
            await ws.send_json({"type": "timestamps", "context_id": context_id, **timing})

    async def _finish(self, ws: web.WebSocketResponse, context_id: str, text: str) -> None:
        answer = self.answers.pop(0) if self.answers else ""
        if answer == "fault":
            await ws.send_json(_error_event("internal_error", ws.headers["api-id"]))
        if answer == "bare":
            await ws.send_json({"type": "notice"})
        if answer == "ping":
            await ws.ping(b"finish")
        await self._send_audio(ws, context_id, text)
        if answer != "stall":
            await self._end(ws, context_id)
        if answer == "again":
            error = _error_event("internal_error", ws.headers["api-id"])
            await ws.send_json({**error, "context_id": context_id})
        if answer == "close":
            await ws.close(code=1012)

    async def _end(self, ws: web.WebSocketResponse, context_id: str, code: str = "") -> None:
        self._open.discard(context_id)
        event = _error_event(code, ws.headers["api-id"]) if code else {"type": "done"}
        await ws.send_json({**event, "context_id": context_id})


@pytest.fixture
async def fake(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[_FakeServer]:
    monkeypatch.setenv("PLIVO_AUTH_ID", AUTH_ID)
    monkeypatch.setenv("PLIVO_AUTH_TOKEN", AUTH_TOKEN)
    fake = _FakeServer()
    await fake.server.start_server()
    yield fake
    await fake.aclose()


def _error_event(code: str, api_id: str) -> dict[str, Any]:
    """An error event: the API sends the HTTP status of the code and the API ID with it."""
    error = {"code": code, "message": MESSAGES[code], "status_code": HTTP_STATUS[code]}
    return {"type": "error", **error, "api_id": api_id}


def _audio_message(context_id: str, char: str = " ") -> bytes:
    """An audio message: 100 ms whose every sample is the character, so a test can read it."""
    samples = char.encode("utf-16-le") * SAMPLES_100MS
    return len(context_id).to_bytes(2, "little") + context_id.encode() + samples


async def _collect_frames(stream: plivo.tts.SynthesizeStream) -> list[rtc.AudioFrame]:
    async with asyncio.timeout(2):  # fail the test, not hang it, if the stream never ends
        return [ev.frame async for ev in stream]


async def _duration_of(stream: plivo.tts.SynthesizeStream) -> float:
    return sum(frame.duration for frame in await _collect_frames(stream))


async def _stream_frames(
    tts: plivo.TTS, *chunks: str, conn_options: APIConnectOptions = NO_RETRY
) -> list[rtc.AudioFrame]:
    """Push `chunks` into one stream, end its input and return its audio frames."""
    async with tts.stream(conn_options=conn_options) as stream:
        for chunk in chunks:
            stream.push_text(chunk)
        stream.end_input()
        return await _collect_frames(stream)


async def _speak(tts: plivo.TTS, *chunks: str, conn_options: APIConnectOptions = NO_RETRY) -> float:
    """The same as _stream_frames, but return the seconds of audio."""
    frames = await _stream_frames(tts, *chunks, conn_options=conn_options)
    return sum(frame.duration for frame in frames)


async def _spoken_text(tts: plivo.TTS, text: str) -> tuple[str, str]:
    """Return the text read back from a stream's audio, and the stream's words joined."""
    frames = await _stream_frames(tts, text)
    words = [word for frame in frames for word in frame.userdata[USERDATA_TIMED_TRANSCRIPT]]
    samples = [sample for frame in frames for sample in frame.data]  # see _audio_message
    spoken = "".join(map(chr, samples[::SAMPLES_100MS])).rstrip("\0")  # the emitter's end marker
    return spoken, "".join(words)


async def _open_connection(tts: plivo.TTS) -> None:
    """Run one short stream, so that a later timeout does not include opening the connection."""
    await _speak(tts, "Hi.")


async def _open_stream(fake: _FakeServer, tts: plivo.TTS) -> plivo.tts.SynthesizeStream:
    """Start a stream whose input stays open, and wait until the server has its first message."""
    stream = tts.stream(conn_options=NO_RETRY)
    stream.push_text("A stream that stays open")
    received = len(fake.received) + 1
    await wait_until(lambda: len(fake.received) == received)
    return stream


async def _interrupt_when(stream: plivo.tts.SynthesizeStream, reached: Callable[[], bool]) -> None:
    """Close `stream` in the first pass of the event loop in which `reached()` is true."""
    async with asyncio.timeout(2):
        while not reached():
            await asyncio.sleep(0)
    await stream.aclose()


def _expected_duration(text: str) -> Any:
    """The seconds of audio the fake sends for `text`.

    The emitter adds 10 ms of silence when it has no audio left to mark as final.
    """
    return pytest.approx(len(text) * 0.1, abs=0.011)


def _held_until_aborted(
    ws: aiohttp.ClientWebSocketResponse[bool],
) -> Callable[..., Awaitable[None]]:
    """A write that backs up until its transport is aborted, as aiohttp's does."""
    lost = ws._writer.protocol.closed

    async def close(*_: Any, **__: Any) -> None:
        await lost

    return close


async def _returning(*_: Any, **__: Any) -> bool:
    """A close that returns at once and leaves the transport to flush."""
    return True


def _context_ids(fake: _FakeServer) -> set[str]:
    return {message["context_id"] for message in fake.received}


def _first_messages(fake: _FakeServer) -> list[dict[str, Any]]:
    return [message for message in fake.received if "voice_id" in message]


def test_missing_credentials_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLIVO_AUTH_ID", AUTH_ID)
    monkeypatch.delenv("PLIVO_AUTH_TOKEN", raising=False)

    with pytest.raises(ValueError, match="PLIVO_AUTH_TOKEN"):
        plivo.TTS()


def test_model_provider_and_defaults() -> None:
    tts = plivo.TTS(auth_id=AUTH_ID, auth_token=AUTH_TOKEN)

    properties = (tts.model, tts.provider, tts.sample_rate, tts.num_channels)
    assert properties == ("koyal-v1", "Plivo", 24000, 1)
    assert (tts.capabilities.streaming, tts.capabilities.aligned_transcript) == (True, True)
    assert tts._tts_url == f"https://api.plivo.com/v1/Account/{AUTH_ID}/TTS"


async def test_only_the_first_message_carries_the_options(fake: _FakeServer) -> None:
    await _speak(fake.tts(**OPTIONS, sample_rate=16000), " ", "Hel", "lo. ")

    context = {"context_id": fake.received[0]["context_id"]}
    output_format = {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 16000}
    first = {**SENT_OPTIONS, "output_format": output_format, "timestamps": True, "text": "Hel"}
    # the leading " " is not sent: the context opens with the first text that is not blank
    assert fake.received == [
        {"type": "text", **context, **first},
        {"type": "text", **context, "text": "lo. "},
        {"type": "flush", **context},
        {"type": "text", **context, "finish": True},
    ]


async def test_first_message_defaults(fake: _FakeServer) -> None:
    await _speak(fake.tts(), "Hello.")

    assert fake.received[0] == {
        "type": "text",
        "context_id": fake.received[0]["context_id"],
        "model_id": "koyal-v1",
        "voice_id": plivo.models.TTSDefaultVoiceId,
        "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 24000},
        "timestamps": True,
        "text": "Hello.",
    }


async def test_word_timestamps_disabled(fake: _FakeServer) -> None:
    tts = fake.tts(word_timestamps=False)

    await _speak(tts, "Hello.")

    assert not tts.capabilities.aligned_transcript
    assert fake.received[0]["timestamps"] is False


async def test_update_options_applies_to_later_streams(fake: _FakeServer) -> None:
    """A stream or request made before the update keeps its options, and the connection stays."""
    tts = fake.tts(voice="v1")
    await _open_connection(tts)
    stream = tts.stream(conn_options=NO_RETRY)
    request = tts.synthesize("One request.", conn_options=NO_RETRY)

    tts.update_options(**OPTIONS)
    tts.update_options(speed=OPTIONS["speed"])

    async with stream:
        stream.push_text("Hello.")
        stream.end_input()
        await _duration_of(stream)
    await request.collect()
    await _speak(tts, "Hello.")

    first_messages = _first_messages(fake)
    assert (first_messages[1]["voice_id"], fake.posted[0]["voice_id"]) == ("v1", "v1")
    assert {name: first_messages[2][name] for name in SENT_OPTIONS} == SENT_OPTIONS
    assert len(fake.sockets) == 1
    assert tts.model == OPTIONS["model"]


async def test_update_options_none_clears_an_option(fake: _FakeServer) -> None:
    tts = fake.tts(**OPTIONS)
    tts.update_options(language=None)

    await _speak(tts, "Hello.")
    await tts.synthesize("Hello.", conn_options=NO_RETRY).collect()

    output_format = {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 24000}
    sent = {**SENT_OPTIONS, "output_format": output_format, "text": "Hello."}
    del sent["language"]
    first = {"type": "text", "context_id": fake.received[0]["context_id"], "timestamps": True}
    assert fake.received[0] == {**first, **sent}
    assert fake.posted == [sent]


async def test_language_keeps_its_region_for_english_only(fake: _FakeServer) -> None:
    tts = fake.tts(language="hi-IN")
    await _speak(tts, "Hello.")
    tts.update_options(language="en-US")
    await _speak(tts, "Hello.")

    assert [first["language"] for first in _first_messages(fake)] == ["hi", "en-US"]


async def test_stream_option_is_sent_only_to_the_realtime_api(fake: _FakeServer) -> None:
    tts = fake.tts(use_normalized_timestamps=False)
    await _speak(tts, "Hello.")
    await tts.synthesize("Hello.", conn_options=NO_RETRY).collect()
    tts.update_options(use_normalized_timestamps=None)
    await _speak(tts, "Hello.")

    first, cleared = _first_messages(fake)
    assert first["use_normalized_timestamps"] is False  # False is a value, sent as any other
    assert "use_normalized_timestamps" not in cleared
    assert "use_normalized_timestamps" not in fake.posted[0]


async def test_client_ref_starts_the_logged_api_id(
    fake: _FakeServer, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="livekit.plugins.plivo")

    await _speak(fake.tts(client_ref="agent-7"), "Hello.")

    logged = [getattr(record, "plivo_api_id", None) for record in caplog.records]
    assert [api_id for api_id in logged if api_id] == ["agent-7-api-1"]


async def test_concurrent_streams_share_one_connection(fake: _FakeServer) -> None:
    """Each stream gets only its own audio and words; the metrics report the reused connection."""
    tts = fake.tts()
    metrics: list[TTSMetrics] = []
    tts.on("metrics_collected", metrics.append)
    fake.audio_interval = 0.001
    texts = ["The first stream.", "A second one.", "A third."]

    first_two = await asyncio.gather(*(_spoken_text(tts, text) for text in texts[:2]))
    after = await _spoken_text(tts, texts[2])

    assert [*first_two, after] == [(text, text) for text in texts]  # its audio, its words
    # the audio of the two contexts came interleaved, not one context after the other
    assert fake.spoke != sorted(fake.spoke, key=fake.spoke.index)
    assert len(fake.sockets) == 1
    reused = sorted((m.connection_reused, m.acquire_time > 0) for m in metrics)
    assert reused == [(False, True), (True, False), (True, False)]
    assert {(m.request_id, m.segment_id) for m in metrics} == {(c, c) for c in _context_ids(fake)}
    assert "cancel" not in {message["type"] for message in fake.received}


async def test_flush_speaks_the_text_so_far(fake: _FakeServer) -> None:
    async with fake.tts().stream(conn_options=NO_RETRY) as stream:
        stream.flush()  # before any text: not sent
        stream.push_text("Hello")
        stream.flush()
        first = await asyncio.wait_for(anext(stream), timeout=1)
        stream.end_input()
        rest = [ev async for ev in stream]

    assert sum(ev.frame.duration for ev in [first, *rest]) == _expected_duration("Hello")
    sent = [(message["type"], message.get("text")) for message in fake.received]
    assert sent == [("text", "Hello"), ("flush", None), ("flush", None), ("text", None)]


async def test_open_input_is_not_timed(fake: _FakeServer) -> None:
    """Only the wait after end_input() is timed: a flush that brings no audio does not time out."""
    tts = fake.tts()
    await _open_connection(tts)

    def sent() -> list[str]:
        return [message["type"] for message in fake.received[3:]]

    # a timeout of 0: a timed wait would fail at once
    async with tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0)) as stream:
        stream.push_text("A")
        stream.flush()
        stream.flush()
        await wait_until(lambda: sent() == ["text", "flush", "flush"])
        stream.flush()
        await wait_until(lambda: sent() == ["text", "flush", "flush", "flush"])


async def test_timestamps_become_timed_words(fake: _FakeServer) -> None:
    frames = await _stream_frames(fake.tts(), "Hi\n there")

    words = [word for frame in frames for word in frame.userdata[USERDATA_TIMED_TRANSCRIPT]]
    assert words == ["Hi\n", " there"]  # a space alone is no word: it joins the next one
    assert [(word.start_time, word.end_time) for word in words] == [(0.0, 0.3), (0.3, 0.9)]


async def test_timestamps_event_gives_its_last_word_at_once(fake: _FakeServer) -> None:
    """An event's last word comes with the audio after it, not with the next event."""
    frames = await _stream_frames(fake.tts(), "Hello world. Again.")

    words = [frame.userdata[USERDATA_TIMED_TRANSCRIPT] for frame in frames]
    assert [word for frame_words in words[:-1] for word in frame_words] == ["Hello ", "world."]
    assert words[-1] == [" Again."]


async def test_audio_has_the_requested_sample_rate(fake: _FakeServer) -> None:
    tts = fake.tts(sample_rate=16000)

    frames = await _stream_frames(tts, "Hello.")
    frame = await tts.synthesize("Hello.", conn_options=NO_RETRY).collect()

    assert {(f.sample_rate, f.num_channels) for f in [*frames, frame]} == {(16000, 1)}


async def test_synthesize_sends_one_request(fake: _FakeServer) -> None:
    tts = fake.tts(voice="v2", language="en_in", speed=1.1)

    async with tts.synthesize("One request. ", conn_options=NO_RETRY) as stream:
        events = [ev async for ev in stream]

    assert sum(ev.frame.duration for ev in events) == _expected_duration("One request. ")
    assert {ev.request_id for ev in events} == {"plivo-api-1"}
    options = {
        "model_id": "koyal-v1",
        "voice_id": "v2",
        "language": "en-IN",
        "pace": 1.1,
        "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 24000},
    }
    assert fake.posted == [{**options, "text": "One request. "}]
    assert fake.sockets == []


async def test_synthesize_error_is_an_api_status_error(fake: _FakeServer) -> None:
    fake.refusals = ["voice_not_found"]

    with pytest.raises(APIStatusError) as exc:
        await fake.tts().synthesize("Hello.", conn_options=ONE_RETRY).collect()

    error = exc.value
    assert (error.status_code, error.request_id, error.retryable) == (404, "plivo-api-1", False)
    assert json.loads(error.body)["code"] == "voice_not_found"
    assert len(fake.posted) == 1


async def test_synthesize_server_error_is_retried(fake: _FakeServer) -> None:
    fake.refusals = ["internal_error"]

    frame = await fake.tts().synthesize("Hello.", conn_options=ONE_RETRY).collect()

    assert frame.duration == _expected_duration("Hello.")
    assert len(fake.posted) == 2


@pytest.mark.parametrize(
    ("answer", "body"),
    [("bad gateway", '{"message": "Bad Gateway"}'), ("bad page", "Passerelle incorrecte �")],
    ids=["bad gateway", "bad page"],
)
async def test_synthesize_error_from_a_proxy_keeps_its_status(
    fake: _FakeServer, answer: str, body: str
) -> None:
    fake.answers = [answer]

    with pytest.raises(APIStatusError) as exc:
        await fake.tts().synthesize("Hello.", conn_options=NO_RETRY).collect()

    assert (exc.value.status_code, exc.value.body) == (502, body)


@pytest.mark.parametrize(
    ("answer", "conn_options", "error"),
    [("abort", NO_RETRY, APIConnectionError), ("stall", SHORT_TIMEOUT, APITimeoutError)],
    ids=["abort", "stall"],
)
async def test_synthesize_body_cut_or_stalled_raises(
    fake: _FakeServer, answer: str, conn_options: APIConnectOptions, error: type[Exception]
) -> None:
    fake.answers = [answer]

    with pytest.raises(error) as exc:
        async with asyncio.timeout(2):
            await fake.tts().synthesize("Hello.", conn_options=conn_options).collect()

    assert type(exc.value) is error  # APITimeoutError is a subclass of APIConnectionError


@pytest.mark.parametrize("timeout", [0.05, 0.0])
async def test_synthesize_connect_timeout(fake: _FakeServer, timeout: float) -> None:
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=1)) as session:
        tts = fake.tts(http_session=session)
        await _open_connection(tts)  # the WebSocket now holds the session's only connection
        conn_options = APIConnectOptions(max_retry=0, timeout=timeout)

        with pytest.raises(APITimeoutError):
            async with asyncio.timeout(2):
                await tts.synthesize("Hello.", conn_options=conn_options).collect()
        await tts.aclose()


async def test_empty_stream_sends_nothing(fake: _FakeServer) -> None:
    tts = fake.tts()

    assert await _speak(tts) == 0
    assert await _speak(tts, " ", "\n") == 0
    assert (fake.sockets, fake.received) == ([], [])


async def test_interrupted_stream_cancels_and_keeps_the_connection(fake: _FakeServer) -> None:
    tts = fake.tts()
    stream = await _open_stream(fake, tts)

    await stream.aclose()

    cancel = {"type": "cancel", "context_id": fake.received[0]["context_id"]}
    await wait_until(lambda: fake.received[-1] == cancel)
    assert await _speak(tts, "The next stream.") == _expected_duration("The next stream.")
    assert len(fake.sockets) == 1


async def test_error_event_fails_the_stream(fake: _FakeServer) -> None:
    """A 4xx is not retried, and the input task stops although the input is still open."""
    fake.refusals = ["voice_not_found"]
    async with fake.tts().stream(conn_options=ONE_RETRY) as stream:
        stream.push_text("Hello.")
        with pytest.raises(APIStatusError) as exc:
            await _duration_of(stream)
        input_tasks = [t for t in asyncio.all_tasks() if "_input_task" in repr(t.get_coro())]

    error = exc.value
    assert error.message == f"Plivo error voice_not_found: {MESSAGES['voice_not_found']}"
    assert (error.status_code, error.request_id, error.retryable) == (404, "plivo-api-1", False)
    assert error.body["code"] == "voice_not_found"
    assert len(_context_ids(fake)) == 1
    assert input_tasks == []


@pytest.mark.parametrize("code", ["concurrency_limit", "at_capacity", "internal_error"])
async def test_retryable_error_event_is_retried(fake: _FakeServer, code: str) -> None:
    fake.refusals = [code]

    duration = await _speak(fake.tts(), "Hello.", conn_options=ONE_RETRY)

    assert duration == _expected_duration("Hello.")
    assert len(_context_ids(fake)) == 2
    assert len(fake.sockets) == 1


async def test_closed_connection_fails_open_streams(fake: _FakeServer) -> None:
    tts = fake.tts()
    first = await _open_stream(fake, tts)
    other = await _open_stream(fake, tts)
    fake.answers = ["close"]  # the server ends the first stream, then closes the connection

    first.end_input()
    assert await _duration_of(first) == _expected_duration("A stream that stays open")
    with pytest.raises(APIConnectionError, match="Plivo connection closed: close code 1012"):
        await _duration_of(other)

    assert await _speak(tts, "Hello again.") == _expected_duration("Hello again.")
    assert len(fake.sockets) == 2


async def test_error_without_a_context_is_logged(
    fake: _FakeServer, caplog: pytest.LogCaptureFixture
) -> None:
    """The connection stays open after such an error."""
    fake.answers = ["fault"]

    assert await _speak(fake.tts(), "Hello.") == _expected_duration("Hello.")

    assert f"Plivo error internal_error: {MESSAGES['internal_error']}" in caplog.messages
    assert len(fake.sockets) == 1


async def test_event_without_code_or_message_is_logged(
    fake: _FakeServer, caplog: pytest.LogCaptureFixture
) -> None:
    """The connection stays open after an event with no context and no code or message."""
    fake.answers = ["bare"]

    assert await _speak(fake.tts(), "Hello.") == _expected_duration("Hello.")

    assert "Plivo notice None: None" in caplog.messages
    assert len(fake.sockets) == 1


async def test_second_end_event_keeps_the_connection(fake: _FakeServer) -> None:
    """An error after a context's done changes nothing, and the other streams go on."""
    tts = fake.tts()
    other = await _open_stream(fake, tts)
    fake.answers = ["again"]

    assert await _speak(tts, "Hello.") == _expected_duration("Hello.")

    other.end_input()
    assert await _duration_of(other) == _expected_duration("A stream that stays open")
    assert len(fake.sockets) == 1


async def test_earlier_error_is_not_the_reason_for_a_later_close(fake: _FakeServer) -> None:
    tts = fake.tts()
    fake.answers = ["fault"]
    assert await _speak(tts, "Hello.") == _expected_duration("Hello.")
    first = await _open_stream(fake, tts)
    other = await _open_stream(fake, tts)
    fake.answers = ["close"]

    first.end_input()
    assert await _duration_of(first) == _expected_duration("A stream that stays open")
    with pytest.raises(APIConnectionError, match="Plivo connection closed: close code 1012"):
        await _duration_of(other)


async def test_error_before_the_close_is_the_streams_error(fake: _FakeServer) -> None:
    """An account with no credit gets the API's error, not a lost connection, and no retry."""
    fake.ended_handshakes = ["insufficient_balance"]

    with pytest.raises(APIStatusError) as exc:
        await _speak(fake.tts(), "Hello.", conn_options=ONE_RETRY)

    error = exc.value
    assert error.message == f"Plivo error insufficient_balance: {MESSAGES['insufficient_balance']}"
    assert (error.status_code, error.retryable) == (400, False)
    assert len(fake.sockets) == 1


async def test_text_over_a_message_limit_keeps_the_other_streams(fake: _FakeServer) -> None:
    """A text too long for one message goes in several, so it closes no connection."""
    tts = fake.tts()
    other = await _open_stream(fake, tts)
    other_id = fake.received[0]["context_id"]
    text = "x" * (2**20 + 1)

    def texts() -> list[str]:
        return [m["text"] for m in fake.received if m["context_id"] != other_id and m.get("text")]

    async with tts.stream(conn_options=NO_RETRY) as stream:
        stream.push_text(text)
        async with asyncio.timeout(2):
            while sum(map(len, texts())) < len(text):
                await asyncio.sleep(0.01)

    other.end_input()
    assert await _duration_of(other) == _expected_duration("A stream that stays open")
    assert len(fake.sockets) == 1
    assert "".join(texts()) == text
    assert max(map(len, texts())) == plivo.tts.MAX_TEXT_LENGTH


async def test_opening_over_the_text_limit_is_refused(fake: _FakeServer) -> None:
    """As the API: a first text over 20,000 characters is refused, not cut."""
    with pytest.raises(APIStatusError, match="text_too_long"):
        await _speak(fake.tts(), "x" * (MAX_TEXT + 1))


async def test_closing_connection_takes_no_new_stream(fake: _FakeServer) -> None:
    """aiohttp marks a dropped WebSocket closed inside receive(), before the receive loop ends."""
    tts = fake.tts()
    await _open_connection(tts)
    assert tts._connection
    tts._connection._ws._closed = True  # as aiohttp sets it when its close begins

    assert await _speak(tts, "Hello.") == _expected_duration("Hello.")
    assert len(fake.sockets) == 2


async def test_silent_connection_is_replaced(
    fake: _FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A connection with no message for SILENT_LIMIT has a dead path: a new stream opens another."""
    monkeypatch.setattr(plivo.tts, "SILENT_LIMIT", 0)  # every connection is silent at once
    tts = fake.tts()
    assert await _speak(tts, "Hello.") == _expected_duration("Hello.")

    assert await _speak(tts, "Hello again.") == _expected_duration("Hello again.")

    assert len(fake.sockets) == 2
    async with asyncio.timeout(2):
        while fake.ended_sockets < 1:
            await asyncio.sleep(0)


async def test_server_ping_is_answered_and_keeps_the_connection(fake: _FakeServer) -> None:
    """A ping from the server is answered and counts as a message, so the connection is kept."""
    tts = fake.tts()
    assert await _speak(tts, "Hello.") == _expected_duration("Hello.")
    assert tts._connection
    tts._connection._received -= plivo.tts.SILENT_LIMIT  # silent until the ping

    await fake.sockets[0].ping(b"alive")
    async with asyncio.timeout(2):
        await fake.ponged.wait()

    assert fake.pongs == [b"alive"]
    assert await _speak(tts, "Hello again.") == _expected_duration("Hello again.")
    assert len(fake.sockets) == 1


async def test_pong_that_cannot_be_written_holds_up_no_audio(fake: _FakeServer) -> None:
    """A pong is written in a task of its own, so the receive loop still routes the audio."""
    tts = fake.tts()
    connection = (await tts._current_connection(1.0))[0]
    connection._ws.pong = _held_until_aborted(connection._ws)  # type: ignore[method-assign]
    fake.answers = ["ping"]

    assert await _speak(tts, "Hello.", conn_options=NO_RETRY) == _expected_duration("Hello.")


async def test_pong_that_fails_logs_nothing(
    fake: _FakeServer, caplog: pytest.LogCaptureFixture
) -> None:
    """A pong written as the connection drops fails, and asyncio reports nothing."""
    tts = fake.tts()

    async def fail_a_pong() -> None:
        connection = (await tts._current_connection(1.0))[0]
        called = asyncio.Event()

        async def reset(*_: Any, **__: Any) -> None:
            called.set()
            raise ConnectionResetError

        connection._ws.pong = reset  # type: ignore[method-assign]
        await fake.sockets[0].ping(b"alive")
        async with asyncio.timeout(2):
            await called.wait()
        await asyncio.wait((connection.close(),))

    # in a function of its own, so that the connection is freed when it returns
    await fail_a_pong()
    await _speak(tts, "Hello.")  # on a new connection: the TTS lets go of the closed one
    gc.collect()  # asyncio reports an unread exception when its task is collected

    assert not [record for record in caplog.records if record.name == "asyncio"]


@pytest.mark.parametrize("ran", [False, True])
async def test_connection_cancelled_by_the_loop_shutdown_still_ends(
    fake: _FakeServer, ran: bool
) -> None:
    """The shutdown cancels the task that owns the connection, even before it ran."""
    tts = fake.tts()
    connection = (await tts._current_connection(1.0))[0]  # before the connection's task runs
    connection._ws.close = _returning  # type: ignore[method-assign]
    if ran:
        await asyncio.sleep(0)  # it waits for a message
    context = connection.open_context("open", MagicMock(), text="Hello.")

    connection._recv_task.cancel()  # as the shutdown does, which then waits for the task
    await asyncio.wait((connection._recv_task,))

    assert isinstance(context.ended.exception(), APIConnectionError)
    assert connection._ws._response.closed  # its connection is back in the session's pool
    async with asyncio.timeout(2):
        while fake.ended_sockets < 1:
            await asyncio.sleep(0)


async def test_stream_ends_when_plivo_ends_its_context_first(fake: _FakeServer) -> None:
    """Plivo ends a context after 60 s with no text or audio as if its finish came."""
    tts = fake.tts()
    stream = await _open_stream(fake, tts)

    context_id = fake.received[-1]["context_id"]  # its held text is spoken, then its done
    await fake.sockets[0].send_bytes(_audio_message(context_id))
    await fake.sockets[0].send_str(json.dumps({"type": "done", "context_id": context_id}))

    await _duration_of(stream)


async def test_ending_streams_do_not_cut_the_answer_to_a_server_close(fake: _FakeServer) -> None:
    """A stream that fails at Plivo's close ends the connection again, while it answers."""
    tts = fake.tts()
    stream = await _open_stream(fake, tts)
    assert tts._connection
    answer, answered = asyncio.Event(), asyncio.Event()

    async def close(*_: Any, **__: Any) -> bool:
        await answer.wait()
        answered.set()
        return True

    tts._connection._ws.close = close  # type: ignore[method-assign]
    server_close = asyncio.create_task(fake.sockets[0].close(code=1012))

    with pytest.raises(APIConnectionError, match="close code 1012"):
        await _duration_of(stream)  # its context is released: the connection is asked to end
    answer.set()

    async with asyncio.timeout(2):
        await answered.wait()
        await server_close


async def test_handshake_that_ends_as_its_stream_ends_leaves_no_socket(fake: _FakeServer) -> None:
    """A handshake that ends in the pass that its stream is interrupted is released, not kept."""
    stream = fake.tts().stream(conn_options=NO_RETRY)
    ws_connect = aiohttp.ClientSession.ws_connect

    async def connect(*args: Any, **kwargs: Any) -> aiohttp.ClientWebSocketResponse[bool]:
        try:
            return await ws_connect(*args, **kwargs)
        finally:
            stream._task.cancel()  # the stream ends in the pass that its handshake ends

    with patch.object(aiohttp.ClientSession, "ws_connect", connect):
        stream.push_text("Hello.")
        await asyncio.wait((stream._task,))

    await wait_until(lambda: fake.ended_sockets == 1)


async def test_close_that_cannot_end_aborts_the_transport(
    fake: _FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plivo.tts, "CLOSE_TIMEOUT", 0)
    tts = fake.tts()
    connection = (await tts._current_connection(1.0))[0]
    connection._ws.close = _held_until_aborted(connection._ws)  # type: ignore[method-assign]
    await asyncio.sleep(0)  # the connection's task waits for a message

    async with asyncio.timeout(2):
        await tts.aclose()
        while fake.ended_sockets < 1:
            await asyncio.sleep(0)


async def test_error_with_no_fields_before_the_close_fails_the_stream(fake: _FakeServer) -> None:
    """An error with no context and none of its fields still ends the stream, as the API's error."""
    fake.ended_handshakes = ["bare"]

    with pytest.raises(APIStatusError, match="Plivo error None: None"):
        await _speak(fake.tts(), "Hello.", conn_options=NO_RETRY)


@pytest.mark.parametrize(
    ("options", "refused", "status", "retryable"),
    [({"auth_token": "wrong"}, [], 401, False), ({}, ["concurrency_limit"], 429, True)],
    ids=["unauthorized", "concurrency_limit"],
)
async def test_refused_handshake_is_an_api_status_error(
    fake: _FakeServer, options: dict[str, str], refused: list[str], status: int, retryable: bool
) -> None:
    fake.refused_handshakes = refused

    with pytest.raises(APIStatusError) as exc:
        await _speak(fake.tts(**options), "Hello.")

    error = exc.value
    # no __cause__: aiohttp's error holds the Authorization header
    refusal = (error.status_code, error.retryable, error.request_id, error.__cause__)
    assert refusal == (status, retryable, "plivo-api-1", None)


async def test_connect_timeout_is_per_stream(fake: _FakeServer) -> None:
    fake.completes_handshakes = False
    tts = fake.tts()
    first = tts.stream(conn_options=SHORT_TIMEOUT)
    first.push_text("Hello.")
    conn_options = APIConnectOptions(max_retry=0, timeout=0.01)

    with pytest.raises(APITimeoutError):
        await _speak(tts, "Hello.", conn_options=conn_options)

    # the second stream waited for the lock without a timeout, then timed only its own connect
    assert not tts._connection_lock.locked()


async def test_unreachable_server_is_a_connection_error(fake: _FakeServer) -> None:
    tts = fake.tts()
    await fake.server.close()

    with pytest.raises(APIConnectionError) as exc:
        await _speak(tts, "Hello.")

    assert exc.value.retryable


async def test_timeout_cancels_only_its_own_context(fake: _FakeServer) -> None:
    """The next stream gets a new connection, and the other stream ends on the first one."""
    tts = fake.tts()
    other = await _open_stream(fake, tts)
    fake.answers = ["stall"]

    with pytest.raises(APITimeoutError):
        await _speak(tts, "A stream that the server never ends.", conn_options=SHORT_TIMEOUT)

    cancel = {"type": "cancel", "context_id": fake.received[1]["context_id"]}
    await wait_until(lambda: fake.received[-1] == cancel)
    assert await _speak(tts, "Hello again.") == _expected_duration("Hello again.")
    assert len(fake.sockets) == 2
    other.end_input()
    assert await _duration_of(other) == _expected_duration("A stream that stays open")
    await wait_until(lambda: fake.ended_sockets == 1)  # the first one closed with its last stream


async def test_unreadable_message_closes_the_connection(
    fake: _FakeServer, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plivo.tts, "CLOSE_TIMEOUT", 0.01)  # the server never answers the close
    tts = fake.tts()
    stream = await _open_stream(fake, tts)
    fake.answers_closes = False

    await fake.sockets[0].send_str("not one of Plivo's messages")
    with pytest.raises(APIConnectionError, match="close code 1006"):
        await _duration_of(stream)

    assert "failed to read a message from Plivo" in caplog.text
    assert await _speak(tts, "Hello again.") == _expected_duration("Hello again.")
    assert len(fake.sockets) == 2


async def test_interrupt_during_handshake_leaks_no_connection(fake: _FakeServer) -> None:
    tts = fake.tts()
    stream = tts.stream(conn_options=NO_RETRY)
    stream.push_text("Hello.")

    await _interrupt_when(stream, lambda: len(fake.sockets) == 1)

    assert await _speak(tts, "Hello.") == _expected_duration("Hello.")
    assert len(fake.sockets) == 2  # the interrupted handshake did not go on to open one
    await tts.aclose()
    await wait_until(lambda: fake.ended_sockets == len(fake.sockets))


class _FakeWebSocket(FakeWebSocket):
    """Upstream's in-memory WebSocket, with aiohttp's writer, response and read error."""

    _writer = MagicMock()

    def __init__(self) -> None:
        super().__init__()
        self._response = MagicMock(headers={})

    def exception(self) -> BaseException | None:
        return None


@pytest.mark.virtual_time
async def test_audio_resets_the_timeout(caplog: pytest.LogCaptureFixture) -> None:
    """In memory: over a real socket, virtual time runs ahead of bytes still on their way."""
    ws = _FakeWebSocket()
    session = MagicMock(spec=aiohttp.ClientSession)
    session.ws_connect.return_value = ws
    tts = plivo.TTS(auth_id=AUTH_ID, auth_token=AUTH_TOKEN, http_session=session)
    conn_options = APIConnectOptions(max_retry=0, timeout=0.1)
    text = "x" * 105  # 105 messages 1 ms apart: longer than the timeout in all

    async with tts, tts.stream(conn_options=conn_options) as stream:
        stream.push_text(text)
        stream.end_input()
        await wait_until(lambda: ws.sent)
        context_id = json.loads(ws.sent[0])["context_id"]
        for char in text:
            await asyncio.sleep(0.001)
            audio = _audio_message(context_id, char)
            ws.incoming.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.BINARY, audio, None))
        ws.respond({"type": "done", "context_id": context_id})

        assert await _duration_of(stream) == _expected_duration(text)

    await asyncio.sleep(2 * plivo.tts.DEAD_LIMIT)  # a closed connection is not watched
    assert "silent" not in caplog.text


@pytest.mark.virtual_time
async def test_silent_connection_fails_a_stream_whose_input_is_open() -> None:
    """In memory: messages keep a connection, and DEAD_LIMIT of silence ends it and its streams."""
    ws = _FakeWebSocket()
    session = MagicMock(spec=aiohttp.ClientSession)
    session.ws_connect.return_value = ws
    tts = plivo.TTS(auth_id=AUTH_ID, auth_token=AUTH_TOKEN, http_session=session)

    async with tts, tts.stream(conn_options=NO_RETRY) as stream:
        stream.push_text("A stream that stays open")
        await wait_until(lambda: ws.sent)
        for _ in range(3):  # a message every half DEAD_LIMIT, for longer than DEAD_LIMIT
            await asyncio.sleep(plivo.tts.DEAD_LIMIT / 2)
            ws.incoming.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.PONG, b"", None))
        assert tts._connection and not tts._connection._recv_task.done()

        async with asyncio.timeout(2 * plivo.tts.DEAD_LIMIT):
            with pytest.raises(APIConnectionError, match="close code 1006"):
                async for _ in stream:
                    pass


async def test_synthesize_releases_its_response(
    fake: _FakeServer, caplog: pytest.LogCaptureFixture
) -> None:
    """A synthesize closed during its audio releases its response, not aiohttp's finalizer."""
    asyncio.get_running_loop().set_debug(True)  # aiohttp reports an unclosed response then
    fake.answers = ["stall"]  # the body stays open
    stream = fake.tts().synthesize("Hello.", conn_options=NO_RETRY)
    await anext(aiter(stream))

    await stream.aclose()
    gc.collect()

    assert "Unclosed response" not in caplog.text


@pytest.mark.virtual_time
async def test_synthesize_times_its_connect_and_reads_apart() -> None:
    """In memory: aiohttp times the connect and each read, not one deadline for both."""

    async def chunks() -> AsyncIterator[tuple[bytes, bool]]:
        yield b"\0\0" * SAMPLES_100MS, False

    async def post(*_: Any, **__: Any) -> MagicMock:
        await asyncio.sleep(1.5)  # a connect and a wait for the headers, each within its 1 s
        resp = MagicMock(status=200, headers={"api-id": "plivo-api-1"})
        resp.content.iter_chunks = chunks
        return resp

    session = MagicMock(spec=aiohttp.ClientSession)
    session.post = post
    tts = plivo.TTS(auth_id=AUTH_ID, auth_token=AUTH_TOKEN, http_session=session)
    conn_options = APIConnectOptions(max_retry=0, timeout=1.0)

    frame = await tts.synthesize("Hello.", conn_options=conn_options).collect()

    assert frame.duration == pytest.approx(0.1, abs=0.011)


@pytest.mark.virtual_time
@pytest.mark.parametrize("kind", ["synthesize", "stream"])
async def test_close_while_connecting_is_not_retried(kind: str) -> None:
    """In memory: a cancel stays a cancel where aiohttp's timer turns it into a timeout (3.10)."""
    calls = 0

    async def connect(*_: Any, **__: Any) -> None:
        nonlocal calls
        calls += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise asyncio.TimeoutError from None  # as aiohttp's timer can on Python 3.10

    session = MagicMock(spec=aiohttp.ClientSession)
    session.post = session.ws_connect = connect
    tts = plivo.TTS(auth_id=AUTH_ID, auth_token=AUTH_TOKEN, http_session=session)
    if kind == "synthesize":
        stream: Any = tts.synthesize("Hello.", conn_options=ONE_RETRY)
    else:
        stream = tts.stream(conn_options=ONE_RETRY)
        stream.push_text("Hello.")
    await wait_until(lambda: calls == 1)

    async with asyncio.timeout(5):
        await stream.aclose()
    assert calls == 1  # closed, not retried as a timeout


async def test_synthesize_timeout_is_per_read(fake: _FakeServer) -> None:
    """The request has no total timeout, so a long text streams for longer: each read is timed."""
    with patch.object(fake.session, "post", wraps=fake.session.post) as post:
        await fake.tts().synthesize("Hello.", conn_options=NO_RETRY).collect()

    timeout = post.call_args.kwargs["timeout"]
    assert (timeout.total, timeout.sock_read) == (None, NO_RETRY.timeout)


async def test_prewarm_opens_the_connection(
    fake: _FakeServer, caplog: pytest.LogCaptureFixture
) -> None:
    """A refused prewarm is logged, not raised, and the next one opens the connection."""
    tts = fake.tts()
    fake.refused_handshakes = ["at_capacity"]

    tts.prewarm()
    await tts._prewarm_task
    assert [record.exception_type for record in caplog.records] == ["WSServerHandshakeError"]
    tts.prewarm()
    await wait_until(lambda: len(fake.sockets) == 1)

    await _speak(tts, "Hello.")
    assert len(fake.sockets) == 1


async def test_aclose_cancels_a_prewarm_still_connecting(fake: _FakeServer) -> None:
    fake.completes_handshakes = False
    tts = fake.tts()

    tts.prewarm()
    first = tts._prewarm_task
    await wait_until(lambda: len(fake.held) == 1)  # the first holds the lock in its handshake
    tts.prewarm()  # starts nothing while the first connects
    await tts.aclose()

    assert first is not None and first.done()
    await wait_until(lambda: fake.held[0].transport is None)  # the client dropped the handshake


async def test_cancelled_aclose_still_ends_streams_and_connection(fake: _FakeServer) -> None:
    """Cancelled while it still ends the streams, aclose() goes on to close the connection."""
    tts = fake.tts()
    stream = await _open_stream(fake, tts)
    ending, end_stream = asyncio.Event(), stream.aclose

    async def aclose() -> None:
        ending.set()
        await end_stream()

    stream.aclose = aclose  # type: ignore[method-assign]
    shutdown = asyncio.create_task(tts.aclose())
    async with asyncio.timeout(2):
        await ending.wait()  # it waits on the stream's end, before any connection

    shutdown.cancel()

    async with asyncio.timeout(2):
        while fake.ended_sockets < 1:
            await asyncio.sleep(0)
    assert fake.received[-1] == {"type": "cancel", "context_id": fake.received[0]["context_id"]}


async def test_aclose_closes_streams_and_connection(fake: _FakeServer) -> None:
    """An aclose() that is itself cancelled is finished by the next one."""
    tts = fake.tts()
    await _open_stream(fake, tts)
    receive_task = tts._connection._recv_task
    closing = asyncio.create_task(tts.aclose())
    await asyncio.sleep(0)
    closing.cancel()

    await tts.aclose()

    assert receive_task.done()
    await wait_until(lambda: fake.ended_sockets == 1)
    assert fake.received[-1] == {"type": "cancel", "context_id": fake.received[0]["context_id"]}


async def test_closed_connection_stops_the_send_loop(fake: _FakeServer) -> None:
    tts = fake.tts()
    await _open_connection(tts)
    send_task = tts._connection._send_task

    await fake.sockets[0].close()

    await wait_until(send_task.done)


async def test_failed_write_is_a_connection_error(
    fake: _FakeServer, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The connection's first write fails here, while the stream's input is still open."""

    async def send_str(*args: object, **kwargs: object) -> None:
        raise ConnectionError("Connection lost")  # as aiohttp on a lost connection

    monkeypatch.setattr(aiohttp.ClientWebSocketResponse, "send_str", send_str)

    async with fake.tts().stream(conn_options=ONE_RETRY) as stream:
        stream.push_text("Hello.")
        with pytest.raises(APIConnectionError):
            await _duration_of(stream)

    assert "failed to send a message to Plivo" in caplog.text
    assert len(fake.sockets) == 2  # the retry did not take the failed connection
    await wait_until(lambda: fake.ended_sockets == 2)


async def test_stream_during_a_failed_write_gets_a_new_connection(
    fake: _FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The next stream starts as the write fails, while the WebSocket is still closing."""
    tts = fake.tts()
    await _open_connection(tts)
    next_stream = tts.stream(conn_options=NO_RETRY)
    send_str = aiohttp.ClientWebSocketResponse.send_str

    async def fail_once(*args: object, **kwargs: object) -> None:
        monkeypatch.setattr(aiohttp.ClientWebSocketResponse, "send_str", send_str)
        next_stream.push_text("Hello again.")
        raise ConnectionError("Connection lost")

    monkeypatch.setattr(aiohttp.ClientWebSocketResponse, "send_str", fail_once)
    with pytest.raises(APIConnectionError):
        await _speak(tts, "Hello.")
    next_stream.end_input()

    assert await _duration_of(next_stream) == _expected_duration("Hello again.")
    assert len(fake.sockets) == 2


async def test_interrupt_at_first_message_keeps_other_streams(fake: _FakeServer) -> None:
    """Its queued first message goes out before its cancel, and the other stream is not cut."""
    tts = fake.tts()
    other = await _open_stream(fake, tts)
    stream = tts.stream(conn_options=NO_RETRY)
    stream.push_text("Hello.")

    await _interrupt_when(stream, lambda: len(tts._connection._contexts) == 2)

    other.end_input()
    assert await _duration_of(other) == _expected_duration("A stream that stays open")
    first, cancel = fake.received[1:3]
    assert first["text"] == "Hello."
    assert cancel == {"type": "cancel", "context_id": first["context_id"]}


async def test_interrupt_during_connection_close_logs_nothing(
    fake: _FakeServer, caplog: pytest.LogCaptureFixture
) -> None:
    """asyncio must not report the error of a context whose stream no longer waits for it."""
    tts = fake.tts()

    async def interrupt_as_the_connection_closes() -> None:
        stream = await _open_stream(fake, tts)
        closing = asyncio.create_task(stream.aclose())
        await asyncio.sleep(0)
        assert tts._connection._contexts  # the cancelled stream's context is still registered
        tts._connection._recv_task.cancel()
        await closing

    # in a function of its own, so that the stream is freed when it returns
    await interrupt_as_the_connection_closes()
    await _speak(tts, "Hello.")  # on a new connection: the TTS lets go of the closed one
    gc.collect()  # asyncio reports an unread exception when its future is collected

    assert not [record for record in caplog.records if record.name == "asyncio"]


async def test_lost_connection_is_retried(fake: _FakeServer) -> None:
    async with fake.tts().stream(conn_options=ONE_RETRY) as stream:
        stream.push_text("Hello.")
        await wait_until(lambda: len(fake.received) == 1)
        await fake.sockets[0].close()  # before any audio
        stream.end_input()
        assert await _duration_of(stream) == _expected_duration("Hello.")

    assert len(fake.sockets) == 2


async def test_warning_is_logged(fake: _FakeServer, caplog: pytest.LogCaptureFixture) -> None:
    """The API drops the text past a context's limit with a warning, and the stream goes on."""
    async with fake.tts().stream(conn_options=NO_RETRY) as stream:
        stream.push_text("x" * MAX_TEXT)
        stream.push_text("y")
        stream.flush()
        await asyncio.wait_for(anext(stream), timeout=1)

    assert "Plivo warning text_too_long: text dropped" in caplog.messages
