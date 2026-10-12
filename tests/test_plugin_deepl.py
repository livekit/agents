"""DeepL Voice duplex model, against a loopback server speaking the Voice API protocol."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import numpy as np
import pytest
from aiohttp import web

from livekit import rtc
from livekit.agents import APIConnectOptions, function_tool, llm
from livekit.agents.types import TimedString
from livekit.plugins.deepl.realtime import VoiceModel, VoiceSession, voice_model

pytestmark = pytest.mark.unit

_FAST = APIConnectOptions(max_retry=2, retry_interval=0.01, timeout=2.0)
_SAMPLE_RATE_OUT = voice_model.OUTPUT_SAMPLE_RATE
_SAMPLE_RATE_IN = voice_model.INPUT_SAMPLE_RATE


@dataclass
class _Request:
    method: str
    token: str | None
    headers: dict[str, str]
    body: dict[str, Any] | None


@dataclass
class _Conn:
    token: str
    ws: web.WebSocketResponse
    received: list[dict[str, Any]] = field(default_factory=list)
    received_at: list[float] = field(default_factory=list)

    def chunks(self) -> list[tuple[float, bytes]]:
        return [
            (at, base64.b64decode(msg["source_media_chunk"]["data"]))
            for at, msg in zip(self.received_at, self.received, strict=True)
            if "source_media_chunk" in msg
        ]


class _FakeDeepL:
    """Issues sessions and tokens over HTTP and accepts their streams, recording all of it."""

    def __init__(self) -> None:
        self.base_url = ""
        self.requests: list[_Request] = []
        self.conns: list[_Conn] = []
        self.new_conn: asyncio.Queue[_Conn] = asyncio.Queue()
        # statuses to answer the next requests of each method with, in order
        self.fail: dict[str, list[int]] = {"POST": [], "GET": []}
        # stream connections to refuse before accepting one
        self.reject_streams = 0
        self.fail_body: dict[str, Any] = {"message": "rejected", "code": "test"}
        self._issued = 0

    async def _session(self, request: web.Request) -> web.Response:
        body = await request.json() if request.method == "POST" else None
        self.requests.append(
            _Request(
                method=request.method,
                token=request.query.get("token"),
                headers=dict(request.headers),
                body=body,
            )
        )
        if self.fail[request.method]:
            return web.json_response(
                self.fail_body,
                status=self.fail[request.method].pop(0),
                headers={"X-Trace-ID": "trace-1"},
            )
        self._issued += 1
        return web.json_response(
            {
                "streaming_url": f"{self.base_url.replace('http', 'ws')}/v3/voice/realtime/connect",
                "token": f"t{self._issued}",
                "session_id": "session-1",
            }
        )

    async def _stream(self, request: web.Request) -> web.StreamResponse:
        if self.reject_streams:
            self.reject_streams -= 1
            return web.Response(status=403)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        conn = _Conn(token=request.query.get("token", ""), ws=ws)
        self.conns.append(conn)
        self.new_conn.put_nowait(conn)
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            event = json.loads(msg.data)
            conn.received.append(event)
            conn.received_at.append(time.monotonic())
            if "end_of_source_media" in event:
                await ws.send_json({"end_of_stream": {}})
                await ws.close()
        return ws


@pytest.fixture
async def server() -> AsyncIterator[_FakeDeepL]:
    fake = _FakeDeepL()
    app = web.Application()
    app.router.add_post("/v3/voice/realtime", fake._session)
    app.router.add_get("/v3/voice/realtime", fake._session)
    app.router.add_get("/v3/voice/realtime/connect", fake._stream)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    fake.base_url = f"http://127.0.0.1:{runner.addresses[0][1]}"
    try:
        yield fake
    finally:
        await runner.cleanup()


@pytest.fixture
def _no_deepl_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPL_API_KEY", raising=False)
    monkeypatch.delenv("DEEPL_AUTH_KEY", raising=False)
    monkeypatch.delenv("DEEPL_BASE_URL", raising=False)


def _model(server: _FakeDeepL, **kwargs: Any) -> VoiceModel:
    kwargs.setdefault("target_language", "de")
    return VoiceModel(api_key="test-key", base_url=server.base_url, conn_options=_FAST, **kwargs)


async def _open(server: _FakeDeepL, **kwargs: Any) -> tuple[VoiceModel, VoiceSession, _Conn]:
    model = _model(server, **kwargs)
    session = model.session()
    conn = await asyncio.wait_for(server.new_conn.get(), 3)
    return model, session, conn


async def _close(model: VoiceModel, session: llm.DuplexSession | llm.RealtimeSession) -> None:
    await session.aclose()
    await model.aclose()


async def _until(predicate: Callable[[], bool], timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


def _speech(ms: int, *, text: str | None = None) -> dict[str, Any]:
    pcm = np.full(_SAMPLE_RATE_OUT * ms // 1000, 1000, dtype=np.int16).tobytes()
    chunk: dict[str, Any] = {
        "language": "de",
        "data": [base64.b64encode(pcm).decode()],
        "duration": ms,
    }
    if text is not None:
        chunk["text"] = text
    return {"target_media_chunk": chunk}


def _segments(*texts: str) -> list[dict[str, Any]]:
    return [{"language": "en", "text": t, "start_time": 0, "end_time": 0} for t in texts]


def _input_frame(ms: int, *, sample_rate: int = _SAMPLE_RATE_IN) -> rtc.AudioFrame:
    samples = sample_rate * ms // 1000
    return rtc.AudioFrame(
        data=np.zeros(samples, dtype=np.int16).tobytes(),
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


async def _collect(stream: Any, out: list[tuple[float, llm.DuplexAudioFrame]]) -> None:
    async for f in stream:
        out.append((time.monotonic(), f))


def _chain(error: BaseException | None) -> Iterator[BaseException]:
    """An exception and every exception it was raised from, as a traceback would print them."""
    while error is not None:
        yield error
        error = error.__cause__ or (None if error.__suppress_context__ else error.__context__)


def _played_s(frames: list[tuple[float, llm.DuplexAudioFrame]]) -> float:
    return sum(f.frame.duration for _, f in frames)


# configuration


@pytest.mark.usefixtures("_no_deepl_env")
def test_keys_pick_the_free_or_pro_endpoint() -> None:
    assert VoiceModel(target_language="de", api_key="k:fx")._opts.base_url == (
        voice_model.DEEPL_FREE_API_URL
    )
    assert VoiceModel(target_language="de", api_key="k")._opts.base_url == voice_model.DEEPL_API_URL
    assert (
        VoiceModel(target_language="de", api_key="k:fx", base_url="http://x")._opts.base_url
        == "http://x"
    )


@pytest.mark.usefixtures("_no_deepl_env")
def test_the_endpoint_can_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPL_BASE_URL", "https://staging.example")
    assert VoiceModel(target_language="de", api_key="k:fx")._opts.base_url == (
        "https://staging.example"
    )
    # an explicit base_url still wins
    assert (
        VoiceModel(target_language="de", api_key="k", base_url="http://x")._opts.base_url
        == "http://x"
    )


@pytest.mark.usefixtures("_no_deepl_env")
def test_an_unencrypted_endpoint_is_warned_about(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="livekit.plugins.deepl"):
        VoiceModel(target_language="de", api_key="k", base_url="http://staging.example")
        VoiceModel(target_language="de", api_key="k", base_url="http://127.0.0.1:8080")
        VoiceModel(target_language="de", api_key="k", base_url="https://staging.example")
    assert ["not https" in r.getMessage() for r in caplog.records] == [True]


def test_the_api_key_stays_out_of_reprs() -> None:
    model = VoiceModel(target_language="de", api_key="secret-key")
    assert "secret-key" not in repr(model._opts)


@pytest.mark.usefixtures("_no_deepl_env")
def test_the_key_falls_back_to_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="DEEPL_API_KEY"):
        VoiceModel(target_language="de")
    monkeypatch.setenv("DEEPL_AUTH_KEY", "auth-key")
    assert VoiceModel(target_language="de")._opts.api_key == "auth-key"
    monkeypatch.setenv("DEEPL_API_KEY", "api-key")
    assert VoiceModel(target_language="de")._opts.api_key == "api-key"


def test_a_source_language_is_fixed_unless_asked_otherwise() -> None:
    opts = VoiceModel(target_language="de", source_language="en", api_key="k")._opts
    assert opts.source_language_mode == "fixed"
    opts = VoiceModel(
        target_language="de", source_language="en", source_language_mode="auto", api_key="k"
    )._opts
    assert opts.source_language_mode == "auto"
    assert VoiceModel(target_language="de", api_key="k")._opts.source_language_mode is None
    with pytest.raises(ValueError, match="source_language"):
        VoiceModel(target_language="de", source_language_mode="fixed", api_key="k")


async def test_the_session_request_carries_the_configuration(server: _FakeDeepL) -> None:
    model, session, conn = await _open(
        server,
        source_language="en",
        voice="match",
        formality="formal",
        glossary_ids=["g1", "g2"],
    )
    await _close(model, session)

    request = server.requests[0]
    assert request.method == "POST"
    assert request.headers["Authorization"] == "DeepL-Auth-Key test-key"
    assert request.body == {
        "source_media_content_type": "audio/pcm;encoding=s16le;rate=16000",
        "target_media_languages": ["de"],
        "target_media_content_type": "audio/pcm;encoding=s16le;rate=24000",
        "message_format": "json",
        "source_language": "en",
        "source_language_mode": "fixed",
        "target_media_voice": "match",
        "formality": "formal",
        "glossary_ids": ["g1", "g2"],
    }
    assert conn.token == "t1"
    assert session.session_id == "session-1"


async def test_without_a_source_language_deepl_detects_it(server: _FakeDeepL) -> None:
    model, session, _ = await _open(server)
    await _close(model, session)

    body = server.requests[0].body
    assert body is not None
    assert "source_language" not in body and "source_language_mode" not in body


async def test_tools_are_rejected_and_instructions_ignored(server: _FakeDeepL) -> None:
    @function_tool
    async def lookup() -> str:
        """Look something up."""
        return ""

    model, session, _ = await _open(server)
    try:
        with pytest.raises(llm.RealtimeError, match="cannot call tools"):
            await session._update_tools([lookup])
        await session._update_tools([])
        await session._update_instructions("Be polite.")
    finally:
        await _close(model, session)


# input


async def test_input_goes_out_as_100ms_pcm_chunks(server: _FakeDeepL) -> None:
    model, session, conn = await _open(server)
    try:
        for _ in range(10):
            session.push_audio(_input_frame(100, sample_rate=48000))
        await _until(lambda: len(conn.chunks()) >= 8)
        assert {len(chunk) for _, chunk in conn.chunks()} == {_SAMPLE_RATE_IN // 10 * 2}
    finally:
        await _close(model, session)


async def test_closing_releases_the_native_resampler(server: _FakeDeepL) -> None:
    model, session, _ = await _open(server)
    session.push_audio(_input_frame(100, sample_rate=48000))
    assert session._input._resampler is not None
    await _close(model, session)
    # held until process exit, its native handle would be dropped after the native side let go
    assert session._input._resampler is None


async def test_a_backlog_goes_out_no_faster_than_twice_real_time(server: _FakeDeepL) -> None:
    model, session, conn = await _open(server)
    try:
        session.push_audio(_input_frame(2000))
        await _until(lambda: len(conn.chunks()) >= 20, timeout=5)
        times = [at for at, _ in conn.chunks()]
        # 20 chunks of 100 ms, each at least 50 ms after the one before
        assert times[-1] - times[0] >= 19 * 0.05 * 0.9
    finally:
        await _close(model, session)


async def test_silence_keeps_an_idle_session_alive(
    server: _FakeDeepL, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(voice_model, "_KEEPALIVE_INTERVAL_S", 0.05)
    model, session, conn = await _open(server)
    try:
        await _until(lambda: len(conn.chunks()) >= 2)
        assert all(chunk == voice_model._SILENCE_CHUNK for _, chunk in conn.chunks())
    finally:
        await _close(model, session)


# source transcript


async def test_source_transcripts_make_speaker_turns(server: _FakeDeepL) -> None:
    model, session, conn = await _open(server)
    events: list[tuple[Any, ...]] = []
    session.on("input_speech_started", lambda _: events.append(("started",)))
    session.on("input_speech_stopped", lambda _: events.append(("stopped",)))
    session.on(
        "input_audio_transcription_completed",
        lambda ev: events.append(
            ("final" if ev.is_final else "interim", ev.transcript, ev.item_id)
        ),
    )
    try:
        for concluded, tentative in (
            ([], ["Hel"]),
            (["Hello,"], [" how"]),
            ([" how are you?"], []),
        ):
            await conn.ws.send_json(
                {
                    "source_transcript_update": {
                        "concluded": _segments(*concluded),
                        "tentative": _segments(*tentative),
                    }
                }
            )
        await _until(lambda: len(events) >= 4)
        # the turn ends on the speaker's own silence, not on the transcript alone
        assert events[-1][0] == "interim"
        session.push_audio(_input_frame(1000))

        item_id = events[1][2]
        assert events == [
            ("started",),
            ("interim", "Hel", item_id),
            ("interim", "Hello, how", item_id),
            ("interim", "Hello, how are you?", item_id),
            ("final", "Hello, how are you?", item_id),
            ("stopped",),
        ]
    finally:
        await _close(model, session)


async def test_a_muted_speaker_turn_ends_on_the_wall_clock(
    server: _FakeDeepL, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a muted track sends no frames, so there is no audio to count the silence with
    monkeypatch.setattr(voice_model, "_TURN_MAX_SILENCE_MS", 200.0)
    model, session, conn = await _open(server)
    events: list[tuple[str, str]] = []
    session.on("input_speech_stopped", lambda _: events.append(("stopped", "")))
    session.on(
        "input_audio_transcription_completed",
        lambda ev: events.append(("final" if ev.is_final else "interim", ev.transcript)),
    )
    try:
        await conn.ws.send_json(
            {"source_transcript_update": {"concluded": _segments("Hello."), "tentative": []}}
        )
        await _until(lambda: ("stopped", "") in events, timeout=2)
        assert events == [("interim", "Hello."), ("final", "Hello."), ("stopped", "")]
    finally:
        await _close(model, session)


async def test_a_turn_deepl_never_concludes_ends_on_long_silence(server: _FakeDeepL) -> None:
    model, session, conn = await _open(server)
    events: list[str] = []
    session.on("input_speech_stopped", lambda _: events.append("stopped"))
    session.on(
        "input_audio_transcription_completed",
        lambda ev: events.append("final" if ev.is_final else "interim"),
    )
    try:
        await conn.ws.send_json(
            {"source_transcript_update": {"concluded": [], "tentative": _segments("Bye")}}
        )
        await _until(lambda: events == ["interim"])
        session.push_audio(_input_frame(1000))
        assert events == ["interim"]
        session.push_audio(_input_frame(2500))
        # nothing was concluded, so the turn ends without a final transcript of its own
        assert events == ["interim", "stopped"]
    finally:
        await _close(model, session)


# translated speech


async def test_translated_speech_is_paced_out_with_its_words(server: _FakeDeepL) -> None:
    model, session, conn = await _open(server)
    deltas: list[llm.DuplexOutputTranscriptDelta] = []
    session.on("transcript_delta", deltas.append)
    frames: list[tuple[float, llm.DuplexAudioFrame]] = []
    collect = asyncio.create_task(_collect(session.audio_stream, frames))
    try:
        await conn.ws.send_json(_speech(300, text="Hallo"))
        await conn.ws.send_json(_speech(300))
        await _until(lambda: _played_s(frames) >= 0.6 - 1e-6)

        # 0.6 s of speech arrived at once but plays out over about as long, less the lead
        assert frames[-1][0] - frames[0][0] >= 0.35
        assert {(f.frame.sample_rate, f.frame.samples_per_channel) for _, f in frames} == {
            (_SAMPLE_RATE_OUT, _SAMPLE_RATE_OUT // 50)
        }
        starts = [f.start_ms for _, f in frames]
        assert starts[0] is not None
        assert starts == [starts[0] + 20 * i for i in range(len(starts))]
        assert [(d.text, d.start_ms) for d in deltas] == [("Hallo", starts[0])]
    finally:
        await _close(model, session)
        await collect


async def test_a_pause_in_the_speech_is_kept_on_the_clock(server: _FakeDeepL) -> None:
    model, session, conn = await _open(server)
    deltas: list[llm.DuplexOutputTranscriptDelta] = []
    session.on("transcript_delta", deltas.append)
    frames: list[tuple[float, llm.DuplexAudioFrame]] = []
    collect = asyncio.create_task(_collect(session.audio_stream, frames))
    try:
        await conn.ws.send_json(_speech(100, text="Eins"))
        await _until(lambda: _played_s(frames) >= 0.1 - 1e-6)
        await asyncio.sleep(0.3)
        # resumes without words of its own, as the rest of a sentence would after a stall
        await conn.ws.send_json(_speech(100))
        await _until(lambda: _played_s(frames) >= 0.2 - 1e-6)

        first, resumed = frames[0][1].start_ms, frames[5][1].start_ms
        assert first is not None and resumed is not None
        assert resumed - (first + 100) >= 150
        assert [(d.text, d.start_ms) for d in deltas] == [("Eins", first), ("", resumed)]
    finally:
        await _close(model, session)
        await collect


async def test_audio_in_another_format_is_refused(server: _FakeDeepL) -> None:
    model, session, conn = await _open(server)
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)
    try:
        chunk = _speech(100, text="Hallo")
        chunk["target_media_chunk"]["content_type"] = "audio/webm;codecs=opus;"
        await conn.ws.send_json(chunk)
        await _until(lambda: bool(errors))
        assert errors[-1].recoverable is False
    finally:
        await _close(model, session)


# through the adapter


async def _drain(ev: llm.GenerationCreatedEvent) -> tuple[list[str], float]:
    texts: list[str] = []
    duration = 0.0
    async for msg in ev.message_stream:

        async def _text(msg: llm.MessageGeneration = msg) -> None:
            async for t in msg.text_stream:
                texts.append(t)

        async def _audio(msg: llm.MessageGeneration = msg) -> None:
            nonlocal duration
            async for f in msg.audio_stream:
                duration += f.duration

        await asyncio.gather(_text(), _audio())
    return texts, duration


async def _adapter_session(
    server: _FakeDeepL,
) -> tuple[VoiceModel, llm.RealtimeSession, _Conn, list[asyncio.Task[tuple[list[str], float]]]]:
    model = _model(server)
    rt = llm.DuplexRealtimeAdapter(model, audio_timeout=0.15).session()
    conn = await asyncio.wait_for(server.new_conn.get(), 3)
    drains: list[asyncio.Task[tuple[list[str], float]]] = []
    rt.on("generation_created", lambda ev: drains.append(asyncio.create_task(_drain(ev))))
    return model, rt, conn, drains


async def test_each_pause_ends_a_reply_that_carries_its_own_words(server: _FakeDeepL) -> None:
    model, rt, conn, drains = await _adapter_session(server)
    try:
        await conn.ws.send_json(_speech(200, text="Hallo."))
        await _until(lambda: len(drains) == 1 and drains[0].done())
        await conn.ws.send_json(_speech(200, text=" Wie geht's?"))
        await _until(lambda: len(drains) == 2 and drains[1].done())

        (texts1, dur1), (texts2, dur2) = drains[0].result(), drains[1].result()
        assert "".join(texts1) == "Hallo." and dur1 == pytest.approx(0.2)
        assert "".join(texts2) == " Wie geht's?" and dur2 == pytest.approx(0.2)
        assert [m.text_content for m in rt.chat_ctx.messages() if m.role == "assistant"] == [
            "Hallo.",
            " Wie geht's?",
        ]
    finally:
        await _close(model, rt)


async def test_asking_for_a_reply_explains_what_an_interpreter_does(server: _FakeDeepL) -> None:
    model, rt, _, drains = await _adapter_session(server)
    try:
        # what an on_enter greeting does: the framework logs this message and carries on
        with pytest.raises(llm.RealtimeError, match="only speaks translations"):
            await rt.generate_reply(instructions="greet the user")
        assert drains == []
    finally:
        await _close(model, rt)


async def test_speech_resuming_mid_sentence_keeps_its_captions_in_place(
    server: _FakeDeepL,
) -> None:
    model, rt, conn, drains = await _adapter_session(server)
    try:
        await conn.ws.send_json(_speech(100, text="Guten"))
        await _until(lambda: len(drains) == 1 and drains[0].done())
        # the rest of a stalled sentence opens a reply, the next words land 100 ms into it
        await conn.ws.send_json(_speech(100))
        await conn.ws.send_json(_speech(100, text=" Morgen"))
        await _until(lambda: len(drains) == 2 and drains[1].done())

        texts, duration = drains[1].result()
        assert duration == pytest.approx(0.2)
        words = [t for t in texts if t]
        assert words == [" Morgen"]
        assert isinstance(words[0], TimedString)
        assert words[0].start_time == pytest.approx(0.1, abs=0.03)
    finally:
        await _close(model, rt)


# connection lifecycle


async def test_a_dropped_connection_resumes_with_the_newest_token(server: _FakeDeepL) -> None:
    model, session, conn = await _open(server)
    reconnected = asyncio.Event()
    session.on("session_reconnected", lambda _: reconnected.set())
    try:
        await conn.ws.close()
        await asyncio.wait_for(server.new_conn.get(), 3)
        await asyncio.wait_for(reconnected.wait(), 3)
        assert [(r.method, r.token) for r in server.requests] == [("POST", None), ("GET", "t1")]
        assert [c.token for c in server.conns] == ["t1", "t2"]
    finally:
        await _close(model, session)


async def test_a_session_deepl_cannot_resume_starts_over(server: _FakeDeepL) -> None:
    model, session, conn = await _open(server)
    server.fail["GET"] = [400]
    try:
        await conn.ws.close()
        await asyncio.wait_for(server.new_conn.get(), 3)
        assert [(r.method, r.token) for r in server.requests] == [
            ("POST", None),
            ("GET", "t1"),
            ("POST", None),
        ]
    finally:
        await _close(model, session)


async def test_a_server_error_message_starts_a_new_session(server: _FakeDeepL) -> None:
    model, session, conn = await _open(server)
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)
    try:
        await conn.ws.send_json(
            {"error": {"error_code": 503, "reason_code": 5030000, "error_message": "busy"}}
        )
        await asyncio.wait_for(server.new_conn.get(), 3)
        # DeepL ended the session, so there is no token to resume it with
        assert [r.method for r in server.requests] == ["POST", "POST"]
        assert [e.recoverable for e in errors] == [True]
    finally:
        await _close(model, session)


async def test_a_client_error_message_ends_the_session(
    server: _FakeDeepL, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    model, session, conn = await _open(server)
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)
    try:
        await conn.ws.send_json(
            {
                "error": {
                    "request_type": "source_media_chunk",
                    "error_code": 400,
                    "reason_code": 4000403,
                    "error_message": "Audio format not supported",
                }
            }
        )
        await _until(lambda: bool(errors))
        await asyncio.sleep(0.2)
        assert [e.recoverable for e in errors] == [False]
        assert len(server.requests) == 1
        # DeepL's own text stays out of the exception, and is logged as pii
        assert "reason 4000403" in str(errors[0].error)
        assert "Audio format not supported" not in str(errors[0].error)
        assert "Audio format not supported" not in caplog.text
        assert any(getattr(r, "lk.pii.error", None) for r in caplog.records)
    finally:
        await _close(model, session)


async def test_a_session_error_body_stays_out_of_errors_and_logs(
    server: _FakeDeepL, caplog: pytest.LogCaptureFixture
) -> None:
    # a provider body can echo the request, so only its error code may reach an exception
    server.fail["POST"] = [403]
    server.fail_body = {"message": "token t-secret-echo is invalid", "code": "invalid_token"}
    caplog.set_level(logging.DEBUG)
    model = _model(server)
    session = model.session()
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)
    try:
        await _until(lambda: bool(errors))
        assert "invalid_token" in str(errors[0].error)
        for e in errors:
            assert all("t-secret-echo" not in str(link) for link in _chain(e.error))
        assert "t-secret-echo" not in caplog.text
        assert any("t-secret-echo" in str(getattr(r, "lk.pii.body", "")) for r in caplog.records)
    finally:
        await _close(model, session)


async def test_errors_and_logs_never_carry_the_session_token(
    server: _FakeDeepL, caplog: pytest.LogCaptureFixture
) -> None:
    # aiohttp quotes the url in its errors, and the streaming url carries the session token
    server.reject_streams = 99
    caplog.set_level(logging.DEBUG)
    model = _model(server)
    session = model.session()
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)
    try:
        await _until(lambda: any(not e.recoverable for e in errors), timeout=5)
        assert any("WSServerHandshakeError 403" in str(e.error) for e in errors)
        for e in errors:
            assert all("token=" not in str(link) for link in _chain(e.error))
        assert "token=" not in caplog.text
    finally:
        await _close(model, session)


@pytest.mark.parametrize("status", [403, 456])
async def test_a_refused_session_request_is_not_retried(server: _FakeDeepL, status: int) -> None:
    server.fail["POST"] = [status]
    model = _model(server)
    session = model.session()
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)
    try:
        await _until(lambda: bool(errors))
        await asyncio.sleep(0.2)
        assert [e.recoverable for e in errors] == [False]
        assert len(server.requests) == 1
        assert "trace-1" in str(errors[0].error)
    finally:
        await _close(model, session)


async def test_closing_ends_the_stream_cleanly(server: _FakeDeepL) -> None:
    model, session, conn = await _open(server)
    session.push_audio(_input_frame(300))
    await _until(lambda: len(conn.chunks()) >= 1)

    started = time.monotonic()
    await _close(model, session)
    assert time.monotonic() - started < 1.0
    assert conn.received[-1] == {"end_of_source_media": {}}
    assert len(server.requests) == 1
