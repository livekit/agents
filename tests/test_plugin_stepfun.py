"""Unit tests for the StepFun STT plugin."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import aiohttp
import numpy as np
import pytest

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, stt
from livekit.plugins import stepfun
from livekit.plugins.stepfun import _utils
from livekit.plugins.stepfun.tts import (
    _await_connection_done,
    _Connection,
    _escape_text,
    _split_for_synthesis,
    _split_for_wire,
)

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time]


@pytest.fixture(autouse=True)
def _stepfun_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests independent of the developer's provider credentials."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    monkeypatch.delenv("STEPFUN_CLUSTER", raising=False)


class _FakeWSMessage:
    def __init__(self, type_: aiohttp.WSMsgType, data: Any, extra: Any = None) -> None:
        self.type = type_
        self.data = data
        # aiohttp carries the WebSocket close code in `data` and reason in
        # `extra` for closed messages.
        self.extra = extra


class _FakeWebSocket:
    """A good-enough stand-in for aiohttp.ClientWebSocketResponse."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self.close_code: int | None = None
        self._incoming: asyncio.Queue[_FakeWSMessage] = asyncio.Queue()
        # Acknowledge commits separately from transcription events. Tests can
        # disable this to simulate a missing acknowledgement.
        self.ack_commits = True
        self._acks: list[_FakeWSMessage] = []

    # -- the interface the plugin calls ------------------------------------

    async def send_str(self, data: str) -> None:
        self.sent.append(json.loads(data))
        if self.ack_commits and self.sent[-1].get("type") == "input_audio_buffer.commit":
            ack = {"type": "input_audio_buffer.committed", "item_id": "item_buffer"}
            self._acks.append(_FakeWSMessage(aiohttp.WSMsgType.TEXT, json.dumps(ack)))

    async def ping(self) -> None:
        pass

    async def close(self) -> None:
        self.closed = True

    async def receive(self) -> _FakeWSMessage:
        if self._acks:
            return self._acks.pop(0)
        return await self._incoming.get()

    # -- the interface tests drive -----------------------------------------

    def queue_server_event(self, event: dict[str, Any]) -> None:
        self._incoming.put_nowait(_FakeWSMessage(aiohttp.WSMsgType.TEXT, json.dumps(event)))

    def queue_binary(self, data: bytes) -> None:
        self._incoming.put_nowait(_FakeWSMessage(aiohttp.WSMsgType.BINARY, data))

    def queue_close(self, code: int | None = None, reason: str | None = None) -> None:
        self._incoming.put_nowait(_FakeWSMessage(aiohttp.WSMsgType.CLOSED, code, extra=reason))

    def queue_error(self) -> None:
        self._incoming.put_nowait(
            _FakeWSMessage(aiohttp.WSMsgType.ERROR, ConnectionResetError("connection reset"))
        )

    # -- assertion helpers -------------------------------------------------

    def frames_of_type(self, type_: str) -> list[dict[str, Any]]:
        """Every frame we sent with the given ``type``."""
        return [f for f in self.sent if f.get("type") == type_]

    def data_of_type(self, type_: str) -> list[dict[str, Any]]:
        """The ``data`` payloads of frames with the given ``type``.

        The streaming TTS protocol wraps every field in ``data``, so asserting
        on the payload directly keeps tests readable.
        """
        return [f.get("data") or {} for f in self.sent if f.get("type") == type_]

    def types_sent(self) -> list[str]:
        return [f.get("type", "") for f in self.sent]


class _FakeConnectionPool:
    """Stands in for ``utils.ConnectionPool``.

    Hands out one preconfigured connection and records whether the plugin
    returned it to the pool or dropped it. Both streaming surfaces are supposed
    to drop theirs, since neither socket is safe to reuse afterwards.
    """

    last_acquire_time = 0.25
    last_connection_reused = False

    def __init__(self, conn: Any) -> None:
        self._conn = conn
        self.removed: list[Any] = []
        self.prewarm_calls = 0
        self.invalidate_calls = 0

    @contextlib.asynccontextmanager
    async def connection(self, *, timeout: float) -> AsyncIterator[Any]:
        # Mirrors utils.ConnectionPool: a failed attempt drops the connection
        # rather than returning it, so callers must not rely on removing it
        # themselves.
        try:
            yield self._conn
        except BaseException:
            self.remove(self._conn)
            raise

    def remove(self, conn: Any) -> None:
        self.removed.append(conn)

    def prewarm(self) -> None:
        self.prewarm_calls += 1

    def invalidate(self) -> None:
        self.invalidate_calls += 1

    async def aclose(self) -> None:
        pass


_SAMPLE_RATE = 16000
_SESSION_READY = {"type": "session.updated", "meta": {"session_id": "sess-1"}}


def _frame(ms: int = 100) -> rtc.AudioFrame:
    samples = _SAMPLE_RATE * ms // 1000
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=_SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=samples,
    )


async def _drive(
    s: stepfun.STT,
    server_events: list[dict],
    *,
    frames: int = 1,
    close_socket: bool = True,
) -> tuple[_FakeWebSocket, list[stt.SpeechEvent]]:
    """Run one stream end to end against a scripted server."""
    ws = _FakeWebSocket()
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]

    for event in server_events:
        ws.queue_server_event(event)
    if close_socket:
        ws.queue_close()

    stream = s.stream()
    for _ in range(frames):
        stream.push_frame(_frame())
    stream.end_input()

    try:
        events = [ev async for ev in stream]
    finally:
        await stream.aclose()
    return ws, events


def _texts(events: list[stt.SpeechEvent], kind: stt.SpeechEventType) -> list[str]:
    return [ev.alternatives[0].text for ev in events if ev.type == kind]


async def _collect(stream: stt.RecognizeStream) -> list[stt.SpeechEvent]:
    return [event async for event in stream]


class _FakeHTTPResponse:
    status = 200
    headers = {"x-request-id": "http-smoke"}

    def __init__(self, lines: list[bytes]) -> None:
        self.content = self._Content(lines)

    class _Content:
        def __init__(self, lines: list[bytes]) -> None:
            self.lines = lines

        async def __aiter__(self):
            for line in self.lines:
                yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _FakeHTTPSession:
    def __init__(self, lines: list[bytes]) -> None:
        self.response = _FakeHTTPResponse(lines)
        self.url = ""
        self.payload: dict[str, Any] = {}
        self.headers: dict[str, str] = {}
        self.requests: list[dict[str, Any]] = []

    def post(self, url: str, *, json: dict, headers: dict, **kwargs: object):
        self.url, self.payload, self.headers = url, json, headers
        self.requests.append(dict(json))
        return self.response

    async def close(self) -> None:
        return None


_DELTA = "conversation.item.input_audio_transcription.delta"
_COMPLETED = "conversation.item.input_audio_transcription.completed"


def test_defaults_to_global_cluster() -> None:
    s = stepfun.STT()
    assert s._opts.cluster == "global"
    assert s._opts.ws_base_url == "wss://api.stepfun.ai"
    assert s._opts.http_base_url == "https://api.stepfun.ai"


def test_env_cluster_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STEPFUN_CLUSTER", "china")
    s = stepfun.STT()
    assert s._opts.cluster == "china"
    assert s._opts.ws_base_url == "wss://api.stepfun.com"


async def test_sends_session_update_before_audio() -> None:
    """The endpoint accepts no query parameters.

    Model, audio format and turn detection are configured by this frame alone,
    so skipping it leaves the session unconfigured and nothing is ever
    recognized.
    """
    s = stepfun.STT(language="zh", prompt="livekit, webrtc")
    ws, _ = await _drive(s, [_SESSION_READY, {"type": _COMPLETED, "transcript": ""}])

    assert ws.types_sent()[0] == "session.update"

    update = ws.frames_of_type("session.update")[0]
    audio_input = update["session"]["audio"]["input"]

    assert audio_input["format"] == {
        "type": "pcm",
        "codec": "pcm_s16le",
        "rate": _SAMPLE_RATE,
        "bits": 16,
        "channel": 1,
    }
    assert audio_input["transcription"]["model"] == "stepaudio-2.5-asr-stream"
    assert audio_input["transcription"]["language"] == "zh"
    assert audio_input["transcription"]["prompt"] == "livekit, webrtc"
    assert audio_input["turn_detection"]["type"] == "server_vad"

    # Every client frame carries its own event_id.
    assert all("event_id" in frame for frame in ws.sent)


async def test_audio_is_sent_as_base64_pcm() -> None:
    s = stepfun.STT()
    ws, _ = await _drive(s, [_SESSION_READY, {"type": _COMPLETED, "transcript": ""}], frames=1)

    appends = ws.frames_of_type("input_audio_buffer.append")
    assert appends, "no audio was sent"
    # 100ms of input at a 50ms frame size.
    assert len(appends) > 2  # real input plus endpointing silence
    assert sum(len(_utils.b64_to_pcm(f["audio"])) for f in appends[2:]) >= (
        _SAMPLE_RATE * s._opts.vad_silence_duration_ms // 1000 * 2
    )
    assert _utils.b64_to_pcm(appends[0]["audio"]) == b"\x00\x00" * (_SAMPLE_RATE // 20)


@pytest.mark.parametrize("sample_rate", [16000, 48000])
@pytest.mark.parametrize("channels", [1, 2, 3])
async def test_stream_downmixes_before_resampling(sample_rate: int, channels: int) -> None:
    count = sample_rate // 10
    # Loud channels catch int16 overflow; averaging must not double duration.
    samples = np.tile(np.array([30000, 10000, -10000][:channels], dtype=np.int16), (count, 1))
    frame = rtc.AudioFrame(
        data=samples.tobytes(),
        sample_rate=sample_rate,
        num_channels=channels,
        samples_per_channel=count,
    )
    mono = rtc.AudioFrame(
        data=samples.mean(axis=1).astype(np.int16).tobytes(),
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=count,
    )
    expected = mono.data.tobytes()
    if sample_rate != _SAMPLE_RATE:
        resampler = rtc.AudioResampler(
            sample_rate, _SAMPLE_RATE, quality=rtc.AudioResamplerQuality.HIGH
        )
        expected = b"".join(f.data.tobytes() for f in [*resampler.push(mono), *resampler.flush()])
    engine = stepfun.STT()
    ws = _FakeWebSocket()
    for event in (_SESSION_READY, {"type": _COMPLETED, "transcript": "hello"}):
        ws.queue_server_event(event)
    ws.queue_close()
    engine._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]
    stream = engine.stream(conn_options=APIConnectOptions(max_retry=0))
    try:
        stream.push_frame(frame)
        stream.end_input()
        await _collect(stream)
        pcm = b"".join(
            _utils.b64_to_pcm(f["audio"]) for f in ws.frames_of_type("input_audio_buffer.append")
        )
        padding = _SAMPLE_RATE * (engine._opts.vad_silence_duration_ms + 50) // 1000 * 2
        assert len(pcm) == len(expected) + padding
        assert pcm[len(expected) :] == b"\x00" * padding
        if sample_rate == _SAMPLE_RATE:
            assert pcm[: len(expected)] == expected
        else:
            # Independent resamplers dither PCM quantization by a sample unit.
            actual_samples = np.frombuffer(pcm[: len(expected)], dtype=np.int16).astype(np.int32)
            expected_samples = np.frombuffer(expected, dtype=np.int16).astype(np.int32)
            assert np.max(np.abs(actual_samples - expected_samples)) <= 2
        assert len(expected) / 2 / _SAMPLE_RATE == pytest.approx(0.1, abs=0.002)
    finally:
        await engine.aclose()


async def test_final_transcript_read_from_transcript_field() -> None:
    # The completed frame carries the final text under `transcript`, not `text`.
    s = stepfun.STT()
    _, events = await _drive(
        s,
        [
            _SESSION_READY,
            {"type": _DELTA, "text": "partial"},
            {"type": _COMPLETED, "transcript": "the final text"},
        ],
    )

    assert _texts(events, stt.SpeechEventType.FINAL_TRANSCRIPT) == ["the final text"]


async def test_batch_asr_posts_pcm_and_reads_sse_final() -> None:
    session = _FakeHTTPSession(
        [
            b'data: {"type":"transcript.text.delta","delta":"Hello "}\n',
            b'data: {"type":"transcript.text.done","text":"Hello world",'
            b'"meta":{"session_id":"batch-1"},"usage":{"input_tokens":4}}\n',
        ]
    )
    engine = stepfun.STT(language="en", http_session=session)  # type: ignore[arg-type]

    event = await engine.recognize(_frame(ms=100))

    assert session.url.endswith("/v1/audio/asr/sse")
    assert session.headers["Accept"] == "text/event-stream"
    assert session.payload["audio"]["input"]["transcription"]["language"] == "en"
    assert _utils.b64_to_pcm(session.payload["audio"]["data"])
    assert event.alternatives[0].text == "Hello world"


async def test_speech_event_id_can_differ_from_transcription_item_id() -> None:
    """StepFun uses separate IDs for VAD and transcription items."""
    s = stepfun.STT()
    _, events = await _drive(
        s,
        [
            _SESSION_READY,
            {"type": "input_audio_buffer.speech_started", "item_id": "vad-item"},
            {
                "type": "conversation.item.created",
                "item": {"id": "transcription-item", "status": "in_progress"},
            },
            {"type": _DELTA, "item_id": "transcription-item", "text": "hello"},
            {"type": "input_audio_buffer.speech_stopped", "item_id": "vad-item"},
            {"type": _COMPLETED, "item_id": "transcription-item", "transcript": "hello"},
        ],
    )

    assert _texts(events, stt.SpeechEventType.FINAL_TRANSCRIPT) == ["hello"]


async def test_stash_delta_only_emits_interim_transcript() -> None:
    """A stash delta must not start and cancel preemptive generation."""
    s = stepfun.STT()
    _, events = await _drive(
        s,
        [
            _SESSION_READY,
            {"type": _DELTA, "text": "hello world", "stash": "world"},
            {"type": _COMPLETED, "transcript": "hello world"},
        ],
    )

    assert _texts(events, stt.SpeechEventType.PREFLIGHT_TRANSCRIPT) == []
    assert _texts(events, stt.SpeechEventType.INTERIM_TRANSCRIPT) == ["hello world"]


async def test_server_vad_turn_settles_without_a_final() -> None:
    """A missing final ends the stream without fabricating a transcript."""
    stream = stepfun.STT().stream(conn_options=APIConnectOptions(timeout=0.01))
    stream._input_done = True
    stream._input_finished_at = 0.0
    stream._last_event_at = 0.0
    stream._consumed_audio = True
    stream._open_items.add("transcription-item")
    stream._interim_texts["transcription-item"] = "partial"
    receiver = asyncio.create_task(asyncio.Event().wait())
    try:
        await stream._await_finalization(receiver)
        assert not receiver.done()
    finally:
        receiver.cancel()
        await asyncio.gather(receiver, return_exceptions=True)
        await stream.aclose()


@pytest.mark.parametrize("result", ["final", "interim", "textless"])
async def test_scripted_stream_drains_without_synthetic_results(
    result: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.plugins.stepfun import stt as stt_impl

    # Drive idle deadlines with the virtual loop, not real wall time.
    monkeypatch.setattr(
        stt_impl, "time", SimpleNamespace(monotonic=asyncio.get_running_loop().time)
    )
    events = [
        _SESSION_READY,
        {"type": "input_audio_buffer.speech_started", "item_id": "vad"},
        {"type": "conversation.item.created", "item": {"id": "turn", "status": "in_progress"}},
        {"type": "input_audio_buffer.speech_stopped", "item_id": "vad"},
    ]
    if result != "textless":
        events.append({"type": _DELTA, "item_id": "turn", "text": "hello"})
    if result == "final":
        events.append({"type": _COMPLETED, "item_id": "turn", "transcript": "hello"})
    if result == "textless":
        with pytest.raises(APIConnectionError, match="cannot be recovered"):
            await _drive(stepfun.STT(), events, close_socket=False)
    else:
        _, output = await _drive(stepfun.STT(), events, close_socket=False)
        assert _texts(output, stt.SpeechEventType.INTERIM_TRANSCRIPT) == ["hello"]
        assert _texts(output, stt.SpeechEventType.FINAL_TRANSCRIPT) == (
            ["hello"] if result == "final" else []
        )


@pytest.mark.parametrize(
    "case", ["empty", "whitespace", "another-item", "earlier-final", "active", "anonymous"]
)
async def test_untranscribed_pending_turn_does_not_succeed(case: str) -> None:
    engine = stepfun.STT()
    stream = engine.stream(conn_options=APIConnectOptions(timeout=0.01))
    stream._input_done = True
    stream._consumed_audio = True
    stream._open_items.add("pending")
    if case == "whitespace":
        stream._interim_texts["pending"] = "  "
    elif case == "another-item":
        stream._interim_texts["other"] = "hello"
    elif case == "earlier-final":
        stream._finals_received = 1
    elif case in ("active", "anonymous"):
        stream._interim_texts["pending"] = "hello"
        stream._speech_active = case == "active"
        stream._anonymous_open = case == "anonymous"
    receiver = asyncio.create_task(asyncio.Event().wait())
    try:
        with pytest.raises(APIConnectionError, match="timed out waiting for final"):
            await stream._await_finalization(receiver)
    finally:
        receiver.cancel()
        await asyncio.gather(receiver, return_exceptions=True)
        await engine.aclose()


async def test_update_options_refreshes_active_stream_options() -> None:
    """Existing streams use the latest options when they reconnect."""
    s = stepfun.STT(language="zh")
    ws = _FakeWebSocket()
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]

    stream = s.stream()
    s.update_options(language="ja")

    assert stream._opts.language == "ja"
    await stream.aclose()


async def test_prompt_update_preserves_stream_language_override() -> None:
    engine = stepfun.STT(language="en")
    stream = engine.stream(language="ja")
    try:
        engine.update_options(prompt="LiveKit")
        config = stream._session_update_message()["session"]["audio"]["input"]["transcription"]
        assert config["language"] == "ja"
        assert config["prompt"] == "LiveKit"
        engine.update_options(model="custom-model")
        assert stream._opts.language == "ja"
        engine.update_options(language="zh")
        assert stream._opts.language == "zh"
    finally:
        await engine.aclose()


async def test_active_stream_sends_full_acknowledged_option_update() -> None:
    """An active session receives a complete update and waits for its ack."""
    s = stepfun.STT(language="zh")
    ws = _ReadyPeer()
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]
    stream = s.stream(conn_options=APIConnectOptions(timeout=1))
    consumer = asyncio.create_task(_collect(stream))
    try:
        await asyncio.wait_for(ws.session_updates.get(), 1)  # initial config
        s.update_options(language="ja", prompt="Use natural punctuation.")
        update = await asyncio.wait_for(ws.session_updates.get(), 1)
        audio_input = update["session"]["audio"]["input"]
        assert audio_input["format"]["codec"] == "pcm_s16le"
        assert audio_input["transcription"]["language"] == "ja"
        assert audio_input["transcription"]["prompt"] == "Use natural punctuation."
        assert audio_input["turn_detection"]["type"] == "server_vad"
        stream.end_input()
        await asyncio.wait_for(consumer, 1)
    finally:
        await stream.aclose()


class _SequencedPool(_FakeConnectionPool):
    """Hands out a different socket per attempt, to observe retries."""

    def __init__(self, sockets: list[_FakeWebSocket]) -> None:
        super().__init__(sockets[0])
        self._sockets = sockets
        self.handed_out = 0

    @contextlib.asynccontextmanager
    async def connection(self, *, timeout: float):
        self.handed_out += 1
        yield self._sockets.pop(0)


class _ReadyPeer(_FakeWebSocket):
    """Acknowledges session.update, then lets tests script the rest."""

    def __init__(self) -> None:
        super().__init__()
        self.session_updates: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def send_str(self, data: str) -> None:
        await super().send_str(data)
        if self.sent[-1]["type"] == "session.update":
            self.session_updates.put_nowait(self.sent[-1])
            self.queue_server_event(_SESSION_READY)


async def test_connection_loss_after_audio_is_not_retried_even_after_final() -> None:
    """Without audio offsets, retrying can still drop or duplicate speech."""

    class Failing(_ReadyPeer):
        async def send_str(self, data: str) -> None:
            await super().send_str(data)
            if self.sent[-1]["type"] == "input_audio_buffer.append":
                for event in (
                    {"type": "input_audio_buffer.speech_started", "item_id": "item-1"},
                    {"type": "input_audio_buffer.speech_stopped", "item_id": "item-1"},
                    {"type": _COMPLETED, "item_id": "item-1", "transcript": "hello"},
                ):
                    self.queue_server_event(event)
                self.queue_error()

    pool = _SequencedPool([Failing(), _ReadyPeer()])
    s = stepfun.STT()
    s._pool = pool  # type: ignore[assignment]
    stream = s.stream(conn_options=APIConnectOptions(max_retry=3, retry_interval=0, timeout=1))
    stream.push_frame(_frame())
    stream.end_input()
    try:
        with pytest.raises(APIConnectionError, match="cannot be recovered") as excinfo:
            await asyncio.wait_for(_collect(stream), 5)
        assert excinfo.value.retryable is False
        assert pool.handed_out == 1
    finally:
        await stream.aclose()


async def test_connection_loss_with_speech_in_flight_fails_instead_of_retrying() -> None:
    """Do not retry consumed audio while its transcription is pending."""

    class Failing(_ReadyPeer):
        async def send_str(self, data: str) -> None:
            await super().send_str(data)
            if self.sent[-1]["type"] == "input_audio_buffer.append":
                self.queue_server_event(
                    {"type": "input_audio_buffer.speech_started", "item_id": "item-1"}
                )
                self.queue_error()

    pool = _SequencedPool([Failing(), _ReadyPeer()])
    s = stepfun.STT()
    s._pool = pool  # type: ignore[assignment]
    stream = s.stream(conn_options=APIConnectOptions(max_retry=3, retry_interval=0, timeout=1))
    stream.push_frame(_frame())
    stream.end_input()
    try:
        with pytest.raises(APIConnectionError, match="cannot be recovered") as excinfo:
            await asyncio.wait_for(_collect(stream), 5)
        assert excinfo.value.retryable is False
        assert pool.handed_out == 1, "retried despite in-flight speech"
    finally:
        await stream.aclose()


async def test_stream_send_error_after_audio_is_not_retryable() -> None:
    """A failed upload consumed audio that a fresh session cannot replay."""

    class BrokenSocket(_FakeWebSocket):
        async def send_str(self, data: str) -> None:
            if "input_audio_buffer.append" in data:
                raise aiohttp.ClientConnectionError("upload failed")
            await super().send_str(data)

    ws = BrokenSocket()
    ws.queue_server_event(_SESSION_READY)
    s = stepfun.STT()
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]
    stream = s.stream(conn_options=APIConnectOptions(max_retry=0, retry_interval=0))
    stream.push_frame(_frame())
    stream.end_input()
    try:
        with pytest.raises(APIConnectionError) as excinfo:
            await asyncio.wait_for(anext(stream), 1)
        assert excinfo.value.message == (
            "stepfun stt: connection lost before final transcription; "
            "in-flight speech cannot be recovered by a retry"
        )
        cause = excinfo.value.__cause__
        assert isinstance(cause, APIConnectionError)
        assert cause.message == "stepfun stt stream: upload failed"
        assert excinfo.value.retryable is False
        assert ws.closed
    finally:
        await s.aclose()


def test_defaults() -> None:
    t = stepfun.TTS()
    assert t.model == "stepaudio-3-tts"
    assert t.provider == "StepFun"
    assert t._opts.cluster == "global"
    # An unknown voice is rejected outright with 400 voice_id_invalid, so the
    # default has to be a real ID present on both clusters.
    assert t._opts.voice == "vibrant-youth"
    assert t.sample_rate == 24000
    assert t.num_channels == 1


async def test_http_tts_streams_sse_audio() -> None:
    pcm = b"\x00\x00" * 1200
    session = _FakeHTTPSession(
        [
            b'data: {"type":"speech.audio.delta","audio":"'
            + _utils.pcm_to_b64(pcm).encode()
            + b'"}\n',
            b'data: {"type":"speech.audio.done"}\n',
        ]
    )
    engine = stepfun.TTS(http_session=session)  # type: ignore[arg-type]
    stream = engine.synthesize("hello")

    audio = [chunk async for chunk in stream]

    assert session.url.endswith("/v1/audio/speech")
    assert session.headers["Accept"] == "text/event-stream"
    assert session.payload["input"] == "hello"
    assert audio
    assert sum(chunk.frame.samples_per_channel for chunk in audio) >= 1200
    await stream.aclose()


@pytest.mark.parametrize("word", ["weather", "weather."])
async def test_long_http_synthesis_keeps_word_intact(word: str) -> None:
    text = "A " * 499 + word + " tomorrow."
    session = _FakeHTTPSession(
        [
            b'data: {"type":"speech.audio.delta","audio":"'
            + _utils.pcm_to_b64(_PCM).encode()
            + b'"}\n',
            b'data: {"type":"speech.audio.done"}\n',
        ]
    )
    engine = stepfun.TTS(http_session=session)  # type: ignore[arg-type]
    stream = engine.synthesize(text, conn_options=APIConnectOptions(max_retry=0))
    try:
        assert [chunk async for chunk in stream]
        chunks = [request["input"] for request in session.requests]
        assert "".join(chunks) == text
        assert all(len(chunk) <= 1000 for chunk in chunks)
        assert any(word in chunk for chunk in chunks)
    finally:
        await engine.aclose()


@pytest.mark.parametrize(
    "text",
    [
        "One sentence. " + "word " * 250,
        "中文。" + "字" * 1200,
        "x" * 2301,
        " " * 1200 + "word",
        "",
        "word " * 400,
        "word " * 198 + "end." + " " * 10,
    ],
)
def test_synthesis_split_preserves_text_and_respects_cap(text: str) -> None:
    chunks = _split_for_synthesis(text)
    assert "".join(chunks) == text
    assert all(len(chunk) <= 1000 for chunk in chunks)
    if text.startswith("One sentence."):
        assert chunks[0] == "One sentence."
    if text.startswith("word "):
        assert all(chunk.strip() for chunk in chunks)
    assert "".join(_split_for_wire(text)) == text


@pytest.mark.parametrize(
    "text,expected", [("select(yes)", "select yes "), ("select（yes）", "select yes ")]
)
def test_parentheses_do_not_merge_spoken_words(text: str, expected: str) -> None:
    engine = stepfun.TTS()
    assert _escape_text(text, engine._opts) == expected
    assert _escape_text(text, stepfun.TTS(enable_inline_instruction=True)._opts) == text
    assert _escape_text(text, stepfun.TTS(model="step-tts-mini")._opts) == text


async def test_handshake_close_reports_code_and_reason() -> None:
    """The close code is the difference between "the socket died" and a
    diagnosable failure, so it must reach the error message."""
    ws = _FakeWebSocket()
    ws.queue_close(code=1001, reason="going away")

    with pytest.raises(APIConnectionError, match="close_code=1001") as excinfo:
        await _await_connection_done(ws)  # type: ignore[arg-type]

    assert "reason='going away'" in str(excinfo.value)


async def test_create_frame_uses_documented_field_names() -> None:
    t = stepfun.TTS(voice="lively-girl", speed=1.2, volume=0.8, language="en")
    ws, _ = await _drive_tts(t, ["hello"], [_CREATED, _audio_delta(), _AUDIO_DONE])

    create = ws.data_of_type("tts.create")
    assert len(create) == 1
    data = create[0]

    assert data["session_id"] == _SESSION
    # Note the suffixed names: the streaming endpoint takes voice_id /
    # speed_ratio / volume_ratio where the HTTP one takes voice / speed / volume.
    assert data["voice_id"] == "lively-girl"
    assert data["speed_ratio"] == 1.2
    assert data["volume_ratio"] == 0.8
    assert data["response_format"] == "pcm"
    assert data["sample_rate"] == 24000
    assert data["language"] == "en"
    assert data["mode"] == "default"


async def test_audio_deltas_decoded_and_emitted() -> None:
    t = stepfun.TTS()
    _, frames = await _drive_tts(
        t, ["hello"], [_CREATED, _audio_delta(), _audio_delta(), _AUDIO_DONE]
    )

    assert frames, "no audio emitted"
    assert sum(f.frame.samples_per_channel for f in frames) > 0
    assert frames[-1].is_final
    # The segment id is the server's session id, which also records it on the
    # trace so a trace can be lined up against the provider's own logs.
    assert frames[0].segment_id == _SESSION


_SESSION = "sess-abc"
_CREATED = {"type": "tts.response.created", "data": {"session_id": _SESSION}}

#: 50ms of silence at 24kHz, so pushed_duration ends up above zero — the base
#: class raises if text was synthesized but no audio ever arrived.
_PCM = b"\x00\x00" * 1200


def _audio_delta(pcm: bytes = _PCM, status: str = "unfinished") -> dict:
    return {
        "type": "tts.response.audio.delta",
        "data": {
            "session_id": _SESSION,
            "status": status,
            "audio": _utils.pcm_to_b64(pcm),
        },
    }


_AUDIO_DONE = {
    "type": "tts.response.audio.done",
    "data": {"session_id": _SESSION, "audio": ""},
}


class _ScriptedWebSocket(_FakeWebSocket):
    """Answers only after the client sends what a real server waits for.

    Events up to tts.response.created are released by tts.create; the rest are
    released by tts.text.done. Pre-queuing everything would let a stream pass
    even if it never sent its text or finished input.
    """

    def __init__(self, server_events: list[dict]) -> None:
        super().__init__()
        split = next(
            (i + 1 for i, e in enumerate(server_events) if e.get("type") == _CREATED["type"]),
            len(server_events),
        )
        self._on_create = server_events[:split]
        self._on_done = server_events[split:]

    async def send_str(self, data: str) -> None:
        await super().send_str(data)
        sent_type = self.sent[-1].get("type")
        if sent_type == "tts.create":
            batch, self._on_create = self._on_create, []
        elif sent_type == "tts.text.done":
            batch, self._on_done = self._on_done, []
        else:
            return
        for event in batch:
            self.queue_server_event(event)


async def _drive_tts(
    t: stepfun.TTS,
    tokens: list[str],
    server_events: list[dict],
    *,
    flush_between: bool = False,
    conn_options: APIConnectOptions | None = None,
) -> tuple[_FakeWebSocket, list]:
    """Run one streaming segment end to end against a scripted server."""
    ws = _ScriptedWebSocket(server_events)
    t._pool = _FakeConnectionPool(_Connection(ws=ws, session_id=_SESSION))  # type: ignore[assignment]

    stream = t.stream(conn_options=conn_options or APIConnectOptions())
    for token in tokens:
        stream.push_text(token)
        if flush_between:
            stream.flush()
    stream.end_input()

    frames = [frame async for frame in stream]
    await stream.aclose()
    return ws, frames
