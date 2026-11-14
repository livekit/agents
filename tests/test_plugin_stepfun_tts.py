"""Unit tests for the StepFun TTS plugin."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import aiohttp
import pytest

from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError
from livekit.agents.types import NOT_GIVEN
from livekit.plugins import stepfun
from livekit.plugins.stepfun import _utils
from livekit.plugins.stepfun.tts import _await_connection_done, _Connection, _escape_text

from .test_plugin_stepfun_stt import _FakeConnectionPool, _FakeWebSocket

pytestmark = pytest.mark.plugin("stepfun")


@pytest.fixture(autouse=True)
def _stepfun_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests independent of the developer's provider credentials."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    monkeypatch.delenv("STEPFUN_CLUSTER", raising=False)


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


async def _drive(
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


def test_capabilities() -> None:
    caps = stepfun.TTS().capabilities
    assert caps.streaming is True
    assert caps.aligned_transcript is False


def test_timestamp_enables_aligned_transcript() -> None:
    assert stepfun.TTS(timestamp=True).capabilities.aligned_transcript is True


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sample_rate": 44100}, "sample_rate"),
        ({"speed": 3.0}, "speed"),
        ({"volume": 0.0}, "volume"),
    ],
)
def test_rejects_out_of_range_options(kwargs: dict, match: str) -> None:
    # Caught locally rather than as an opaque 400 from the server.
    with pytest.raises(ValueError, match=match):
        stepfun.TTS(**kwargs)


def test_instruction_capped_per_model() -> None:
    # stepaudio-2.5-tts caps instruction at 200 characters, stepaudio-3-tts at 500.
    with pytest.raises(ValueError, match="200 characters"):
        stepfun.TTS(model="stepaudio-2.5-tts", instruction="x" * 201)

    stepfun.TTS(model="stepaudio-3-tts", instruction="x" * 500)


def test_instruction_dropped_on_unsupported_model() -> None:
    # step-tts-mini rejects `instruction` outright, so sending it would turn a
    # cosmetic option into a hard failure.
    t = stepfun.TTS(model="step-tts-mini", instruction="be cheerful")
    assert t._opts.instruction is NOT_GIVEN
    assert not t._opts.supports_instruction


async def test_handshake_captures_server_session_id() -> None:
    """The session id comes from the server, not from us.

    Inventing one client-side cannot work: every later frame has to quote the id
    the server issued in tts.connection.done.
    """
    ws = _FakeWebSocket()
    ws.queue_server_event({"type": "tts.connection.done", "data": {"session_id": "from-server"}})

    assert await _await_connection_done(ws) == "from-server"  # type: ignore[arg-type]


async def test_handshake_propagates_error_frame() -> None:
    ws = _FakeWebSocket()
    ws.queue_server_event(
        {
            "type": "tts.response.error",
            "data": {"session_id": "x", "code": "401", "message": "bad key"},
        }
    )

    with pytest.raises(APIStatusError) as excinfo:
        await _await_connection_done(ws)  # type: ignore[arg-type]
    assert excinfo.value.retryable is False


def test_tts_ws_error_text_is_not_in_exception_formatting() -> None:
    # Retry logs format the exception; the provider message may echo input text.
    secret = "sensitive speech echoed by provider"
    frame = {"type": "tts.response.error", "data": {"code": "500", "message": secret}}
    error = _utils.translate_tts_ws_error(frame, context="TTS")
    assert isinstance(error, APIStatusError)
    assert error.body == frame
    assert error.retryable
    assert secret not in str(error)
    assert secret not in repr(error)


async def test_handshake_rejects_close() -> None:
    ws = _FakeWebSocket()
    ws.queue_close()

    with pytest.raises(APIConnectionError):
        await _await_connection_done(ws)  # type: ignore[arg-type]


async def test_handshake_close_reports_code_and_reason() -> None:
    """The close code is the difference between "the socket died" and a
    diagnosable failure, so it must reach the error message."""
    ws = _FakeWebSocket()
    ws.queue_close(code=1001, reason="going away")

    with pytest.raises(APIConnectionError, match="close_code=1001") as excinfo:
        await _await_connection_done(ws)  # type: ignore[arg-type]

    assert "reason='going away'" in str(excinfo.value)


async def test_handshake_close_without_code_stays_readable() -> None:
    # Both fields are routinely absent; the suffix must not turn into "None".
    ws = _FakeWebSocket()
    ws.queue_close()

    with pytest.raises(APIConnectionError) as excinfo:
        await _await_connection_done(ws)  # type: ignore[arg-type]

    assert "close_code" not in str(excinfo.value)
    assert "(" not in str(excinfo.value)


async def test_create_frame_uses_documented_field_names() -> None:
    t = stepfun.TTS(voice="lively-girl", speed=1.2, volume=0.8, language="en")
    ws, _ = await _drive(t, ["hello"], [_CREATED, _audio_delta(), _AUDIO_DONE])

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


async def test_create_precedes_any_text() -> None:
    t = stepfun.TTS()
    ws, _ = await _drive(t, ["hello"], [_CREATED, _audio_delta(), _AUDIO_DONE])
    assert ws.types_sent()[0] == "tts.create"


async def test_flush_sends_flush_not_done() -> None:
    """Regression test for the segment-boundary mix-up."""
    t = stepfun.TTS()
    ws, _ = await _drive(
        t,
        ["hello"],
        [_CREATED, _audio_delta(), _AUDIO_DONE],
        flush_between=True,
    )

    types = ws.types_sent()
    assert "tts.text.flush" in types

    # done must come last, after every flush, and exactly once.
    assert types.count("tts.text.done") == 1
    assert types[-1] == "tts.text.done"
    assert types.index("tts.text.flush") < types.index("tts.text.done")


async def test_done_sent_once_input_ends() -> None:
    t = stepfun.TTS()
    ws, _ = await _drive(t, ["hello"], [_CREATED, _audio_delta(), _AUDIO_DONE])
    assert ws.data_of_type("tts.text.done") == [{"session_id": _SESSION}]


async def test_text_frames_carry_session_id() -> None:
    t = stepfun.TTS()
    ws, _ = await _drive(t, ["hi ", "there"], [_CREATED, _audio_delta(), _AUDIO_DONE])

    deltas = ws.data_of_type("tts.text.delta")
    assert [d["text"] for d in deltas] == ["hi ", "there"]
    assert all(d["session_id"] == _SESSION for d in deltas)


async def test_long_text_split_to_respect_frame_cap() -> None:
    # An over-long frame is rejected outright; LLM tokens never approach the cap
    # but session.say() with a long string routed through stream() can.
    t = stepfun.TTS()
    ws, _ = await _drive(t, ["x" * 2500], [_CREATED, _audio_delta(), _AUDIO_DONE])

    deltas = ws.data_of_type("tts.text.delta")
    assert [len(d["text"]) for d in deltas] == [1000, 1000, 500]


async def test_audio_deltas_decoded_and_emitted() -> None:
    t = stepfun.TTS()
    _, frames = await _drive(t, ["hello"], [_CREATED, _audio_delta(), _audio_delta(), _AUDIO_DONE])

    assert frames, "no audio emitted"
    assert sum(f.frame.samples_per_channel for f in frames) > 0
    assert frames[-1].is_final
    # The segment id is the server's session id, which also records it on the
    # trace so a trace can be lined up against the provider's own logs.
    assert frames[0].segment_id == _SESSION


async def test_audio_done_payload_not_pushed_twice() -> None:
    """audio.done also carries the whole session's audio.

    Every byte already arrived as deltas, so pushing it again would duplicate
    the utterance.
    """
    t = stepfun.TTS()
    one_delta = await _drive(t, ["hello"], [_CREATED, _audio_delta(), _AUDIO_DONE])

    t2 = stepfun.TTS()
    with_payload = await _drive(
        t2,
        ["hello"],
        [
            _CREATED,
            _audio_delta(),
            {
                "type": "tts.response.audio.done",
                "data": {"session_id": _SESSION, "audio": _utils.pcm_to_b64(_PCM)},
            },
        ],
    )

    def total(frames: list) -> int:
        return sum(f.frame.samples_per_channel for f in frames)

    assert total(one_delta[1]) == total(with_payload[1])


async def test_connection_is_dropped_not_pooled() -> None:
    """The server releases the socket once generation ends.

    Returning it to the pool would hand the next segment a dead connection, so
    it is dropped and the pool re-armed instead.
    """
    t = stepfun.TTS()
    _, _ = await _drive(t, ["hello"], [_CREATED, _audio_delta(), _AUDIO_DONE])

    pool = t._pool
    assert len(pool.removed) == 1  # type: ignore[union-attr]
    assert pool.prewarm_calls >= 1  # type: ignore[union-attr]


async def test_ttfb_metrics_emitted() -> None:
    """_mark_started() is the TTFB anchor, and this asserts the consequence."""
    t = stepfun.TTS()
    collected: list = []
    t.on("metrics_collected", collected.append)

    await _drive(t, ["hello"], [_CREATED, _audio_delta(), _AUDIO_DONE])

    assert collected, "no TTS metrics emitted — was _mark_started() called?"
    assert collected[0].ttfb > 0
    assert collected[0].audio_duration > 0


async def test_subtitles_pushed_as_timed_transcript() -> None:
    t = stepfun.TTS(timestamp=True)
    _, frames = await _drive(
        t,
        ["hello world"],
        [
            _CREATED,
            {
                "type": "tts.response.subtitle",
                "data": {
                    "session_id": _SESSION,
                    "request_id": f"{_SESSION}-tts.1",
                    "text": "hello world",
                    # Milliseconds on the wire, seconds in the framework.
                    "items": [
                        {"text": "hello", "start_time": 0, "end_time": 400},
                        {"text": "world", "start_time": 400, "end_time": 900},
                    ],
                },
            },
            _audio_delta(),
            _AUDIO_DONE,
        ],
    )

    assert frames, "no audio emitted"


@pytest.mark.parametrize(
    ("code", "retryable"),
    [
        ("503", True),  # engine overloaded
        ("500", True),
        ("429", True),
        ("401", False),
        ("400", False),
    ],
)
def test_tts_ws_error_mapping(code: str, retryable: bool) -> None:
    # Unlike the ASR surface, `code` here is an HTTP status rendered as a
    # string, so retryability comes from the HTTP table.
    err = _utils.translate_tts_ws_error(
        {
            "type": "tts.response.error",
            "data": {"session_id": _SESSION, "code": code, "message": "nope"},
        },
        context="test",
    )
    assert isinstance(err, APIStatusError)
    assert err.status_code == int(code)
    assert err.retryable is retryable
    assert err.request_id == _SESSION


def test_unparseable_error_code_is_terminal() -> None:
    # An unknown code carries no retry semantics; failing over beats retrying a
    # failure we cannot reason about.
    err = _utils.translate_tts_ws_error(
        {"type": "tts.response.error", "data": {"code": None, "message": "?"}},
        context="test",
    )
    assert isinstance(err, APIStatusError)
    assert err.retryable is False


async def test_error_frame_aborts_stream() -> None:
    t = stepfun.TTS()
    ws = _FakeWebSocket()
    t._pool = _FakeConnectionPool(_Connection(ws=ws, session_id=_SESSION))  # type: ignore[assignment]
    ws.queue_server_event(_CREATED)
    ws.queue_server_event(
        {
            "type": "tts.response.error",
            "data": {"session_id": _SESSION, "code": "400", "message": "bad voice"},
        }
    )

    stream = t.stream(conn_options=APIConnectOptions(max_retry=0))
    stream.push_text("hello")
    stream.end_input()

    with pytest.raises(APIStatusError):
        async for _ in stream:
            pass
    await stream.aclose()


def _opts(**kwargs):
    return stepfun.TTS(**kwargs)._opts


@pytest.mark.parametrize("brackets", [("(", ")"), ("（", "）")])
def test_parenthesised_directives_neutralized_by_default(brackets: tuple[str, str]) -> None:
    """LLM-generated text is not under our control.

    On the models that parse them, text inside parentheses is read as
    performance direction and is not spoken — so a stray "(as you mentioned)"
    would be silently swallowed, and a user who can steer the LLM's wording
    could reach the voice settings through it.
    """
    open_b, close_b = brackets
    text = f"sure{open_b}excited{close_b} let's go"
    assert _escape_text(text, _opts()) == "sureexcited let's go"


def test_escaping_can_be_opted_out() -> None:
    opts = _opts(enable_inline_instruction=True)
    assert _escape_text("sure(excited) go", opts) == "sure(excited) go"


def test_escaping_skipped_on_models_without_directives() -> None:
    # step-tts-mini does not interpret parentheses, so stripping them there
    # would be gratuitous mangling.
    opts = _opts(model="step-tts-mini")
    assert _escape_text("sure(excited) go", opts) == "sure(excited) go"


async def test_escaping_applied_on_the_wire() -> None:
    t = stepfun.TTS()
    ws, _ = await _drive(t, ["ok (laughs) fine"], [_CREATED, _audio_delta(), _AUDIO_DONE])
    assert ws.data_of_type("tts.text.delta")[0]["text"] == "ok laughs fine"


async def test_aclose_does_not_close_external_session() -> None:

    session = aiohttp.ClientSession()
    try:
        t = stepfun.TTS(http_session=session)
        await t.aclose()
        assert not session.closed
    finally:
        await session.close()


async def test_update_options_invalidates_pool() -> None:
    """A pooled socket is bound to the model it was opened with, via the query
    parameter, so reusing one after a model change would synthesize with the old
    one.
    """
    t = stepfun.TTS()
    pool = _FakeConnectionPool(_Connection(ws=_FakeWebSocket(), session_id=_SESSION))
    t._pool = pool  # type: ignore[assignment]

    t.update_options(voice="lively-girl", speed=1.5)

    assert t._opts.voice == "lively-girl"
    assert t._opts.speed == 1.5
    assert pool.invalidate_calls == 1


async def test_chunked_stream_snapshots_options() -> None:
    t = stepfun.TTS(voice="vibrant-youth")
    chunked = t.synthesize("hello")
    t.update_options(voice="lively-girl")
    assert chunked._opts.voice == "vibrant-youth"
    await chunked.aclose()


class _FakeResponse:
    """An SSE response: one-shot synthesis streams base64 PCM in JSON events."""

    status = 200
    headers = {"x-request-id": "req-1"}

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    class _Content:
        def __init__(self, lines: list[bytes]) -> None:
            self._lines = lines

        async def __aiter__(self):
            for line in self._lines:
                yield line

    @property
    def content(self) -> _Content:
        return self._Content(self._lines)

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


def _sse(*pcm_chunks: bytes, done: bool = True) -> list[bytes]:
    """Encode PCM deltas followed by completion events."""
    lines = [
        b'data: {"type":"speech.audio.delta","audio":"' + _utils.pcm_to_b64(c).encode() + b'"}\n'
        for c in pcm_chunks
    ]
    if done:
        lines += [b'data: {"type":"speech.audio.done","audio":""}\n', b"data: [DONE]\n"]
    return lines


def _pcm(num_samples: int = 1200, value: int = 0) -> bytes:
    import struct

    return struct.pack("<h", value) * num_samples


class _FakeHTTPSession:
    """Records every POST and answers with a fixed SSE body."""

    def __init__(self, lines: list[bytes]) -> None:
        self.lines = lines
        self.url: str = ""
        self.json: dict = {}
        self.headers: dict = {}
        self.requests: list[dict] = []
        self.timeout: object = None
        self.closed = False

    def post(self, url: str, *, json: dict, **kwargs: object) -> _FakeResponse:
        self.url = url
        self.json = dict(json)
        self.requests.append(dict(json))
        self.headers = dict(kwargs.get("headers") or {})  # type: ignore[call-overload]
        self.timeout = kwargs.get("timeout")
        return _FakeResponse(self.lines)

    async def close(self) -> None:
        self.closed = True


async def test_chunked_request_uses_http_field_names() -> None:
    """The two surfaces disagree on parameter names.

    HTTP takes voice / speed / volume where the WebSocket takes voice_id /
    speed_ratio / volume_ratio, so a name copied across either silently
    defaults or 400s.
    """
    session = _FakeHTTPSession(_sse(_pcm()))
    t = stepfun.TTS(
        voice="lively-girl",
        speed=1.3,
        volume=0.9,
        text_normalization="enhanced",
        instruction="warm",
        http_session=session,  # type: ignore[arg-type]
    )

    frames = [frame async for frame in t.synthesize("hello")]

    assert session.url == "https://api.stepfun.ai/v1/audio/speech"
    assert session.json["voice"] == "lively-girl"
    assert session.json["speed"] == 1.3
    assert session.json["volume"] == 0.9
    assert session.json["text_normalization"] == "enhanced"
    assert session.json["instruction"] == "warm"
    assert session.json["input"] == "hello"
    assert "voice_id" not in session.json
    assert "speed_ratio" not in session.json

    # SSE supports incremental output and requires PCM rather than WAV.
    assert session.json["stream_format"] == "sse"
    assert session.json["response_format"] == "pcm"
    assert session.headers["Accept"] == "text/event-stream"
    assert frames, "no audio emitted"


async def test_chunked_drops_instruction_on_unsupported_model() -> None:
    # step-tts-mini rejects `instruction` outright.
    session = _FakeHTTPSession(_sse(_pcm()))
    t = stepfun.TTS(
        model="step-tts-mini",
        instruction="warm",
        http_session=session,  # type: ignore[arg-type]
    )

    [_ async for _ in t.synthesize("hello")]

    assert "instruction" not in session.json


@pytest.mark.parametrize("length", [1000, 1001, 2501])
async def test_long_http_tts_splits_into_one_audio_segment(length: int) -> None:
    """The HTTP endpoint caps `input` at 1,000 characters per request."""
    session = _FakeHTTPSession(_sse(_pcm()))
    t = stepfun.TTS(http_session=session)  # type: ignore[arg-type]
    text = "你" * length
    stream = t.synthesize(text, conn_options=APIConnectOptions(max_retry=0, timeout=7))
    try:
        frames = [frame async for frame in stream]
        assert "".join(r["input"] for r in session.requests) == text
        assert all(len(r["input"]) <= 1000 for r in session.requests)
        assert len(session.requests) == (length + 999) // 1000
        # The emitter pads the last PCM block to its frame boundary.
        samples = sum(f.frame.samples_per_channel for f in frames)
        assert 1200 * len(session.requests) <= samples < 1200 * len(session.requests) + 480
        assert len({f.segment_id for f in frames}) == 1
        assert sum(f.is_final for f in frames) == 1
        # The deadline scales with each request's text: a fixed total would cut
        # off a healthy long download, and none would let a trickle run forever.
        assert isinstance(session.timeout, aiohttp.ClientTimeout)
        last = session.requests[-1]["input"]
        assert session.timeout.total == pytest.approx(30.0 + 0.3 * len(last))
        assert session.timeout.connect == 7
        assert session.timeout.sock_connect == 7
        # Allow synthesis pauses independently of the connection timeout.
        assert session.timeout.sock_read == 30
    finally:
        await stream.aclose()


async def test_http_tts_deadline_is_always_finite() -> None:
    """Even short text gets a bounded total, never None."""
    session = _FakeHTTPSession(_sse(_pcm()))
    t = stepfun.TTS(http_session=session)  # type: ignore[arg-type]
    [_ async for _ in t.synthesize("hi", conn_options=APIConnectOptions(max_retry=0, timeout=60))]
    assert isinstance(session.timeout, aiohttp.ClientTimeout)
    # The base allowance and idle limit never drop below the caller's timeout.
    assert session.timeout.total == pytest.approx(60.0 + 0.3 * 2)
    assert session.timeout.sock_read == 60


async def test_http_tts_preserves_audio_order_across_requests() -> None:
    class Session(_FakeHTTPSession):
        def post(self, url: str, *, json: dict, **kwargs: object) -> _FakeResponse:
            super().post(url, json=json, **kwargs)
            # Distinct constant amplitudes make request order observable; two
            # deltas per request exercise concatenation inside one response.
            value = len(self.requests) * 100
            return _FakeResponse(_sse(_pcm(600, value), _pcm(600, value)))

    session = Session([])
    t = stepfun.TTS(http_session=session)  # type: ignore[arg-type]
    stream = t.synthesize("x" * 2501, conn_options=APIConnectOptions(max_retry=0))
    try:
        frames = [frame async for frame in stream]
        pcm = b"".join(f.frame.data.tobytes() for f in frames)
        for i in range(3):
            assert pcm[i * 2400 : (i + 1) * 2400] == _pcm(1200, (i + 1) * 100)
    finally:
        await stream.aclose()


async def test_http_tts_stream_without_done_is_retryable_truncation() -> None:
    """A stream that ends before speech.audio.done lost audio; the base class
    retries it under a fresh request id, marking partial audio as stale."""
    session = _FakeHTTPSession(_sse(_pcm(), done=False))
    t = stepfun.TTS(http_session=session)  # type: ignore[arg-type]
    stream = t.synthesize("hello", conn_options=APIConnectOptions(max_retry=0))
    try:
        with pytest.raises(APIConnectionError, match="speech.audio.done") as excinfo:
            [_ async for _ in stream]
        assert excinfo.value.retryable
    finally:
        await stream.aclose()


async def test_http_tts_error_event_raises_without_echoing_provider_text() -> None:
    secret = "sensitive text echoed by provider"
    error = b'data: {"type":"speech.audio.error","message":"' + secret.encode() + b'"}\n'
    session = _FakeHTTPSession([*_sse(_pcm(), done=False), error])
    t = stepfun.TTS(http_session=session)  # type: ignore[arg-type]
    stream = t.synthesize("hello", conn_options=APIConnectOptions(max_retry=0))
    try:
        with pytest.raises(APIStatusError) as excinfo:
            [_ async for _ in stream]
        assert excinfo.value.retryable is False
        assert excinfo.value.body == {"type": "speech.audio.error", "message": secret}
        assert secret not in str(excinfo.value)
    finally:
        await stream.aclose()


async def test_http_tts_stalled_download_surfaces_timeout() -> None:
    class Response(_FakeResponse):
        class _Content(_FakeResponse._Content):
            async def __aiter__(self):
                yield self._lines[0]
                raise asyncio.TimeoutError

    class Session(_FakeHTTPSession):
        def post(self, url: str, *, json: dict, **kwargs: object) -> _FakeResponse:
            return Response(self.lines)

    from livekit.agents import APITimeoutError

    session = Session(_sse(_pcm()))
    t = stepfun.TTS(http_session=session)  # type: ignore[arg-type]
    stream = t.synthesize("hello", conn_options=APIConnectOptions(max_retry=0))
    try:
        with pytest.raises(APITimeoutError):
            [_ async for _ in stream]
    finally:
        await stream.aclose()


async def test_real_pool_round_trip() -> None:
    """Acquire, return and retire a _Connection through the real pool."""
    from livekit.agents import utils

    closed: list[_Connection] = []

    async def _connect(timeout: float) -> _Connection:
        return _Connection(ws=_FakeWebSocket(), session_id=_SESSION)  # type: ignore[arg-type]

    async def _close(conn: _Connection) -> None:
        closed.append(conn)

    pool = utils.ConnectionPool[_Connection](connect_cb=_connect, close_cb=_close)

    async with pool.connection(timeout=5) as first:
        assert isinstance(first, _Connection)
    # Returned to the pool, so the next acquisition reuses it.
    async with pool.connection(timeout=5) as second:
        assert second is first
        assert pool.last_connection_reused is True
        # What SynthesizeStream does once generation ends: the server releases
        # the socket, so it must never go back into the pool.
        pool.remove(second)

    async with pool.connection(timeout=5) as third:
        assert third is not first
        assert pool.last_connection_reused is False
        pool.remove(third)

    await pool.aclose()
    assert first in closed


async def test_real_pool_prewarm() -> None:
    from livekit.agents import utils

    connects = 0

    async def _connect(timeout: float) -> _Connection:
        nonlocal connects
        connects += 1
        return _Connection(ws=_FakeWebSocket(), session_id=_SESSION)  # type: ignore[arg-type]

    pool = utils.ConnectionPool[_Connection](connect_cb=_connect, close_cb=None)
    pool.prewarm()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    async with pool.connection(timeout=5) as conn:
        assert isinstance(conn, _Connection)
    # The prewarmed connection was reused rather than a second one opened.
    assert connects == 1
    await pool.aclose()


_NO_RETRY = APIConnectOptions(max_retry=0)


@pytest.mark.parametrize(
    "options",
    [
        {"speed": 0.1},
        {"volume": 3.0},
        {"instruction": "x" * 501},
        {"model": "stepaudio-2.5-tts", "instruction": "x" * 201},
    ],
)
def test_invalid_update_is_atomic(options):
    tts = stepfun.TTS()
    previous = tts._opts
    with pytest.raises(ValueError):
        tts.update_options(voice="lively-girl", **options)
    assert tts._opts == previous


def test_model_update_checks_retained_instruction():
    tts = stepfun.TTS(instruction="x" * 300)
    with pytest.raises(ValueError, match="200 characters"):
        tts.update_options(model="stepaudio-2.5-tts")
    assert tts.model == "stepaudio-3-tts"
    tts.update_options(model="step-tts-mini")
    assert tts._opts.instruction is NOT_GIVEN


async def test_handshake_surfaces_transport_error():
    ws = _FakeWebSocket()
    ws.queue_error()
    with pytest.raises(APIConnectionError, match="websocket error"):
        await asyncio.wait_for(_await_connection_done(ws), 1)


@pytest.mark.parametrize("configured", [False, True])
async def test_transport_error_closes_socket(configured):
    ws = _FakeWebSocket()
    if configured:
        ws.queue_server_event(_CREATED)
    ws.queue_error()
    tts = stepfun.TTS()
    tts._pool = _FakeConnectionPool(_Connection(ws=ws, session_id=_SESSION))  # type: ignore[assignment]
    stream = tts.stream(conn_options=_NO_RETRY)
    stream.push_text("hello")
    stream.end_input()
    try:
        with pytest.raises(APIConnectionError, match="websocket error"):
            await asyncio.wait_for(anext(stream), 1)
        assert ws.closed
    finally:
        await tts.aclose()


async def test_closed_pooled_socket_is_replaced_instead_of_failing(monkeypatch):
    """Replace a closed pooled socket before sending the segment."""
    tts = stepfun.TTS()
    dead = _FakeWebSocket()
    dead.closed = True
    stale = _Connection(ws=dead, session_id="stale")
    tts._pool = _FakeConnectionPool(stale)  # type: ignore[assignment]

    live = _FakeWebSocket()
    for event in (_CREATED, _audio_delta(), _AUDIO_DONE):
        live.queue_server_event(event)
    connect = AsyncMock(return_value=_Connection(ws=live, session_id=_SESSION))
    monkeypatch.setattr(tts, "_connect_ws", connect)

    stream = tts.stream(conn_options=_NO_RETRY)
    stream.push_text("hello")
    stream.end_input()
    try:
        frames = [frame async for frame in stream]
        assert frames, "the turn must still produce audio"
        connect.assert_awaited_once()
        assert stale in tts._pool.removed
    finally:
        await tts.aclose()


async def test_model_update_does_not_change_existing_stream(monkeypatch):
    old_model = "stepaudio-3-tts"
    tts = stepfun.TTS(model=old_model)
    ws = _FakeWebSocket()
    conn = _Connection(ws=ws, session_id=_SESSION, model=old_model)
    for event in (_CREATED, _audio_delta(), _AUDIO_DONE):
        ws.queue_server_event(event)
    stale = _Connection(ws=_FakeWebSocket(), session_id="stale", model="stepaudio-2.5-tts")
    tts._pool = _FakeConnectionPool(stale)  # type: ignore[assignment]
    connect = AsyncMock(return_value=conn)
    monkeypatch.setattr(tts, "_connect_ws", connect)
    stream = tts.stream(conn_options=_NO_RETRY)
    tts.update_options(model="stepaudio-2.5-tts")
    stream.push_text("hello")
    stream.end_input()
    try:
        frames = [frame async for frame in stream]
        assert frames
        connect.assert_awaited_once_with(_NO_RETRY.timeout, model=old_model)
        assert stale.ws.closed
        assert ws.closed
    finally:
        await tts.aclose()


async def test_aclose_cancels_active_http_synthesis(monkeypatch):
    started, released = asyncio.Event(), asyncio.Event()

    class BlockingResponse:
        status = 200
        headers = {}

        @property
        def content(self):
            return self

        async def __aiter__(self):
            started.set()
            await asyncio.Event().wait()
            yield b""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            released.set()

    session = _FakeHTTPSession([])
    monkeypatch.setattr(session, "post", lambda *args, **kwargs: BlockingResponse())
    tts = stepfun.TTS(http_session=session)  # type: ignore[arg-type]
    stream = tts.synthesize("hello", conn_options=_NO_RETRY)
    try:
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.wait_for(tts.aclose(), 1)
        assert released.is_set()
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        assert not session.closed
    finally:
        await stream.aclose()


@pytest.mark.parametrize("component", [stepfun.STT, stepfun.TTS])
def test_explicit_cluster_is_validated(component):
    with pytest.raises(ValueError, match="STEPFUN_CLUSTER"):
        component(cluster="invalid")


@pytest.mark.parametrize("component", [stepfun.STT, stepfun.TTS])
async def test_interrupted_stream_closes_socket_immediately(component):
    started = asyncio.Event()

    class ListeningSocket(_FakeWebSocket):
        async def receive(self):
            if self._incoming.empty():
                started.set()
            return await super().receive()

    ws = ListeningSocket()
    plugin = component()
    if component is stepfun.STT:
        ws.queue_server_event({"type": "session.updated"})
        plugin._pool = _FakeConnectionPool(ws)  # type: ignore[union-attr]
    else:
        ws.queue_server_event(_CREATED)
        plugin._pool = _FakeConnectionPool(_Connection(ws=ws, session_id=_SESSION))  # type: ignore[union-attr]
    stream = plugin.stream(conn_options=_NO_RETRY)
    try:
        await asyncio.wait_for(started.wait(), 1)
        await stream.aclose()
        assert ws.closed
    finally:
        await plugin.aclose()
