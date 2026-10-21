"""Unit tests for the StepFun STT plugin."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError, stt
from livekit.plugins import stepfun
from livekit.plugins.stepfun import _utils

pytestmark = pytest.mark.plugin("stepfun")


@pytest.fixture(autouse=True)
def _stepfun_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests independent of the developer's provider credentials."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    monkeypatch.delenv("STEPFUN_CLUSTER", raising=False)


class _FakeWSMessage:
    def __init__(self, type_: aiohttp.WSMsgType, data: Any, extra: Any = None) -> None:
        self.type = type_
        self.data = data
        # aiohttp carries the WebSocket close code here and the reason in
        # `data`, which is the only evidence of why a socket died.
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
        self._incoming.put_nowait(_FakeWSMessage(aiohttp.WSMsgType.CLOSED, reason, extra=code))

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
) -> tuple[_FakeWebSocket, list[stt.SpeechEvent]]:
    """Run one stream end to end against a scripted server."""
    ws = _FakeWebSocket()
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]

    for event in server_events:
        ws.queue_server_event(event)
    # Finalize every pending item before closing; truncation tests script their
    # own socket and omit these responses.
    item_ids = {event.get("item_id") or "" for event in server_events}
    for item_id in item_ids | {""}:
        ws.queue_server_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": item_id,
                "transcript": "",
            }
        )
    # A stop closes the last reported utterance so the stream can settle. No
    # end position is sent: nothing consumes one.
    ws.queue_server_event({"type": "input_audio_buffer.speech_stopped"})
    ws.queue_close()

    stream = s.stream()
    for _ in range(frames):
        stream.push_frame(_frame())
    stream.end_input()

    events = [ev async for ev in stream]
    await stream.aclose()
    return ws, events


def _texts(events: list[stt.SpeechEvent], kind: stt.SpeechEventType) -> list[str]:
    return [ev.alternatives[0].text for ev in events if ev.type == kind]


def test_reads_api_key_from_env() -> None:
    assert stepfun.STT()._api_key == "test-key"


def test_explicit_api_key_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STEPFUN_API_KEY", "from-env")
    assert stepfun.STT(api_key="explicit")._api_key == "explicit"


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STEPFUN_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        stepfun.STT()


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


def test_invalid_env_cluster_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STEPFUN_CLUSTER", "eu-west")
    with pytest.raises(ValueError, match="STEPFUN_CLUSTER"):
        stepfun.STT()


def test_model_and_provider_reported() -> None:
    # Without these the framework labels every metric "unknown".
    s = stepfun.STT()
    assert s.model == "stepaudio-2.5-asr-stream"
    assert s.provider == "StepFun"


def test_capabilities_declared_honestly() -> None:
    caps = stepfun.STT().capabilities
    assert caps.streaming is True
    assert caps.interim_results is True
    assert caps.offline_recognize is True
    assert caps.diarization is False
    assert caps.aligned_transcript is False
    # The endpoint takes free-text `prompt` hints, not a keyterm list.
    assert caps.keyterms is False


def test_timestamp_align_enables_word_alignment() -> None:
    caps = stepfun.STT(enable_timestamp_align=True).capabilities
    assert caps.aligned_transcript == "word"


def test_timestamp_align_ignored_on_unsupported_model() -> None:
    # Only stepaudio-2.5-asr-stream implements it; claiming the capability on a
    # model that cannot deliver it would be over-declaring.
    caps = stepfun.STT(model="step-asr-1.1-stream", enable_timestamp_align=True).capabilities
    assert caps.aligned_transcript is False


async def test_sends_session_update_before_audio() -> None:
    """The endpoint accepts no query parameters.

    Model, audio format and turn detection are configured by this frame alone,
    so skipping it leaves the session unconfigured and nothing is ever
    recognized.
    """
    s = stepfun.STT(language="zh", prompt="livekit, webrtc")
    ws, _ = await _drive(s, [_SESSION_READY])

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


async def test_turn_detection_omitted_when_server_vad_disabled() -> None:
    s = stepfun.STT(server_vad=False)
    ws, _ = await _drive(s, [_SESSION_READY])

    audio_input = ws.frames_of_type("session.update")[0]["session"]["audio"]["input"]
    assert "turn_detection" not in audio_input


async def test_timestamp_align_requested_in_session_update() -> None:
    s = stepfun.STT(enable_timestamp_align=True)
    ws, _ = await _drive(s, [_SESSION_READY])

    transcription = ws.frames_of_type("session.update")[0]["session"]["audio"]["input"][
        "transcription"
    ]
    assert transcription["enable_timestamp_align"] is True


async def test_audio_is_sent_as_base64_pcm() -> None:
    s = stepfun.STT()
    ws, _ = await _drive(s, [_SESSION_READY], frames=1)

    appends = ws.frames_of_type("input_audio_buffer.append")
    assert appends, "no audio was sent"
    # 100ms of input at a 50ms frame size.
    assert len(appends) > 2  # real input plus endpointing silence
    assert sum(len(_utils.b64_to_pcm(f["audio"])) for f in appends[2:]) >= (
        _SAMPLE_RATE * s._opts.vad_silence_duration_ms // 1000 * 2
    )
    assert _utils.b64_to_pcm(appends[0]["audio"]) == b"\x00\x00" * (_SAMPLE_RATE // 20)


async def test_commit_sent_after_silence_under_server_vad() -> None:
    """A best-effort commit follows the endpointing silence.

    The public gateway ignores commits while server VAD is on, so it is sent
    untracked: the stream must not wait for an acknowledgement. It starts to
    finalize the segment once the gateway forwards commits.
    """
    s = stepfun.STT(server_vad=True)
    ws, events = await _drive(s, [_SESSION_READY])
    types = ws.types_sent()
    assert types.count("input_audio_buffer.commit") == 1
    assert types[-1] == "input_audio_buffer.commit"
    # Silence still precedes it, so today's endpointing path is unchanged.
    assert types.index("input_audio_buffer.append") < types.index("input_audio_buffer.commit")


async def test_commit_sent_when_server_vad_disabled() -> None:
    """With server VAD off, commit is the documented path and it works.

    Measured against the live service: a final arrived 438ms after input ended.
    If a flush stops producing a commit, the final never arrives and a
    standalone caller goes mute — AgentSession does not exercise this path, so
    the stream has to be driven directly.
    """
    s = stepfun.STT(server_vad=False)
    ws, _ = await _drive(s, [_SESSION_READY])
    assert ws.frames_of_type("input_audio_buffer.commit")
    # The commit must be the last frame: anything after it would be audio the
    # server has already been told not to expect.
    assert ws.types_sent()[-1] == "input_audio_buffer.commit"


async def test_flush_drains_partial_frame() -> None:
    """The sub-frame tail inside AudioByteStream must be drained on flush.

    A 30ms push does not fill a 50ms frame, so without the explicit drain that
    audio is silently dropped.
    """
    s = stepfun.STT(server_vad=False)
    ws = _FakeWebSocket()
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]
    ws.queue_server_event(_SESSION_READY)
    ws.queue_server_event(
        {"type": "conversation.item.input_audio_transcription.completed", "transcript": ""}
    )
    ws.queue_close()

    stream = s.stream()
    stream.push_frame(_frame(ms=30))
    stream.end_input()
    [_ async for _ in stream]
    await stream.aclose()

    assert ws.frames_of_type("input_audio_buffer.append"), "tail frame was lost"


_DELTA = "conversation.item.input_audio_transcription.delta"
_COMPLETED = "conversation.item.input_audio_transcription.completed"


async def test_cumulative_deltas_replace_rather_than_append() -> None:
    """Regression test for the most damaging protocol mistake available here.

    stepaudio-2.5-asr-stream sends the whole transcript so far on every delta,
    corrections included. Appending them turns "你好" / "你好世界" into
    "你好你好世界" and gets worse with every partial.
    """
    s = stepfun.STT(language="zh")
    _, events = await _drive(
        s,
        [
            _SESSION_READY,
            {"type": _DELTA, "text": "你好"},
            {"type": _DELTA, "text": "你好世界"},
        ],
    )

    assert _texts(events, stt.SpeechEventType.INTERIM_TRANSCRIPT) == ["你好", "你好世界"]


async def test_incremental_deltas_are_appended_on_legacy_model() -> None:
    # step-asr-1.1-stream sends fragments instead, so the two models genuinely
    # need different handling.
    s = stepfun.STT(model="step-asr-1.1-stream")
    _, events = await _drive(
        s,
        [
            _SESSION_READY,
            {"type": _DELTA, "text": "hello "},
            {"type": _DELTA, "text": "world"},
        ],
    )

    assert _texts(events, stt.SpeechEventType.INTERIM_TRANSCRIPT) == [
        "hello ",
        "hello world",
    ]


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


async def test_stash_produces_preflight_transcript() -> None:
    """`stash` marks the trailing characters the server may still rewrite.

    Everything before it is settled, which is what PREFLIGHT_TRANSCRIPT is for:
    stable enough for preemptive generation while the rest is still in flux.
    """
    s = stepfun.STT()
    _, events = await _drive(
        s,
        [_SESSION_READY, {"type": _DELTA, "text": "hello world", "stash": "world"}],
    )

    assert _texts(events, stt.SpeechEventType.PREFLIGHT_TRANSCRIPT) == ["hello "]
    assert _texts(events, stt.SpeechEventType.INTERIM_TRANSCRIPT) == ["hello world"]


async def test_word_timestamps_mapped_when_enabled() -> None:
    s = stepfun.STT(enable_timestamp_align=True)
    _, events = await _drive(
        s,
        [
            _SESSION_READY,
            {
                "type": _COMPLETED,
                "transcript": "hi there",
                "words": [
                    {"word": "hi", "start": 0.0, "end": 0.4, "final": True},
                    {"word": "there", "start": 0.4, "end": 0.9, "final": True},
                ],
            },
        ],
    )

    finals = [ev for ev in events if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    words = finals[0].alternatives[0].words
    assert words is not None
    assert [str(w) for w in words] == ["hi", "there"]
    assert words[0].start_time == 0.0
    assert words[1].end_time == 0.9


async def test_non_final_words_are_skipped() -> None:
    # A word with final=false sits in the stash region and its timing is only an
    # estimate, so publishing it as settled would be a lie.
    s = stepfun.STT(enable_timestamp_align=True)
    _, events = await _drive(
        s,
        [
            _SESSION_READY,
            {
                "type": _DELTA,
                "text": "hi there",
                "words": [
                    {"word": "hi", "start": 0.0, "end": 0.4, "final": True},
                    {"word": "there", "start": 0.4, "end": 0.9, "final": False},
                ],
            },
        ],
    )

    interim = [ev for ev in events if ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT]
    words = interim[0].alternatives[0].words
    assert words is not None
    assert [str(w) for w in words] == ["hi"]


async def test_words_omitted_when_alignment_disabled() -> None:
    s = stepfun.STT()
    _, events = await _drive(
        s,
        [
            _SESSION_READY,
            {
                "type": _COMPLETED,
                "transcript": "hi",
                "words": [{"word": "hi", "start": 0.0, "end": 0.4, "final": True}],
            },
        ],
    )

    finals = [ev for ev in events if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert finals[0].alternatives[0].words is None


async def test_server_vad_events_are_translated() -> None:
    """These are genuine server events, so translating them is not fabrication.

    The design boundary that matters is the opposite one: nothing here may be
    inferred from client-side energy detection, which would fight silero.VAD and
    the turn detector in a mixed setup.
    """
    s = stepfun.STT()
    _, events = await _drive(
        s,
        [
            _SESSION_READY,
            {"type": "input_audio_buffer.speech_started", "audio_start_ms": 0},
            {"type": "input_audio_buffer.speech_stopped", "audio_start_ms": 900},
        ],
    )

    types = [ev.type for ev in events]
    assert stt.SpeechEventType.START_OF_SPEECH in types
    assert stt.SpeechEventType.END_OF_SPEECH in types


async def test_usage_reported_with_server_token_counts() -> None:
    """RECOGNITION_USAGE is the only trigger for streamed STTMetrics.

    The base class's metrics monitor listens for that event alone, so not
    emitting it means no STT metrics for the entire session.
    """
    s = stepfun.STT()
    _, events = await _drive(
        s,
        [
            _SESSION_READY,
            {
                "type": _COMPLETED,
                "transcript": "hi",
                "usage": {"prompt_tokens": 120, "completion_tokens": 7},
            },
        ],
    )

    usage = [ev for ev in events if ev.type == stt.SpeechEventType.RECOGNITION_USAGE]
    assert usage, "no usage event emitted"
    assert usage[0].recognition_usage is not None
    assert usage[0].recognition_usage.input_tokens == 120
    assert usage[0].recognition_usage.output_tokens == 7
    assert usage[0].recognition_usage.audio_duration > 0


async def test_connection_is_dropped_not_pooled() -> None:
    """The socket carries per-utterance state with no documented reset.

    Returning it to the pool would hand the next stream a half-used session, so
    it is dropped and the pool re-armed instead.
    """
    s = stepfun.STT()
    ws, _ = await _drive(s, [_SESSION_READY])

    pool = s._pool
    assert pool.removed == [ws]  # type: ignore[union-attr]
    assert pool.prewarm_calls >= 1  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("status", "retryable"),
    [
        (400, False),  # bad parameters: a retry fails identically
        (401, False),  # marking auth failures retryable makes fallback chains hammer
        (402, False),  # no balance
        (404, False),
        (451, False),  # content moderation
        (429, True),  # plain rate limiting
        (500, True),
        (503, True),
    ],
)
def test_status_error_retryable_mapping(status: int, retryable: bool) -> None:
    err = _utils.status_error(status_code=status, message="x")
    assert isinstance(err, APIStatusError)
    assert err.retryable is retryable


@pytest.mark.parametrize(
    "identifier",
    [
        "insufficient_credit",
        "project_credit_limit_exceeded",
        "member_project_credit_limit_exceeded",
    ],
)
def test_quota_429_is_not_retryable(identifier: str) -> None:
    """429 is overloaded on this API.

    Ordinary rate limiting clears on retry; a Credit ceiling does not clear
    until the limit is raised or the quota resets on the 1st of next month.
    Treating the latter as retryable pins a FallbackAdapter chain to a provider
    that is guaranteed to fail for the rest of the month.
    """
    err = _utils.status_error(
        status_code=429,
        message="quota",
        body=f'{{"error": {{"code": "{identifier}"}}}}',
    )
    assert err.retryable is False


@pytest.mark.parametrize(
    ("code", "retryable"),
    [
        ("internal_error", True),
        ("max_idle_timeout", True),  # stale connection; reconnecting fixes it
        ("pong_timeout", True),
        ("invalid_value", False),
        ("missing_param", False),
        ("risk_blocked", False),  # content safety: a retry fails identically
    ],
)
def test_asr_ws_error_retryable_mapping(code: str, retryable: bool) -> None:
    err = _utils.translate_asr_ws_error(
        {
            "type": "error",
            "meta": {"session_id": "sess-1"},
            "error": {"type": "invalid_request_error", "code": code, "message": "nope"},
        },
        context="test",
    )
    assert isinstance(err, APIStatusError)
    assert err.retryable is retryable
    assert err.request_id == "sess-1"


async def test_http_error_retains_body_but_does_not_log_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "sensitive speech echoed by provider"

    class Response:
        status = 429
        headers = {"x-request-id": "req-safe"}

        async def text(self) -> str:
            return secret

    with pytest.raises(APIStatusError) as excinfo:
        await _utils.raise_for_status(Response())  # type: ignore[arg-type]
    error = excinfo.value
    assert error.body == secret
    assert error.retryable
    assert error.request_id == "req-safe"
    with caplog.at_level(logging.WARNING):
        logging.getLogger("stepfun-retry-log").warning("failed, retrying: %s", error)
        logging.getLogger("stepfun-retry-log").warning("exception representation: %r", error)
    assert secret not in caplog.text
    assert "429" in caplog.text
    assert "req-safe" in caplog.text


def test_asr_ws_error_text_is_not_in_exception_formatting() -> None:
    # Retry logs format the exception; the provider message may echo speech.
    secret = "sensitive speech echoed by provider"
    frame = {"type": "error", "error": {"code": "internal_error", "message": secret}}
    error = _utils.translate_asr_ws_error(frame, context="ASR")
    assert isinstance(error, APIStatusError)
    assert error.body == frame
    assert error.retryable
    assert secret not in str(error)
    assert secret not in repr(error)


def test_raise_translated_chains_only_changed_errors() -> None:
    """A translated error is chained to its original; a passed-through one
    is re-raised as-is, never made its own cause."""
    original = aiohttp.ClientConnectionError("reset")
    with pytest.raises(APIConnectionError) as translated:
        _utils.raise_translated(original, context="ctx")
    assert translated.value.__cause__ is original

    already = APIConnectionError("already translated")
    with pytest.raises(APIConnectionError) as passed:
        _utils.raise_translated(already, context="ctx")
    assert passed.value is already
    assert already.__cause__ is not already


async def test_recognize_error_is_not_its_own_cause() -> None:
    # The SSE error event raises an APIStatusError that passes through
    # translate_exception unchanged, which is the self-cause path.
    session = _FakeSSESession(
        ['data: {"type":"error","message":"bad audio","meta":{"session_id":"sse-x"}}']
    )
    s = stepfun.STT(http_session=session)  # type: ignore[arg-type]
    with pytest.raises(APIStatusError) as excinfo:
        await s.recognize([_frame()], conn_options=_no_retry())
    assert excinfo.value.__cause__ is not excinfo.value


async def test_error_frame_raises() -> None:
    s = stepfun.STT()
    ws = _FakeWebSocket()
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]
    ws.queue_server_event(_SESSION_READY)
    ws.queue_server_event(
        {
            "type": "error",
            "meta": {"session_id": "sess-1"},
            "error": {"code": "invalid_value", "message": "bad rate"},
        }
    )

    stream = s.stream(conn_options=_no_retry())
    stream.push_frame(_frame())
    stream.end_input()

    with pytest.raises(APIStatusError):
        async for _ in stream:
            pass
    await stream.aclose()


def _no_retry():
    from livekit.agents import APIConnectOptions

    # A non-retryable error must surface immediately; retries would only slow
    # the test down and obscure which error arrived.
    return APIConnectOptions(max_retry=0)


async def test_aclose_does_not_close_external_session() -> None:
    import aiohttp

    session = aiohttp.ClientSession()
    try:
        s = stepfun.STT(http_session=session)
        await s.aclose()
        # Closing a caller-supplied session is a common way for plugins to
        # break their callers.
        assert not session.closed
    finally:
        await session.close()


async def test_update_options_invalidates_pool() -> None:
    """Pooled sockets were configured by a session.update built from the old
    options, so they must stop being handed out.
    """
    s = stepfun.STT()
    pool = _FakeConnectionPool(_FakeWebSocket())
    s._pool = pool  # type: ignore[assignment]

    s.update_options(language="ja")

    assert s._opts.language == "ja"
    assert pool.invalidate_calls == 1


async def test_stream_snapshots_options() -> None:
    """Options are snapshotted in the stream constructor, never read lazily in
    _run().

    Otherwise this races with the next update_options() call and shows up as
    intermittently using the previous turn's language — hard enough to reproduce
    that it needs a test.
    """
    s = stepfun.STT(language="zh")
    ws = _FakeWebSocket()
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]

    stream = s.stream()
    s.update_options(language="ja")

    assert stream._opts.language == "zh"
    await stream.aclose()


class _FakeSSEResponse:
    status = 200
    headers: dict[str, str] = {}

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    @property
    def content(self):
        async def _gen():
            for line in self._lines:
                yield (line + "\n").encode()

        return _gen()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _FakeSSESession:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.url = ""
        self.json: dict = {}
        self.headers: dict[str, str] = {}

    def post(self, url: str, *, json: dict, headers: dict, **kwargs: object):
        self.url = url
        self.json = json
        self.headers = headers
        return _FakeSSEResponse(self.lines)

    async def close(self) -> None:
        pass


async def test_recognize_posts_nested_pcm_body() -> None:
    session = _FakeSSESession(
        [
            'data: {"type":"transcript.text.delta","delta":"hel"}',
            'data: {"type":"transcript.text.delta","delta":"lo"}',
            'data: {"type":"transcript.text.done","text":"hello",'
            '"meta":{"session_id":"sse-1"},'
            '"usage":{"input_tokens":90,"output_tokens":4}}',
            "data: [DONE]",
        ]
    )
    s = stepfun.STT(language="zh", http_session=session)  # type: ignore[arg-type]

    event = await s.recognize([_frame(ms=200)])

    assert s._opts.http_base_url == "https://api.stepfun.ai"
    assert session.url == "https://api.stepfun.ai/v1/audio/asr/sse"
    # SSE requires this Accept header; without it the server answers with plain
    # JSON and the parse yields nothing.
    assert session.headers["Accept"] == "text/event-stream"

    audio = session.json["audio"]
    transcription = audio["input"]["transcription"]
    assert transcription["model"] == "stepaudio-3-asr-max"
    assert transcription["language"] == "zh"
    assert audio["input"]["format"] == {
        "type": "pcm",
        "codec": "pcm_s16le",
        "rate": _SAMPLE_RATE,
        "bits": 16,
        "channel": 1,
    }
    assert _utils.b64_to_pcm(audio["data"])

    assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    # The full transcript lives in `text` on the done event; deltas are
    # incremental on this route, unlike the WebSocket's cumulative ones.
    assert event.alternatives[0].text == "hello"
    assert event.request_id == "sse-1"
    assert event.recognition_usage is not None
    assert event.recognition_usage.input_tokens == 90
    assert event.recognition_usage.audio_duration == pytest.approx(0.2)


async def test_recognize_resamples_high_rate_input_to_stt_rate() -> None:
    # The VAD hands batch segments over at the input rate (e.g. 48 kHz). The
    # plugin must resample to the STT's sample_rate so the upload matches the
    # streaming path instead of shipping ~3x the bytes.
    session = _FakeSSESession(
        [
            'data: {"type":"transcript.text.done","text":"hi",'
            '"meta":{"session_id":"sse-rs"},"usage":{"input_tokens":1,"output_tokens":1}}',
        ]
    )
    s = stepfun.STT(http_session=session)  # type: ignore[arg-type]  # sample_rate=16000

    samples_48k = 48000 * 200 // 1000  # 200 ms at 48 kHz
    frame_48k = rtc.AudioFrame(
        data=b"\x00\x00" * samples_48k,
        sample_rate=48000,
        num_channels=1,
        samples_per_channel=samples_48k,
    )
    await s.recognize([frame_48k])

    audio = session.json["audio"]
    assert audio["input"]["format"]["rate"] == _SAMPLE_RATE  # 16000, not 48000
    decoded = _utils.b64_to_pcm(audio["data"])
    samples_out = len(decoded) // 2
    # ~200 ms at 16 kHz = 3200 samples (resampler padding allows a little slack),
    # and strictly smaller than the 48 kHz input.
    assert samples_out < samples_48k
    assert samples_out == pytest.approx(3200, rel=0.1)


async def test_recognize_falls_back_to_accumulated_deltas() -> None:
    # A done event that omits `text` should not lose the transcript.
    session = _FakeSSESession(
        [
            'data: {"type":"transcript.text.delta","delta":"partial"}',
            'data: {"type":"transcript.text.done","meta":{"session_id":"sse-2"}}',
        ]
    )
    s = stepfun.STT(http_session=session)  # type: ignore[arg-type]

    event = await s.recognize([_frame()])
    assert event.alternatives[0].text == "partial"


async def test_recognize_raises_on_error_event() -> None:
    session = _FakeSSESession(
        ['data: {"type":"error","message":"bad audio","meta":{"session_id":"sse-3"}}']
    )
    s = stepfun.STT(http_session=session)  # type: ignore[arg-type]

    with pytest.raises(APIStatusError):
        await s.recognize([_frame()], conn_options=_no_retry())


async def test_recognize_omits_hotwords_on_global_cluster() -> None:
    # hotwords is documented for the mainland-China cluster only; sending it
    # globally risks a 400 on an unknown parameter.
    session = _FakeSSESession(['data: {"type":"transcript.text.done","text":"x"}'])
    s = stepfun.STT(hotwords=["livekit"], http_session=session)  # type: ignore[arg-type]

    await s.recognize([_frame()])
    assert "hotwords" not in session.json["audio"]["input"]["transcription"]


async def test_recognize_sends_hotwords_on_china_cluster() -> None:
    session = _FakeSSESession(['data: {"type":"transcript.text.done","text":"x"}'])
    s = stepfun.STT(
        cluster="china",
        hotwords=["livekit"],
        http_session=session,  # type: ignore[arg-type]
    )

    await s.recognize([_frame()])
    assert session.json["audio"]["input"]["transcription"]["hotwords"] == ["livekit"]


def test_http_only_declares_no_streaming() -> None:
    caps = stepfun.STT(use_streaming=False).capabilities
    assert caps.streaming is False
    # StreamAdapter surfaces only the final transcript, so claiming interim
    # results would be a lie — and the framework would believe it.
    assert caps.interim_results is False
    # recognize() still works; that is the whole point of this mode.
    assert caps.offline_recognize is True


def test_http_only_reports_the_batch_model() -> None:
    """StreamAdapter delegates `model` to the wrapped STT.

    Reporting the streaming model while the batch route does the work would
    mislabel every metric for the session.
    """
    s = stepfun.STT(use_streaming=False)
    assert s.model == "stepaudio-3-asr-max"

    s2 = stepfun.STT(use_streaming=False, batch_model="stepaudio-2.5-asr")
    assert s2.model == "stepaudio-2.5-asr"


def test_streaming_mode_reports_the_streaming_model() -> None:
    assert stepfun.STT().model == "stepaudio-2.5-asr-stream"


def test_http_only_refuses_to_stream() -> None:
    # Serving a stream here would send audio to the streaming endpoint with a
    # model it does not accept, and would contradict the declared capability.
    s = stepfun.STT(use_streaming=False)
    with pytest.raises(RuntimeError, match="use_streaming=False"):
        s.stream()


def test_http_only_ignores_timestamp_align() -> None:
    # Word alignment is a streaming-endpoint feature.
    caps = stepfun.STT(use_streaming=False, enable_timestamp_align=True).capabilities
    assert caps.aligned_transcript is False


async def test_http_only_still_recognizes() -> None:
    session = _FakeSSESession(
        ['data: {"type":"transcript.text.done","text":"hello","meta":{"session_id":"s"}}']
    )
    s = stepfun.STT(use_streaming=False, http_session=session)  # type: ignore[arg-type]

    event = await s.recognize([_frame()])

    assert event.alternatives[0].text == "hello"
    assert session.json["audio"]["input"]["transcription"]["model"] == "stepaudio-3-asr-max"


async def test_stream_adapter_wraps_http_only_stt() -> None:
    """The framework's own adapter must accept this STT.

    This is the configuration an AgentSession builds automatically when given a
    non-streaming STT plus a vad, so it is worth pinning that the wrapping
    works and that the adapter reports our model rather than its own.
    """
    from livekit.agents import stt as stt_module

    class _FakeVAD:
        pass

    s = stepfun.STT(use_streaming=False)
    adapter = stt_module.StreamAdapter(stt=s, vad=_FakeVAD())  # type: ignore[arg-type]

    assert adapter.capabilities.streaming is True
    assert adapter.capabilities.interim_results is False
    assert adapter.model == "stepaudio-3-asr-max"
    assert adapter.provider == "StepFun"
    assert adapter.wrapped_stt is s


async def test_stream_terminates_when_server_keeps_socket_open() -> None:
    """A missing final must surface as an error, not as success.

    Speech was in flight (an interim arrived), so the error is non-retryable:
    a fresh session could not recover the consumed audio.
    """
    s = stepfun.STT()
    ws = _FakeWebSocket()
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]
    ws.queue_server_event(_SESSION_READY)
    ws.queue_server_event({"type": _DELTA, "text": "hello"})

    stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.02))
    stream.push_frame(_frame())
    stream.end_input()
    try:
        with pytest.raises(APIConnectionError) as excinfo:
            [_ async for _ in stream]
        assert excinfo.value.retryable is False
    finally:
        await stream.aclose()


async def test_stream_still_surfaces_recv_errors() -> None:
    """The grace period must not swallow a genuine failure.

    asyncio.wait() does not raise, unlike gather(), so the task results have to
    be checked explicitly afterwards or every receive-side error would be
    silently discarded.
    """
    s = stepfun.STT()
    ws = _FakeWebSocket()
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]
    ws.queue_server_event(_SESSION_READY)
    ws.queue_server_event(
        {
            "type": "error",
            "meta": {"session_id": "sess-1"},
            "error": {"code": "invalid_value", "message": "bad rate"},
        }
    )

    stream = s.stream(conn_options=_no_retry())
    stream.push_frame(_frame())
    stream.end_input()

    with pytest.raises(APIStatusError) as excinfo:
        async for _ in stream:
            pass
    # The error passes through translation unchanged; it must not be its own cause.
    assert excinfo.value.__cause__ is not excinfo.value
    await stream.aclose()


async def test_unknown_model_deltas_replace_not_append() -> None:
    # The end-to-end consequence of the above, for any model not on the
    # known-incremental denylist.
    s = stepfun.STT(model="stepaudio-9-asr-stream")
    _, events = await _drive(
        s,
        [
            _SESSION_READY,
            {"type": _DELTA, "text": "hello"},
            {"type": _DELTA, "text": "hello world"},
        ],
    )
    assert _texts(events, stt.SpeechEventType.INTERIM_TRANSCRIPT) == [
        "hello",
        "hello world",
    ]


def test_unknown_model_does_not_overdeclare_alignment() -> None:
    # Alignment stays an allowlist: claiming "word" for a model that never
    # emits `words` would break adaptive interruption silently.
    caps = stepfun.STT(
        model="some-future-model-we-have-never-seen", enable_timestamp_align=True
    ).capabilities
    assert caps.aligned_transcript is False


async def test_arbitrary_model_sent_in_session_update() -> None:
    # The Literal does not restrict what can be sent; whatever the caller
    # chooses reaches the server verbatim.
    s = stepfun.STT(model="stepaudio-9-asr-stream")
    ws, _ = await _drive(s, [_SESSION_READY])
    transcription = ws.frames_of_type("session.update")[0]["session"]["audio"]["input"][
        "transcription"
    ]
    assert transcription["model"] == "stepaudio-9-asr-stream"


async def test_session_created_does_not_acknowledge_configuration() -> None:
    s = stepfun.STT()
    ws = _FakeWebSocket()
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]
    ws.queue_server_event({"type": "session.created"})
    ws.queue_server_event(
        {"type": "error", "error": {"code": "invalid_value", "message": "bad configuration"}}
    )
    stream = s.stream(conn_options=_no_retry())
    stream.push_frame(_frame())
    stream.end_input()
    try:
        with pytest.raises(APIStatusError):
            [_ async for _ in stream]
        assert not ws.frames_of_type("input_audio_buffer.append")
        assert ws in s._pool.removed
    finally:
        await stream.aclose()


async def test_session_created_followed_by_updated_is_accepted() -> None:
    ws, _ = await _drive(stepfun.STT(), [{"type": "session.created"}, _SESSION_READY])
    assert ws.frames_of_type("input_audio_buffer.append")


async def test_end_input_commits_only_once() -> None:
    ws, _ = await _drive(stepfun.STT(server_vad=False), [_SESSION_READY])
    assert len(ws.frames_of_type("input_audio_buffer.commit")) == 1


async def test_empty_stream_does_not_commit() -> None:
    ws, _ = await _drive(stepfun.STT(server_vad=False), [_SESSION_READY], frames=0)
    assert not ws.frames_of_type("input_audio_buffer.commit")


async def _collect(stream):
    return [event async for event in stream]


@pytest.mark.parametrize("server_vad", [True, False])
@pytest.mark.parametrize("delay", [0.03, 3.1])
async def test_delayed_final_is_delivered_without_server_closing(
    server_vad: bool, delay: float
) -> None:
    """Deliver delayed finals after end_input without requiring socket closure."""

    class Peer(_FakeWebSocket):
        async def send_str(self, data: str) -> None:
            await super().send_str(data)
            if self.sent[-1]["type"] == "session.update":
                self.queue_server_event(_SESSION_READY)

    ws = Peer()
    s = stepfun.STT(server_vad=server_vad)
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]
    stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=4))
    stream.push_frame(_frame(ms=30))  # exercises end-input tail draining
    stream.end_input()

    async def finish() -> None:
        for _ in range(1000):
            if stream._input_done:
                break
            await asyncio.sleep(0.001)
        assert stream._input_done, "sender did not finish input"
        ws.queue_server_event({"type": _DELTA, "item_id": "last", "text": "hello"})
        await asyncio.sleep(delay)
        if server_vad:
            ws.queue_server_event(
                {
                    "type": "input_audio_buffer.speech_stopped",
                    "item_id": "last",
                    "audio_start_ms": 30,
                }
            )
        ws.queue_server_event({"type": _COMPLETED, "item_id": "last", "transcript": "hello"})

    peer_task = asyncio.create_task(finish())
    try:
        events = await asyncio.wait_for(_collect(stream), timeout=5)
        finals = [e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
        assert [e.alternatives[0].text for e in finals] == ["hello"]
        # Server VAD also commits, but only as a best-effort request.
        assert ws.frames_of_type("input_audio_buffer.commit")
        appends = ws.frames_of_type("input_audio_buffer.append")
        real_bytes = len(_utils.b64_to_pcm(appends[0]["audio"]))
        assert real_bytes == 16000 * 30 // 1000 * 2
        if server_vad:
            assert len(appends) > 1  # trailing endpointing silence
        usage = [e for e in events if e.recognition_usage]
        assert sum(e.recognition_usage.audio_duration for e in usage) == pytest.approx(0.03)
    finally:
        await asyncio.gather(peer_task, return_exceptions=True)
        await stream.aclose()


async def test_socket_close_before_pending_final_is_an_error() -> None:
    class Peer(_FakeWebSocket):
        async def send_str(self, data: str) -> None:
            await super().send_str(data)
            if self.sent[-1]["type"] == "session.update":
                self.queue_server_event(_SESSION_READY)
            elif self.sent[-1]["type"] == "input_audio_buffer.commit":
                # The transcription starts, then the socket dies before the
                # final for it arrives.
                self.queue_server_event(_item("item_x", "in_progress"))
                self.queue_close()

    ws = Peer()
    s = stepfun.STT(server_vad=False)
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]
    stream = s.stream(conn_options=APIConnectOptions(max_retry=0))
    stream.push_frame(_frame())
    stream.end_input()
    try:
        with pytest.raises(APIConnectionError, match="before final transcription") as excinfo:
            await _collect(stream)
        # The audio was consumed and its final never arrived, so a retry on a
        # fresh session cannot recover it.
        assert excinfo.value.retryable is False
        assert ws.closed
        assert ws in s._pool.removed
    finally:
        await stream.aclose()


async def test_empty_stt_stream_finishes_without_final_response() -> None:
    ws = _FakeWebSocket()
    ws.queue_server_event(_SESSION_READY)
    s = stepfun.STT()
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]
    stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.2))
    stream.end_input()
    try:
        assert await asyncio.wait_for(_collect(stream), 1) == []
        assert not ws.frames_of_type("input_audio_buffer.append")
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

    async def send_str(self, data: str) -> None:
        await super().send_str(data)
        if self.sent[-1]["type"] == "session.update":
            self.queue_server_event(_SESSION_READY)


async def test_connection_loss_after_input_ends_is_not_retried() -> None:
    """A retry cannot replay consumed audio, even once input has ended.

    The stop and the final arrive after both frames were uploaded, and the
    transport then fails rather than closing cleanly. The audio is gone from
    the input channel either way, so the loss is reported instead of retried.
    """

    class Failing(_ReadyPeer):
        async def send_str(self, data: str) -> None:
            await super().send_str(data)
            if self.sent[-1]["type"] == "input_audio_buffer.append":
                self.queue_server_event(
                    {"type": "input_audio_buffer.speech_started", "item_id": "a"}
                )

    failing = Failing()
    pool = _SequencedPool([failing, _ReadyPeer()])
    s = stepfun.STT()
    s._pool = pool  # type: ignore[assignment]
    stream = s.stream(conn_options=APIConnectOptions(max_retry=1, retry_interval=0, timeout=2))
    try:
        stream.push_frame(_frame())
        stream.push_frame(_frame())
        for _ in range(100):
            if failing.frames_of_type("input_audio_buffer.append"):
                break
            await asyncio.sleep(0.01)
        # The stop arrives late and carries no position.
        failing.queue_server_event({"type": "input_audio_buffer.speech_stopped", "item_id": "a"})
        failing.queue_server_event({"type": _COMPLETED, "item_id": "a", "transcript": "A"})
        failing.queue_error()
        stream.end_input()
        with pytest.raises(APIConnectionError, match="cannot be recovered") as excinfo:
            await asyncio.wait_for(_collect(stream), 5)
        assert excinfo.value.retryable is False
        assert pool.handed_out == 1
    finally:
        await stream.aclose()


async def test_connection_loss_after_a_final_is_not_retried() -> None:
    """A retry cannot replay consumed audio, even after a delivered final."""

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


async def test_connection_loss_before_consuming_audio_still_retries() -> None:
    """Connection setup may retry because no caller audio has been consumed."""

    class Failing(_FakeWebSocket):
        async def send_str(self, data: str) -> None:
            await super().send_str(data)
            self.queue_error()

    class Healthy(_ReadyPeer):
        async def send_str(self, data: str) -> None:
            await super().send_str(data)
            if self.sent[-1]["type"] == "input_audio_buffer.append":
                for event in (
                    {"type": "input_audio_buffer.speech_stopped", "audio_start_ms": 10_000},
                    {"type": _COMPLETED, "item_id": "item-1", "transcript": "hello"},
                ):
                    self.queue_server_event(event)

    healthy = Healthy()
    pool = _SequencedPool([Failing(), healthy])
    s = stepfun.STT()
    s._pool = pool  # type: ignore[assignment]
    stream = s.stream(conn_options=APIConnectOptions(max_retry=1, retry_interval=0, timeout=1))
    stream.push_frame(_frame())
    stream.end_input()
    try:
        events = await asyncio.wait_for(_collect(stream), 5)
        assert pool.handed_out == 2
        silence_bytes = 16000 * (s._opts.vad_silence_duration_ms + 50) // 1000 * 2
        uploaded = sum(
            len(_utils.b64_to_pcm(f["audio"]))
            for f in healthy.frames_of_type("input_audio_buffer.append")
        )
        assert uploaded == 16000 * 100 // 1000 * 2 + silence_bytes
        usage = [e for e in events if e.recognition_usage]
        assert sum(e.recognition_usage.audio_duration for e in usage) == pytest.approx(0.1)
    finally:
        await stream.aclose()


# -- commit acknowledgements and end-of-input handling ---------------------


def _item(item_id: str, status: str) -> dict:
    return {"type": "conversation.item.created", "item": {"id": item_id, "status": status}}


async def test_manual_completed_turn_does_not_wait_for_the_item_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A final closes the transcription item, not the commit bookkeeping item."""

    from livekit.plugins.stepfun import stt as stt_impl

    # Longer than the assertion below, so waiting it out would fail here.
    monkeypatch.setattr(stt_impl, "_COMMIT_ITEM_WINDOW", 5.0)

    class Peer(_ReadyPeer):
        async def send_str(self, data: str) -> None:
            await super().send_str(data)
            if self.sent[-1]["type"] == "input_audio_buffer.commit":
                for event in (
                    _item("item_buffer", "completed"),
                    _item("item_x_0", "in_progress"),
                    {"type": _COMPLETED, "item_id": "item_x_0", "transcript": "hello"},
                ):
                    self.queue_server_event(event)

    ws = Peer()
    s = stepfun.STT(server_vad=False)
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]
    stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
    stream.push_frame(_frame())
    stream.end_input()
    try:
        started = time.monotonic()
        events = await asyncio.wait_for(_collect(stream), 5)
        # A final cannot resolve the arrival window of an unrelated commit.
        assert time.monotonic() - started < 0.5
        finals = [e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
        assert [e.alternatives[0].text for e in finals] == ["hello"]
    finally:
        await stream.aclose()


async def test_manual_in_progress_item_before_commit_ack() -> None:
    """Handle transcription starting before the commit acknowledgement."""

    class Peer(_ReadyPeer):
        async def send_str(self, data: str) -> None:
            await super().send_str(data)
            if len(self.frames_of_type("input_audio_buffer.append")) == 1 and (
                self.sent[-1]["type"] == "input_audio_buffer.append"
            ):
                self.queue_server_event(_item("item_y_0", "in_progress"))

    ws = Peer()
    s = stepfun.STT(server_vad=False)
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]
    stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
    stream.push_frame(_frame())
    stream.end_input()

    async def finish() -> None:
        for _ in range(1000):
            if stream._input_done:
                break
            await asyncio.sleep(0.001)
        assert stream._input_done, "sender did not finish input"
        await asyncio.sleep(0.2)  # the final arrives well after the ack
        ws.queue_server_event({"type": _COMPLETED, "item_id": "item_y_0", "transcript": "long"})

    peer = asyncio.create_task(finish())
    try:
        events = await asyncio.wait_for(_collect(stream), 5)
        finals = [e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
        assert [e.alternatives[0].text for e in finals] == ["long"]
    finally:
        await asyncio.gather(peer, return_exceptions=True)
        await stream.aclose()


async def test_manual_commit_of_silence_completes_without_a_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An acknowledged silent commit needs no final transcript."""
    from livekit.plugins.stepfun import stt as stt_impl

    monkeypatch.setattr(stt_impl, "_COMMIT_ITEM_WINDOW", 0.2)
    ws = _ReadyPeer()
    s = stepfun.STT(server_vad=False)
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]
    stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
    stream.push_frame(_frame())
    stream.end_input()
    try:
        started = time.monotonic()
        events = await asyncio.wait_for(_collect(stream), 5)
        # It waited out the window for an in_progress item, then finished.
        assert time.monotonic() - started >= 0.2
        assert not [e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    finally:
        await stream.aclose()


async def test_manual_unacknowledged_commit_times_out_as_lost_speech() -> None:
    ws = _ReadyPeer()
    ws.ack_commits = False
    s = stepfun.STT(server_vad=False)
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]
    stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.2))
    stream.push_frame(_frame())
    stream.end_input()
    try:
        with pytest.raises(APIConnectionError) as excinfo:
            await asyncio.wait_for(_collect(stream), 5)
        assert excinfo.value.retryable is False
    finally:
        await stream.aclose()


async def test_commit_acknowledged_while_the_send_is_suspended() -> None:
    """The acknowledgement can overtake the send that triggered it.

    aiohttp writes the frame and then awaits the transport drain, so the
    receiver may process `committed` before the sender records the commit.
    Accounting registered only after the await loses that acknowledgement and
    waits for a second one that never comes.
    """

    class Peer(_ReadyPeer):
        async def send_str(self, data: str) -> None:
            await super().send_str(data)
            if self.sent[-1].get("type") == "input_audio_buffer.commit":
                await asyncio.sleep(0)  # the post-write drain window

    ws = Peer()
    s = stepfun.STT(server_vad=False)
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]
    stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
    stream.push_frame(_frame())
    stream.end_input()

    async def server() -> None:
        # the transcription the acknowledged commit started
        ws.queue_server_event(_item("item_x", "in_progress"))
        ws.queue_server_event({"type": _COMPLETED, "item_id": "item_x", "transcript": "hello"})

    peer = asyncio.create_task(server())
    try:
        events = await asyncio.wait_for(_collect(stream), 3)
        finals = [
            e.alternatives[0].text for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT
        ]
        assert finals == ["hello"]
    finally:
        await asyncio.gather(peer, return_exceptions=True)
        await stream.aclose()


async def test_final_for_an_earlier_commit_does_not_resolve_a_later_one() -> None:
    """Two commits in flight: the first final must not release the second.

    Transcription items can appear well after their acknowledgement, so a
    final belonging to one commit says nothing about another commit whose item
    has not arrived yet.
    """

    class Peer(_ReadyPeer):
        async def send_str(self, data: str) -> None:
            await super().send_str(data)
            commits = self.frames_of_type("input_audio_buffer.commit")
            if commits and len(commits) == 1:
                # only the first commit's item shows up early
                self.queue_server_event(_item("item_a", "in_progress"))

    ws = Peer()
    s = stepfun.STT(server_vad=False)
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]
    stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))

    async def collect() -> list:
        return [event async for event in stream]

    task = asyncio.create_task(collect())
    stream.push_frame(_frame())
    stream.flush()  # commit A, item A follows
    await asyncio.sleep(0.05)
    stream.push_frame(_frame())
    stream.flush()  # commit B, its item is late
    await asyncio.sleep(0.05)
    ws.queue_server_event({"type": _COMPLETED, "item_id": "item_a", "transcript": "A"})
    await asyncio.sleep(0.05)
    stream.end_input()
    await asyncio.sleep(0.2)  # B's item only now
    ws.queue_server_event(_item("item_b", "in_progress"))
    ws.queue_server_event({"type": _COMPLETED, "item_id": "item_b", "transcript": "B"})
    ws.queue_close()
    try:
        events = await asyncio.wait_for(task, 5)
        finals = [
            e.alternatives[0].text for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT
        ]
        assert finals == ["A", "B"]
    finally:
        await stream.aclose()


async def test_server_vad_waits_for_a_late_turn_within_the_settle_window() -> None:
    """An earlier final does not immediately end the stream.

    A turn reported within the quiet settle is still delivered. Later turns are
    best-effort: the API gives no proof that no further audio will be reported.
    """
    ws = _ReadyPeer()
    s = stepfun.STT()
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]
    stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=3))
    consumer = asyncio.create_task(_collect(stream))
    try:
        for _ in range(5):
            stream.push_frame(_frame(ms=20))
            await asyncio.sleep(0.02)
        for event in (
            {"type": "input_audio_buffer.speech_started", "item_id": "first"},
            {
                "type": "input_audio_buffer.speech_stopped",
                "item_id": "first",
                "audio_start_ms": 100,
            },
            {"type": _COMPLETED, "item_id": "first", "transcript": "first"},
        ):
            ws.queue_server_event(event)
        await asyncio.sleep(0.02)
        for _ in range(5):
            stream.push_frame(_frame(ms=20))
            await asyncio.sleep(0.02)
        stream.end_input()
        await asyncio.sleep(0.5)
        assert not consumer.done(), "finished during the quiet settle"
        for event in (
            {"type": "input_audio_buffer.speech_started", "item_id": "last"},
            {"type": "input_audio_buffer.speech_stopped", "item_id": "last", "audio_start_ms": 200},
            {"type": _COMPLETED, "item_id": "last", "transcript": "last"},
        ):
            ws.queue_server_event(event)
        events = await asyncio.wait_for(consumer, 2)
        assert _texts(events, stt.SpeechEventType.FINAL_TRANSCRIPT) == ["first", "last"]
    finally:
        await stream.aclose()
        await asyncio.gather(consumer, return_exceptions=True)


async def test_server_vad_trailing_silence_completes_without_boundary() -> None:
    """A VAD boundary need not cover silence sent after the last turn.

    The final arrives before two seconds of trailing silence; no second
    speech_stopped is promised, and that must not turn a delivered final into
    a timeout.
    """
    ws = _ReadyPeer()
    s = stepfun.STT()
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]
    stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=3))
    consumer = asyncio.create_task(_collect(stream))
    try:
        for _ in range(10):
            stream.push_frame(_frame(ms=20))
            await asyncio.sleep(0.01)
        for event in (
            {"type": "input_audio_buffer.speech_started", "item_id": "last"},
            {"type": "input_audio_buffer.speech_stopped", "item_id": "last", "audio_start_ms": 200},
            {"type": _COMPLETED, "item_id": "last", "transcript": "hello"},
        ):
            ws.queue_server_event(event)
        for _ in range(100):  # two seconds of trailing silence
            stream.push_frame(_frame(ms=20))
        started = time.monotonic()
        stream.end_input()
        events = await asyncio.wait_for(consumer, 5)
        # A quiet settle, not the full connect timeout.
        assert time.monotonic() - started < 2
        assert _texts(events, stt.SpeechEventType.FINAL_TRANSCRIPT) == ["hello"]
    finally:
        await stream.aclose()
        await asyncio.gather(consumer, return_exceptions=True)


async def test_manual_silent_commit_with_short_timeout_reports_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timeout shorter than the item window cannot confirm the commit."""
    from livekit.plugins.stepfun import stt as stt_impl

    monkeypatch.setattr(stt_impl, "_COMMIT_ITEM_WINDOW", 5.0)
    ws = _ReadyPeer()
    s = stepfun.STT(server_vad=False)
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]
    stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.2))
    stream.push_frame(_frame())
    stream.end_input()
    try:
        with pytest.raises(APIConnectionError, match="cannot confirm"):
            await asyncio.wait_for(_collect(stream), 2)
    finally:
        await stream.aclose()


async def test_close_without_a_final_reports_missing_transcription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A close is not evidence that the uploaded audio was transcribed."""
    from livekit.plugins.stepfun import stt as stt_impl

    ws = _ReadyPeer()
    original = stt_impl.RecognizeStream._finish_input

    async def finish(self, ws_, samples_per_frame):  # type: ignore[no-untyped-def]
        await original(self, ws_, samples_per_frame)
        ws.queue_close()  # closes with no final reported

    monkeypatch.setattr(stt_impl.RecognizeStream, "_finish_input", finish)
    s = stepfun.STT()
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]
    stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
    stream.push_frame(_frame())
    stream.end_input()
    try:
        with pytest.raises(APIConnectionError, match="closed before final"):
            await asyncio.wait_for(_collect(stream), 3)
    finally:
        await stream.aclose()


async def test_server_vad_without_any_final_reports_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No final means no evidence the uploaded audio was processed."""
    from livekit.plugins.stepfun import stt as stt_impl

    monkeypatch.setattr(stt_impl, "_NO_EVIDENCE_DEADLINE", 0.2)
    ws = _ReadyPeer()
    s = stepfun.STT()
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]
    stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
    try:
        consumer = asyncio.create_task(_collect(stream))
        for _ in range(5):
            stream.push_frame(_frame(ms=20))
            await asyncio.sleep(0.01)
        stream.end_input()
        with pytest.raises(APIConnectionError, match="cannot confirm"):
            await asyncio.wait_for(consumer, 3)
    finally:
        await stream.aclose()


async def test_final_during_upload_followed_by_silence_completes() -> None:
    """Trailing silence after a final must not reopen the stream.

    The server finalizes "hello" while the client is still uploading; the
    remaining frames are silence, so no second transcription ever arrives.
    """

    class Peer(_ReadyPeer):
        async def send_str(self, data: str) -> None:
            await super().send_str(data)
            appends = self.frames_of_type("input_audio_buffer.append")
            if self.sent[-1]["type"] == "input_audio_buffer.append" and len(appends) == 1:
                for event in (
                    {"type": "input_audio_buffer.speech_started", "item_id": "item-1"},
                    {
                        "type": "input_audio_buffer.speech_stopped",
                        "item_id": "item-1",
                        "audio_start_ms": 200,
                    },
                    {"type": _COMPLETED, "item_id": "item-1", "transcript": "hello"},
                ):
                    self.queue_server_event(event)

    ws = Peer()
    s = stepfun.STT()
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]
    stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1))
    for _ in range(4):
        stream.push_frame(_frame(ms=50))
    stream.end_input()
    try:
        events = await asyncio.wait_for(_collect(stream), 5)
        finals = [e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
        assert [e.alternatives[0].text for e in finals] == ["hello"]
    finally:
        await stream.aclose()


@pytest.mark.parametrize(
    "events",
    [
        # speech_started without an id; the final names the item.
        [
            {"type": "input_audio_buffer.speech_started"},
            {"type": "input_audio_buffer.speech_stopped"},
            {"type": _COMPLETED, "item_id": "item-1", "transcript": "hello"},
        ],
        # id-less start, then an identified commit for the same utterance.
        [
            {"type": "input_audio_buffer.speech_started"},
            {"type": "input_audio_buffer.committed", "item_id": "item-1"},
            {"type": "input_audio_buffer.speech_stopped", "item_id": "item-1"},
            {"type": _COMPLETED, "item_id": "item-1", "transcript": "hello"},
        ],
        # interim without an id, final with one.
        [
            {"type": "input_audio_buffer.speech_started", "item_id": "item-1"},
            {"type": _DELTA, "text": "hel"},
            {"type": "input_audio_buffer.speech_stopped", "item_id": "item-1"},
            {"type": _COMPLETED, "item_id": "item-1", "transcript": "hello"},
        ],
        # identified and anonymous interims both in flight; the identified
        # final must clear the anonymous one too.
        [
            {"type": "input_audio_buffer.speech_started", "item_id": "item-1"},
            {"type": _DELTA, "item_id": "item-1", "text": "hel"},
            {"type": _DELTA, "text": "hello"},
            {"type": "input_audio_buffer.speech_stopped", "item_id": "item-1"},
            {"type": _COMPLETED, "item_id": "item-1", "transcript": "hello"},
        ],
    ],
)
async def test_mixed_identified_and_unidentified_events_complete(events: list[dict]) -> None:
    """An id-less report must be closed by the final for its utterance."""

    class Peer(_ReadyPeer):
        async def send_str(self, data: str) -> None:
            await super().send_str(data)
            appends = self.frames_of_type("input_audio_buffer.append")
            if self.sent[-1]["type"] == "input_audio_buffer.append" and len(appends) == 1:
                for event in events:
                    if event.get("type") == "input_audio_buffer.speech_stopped":
                        event = {**event, "audio_start_ms": 100}
                    self.queue_server_event(event)

    ws = Peer()
    s = stepfun.STT()
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]
    stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1))
    stream.push_frame(_frame())
    stream.end_input()
    try:
        out = await asyncio.wait_for(_collect(stream), 5)
        finals = [e for e in out if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
        assert [e.alternatives[0].text for e in finals] == ["hello"]
    finally:
        await stream.aclose()


async def test_streaming_reports_detected_language() -> None:
    _, events = await _drive(
        stepfun.STT(language="en"),
        [
            _SESSION_READY,
            {"type": _DELTA, "text": "你好", "language": "zh"},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "你好",
                "language": "zh",
            },
        ],
    )
    transcripts = [e for e in events if e.alternatives]
    assert transcripts
    assert all(e.alternatives[0].language == "zh" for e in transcripts)


async def test_batch_prewarm_does_not_open_streaming_socket() -> None:
    session = _FakeSSESession([])
    s = stepfun.STT(use_streaming=False, http_session=session)  # type: ignore[arg-type]
    pool = _FakeConnectionPool(_FakeWebSocket())
    s._pool = pool  # type: ignore[assignment]
    s.prewarm()
    assert pool.prewarm_calls == 0
    await s.aclose()


@pytest.mark.parametrize(
    "lines",
    [
        [],
        [
            'data: {"type":"transcript.text.delta","delta":"unfinished"}',
            "data: [DONE]",
        ],
    ],
)
async def test_batch_rejects_truncated_response(lines: list[str]) -> None:
    from livekit.agents import APIConnectionError

    s = stepfun.STT(http_session=_FakeSSESession(lines))  # type: ignore[arg-type]
    with pytest.raises(APIConnectionError, match="before transcript.text.done"):
        await s.recognize([_frame()], conn_options=_no_retry())


async def test_batch_returns_at_done_without_waiting_for_eof() -> None:
    session = _FakeSSESession(
        [
            'data: {"type":"transcript.text.done","text":"complete"}',
            'data: {"type":"error","message":"must not read past done"}',
        ]
    )
    s = stepfun.STT(http_session=session)  # type: ignore[arg-type]
    event = await s.recognize([_frame()], conn_options=_no_retry())
    assert event.alternatives[0].text == "complete"


async def test_interleaved_items_keep_separate_transcripts() -> None:
    _, events = await _drive(
        stepfun.STT(),
        [
            _SESSION_READY,
            {"type": _DELTA, "item_id": "first", "text": "first turn"},
            {"type": _DELTA, "item_id": "second", "text": "second turn"},
            {"type": "conversation.item.input_audio_transcription.completed", "item_id": "first"},
            {"type": "conversation.item.input_audio_transcription.completed", "item_id": "second"},
        ],
    )
    assert _texts(events, stt.SpeechEventType.FINAL_TRANSCRIPT) == ["first turn", "second turn"]


async def test_empty_final_does_not_restore_rejected_partial() -> None:
    _, events = await _drive(
        stepfun.STT(),
        [
            _SESSION_READY,
            {"type": _DELTA, "text": "discarded"},
            {"type": "conversation.item.input_audio_transcription.completed", "transcript": ""},
        ],
    )
    assert not _texts(events, stt.SpeechEventType.FINAL_TRANSCRIPT)


def test_model_update_refreshes_word_alignment_capability() -> None:
    speech = stepfun.STT(enable_timestamp_align=True)
    assert speech.capabilities.aligned_transcript == "word"
    speech.update_options(model="step-asr-1.1-stream")
    assert speech.capabilities.aligned_transcript is False
    speech.update_options(model="stepaudio-2.5-asr-stream")
    assert speech.capabilities.aligned_transcript == "word"


async def test_batch_empty_final_does_not_restore_rejected_delta() -> None:
    session = _FakeSSESession(
        [
            'data: {"type":"transcript.text.delta","delta":"discarded"}',
            'data: {"type":"transcript.text.done","text":""}',
        ]
    )
    speech = stepfun.STT(http_session=session)  # type: ignore[arg-type]
    event = await speech.recognize([_frame()])
    assert event.alternatives[0].text == ""


@pytest.mark.parametrize("configured", [False, True])
async def test_transport_error_closes_socket(configured):
    ws = _FakeWebSocket()
    if configured:
        ws.queue_server_event(_SESSION_READY)
    ws.queue_error()
    s = stepfun.STT()
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]
    stream = s.stream(conn_options=_no_retry())
    stream.push_frame(_frame())
    stream.end_input()
    try:
        with pytest.raises(APIConnectionError, match="websocket error"):
            await asyncio.wait_for(anext(stream), 1)
        assert ws.closed
    finally:
        await s.aclose()


async def test_stream_send_error_after_audio_is_not_retried() -> None:
    """A failed upload is mapped, and never retried once audio was consumed.

    The frame has already left the input channel, so a retry would resume
    from after it and drop that speech silently. Marking the error
    non-retryable is what makes consuming audio without retaining it safe.
    """

    class BrokenSocket(_FakeWebSocket):
        async def send_str(self, data: str) -> None:
            if "input_audio_buffer.append" in data:
                raise aiohttp.ClientConnectionError("upload failed")
            await super().send_str(data)

    ws = BrokenSocket()
    ws.queue_server_event(_SESSION_READY)
    s = stepfun.STT()
    s._pool = _FakeConnectionPool(ws)  # type: ignore[assignment]
    stream = s.stream(conn_options=APIConnectOptions(max_retry=1, retry_interval=0))
    stream.push_frame(_frame())
    stream.end_input()
    try:
        with pytest.raises(APIConnectionError, match="upload failed") as excinfo:
            await asyncio.wait_for(anext(stream), 1)
        assert excinfo.value.retryable is False
        assert ws.closed
    finally:
        await s.aclose()
