"""Gradium TTS error handling, against a scripted fake websocket."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
from dataclasses import dataclass
from typing import Any

import pytest

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
)

# hermetic: every test talks to a scripted fake websocket, so this is a unit module
pytestmark = pytest.mark.unit

# no retries unless a test asks for them: they assert on the error of the single attempt
CONN_OPTS = APIConnectOptions(max_retry=0, timeout=5)

SAMPLE_RATE = 48000
# 0.5s of int16 mono silence, enough to fill the emitter's 200ms frames
AUDIO_CHUNK = bytes(SAMPLE_RATE // 2 * 2)

READY = {"type": "ready", "request_id": "req", "model_name": "default", "sample_rate": SAMPLE_RATE}
AUDIO = {
    "type": "audio",
    "audio": base64.b64encode(AUDIO_CHUNK).decode(),
    "start_s": 0,
    "stop_s": 0.5,
}
EOS = {"type": "end_of_stream"}
ERROR = {"type": "error", "message": "Unknown voice 'nope'.", "code": 1011}


@dataclass
class _Msg:
    type: Any
    data: Any = None
    extra: Any = None


class FakeWebSocket:
    """Scripted server side of one socket carrying any number of sessions.

    `scripts` holds one list of frames per session, consumed in the order the plugin
    sends `setup` messages; a script's frames are tagged with that session's
    client_req_id unless they carry one already. Frames are JSON dicts (sent as TEXT),
    the string "close" (a CLOSE frame with `close_code`), the string "timeout"
    (receive() raises asyncio.TimeoutError, as aiohttp does when the receive timeout
    elapses), ("error", exc) (an ERROR frame, with `exc` as the socket's exception) or
    ("raise", exc) (receive() raises `exc`).
    receive() blocks while nothing is queued, like a live socket between replies.

    `fail_send_on` makes send_str raise a connection error for that message type.
    """

    def __init__(
        self,
        scripts: list[list[Any]],
        *,
        close_code: int | None = None,
        close_reason: str = "",
        fail_send_on: str | None = None,
    ) -> None:
        import aiohttp

        self._ws_type = aiohttp.WSMsgType
        self._scripts = [list(script) for script in scripts]
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._fail_send_on = fail_send_on
        self.close_code = close_code
        self._close_reason = close_reason
        self.closed = False
        self.sent: list[dict[str, Any]] = []
        self._exception: BaseException | None = None

    def exception(self) -> BaseException | None:
        return self._exception

    def queue(self, *frames: Any) -> None:
        """Queue frames the server sends on its own, e.g. for an earlier session."""
        for frame in frames:
            self._queue.put_nowait(frame)

    async def send_str(self, data: str) -> None:
        import aiohttp

        msg = json.loads(data)
        if msg.get("type") == self._fail_send_on:
            raise aiohttp.ClientConnectionResetError("Cannot write to closing transport")
        self.sent.append(msg)
        if msg["type"] == "setup":
            script = self._scripts.pop(0) if self._scripts else []
            for frame in script:
                if isinstance(frame, dict) and "client_req_id" not in frame:
                    frame = {**frame, "client_req_id": msg.get("client_req_id")}
                self._queue.put_nowait(frame)

    async def receive(self, timeout: float | None = None) -> _Msg:
        frame = await self._queue.get()
        if frame == "timeout":
            raise asyncio.TimeoutError
        if frame == "close":
            return _Msg(self._ws_type.CLOSE, self.close_code, self._close_reason)
        if isinstance(frame, tuple) and frame[0] == "error":
            self._exception = frame[1]
            return _Msg(self._ws_type.ERROR, frame[1])
        if isinstance(frame, tuple) and frame[0] == "raise":
            raise frame[1]
        return _Msg(self._ws_type.TEXT, json.dumps(frame))

    async def close(self) -> bool:
        self.closed = True
        return True

    async def __aenter__(self) -> FakeWebSocket:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


class FakeSession:
    """Hands out `sockets` in order, one per connect; the last one is reused for retries."""

    def __init__(self, *sockets: FakeWebSocket) -> None:
        self._sockets = list(sockets)
        self.connects: list[tuple[str, dict[str, Any]]] = []

    async def ws_connect(self, url: str, **kwargs: Any) -> FakeWebSocket:
        self.connects.append((url, kwargs))
        return self._sockets.pop(0) if len(self._sockets) > 1 else self._sockets[0]


class RejectingSession:
    """A server that refuses the websocket handshake with an HTTP status."""

    def __init__(self, status: int) -> None:
        self.status = status

    async def ws_connect(self, url: str, **kwargs: Any) -> FakeWebSocket:
        from unittest.mock import MagicMock

        import aiohttp

        raise aiohttp.ClientResponseError(
            MagicMock(), (), status=self.status, message="Unauthorized"
        )


_open_tts: list[Any] = []


@pytest.fixture(autouse=True)
async def _close_tts_instances():
    """Every TTS keeps a shared socket and its reader task; close them like the agent does."""
    yield
    for tts in _open_tts:
        await tts.aclose()
    _open_tts.clear()


def _new_tts(http_session: Any, **kwargs: Any):
    from livekit.plugins.gradium import TTS

    tts = TTS(api_key="test-key", http_session=http_session, **kwargs)
    _open_tts.append(tts)
    return tts


def _make_tts(frames: list[Any], *, close_code: int | None = None, **ws_kwargs: Any):
    ws = FakeWebSocket([frames], close_code=close_code, **ws_kwargs)
    session = FakeSession(ws)
    return _new_tts(session), session, ws


def _make_tts_scripts(scripts: list[list[Any]]):
    """A TTS over one fake socket that will carry several sessions."""
    ws = FakeWebSocket(scripts)
    session = FakeSession(ws)
    return _new_tts(session), session, ws


async def _drain(tts, conn_options: APIConnectOptions) -> float:
    """Run one synthesize() and return the audio duration; bounded so a hang fails."""

    async def _go() -> float:
        duration = 0.0
        async for ev in tts.synthesize("hello world", conn_options=conn_options):
            duration += ev.frame.duration
        return duration

    return await asyncio.wait_for(_go(), timeout=5)


async def _drain_stream(tts, conn_options: APIConnectOptions) -> float:
    async def _go() -> float:
        duration = 0.0
        async with tts.stream(conn_options=conn_options) as stream:
            stream.push_text("hello world")
            stream.end_input()
            async for ev in stream:
                duration += ev.frame.duration
        return duration

    return await asyncio.wait_for(_go(), timeout=5)


async def test_synthesize_emits_audio_and_sends_eos():
    from livekit.plugins.gradium.tts import WS_HEARTBEAT_S

    tts, session, ws = _make_tts([READY, AUDIO, AUDIO, EOS])

    duration = 0.0
    async for ev in tts.synthesize("hello world", conn_options=CONN_OPTS):
        duration += ev.frame.duration
    # two 0.5s chunks; the emitter may pad the tail by a few ms
    assert 1.0 <= duration <= 1.1

    assert [m["type"] for m in ws.sent] == ["setup", "text", "end_of_stream"]
    assert ws.sent[1]["text"] == "hello world"
    # pings are how a dead socket is noticed while the server is silent waiting for text
    assert session.connects[0][1]["heartbeat"] == WS_HEARTBEAT_S


async def test_synthesize_raises_on_error_frame():
    tts, _, _ = _make_tts([READY, ERROR, "close"], close_code=1011)

    with pytest.raises(APIStatusError, match="Unknown voice 'nope'") as excinfo:
        await _drain(tts, CONN_OPTS)
    # the frame must escape as-is, not wrapped in an APIConnectionError
    assert type(excinfo.value) is APIStatusError
    # the server tags the frame with the session id; the rest is the frame as sent
    assert {k: v for k, v in excinfo.value.body.items() if k != "client_req_id"} == ERROR
    assert excinfo.value.status_code == 1011
    assert excinfo.value.retryable is True


async def test_synthesize_raises_on_close_before_eos():
    tts, _, _ = _make_tts([READY, "close"], close_code=1011, close_reason="Worker failure")

    with pytest.raises(APIStatusError, match="closed unexpectedly: Worker failure") as excinfo:
        await _drain(tts, CONN_OPTS)
    assert type(excinfo.value) is APIStatusError
    assert excinfo.value.status_code == 1011
    assert excinfo.value.retryable is True


async def test_synthesize_is_retried_after_partial_audio():
    """A retryable failure after some audio is retried like any other: the base
    class restarts the synthesis under a fresh request_id (livekit/agents#6346).
    Here the socket dies with the first reply; the retry opens a new one."""
    sockets = [
        FakeWebSocket([[READY, AUDIO, "close"]], close_code=1011),
        FakeWebSocket([[READY, AUDIO, EOS]]),
    ]
    session = FakeSession(*sockets)
    tts = _new_tts(session)

    duration = await _drain(tts, APIConnectOptions(max_retry=2, retry_interval=0.05, timeout=5))
    assert 0.5 <= duration <= 0.6  # only the retry's audio reaches the caller
    assert len(session.connects) == 2
    assert sockets[0].closed


async def test_error_frame_with_client_error_code_is_not_retried():
    error = {"type": "error", "message": "Unknown model 'nope'.", "code": 1003}
    tts, session, _ = _make_tts([READY, error, "close"], close_code=1003)

    with pytest.raises(APIStatusError, match="Unknown model") as excinfo:
        await _drain(tts, APIConnectOptions(max_retry=2, timeout=5))
    assert excinfo.value.retryable is False
    assert len(session.connects) == 1


async def test_error_frame_with_server_error_code_is_retried():
    sockets = [FakeWebSocket([[READY, ERROR, "close"]], close_code=1011) for _ in range(2)]
    session = FakeSession(*sockets)
    tts = _new_tts(http_session=session)  # type: ignore[arg-type]

    with pytest.raises(APIStatusError, match="Unknown voice 'nope'"):
        await _drain(tts, APIConnectOptions(max_retry=1, timeout=5))
    assert len(session.connects) == 2
    assert all(ws.sent[-1]["type"] == "end_of_stream" for ws in sockets)


async def test_close_with_unsupported_data_code_is_not_retried():
    tts, session, _ = _make_tts([READY, "close"], close_code=1003)

    with pytest.raises(APIStatusError) as excinfo:
        await _drain(tts, APIConnectOptions(max_retry=2, timeout=5))
    assert excinfo.value.status_code == 1003
    assert excinfo.value.retryable is False
    assert len(session.connects) == 1


async def test_concurrency_limit_error_frame_is_retried():
    """The org's concurrency cap (1008) frees up when another session ends."""
    limit = {
        "type": "error",
        "message": "Concurrency limit exceeded: 15 active sessions",
        "code": 1008,
    }
    sockets = [
        FakeWebSocket([[READY, limit, "close"]], close_code=1008),
        FakeWebSocket([[READY, AUDIO, EOS]]),
    ]
    session = FakeSession(*sockets)
    tts = _new_tts(session)

    duration = await _drain(tts, APIConnectOptions(max_retry=1, timeout=5))
    assert 0.5 <= duration <= 0.6
    assert len(session.connects) == 2


async def test_synthesize_raises_on_transport_error():
    err = RuntimeError("frame too large")
    tts, _, _ = _make_tts([READY, ("error", err)])

    with pytest.raises(APIConnectionError) as excinfo:
        await _drain(tts, CONN_OPTS)
    assert excinfo.value.__cause__ is err


async def test_stream_emits_audio_and_sends_words_then_eos():
    tts, _, ws = _make_tts([READY, AUDIO, AUDIO, EOS])

    duration = 0.0
    async with tts.stream(conn_options=CONN_OPTS) as stream:
        stream.push_text("hello ")
        stream.push_text("world")
        stream.end_input()
        async for ev in stream:
            duration += ev.frame.duration
    assert 1.0 <= duration <= 1.1

    assert [m["type"] for m in ws.sent] == ["setup", "text", "text", "end_of_stream"]
    assert [m["text"] for m in ws.sent[1:3]] == ["hello ", "world "]


async def test_every_message_carries_the_session_id():
    tts, _, ws = _make_tts([READY, AUDIO, EOS])
    await _drain_stream(tts, CONN_OPTS)
    ids = {m.get("client_req_id") for m in ws.sent}
    assert len(ids) == 1
    assert None not in ids


async def test_stream_raises_on_error_frame():
    tts, _, _ = _make_tts([READY, ERROR, "close"], close_code=1011)

    with pytest.raises(APIStatusError, match="Unknown voice 'nope'") as excinfo:
        await _drain_stream(tts, CONN_OPTS)
    assert type(excinfo.value) is APIStatusError
    # the server tags the frame with the session id; the rest is the frame as sent
    assert {k: v for k, v in excinfo.value.body.items() if k != "client_req_id"} == ERROR


async def test_stream_raises_on_close_before_eos():
    tts, _, _ = _make_tts([READY, AUDIO, "close"], close_code=1011)

    with pytest.raises(APIStatusError) as excinfo:
        await _drain_stream(tts, CONN_OPTS)
    assert excinfo.value.status_code == 1011


async def test_stream_raises_on_transport_error():
    err = RuntimeError("frame too large")
    tts, _, _ = _make_tts([READY, ("error", err)])

    with pytest.raises(APIConnectionError) as excinfo:
        await _drain_stream(tts, CONN_OPTS)
    assert excinfo.value.__cause__ is err


async def test_close_without_code_is_retryable():
    """A socket that just dies has no close code; that is a transient failure."""
    tts, _, _ = _make_tts([READY, "close"])

    with pytest.raises(APIStatusError) as excinfo:
        await _drain(tts, CONN_OPTS)
    assert excinfo.value.status_code == -1
    assert excinfo.value.retryable is True


async def test_error_frame_without_fields_is_still_an_error():
    tts, _, _ = _make_tts([READY, {"type": "error"}, "close"])

    with pytest.raises(APIStatusError, match="unknown error") as excinfo:
        await _drain(tts, CONN_OPTS)
    assert excinfo.value.status_code == -1
    assert excinfo.value.retryable is True


@pytest.mark.parametrize("drain", [_drain, _drain_stream])
async def test_waiting_for_ready_times_out(drain):
    """The server never answers the setup: fail after conn_options.timeout."""
    tts, _, _ = _make_tts([])

    with pytest.raises(APITimeoutError):
        await drain(tts, APIConnectOptions(max_retry=0, timeout=0.2))


@pytest.mark.parametrize("drain", [_drain, _drain_stream])
async def test_no_timeout_after_ready(drain):
    """Once the session is ready, silence is the server waiting for text, not a fault."""
    tts, _, ws = _make_tts([READY])

    async def late_audio() -> None:
        await asyncio.sleep(0.5)  # longer than the timeout below
        client_req_id = ws.sent[0]["client_req_id"]
        ws.queue({**AUDIO, "client_req_id": client_req_id}, {**EOS, "client_req_id": client_req_id})

    late = asyncio.create_task(late_audio())
    try:
        duration = await drain(tts, APIConnectOptions(max_retry=0, timeout=0.2))
    finally:
        await late
    assert 0.5 <= duration <= 0.6


@pytest.mark.parametrize("drain", [_drain, _drain_stream])
async def test_rejected_handshake_raises_status_error(drain):
    tts = _new_tts(http_session=RejectingSession(401))  # type: ignore[arg-type]

    with pytest.raises(APIStatusError, match="Unauthorized") as excinfo:
        await drain(tts, APIConnectOptions(max_retry=2, timeout=5))
    assert excinfo.value.status_code == 401
    # 4xx is final: the base class must not have retried
    assert excinfo.value.retryable is False


@pytest.mark.parametrize("drain", [_drain, _drain_stream])
async def test_malformed_audio_frame_raises_connection_error(drain):
    bad_audio = {"type": "audio", "audio": "abc"}  # not valid base64 (bad padding)
    tts, _, _ = _make_tts([READY, bad_audio, EOS])

    with pytest.raises(APIConnectionError) as excinfo:
        await drain(tts, CONN_OPTS)
    assert isinstance(excinfo.value.__cause__, binascii.Error)


async def test_stream_send_failure_raises_connection_error():
    """A send that fails mid-stream must surface, not hang behind the blocked receive."""
    tts, _, _ = _make_tts([READY], fail_send_on="text")

    with pytest.raises(APIConnectionError) as excinfo:
        await asyncio.wait_for(_drain_stream(tts, CONN_OPTS), timeout=5)
    assert "closing transport" in str(excinfo.value)


# --- multiplexing: one socket per TTS instance, sessions in parallel on it -----------
