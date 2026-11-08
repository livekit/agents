from __future__ import annotations

import asyncio
import json
from collections import deque
from unittest.mock import MagicMock

import aiohttp
import pytest
from aiohttp import RequestInfo, WSServerHandshakeError
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError

pytestmark = pytest.mark.plugin("vartaloop")

SECRET_API_KEY = "vt_live_secret-api-key-do-not-log"


def _handshake_error_with_api_key(api_key: str) -> WSServerHandshakeError:
    url = URL("wss://api.vartaloop.ai/v1/tts/stream")
    headers = CIMultiDict({"Host": "api.vartaloop.ai", "Authorization": f"Bearer {api_key}"})
    request_info = RequestInfo(
        url=url, method="GET", headers=CIMultiDictProxy(headers), real_url=url
    )
    return WSServerHandshakeError(request_info, (), status=401, message="Unauthorized")


@pytest.mark.asyncio
async def test_connect_ws_redacts_api_key_from_handshake_error():
    from livekit.plugins.vartaloop import TTS

    tts = TTS(api_key=SECRET_API_KEY)

    async def _raise(*_args, **_kwargs):
        raise _handshake_error_with_api_key(SECRET_API_KEY)

    session = MagicMock()
    session.ws_connect = MagicMock(side_effect=_raise)
    tts._session = session

    with pytest.raises(APIStatusError) as exc_info:
        await tts._connect_ws(timeout=5.0)

    err = exc_info.value
    assert err.status_code == 401
    assert SECRET_API_KEY not in repr(err)
    assert err.__cause__ is None


@pytest.mark.asyncio
async def test_connect_ws_generic_error_does_not_chain_cause():
    from livekit.plugins.vartaloop import TTS

    tts = TTS(api_key=SECRET_API_KEY)
    leaky = ConnectionError(f"wss://api.vartaloop.ai/v1/tts/stream?api_key={SECRET_API_KEY}")

    async def _raise(*_args, **_kwargs):
        raise leaky

    session = MagicMock()
    session.ws_connect = MagicMock(side_effect=_raise)
    tts._session = session

    with pytest.raises(APIConnectionError) as exc_info:
        await tts._connect_ws(timeout=5.0)

    err = exc_info.value
    assert err.__cause__ is None
    assert SECRET_API_KEY not in repr(err)


def test_request_body_sends_only_what_differs_from_the_server_defaults():
    from livekit.plugins.vartaloop import TTS
    from livekit.plugins.vartaloop.tts import _request_body

    tts = TTS(api_key=SECRET_API_KEY)
    body = _request_body(tts._opts, text="नमस्ते", request_id="r1")
    assert body == {
        "text": "नमस्ते",
        "voice": "saanvi",
        "format": "pcm16",
        "request_id": "r1",
        "digits": "auto",
    }

    tts.update_options(
        voice="sahil",
        expression="excited",
        digits="en",
        speed=1.2,
        profile="telephony",
        seed=7,
        temperature=0.5,
        postfx=False,
        raw=True,
    )
    body = _request_body(tts._opts, text="hi", request_id="r2")
    assert body["voice"] == "sahil"
    assert body["style"] == "excited"  # the API's name for the expression
    assert body["digits"] == "en"
    assert body["speed"] == 1.2
    assert body["profile"] == "telephony"
    assert body["seed"] == 7
    assert body["temperature"] == 0.5
    assert body["postfx"] is False
    assert body["raw"] is True
    assert "model" not in body  # the server picks the model from the voice

    tts.update_options(expression="neutral", profile="default", speed=None)
    body = _request_body(tts._opts, text="hi", request_id="r3")
    assert "style" not in body and "profile" not in body and "speed" not in body


def test_options_are_validated():
    from livekit.plugins.vartaloop import TTS

    with pytest.raises(ValueError):
        TTS(api_key=SECRET_API_KEY, speed=3.0)
    with pytest.raises(ValueError):
        TTS(api_key=SECRET_API_KEY, digits="de")
    with pytest.raises(ValueError):
        TTS(api_key=SECRET_API_KEY, sample_rate=16000)
    tts = TTS(api_key=SECRET_API_KEY)
    with pytest.raises(ValueError):
        tts.update_options(profile="radio")
    assert tts.model == "narad"
    assert tts.provider == "Vartaloop"
    assert tts.capabilities.streaming is True


# --- the WebSocket protocol: start -> binary audio -> end per utterance, in order ---


class _FakeMsg:
    def __init__(self, type: aiohttp.WSMsgType, data=None, extra=None) -> None:
        self.type = type
        self.data = data
        self.extra = extra

    @classmethod
    def text(cls, payload: dict) -> _FakeMsg:
        return cls(aiohttp.WSMsgType.TEXT, json.dumps(payload))

    @classmethod
    def binary(cls, data: bytes) -> _FakeMsg:
        return cls(aiohttp.WSMsgType.BINARY, data)


class _FakeWS:
    """A Vartaloop server: every utterance message gets start, one PCM frame, end."""

    def __init__(
        self,
        *,
        stale: bool = False,
        error: dict | None = None,
        stale_error: bool = False,
        silent_from: int | None = None,
    ) -> None:
        self.sent: list[dict] = []
        # utterances from this index on get no answer at all (a stalled server)
        self._silent_from = silent_from
        self._queue: deque[_FakeMsg] = deque()
        self._has_msg = asyncio.Event()
        self.close_code = None
        if stale_error:
            # a late error of an utterance from before this stream
            self._queue.append(
                _FakeMsg.text(
                    {"type": "error", "request_id": "old", "status": 503, "error": "busy"}
                )
            )
            self._has_msg.set()
        if stale:
            # leftovers of an utterance from before this stream, as a pooled socket may carry
            self._queue.append(_FakeMsg.text({"type": "start", "request_id": "stale"}))
            self._queue.append(_FakeMsg.binary(b"\x00" * 3840))
            self._queue.append(_FakeMsg.text({"type": "end", "request_id": "stale"}))
            self._has_msg.set()
        self._error = error

    async def send_str(self, data: str) -> None:
        body = json.loads(data)
        self.sent.append(body)
        rid = body["request_id"]
        if self._silent_from is not None and len(self.sent) > self._silent_from:
            return
        if self._error is not None:
            self._queue.append(_FakeMsg.text({"type": "error", "request_id": rid, **self._error}))
        else:
            n = len(self.sent)
            self._queue.append(_FakeMsg.text({"type": "start", "request_id": rid, "chunks": 1}))
            self._queue.append(_FakeMsg.binary(bytes([n]) * 3840))
            self._queue.append(
                _FakeMsg.text({"type": "first_audio_ms", "request_id": rid, "ms": 20})
            )
            self._queue.append(_FakeMsg.binary(bytes([n]) * 3840))
            self._queue.append(_FakeMsg.text({"type": "end", "request_id": rid, "audio_s": 0.16}))
        self._has_msg.set()

    async def receive(self, timeout: float | None = None) -> _FakeMsg:
        while not self._queue:
            self._has_msg.clear()
            await asyncio.wait_for(self._has_msg.wait(), timeout)
        return self._queue.popleft()


class _FakePool:
    def __init__(self, ws: _FakeWS) -> None:
        self._ws = ws
        self.last_acquire_time = 0.0
        self.last_connection_reused = False
        self.removed = False

    def connection(self, timeout: float):  # noqa: ANN201
        pool = self

        class _Ctx:
            async def __aenter__(self_):  # noqa: N805
                return pool._ws

            async def __aexit__(self_, exc_type, *exc):  # noqa: N805
                if exc_type is not None:
                    pool.removed = True
                return False

        return _Ctx()


@pytest.mark.asyncio
async def test_stream_sends_one_utterance_per_sentence_and_plays_them_in_order():
    from livekit.plugins.vartaloop import TTS

    tts = TTS(api_key=SECRET_API_KEY, voice="sahil", digits="hi")
    ws = _FakeWS(stale=True)
    tts._pool = _FakePool(ws)  # type: ignore[assignment]

    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5.0))
    stream.push_text("नमस्ते, आपका loan approve हो गया है। ")
    stream.push_text("अगला step क्या है? ")
    stream.end_input()

    frames: list[bytes] = []
    async for ev in stream:
        frames.append(bytes(ev.frame.data))

    # two sentences -> two utterance messages, each with its own request_id and the options
    assert len(ws.sent) == 2
    assert ws.sent[0]["text"].startswith("नमस्ते")
    assert ws.sent[1]["text"].startswith("अगला")
    assert ws.sent[0]["request_id"] != ws.sent[1]["request_id"]
    assert all(
        m["voice"] == "sahil" and m["digits"] == "hi" and m["format"] == "pcm16" for m in ws.sent
    )

    # the stale utterance's audio (zeros) is dropped; ours arrives in order
    audio = b"".join(frames)
    assert audio, "no audio frames were synthesized"
    assert b"\x00" * 3840 not in audio
    assert audio.index(b"\x01") < audio.index(b"\x02")
    assert len(audio) == 4 * 3840


@pytest.mark.asyncio
async def test_stream_server_error_is_an_api_status_error_and_drops_the_socket():
    from livekit.plugins.vartaloop import TTS

    tts = TTS(api_key=SECRET_API_KEY)
    pool = _FakePool(_FakeWS(error={"status": 503, "retry_after": 1, "error": "busy"}))
    tts._pool = pool  # type: ignore[assignment]

    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5.0))
    stream.push_text("Hello there, how are you doing today?")
    stream.end_input()

    with pytest.raises(APIStatusError) as exc_info:
        async for _ in stream:
            pass
    assert exc_info.value.status_code == 503
    assert exc_info.value.retryable is True
    assert pool.removed is True


@pytest.mark.asyncio
async def test_stream_bad_request_is_not_retryable():
    from livekit.plugins.vartaloop import TTS

    tts = TTS(api_key=SECRET_API_KEY)
    tts._pool = _FakePool(_FakeWS(error={"status": 400, "error": "unknown voice x"}))  # type: ignore[assignment]

    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5.0))
    stream.push_text("Hello there, how are you doing today?")
    stream.end_input()

    with pytest.raises(APIStatusError) as exc_info:
        async for _ in stream:
            pass
    assert exc_info.value.status_code == 400
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_stream_ignores_a_stale_error_of_another_utterance():
    from livekit.plugins.vartaloop import TTS

    tts = TTS(api_key=SECRET_API_KEY)
    tts._pool = _FakePool(_FakeWS(stale_error=True))  # type: ignore[assignment]

    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5.0))
    stream.push_text("Hello there, how are you doing today?")
    stream.end_input()

    frames = [bytes(ev.frame.data) async for ev in stream]
    assert b"".join(frames) == b"\x01" * (2 * 3840)


@pytest.mark.asyncio
async def test_update_options_applies_to_the_next_sentence_of_a_running_stream():
    from livekit.plugins.vartaloop import TTS

    tts = TTS(api_key=SECRET_API_KEY, voice="saanvi")
    ws = _FakeWS()
    tts._pool = _FakePool(ws)  # type: ignore[assignment]

    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5.0))
    stream.push_text("First sentence of the reply, long enough. ")
    # the tokenizer releases a sentence once the next one starts: this push sends the first
    stream.push_text("Second sentence of the reply, long enough. ")
    async with asyncio.timeout(5):
        while not ws.sent:
            await asyncio.sleep(0.01)
    tts.update_options(voice="sahil", expression="excited")
    stream.push_text("Third sentence of the reply, long enough.")
    stream.end_input()
    async for _ in stream:
        pass

    assert [m["voice"] for m in ws.sent] == ["saanvi", "sahil", "sahil"]
    assert "style" not in ws.sent[0] and ws.sent[1]["style"] == "excited"


@pytest.mark.asyncio
async def test_stream_finishes_promptly_when_input_closes_after_the_last_end():
    """The first sentence's audio has ended while the LLM is still producing text; closing
    the input then must finish the stream at once, not after a receive timeout."""
    import time

    from livekit.plugins.vartaloop import TTS

    tts = TTS(api_key=SECRET_API_KEY)
    ws = _FakeWS()
    tts._pool = _FakePool(ws)  # type: ignore[assignment]

    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5.0))
    stream.push_text("First sentence of the reply, long enough. ")
    stream.push_text("Second sentence of the reply, long enough. ")  # releases the first
    async with asyncio.timeout(5):
        while not ws.sent:
            await asyncio.sleep(0.01)
    await asyncio.sleep(0.2)  # the server has long sent its end event
    t0 = time.perf_counter()
    stream.end_input()  # the second sentence goes out now and ends
    frames = [bytes(ev.frame.data) async for ev in stream]
    assert time.perf_counter() - t0 < 2.0
    assert len(ws.sent) == 2 and len(b"".join(frames)) == 4 * 3840


def test_a_rejected_update_leaves_the_options_untouched():
    from livekit.plugins.vartaloop import TTS

    tts = TTS(api_key=SECRET_API_KEY, speed=1.2)
    with pytest.raises(ValueError):
        tts.update_options(speed=3.0, voice="sahil")
    assert tts._opts.speed == 1.2
    assert tts._opts.voice == "saanvi"


@pytest.mark.asyncio
async def test_a_pause_in_the_input_longer_than_the_timeout_does_not_fail_the_stream():
    """The first sentence has been spoken and the LLM pauses for longer than the connection
    timeout before the next one: nothing is pending, so nothing is owed by the server."""
    from livekit.plugins.vartaloop import TTS

    tts = TTS(api_key=SECRET_API_KEY)
    ws = _FakeWS()
    tts._pool = _FakePool(ws)  # type: ignore[assignment]

    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.3))
    stream.push_text("First sentence of the reply, long enough. ")
    stream.push_text("Second sentence of the reply, long enough. ")  # releases the first
    async with asyncio.timeout(5):
        while not ws.sent:
            await asyncio.sleep(0.01)
    await asyncio.sleep(1.0)  # the LLM thinks; longer than the 0.3 s timeout
    stream.push_text("Third sentence of the reply, long enough.")
    stream.end_input()

    frames = [bytes(ev.frame.data) async for ev in stream]
    assert len(ws.sent) == 3 and len(b"".join(frames)) == 6 * 3840


@pytest.mark.asyncio
async def test_a_sentence_sent_after_an_idle_gap_still_times_out():
    """The first sentence completes, the LLM pauses, then sends a second sentence the server
    never answers: the timeout must apply to it even though the wait started idle."""
    import time

    from livekit.agents import APITimeoutError
    from livekit.plugins.vartaloop import TTS

    tts = TTS(api_key=SECRET_API_KEY)
    ws = _FakeWS(silent_from=1)
    tts._pool = _FakePool(ws)  # type: ignore[assignment]

    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.5))
    stream.push_text("First sentence of the reply, long enough. ")
    stream.push_text("Second sentence of the reply, long enough. ")  # releases the first
    async with asyncio.timeout(5):
        while not ws.sent:
            await asyncio.sleep(0.01)
    await asyncio.sleep(0.8)  # idle: longer than the timeout, nothing pending
    stream.push_text("Third sentence of the reply, long enough.")  # releases the second: unanswered
    t0 = time.perf_counter()
    with pytest.raises(APITimeoutError):
        async for _ in stream:
            pass
    assert time.perf_counter() - t0 < 3.0
    assert len(ws.sent) == 2
