"""Unit tests for Soniox TTS sentence buffering and stream rotation.

The plugin buffers LLM chunks into complete sentences, feeds them into one
shared Soniox stream, and rotates to a fresh ``stream_id`` when input goes
idle for ``stream_idle_timeout`` or after a transient failure (batch replay).
These tests drive the real ``SynthesizeStream`` against a fake ``_Connection``, and
the real ``_Connection`` against a fake websocket for what it makes of the frames.

Idle-timeout tests use ``virtual_time`` so timers advance deterministically.
"""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from livekit.agents import APIStatusError
from livekit.agents.tts import AudioEmitter
from livekit.plugins import soniox
from livekit.plugins.soniox.tts import _Connection

pytestmark = [
    pytest.mark.plugin("soniox"),
    pytest.mark.virtual_time,
    pytest.mark.no_concurrent,
]

# 10 ms of mono s16le silence at 24 kHz
_SILENCE_PCM = b"\x00\x00" * 240

SENTENCES = [
    "Hello there, this is the first sentence. ",
    "And here comes a second sentence! ",
    "Finally a third one?",
]


async def test_websocket_authenticates_once_for_multiple_streams() -> None:
    sent: asyncio.Queue[str] = asyncio.Queue()
    received: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()
    ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
    ws.closed = False
    ws.send_str = AsyncMock(side_effect=sent.put)
    ws.receive = AsyncMock(side_effect=received.get)
    session = MagicMock(spec=aiohttp.ClientSession)
    session.ws_connect = AsyncMock(return_value=ws)
    tts = soniox.TTS(api_key="test-key", http_session=session)

    try:
        for stream_id in ("first", "second"):
            connection, _, _ = await tts._current_connection(timeout=1.0)
            waiter = asyncio.get_running_loop().create_future()
            connection.register_stream(stream_id, MagicMock(), waiter, opts=tts._opts)
            try:
                connection.send_text(stream_id, "Hello!", text_end=True)
                config = json.loads(await asyncio.wait_for(sent.get(), timeout=1.0))
                text = json.loads(await asyncio.wait_for(sent.get(), timeout=1.0))

                assert "api_key" not in config
                assert config["model"] == tts.model
                assert config["stream_id"] == stream_id
                assert text == {"stream_id": stream_id, "text": "Hello!", "text_end": True}
            finally:
                connection.unregister_stream(stream_id)
                waiter.cancel()

        session.ws_connect.assert_awaited_once_with(
            tts._opts.websocket_url, headers={"Authorization": "Bearer test-key"}
        )
    finally:
        await tts.aclose()


@pytest.mark.parametrize("stream_id", [None, ""])
@pytest.mark.parametrize("status_code", [401, 429])
@pytest.mark.parametrize("register_before_error", [False, True])
async def test_connection_error_reaches_all_streams(
    stream_id: str | None, status_code: int, register_before_error: bool
) -> None:
    tts = soniox.TTS(api_key="test-key")
    connection = _Connection(tts._opts, MagicMock())
    payload: dict[str, Any] = {
        "error_code": status_code,
        "error_message": "Connection rejected",
        "request_id": "test-request",
    }
    if stream_id is not None:
        payload["stream_id"] = stream_id

    ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
    ws.closed = False
    ws.close_code = 1008
    ws.receive = AsyncMock(
        side_effect=[
            aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(payload), ""),
            aiohttp.WSMessage(aiohttp.WSMsgType.CLOSE, 1008, ""),
        ]
    )
    connection._ws = ws
    waiters: list[asyncio.Future[None]] = []

    def register_stream(name: str) -> None:
        waiter = asyncio.get_running_loop().create_future()
        waiters.append(waiter)
        connection.register_stream(name, MagicMock(), waiter, opts=tts._opts)

    try:
        if register_before_error:
            register_stream("first")
            register_stream("second")

        await connection._recv_loop()
        register_stream("closing")
        assert connection._close_task is not None
        await connection._close_task
        register_stream("closed")

        for waiter in waiters:
            with pytest.raises(APIStatusError) as exc_info:
                await asyncio.wait_for(waiter, timeout=1.0)
            assert exc_info.value.status_code == status_code
            assert exc_info.value.message == "Connection rejected"
            assert exc_info.value.retryable is (status_code == 429)
            assert exc_info.value.request_id == "test-request"

        assert not connection.is_current
        assert connection.closed
        assert connection.num_active_streams == 0
        ws.receive.assert_awaited_once()
    finally:
        await connection.aclose()
        if connection._close_task is not None:
            await connection._close_task
        for waiter in waiters:
            if waiter.done() and not waiter.cancelled():
                waiter.exception()
            else:
                waiter.cancel()
        await tts.aclose()


@dataclass
class _StreamSlot:
    emitter: AudioEmitter
    waiter: asyncio.Future[None]


class _FakeConnection:
    """Stand-in for ``_Connection``: text produces audio, ``text_end`` terminates."""

    def __init__(self, *, fail_on_send: int | None = None) -> None:
        self.is_current = True
        self.closed = False
        self.registered_ids: list[str] = []
        self.time_offsets: list[float] = []
        self.send_calls: list[tuple[str, str, bool]] = []
        self.max_open_streams = 0
        # Where a released stream's words reached; the real connection reports
        # its own, this fake lets a test set one.
        self.timed_end = 0.0
        self.unregister_calls: list[tuple[str, bool]] = []
        self._streams: dict[str, _StreamSlot] = {}
        self._sends = 0
        self._fail_on_send = fail_on_send

    def register_stream(
        self,
        stream_id: str,
        emitter: AudioEmitter,
        waiter: asyncio.Future[None],
        *,
        opts: Any,
        time_offset: float = 0.0,
    ) -> None:
        self.registered_ids.append(stream_id)
        self.time_offsets.append(time_offset)
        self._streams[stream_id] = _StreamSlot(emitter, waiter)
        self.max_open_streams = max(self.max_open_streams, len(self._streams))

    def unregister_stream(self, stream_id: str, *, flush: bool = False) -> float:
        self._streams.pop(stream_id, None)
        self.unregister_calls.append((stream_id, flush))
        return self.timed_end

    def send_text(self, stream_id: str, text: str, *, text_end: bool = False) -> None:
        self.send_calls.append((stream_id, text, text_end))
        slot = self._streams.get(stream_id)
        if slot is None:
            return
        if text_end and not text:
            # server: final audio + audio_end + terminated
            if not slot.waiter.done():
                slot.waiter.set_result(None)
            return
        self._sends += 1
        if self._sends == self._fail_on_send:
            self._fail_on_send = None
            if not slot.waiter.done():
                slot.waiter.set_exception(
                    APIStatusError("transient", status_code=429, retryable=True)
                )
            return
        slot.emitter.push(_SILENCE_PCM)

    def cancel_stream(self, stream_id: str) -> None:
        slot = self._streams.get(stream_id)
        if slot is not None and not slot.waiter.done():
            slot.waiter.set_result(None)


async def _synthesize(
    fake: _FakeConnection,
    *,
    chunk_delays: list[float] | None = None,
    **tts_kwargs: Any,
) -> int:
    tts = soniox.TTS(api_key="fake-key", **tts_kwargs)

    async def _fake_current_connection(*, timeout: float) -> tuple[Any, float, bool]:
        return fake, 0.0, True

    tts._current_connection = _fake_current_connection  # type: ignore[method-assign]

    stream = tts.stream()
    delays = chunk_delays or [0.01] * len(SENTENCES)

    async def _push() -> None:
        for chunk, delay in zip(SENTENCES, delays, strict=True):
            stream.push_text(chunk)
            await asyncio.sleep(delay)
        stream.end_input()

    push_t = asyncio.create_task(_push())
    frames = 0
    async for _ in stream:
        frames += 1
    await push_t
    await stream.aclose()
    return frames


async def test_steady_flow_uses_single_stream() -> None:
    fake = _FakeConnection()
    frames = await _synthesize(fake)

    assert frames > 0
    assert len(fake.registered_ids) == 1
    text_sends = [c for c in fake.send_calls if not c[2]]
    end_sends = [c for c in fake.send_calls if c[2]]
    assert len(text_sends) == len(SENTENCES)
    assert len(end_sends) == 1


async def test_idle_stall_rotates_stream() -> None:
    fake = _FakeConnection()
    frames = await _synthesize(
        fake,
        chunk_delays=[3.0, 0.01, 0.01],
        stream_idle_timeout=1.0,
    )

    assert frames > 0
    # sentence 1 on the first stream, finalized during the stall; 2 and 3 on the next
    assert len(fake.registered_ids) == 2
    assert sum(1 for c in fake.send_calls if c[2]) == 2
    assert fake.max_open_streams == 1


async def test_transient_failure_replays_batch() -> None:
    fake = _FakeConnection(fail_on_send=1)
    frames = await _synthesize(fake)

    assert frames > 0
    assert len(fake.registered_ids) == 2
    failed_id, replacement_id = fake.registered_ids
    replayed = [text for sid, text, end in fake.send_calls if sid == replacement_id and not end]
    assert replayed[0].strip() == SENTENCES[0].strip()
    assert fake.max_open_streams == 1


async def test_invalid_stream_idle_timeout_rejected() -> None:
    with pytest.raises(ValueError):
        soniox.TTS(api_key="fake-key", stream_idle_timeout=0)
    with pytest.raises(ValueError):
        soniox.TTS(api_key="fake-key", stream_idle_timeout=-1.0)


@pytest.mark.parametrize("client_reference_id", [None, "session-123"])
async def test_client_reference_id_in_config(client_reference_id: str | None) -> None:
    sent: asyncio.Queue[str] = asyncio.Queue()
    received: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()
    ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
    ws.closed = False
    ws.send_str = AsyncMock(side_effect=sent.put)
    ws.receive = AsyncMock(side_effect=received.get)
    session = MagicMock(spec=aiohttp.ClientSession)
    session.ws_connect = AsyncMock(return_value=ws)
    tts = soniox.TTS(
        api_key="test-key", http_session=session, client_reference_id=client_reference_id
    )

    try:
        connection, _, _ = await tts._current_connection(timeout=1.0)
        waiter = asyncio.get_running_loop().create_future()
        connection.register_stream("s1", MagicMock(), waiter, opts=tts._opts)
        try:
            connection.send_text("s1", "Hello!", text_end=True)
            config = json.loads(await asyncio.wait_for(sent.get(), timeout=1.0))
            if client_reference_id is None:
                assert "client_reference_id" not in config
            else:
                assert config["client_reference_id"] == client_reference_id
        finally:
            connection.unregister_stream("s1")
            waiter.cancel()
    finally:
        await tts.aclose()


async def test_reduce_silence_propagates_to_options() -> None:
    tts = soniox.TTS(api_key="fake-key", model="tts-rt-v2")
    assert tts._opts.reduce_silence is False

    tts.update_options(reduce_silence=True)
    assert tts._opts.reduce_silence is True

    tts = soniox.TTS(api_key="fake-key", model="tts-rt-v2", reduce_silence=True)
    assert tts._opts.reduce_silence is True


async def test_reduce_silence_rejected_on_unsupported_model() -> None:
    # default model is tts-rt-v1-preview, which lacks supports_silence_reduction
    with pytest.raises(ValueError, match="reduce_silence"):
        soniox.TTS(api_key="fake-key", reduce_silence=True)
    with pytest.raises(ValueError, match="reduce_silence"):
        soniox.TTS(api_key="fake-key", model="tts-rt-v1", reduce_silence=True)

    tts = soniox.TTS(api_key="fake-key")
    with pytest.raises(ValueError, match="reduce_silence"):
        tts.update_options(reduce_silence=True)
    assert tts._opts.reduce_silence is False

    tts = soniox.TTS(api_key="fake-key", model="tts-rt-v2", reduce_silence=True)
    with pytest.raises(ValueError, match="reduce_silence"):
        tts.update_options(model="tts-rt-v1")
    assert tts._opts.model == "tts-rt-v2"


async def test_word_timestamps_propagates_to_options_and_capabilities() -> None:
    tts = soniox.TTS(api_key="fake-key")
    assert tts._opts.word_timestamps is False
    assert tts.capabilities.aligned_transcript is False

    tts = soniox.TTS(api_key="fake-key", word_timestamps=True)
    assert tts._opts.word_timestamps is True
    assert tts.capabilities.aligned_transcript is True


def _timed(
    text: str, *, start: float = 0.0, step: float = 0.1, dur: float = 0.08
) -> dict[str, Any]:
    """One Soniox timestamps frame: a character every `step` seconds, `dur` long.

    `dur` is shorter than `step`, as a real character is: the gap between one
    character's end and the next one's start is what tells a word's own end
    apart from where the next word begins.
    """
    return {
        "characters": list(text),
        "character_start_times_seconds": [start + i * step for i in range(len(text))],
        "character_end_times_seconds": [start + i * step + dur for i in range(len(text))],
    }


def _timed_stream(**kwargs: Any) -> Any:
    from livekit.plugins.soniox.tts import _StreamData

    return _StreamData(
        emitter=MagicMock(),
        waiter=asyncio.get_running_loop().create_future(),
        opts=soniox.TTS(api_key="fake-key", word_timestamps=True)._opts,
        **kwargs,
    )


async def test_timed_words_are_assembled_from_the_timed_characters() -> None:
    from livekit.plugins.soniox.tts import _flush_timed_word, _push_timed_words, _StreamData

    emitter = MagicMock()
    stream = _StreamData(
        emitter=emitter,
        waiter=asyncio.get_running_loop().create_future(),
        opts=soniox.TTS(api_key="fake-key", word_timestamps=True)._opts,
    )

    # "there" straddles the two frames, so it is emitted only once it is whole.
    _push_timed_words(stream, _timed("Hi th"))
    assert [str(call.args[0]) for call in emitter.push_timed_transcript.call_args_list] == ["Hi "]

    _push_timed_words(stream, _timed("ere", start=0.5))
    _flush_timed_word(stream)

    words = [call.args[0] for call in emitter.push_timed_transcript.call_args_list]
    # The words join back into exactly what was spoken, spaces included.
    assert "".join(str(word) for word in words) == "Hi there"
    assert [str(word) for word in words] == ["Hi ", "there"]
    # The word's own end: 'i' ends 0.08s after it starts, where the space that
    # follows it starts at 0.2s and belongs to no word's timing.
    assert (words[0].start_time, words[0].end_time) == pytest.approx((0.0, 0.18))
    assert (words[1].start_time, words[1].end_time) == pytest.approx((0.3, 0.78))


async def test_timed_words_sit_on_the_emitters_timeline_not_the_streams() -> None:
    from livekit.plugins.soniox.tts import _flush_timed_word, _push_timed_words

    # A reply that rotated to a second Soniox stream: its character times start
    # at 0 again, while the emitter's segment already holds 4 seconds of audio.
    stream = _timed_stream(time_offset=4.0)

    _push_timed_words(stream, _timed("Bye"))
    _flush_timed_word(stream)

    word = stream.emitter.push_timed_transcript.call_args_list[0].args[0]
    assert str(word) == "Bye"
    assert (word.start_time, word.end_time) == pytest.approx((4.0, 4.28))


async def test_timed_words_drop_a_frame_whose_arrays_disagree() -> None:
    from livekit.plugins.soniox.tts import _push_timed_words

    stream = _timed_stream()

    _push_timed_words(
        stream,
        {
            "characters": ["a", "b"],
            "character_start_times_seconds": [0.0],
            "character_end_times_seconds": [0.1],
        },
    )

    # Taking the short array would have left a truncated word behind.
    assert stream.text == ""
    stream.emitter.push_timed_transcript.assert_not_called()


async def test_rotating_streams_carry_the_emitters_duration_as_the_time_offset() -> None:
    fake = _FakeConnection()

    await _synthesize(
        fake,
        chunk_delays=[3.0, 0.01, 0.01],
        stream_idle_timeout=1.0,
        word_timestamps=True,
    )

    assert len(fake.registered_ids) == 2, "expected the idle stall to rotate the stream"
    # The first stream starts the segment; the second begins where the audio
    # already written leaves off, which is where its own timestamps start at 0.
    assert fake.time_offsets[0] == 0.0
    assert fake.time_offsets[1] > 0.0


async def test_rotating_stream_starts_after_the_previous_streams_last_word() -> None:
    fake = _FakeConnection()
    # The words of the first stream ran past the audio the emitter has handed
    # on, as they do while frames are still queued.
    fake.timed_end = 9.0

    await _synthesize(
        fake,
        chunk_delays=[3.0, 0.01, 0.01],
        stream_idle_timeout=1.0,
        word_timestamps=True,
    )

    assert len(fake.registered_ids) == 2, "expected the idle stall to rotate the stream"
    # Taking the emitter's duration alone would place this stream's first word
    # before the previous stream's last word ended, which reads as time going
    # backwards to anything pacing against the words.
    assert fake.time_offsets[1] == 9.0


async def test_a_transient_failure_after_a_rotation_still_replays() -> None:
    fake = _FakeConnection(fail_on_send=2)
    # The rotated stream's words run past the audio the emitter has handed on,
    # which is what the timeline fix is for; the replay must not depend on it.
    fake.timed_end = 9.0

    await _synthesize(
        fake,
        chunk_delays=[3.0, 0.01, 0.01],
        stream_idle_timeout=1.0,
        word_timestamps=True,
    )

    # Rotation, failure, replay: gating the replay on the words' timeline
    # instead of the emitter's would have raised here.
    assert len(fake.registered_ids) == 3
    # And the replay is of the rotated stream's own sentence, so the three
    # streams are open, fail, replay rather than any other shape.
    replayed = [sid for sid, text, _ in fake.send_calls if text.strip() == SENTENCES[1].strip()]
    assert replayed == fake.registered_ids[1:]


async def test_an_interrupted_turn_releases_its_stream_with_a_flush() -> None:
    fake = _FakeConnection()
    tts = soniox.TTS(api_key="fake-key", word_timestamps=True)

    async def _fake_current_connection(*, timeout: float) -> tuple[Any, float, bool]:
        return fake, 0.0, True

    tts._current_connection = _fake_current_connection  # type: ignore[method-assign]

    stream = tts.stream()
    stream.push_text(SENTENCES[0])
    # Cut the turn short, as an interruption does. `aclose()` cancels the run
    # task without awaiting anything in between, so the cancellation reaches
    # `_run` before the stream's termination does and the finally still has the
    # stream in hand, holding a word back. The sleep is what lets the stream
    # open at all.
    await asyncio.sleep(0.05)
    await stream.aclose()

    # The word held back has had its audio pushed, so releasing the stream has
    # to flush it; a stream released any other way must not be flushed.
    assert fake.unregister_calls[-1][1] is True


class _RecordingEmitter:
    """Records what the connection pushes, in order, as the real emitter would see it."""

    def __init__(self) -> None:
        self.pushes: list[tuple[str, Any]] = []

    def push(self, audio: bytes) -> None:
        self.pushes.append(("audio", audio))

    def push_timed_transcript(self, timed_words: Any) -> None:
        words = timed_words if isinstance(timed_words, list) else [timed_words]
        for word in words:
            self.pushes.append(("word", word))

    @property
    def words(self) -> list[Any]:
        return [push for kind, push in self.pushes if kind == "word"]


def _audio_frame(
    stream_id: str, text: str, *, start: float = 0.0, **extra: Any
) -> aiohttp.WSMessage:
    payload: dict[str, Any] = {
        "stream_id": stream_id,
        "audio": base64.b64encode(b"\x00\x00").decode(),
    }
    if text:
        payload["timestamps"] = _timed(text, start=start)
    payload.update(extra)
    return aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(payload), "")


async def _drive_recv_loop(frames: list[aiohttp.WSMessage], **tts_kwargs: Any) -> _RecordingEmitter:
    """Run the real receive loop over `frames` for one registered stream."""
    tts = soniox.TTS(api_key="test-key", **tts_kwargs)
    connection = _Connection(tts._opts, MagicMock())
    ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
    ws.closed = False
    ws.close_code = 1000
    ws.receive = AsyncMock(
        side_effect=[*frames, aiohttp.WSMessage(aiohttp.WSMsgType.CLOSE, 1000, "")]
    )
    connection._ws = ws

    emitter = _RecordingEmitter()
    waiter = asyncio.get_running_loop().create_future()
    connection.register_stream("s1", emitter, waiter, opts=tts._opts)  # type: ignore[arg-type]
    try:
        await connection._recv_loop()
    finally:
        waiter.cancel()
        await connection.aclose()
        if connection._close_task is not None:
            await connection._close_task
        await tts.aclose()
    return emitter


async def test_word_timestamps_are_requested_in_the_start_config() -> None:
    for word_timestamps, expected in ((False, None), (True, True)):
        sent: asyncio.Queue[str] = asyncio.Queue()
        ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
        ws.closed = False
        ws.send_str = AsyncMock(side_effect=sent.put)
        ws.receive = AsyncMock(side_effect=asyncio.Queue().get)
        session = MagicMock(spec=aiohttp.ClientSession)
        session.ws_connect = AsyncMock(return_value=ws)
        tts = soniox.TTS(api_key="test-key", http_session=session, word_timestamps=word_timestamps)

        try:
            connection, _, _ = await tts._current_connection(timeout=1.0)
            waiter = asyncio.get_running_loop().create_future()
            connection.register_stream("s1", MagicMock(), waiter, opts=tts._opts)
            try:
                connection.send_text("s1", "Hello!", text_end=True)
                config = json.loads(await asyncio.wait_for(sent.get(), timeout=1.0))

                assert config.get("return_timestamps") is expected
            finally:
                connection.unregister_stream("s1")
                waiter.cancel()
        finally:
            await tts.aclose()


async def test_timed_words_are_pushed_with_the_audio_they_describe() -> None:
    emitter = await _drive_recv_loop(
        [
            _audio_frame("s1", "Hi "),
            _audio_frame("s1", "there", start=0.3),
            _audio_frame("s1", "", audio_end=True),
        ],
        word_timestamps=True,
    )

    # The order the recv loop pushes into the emitter, not what a frame ends up
    # carrying: a word is emitted once the next one starts, so it is pushed after
    # the audio frame that carried its own characters, and the last word is
    # flushed when the audio ends rather than being left behind.
    assert [kind for kind, _ in emitter.pushes] == ["audio", "word", "audio", "word", "audio"]
    assert [str(word) for word in emitter.words] == ["Hi ", "there"]


async def test_a_released_stream_still_reports_where_its_words_reached() -> None:
    """The recv loop releases the stream on `terminated`, the turn asks after."""
    tts = soniox.TTS(api_key="test-key", word_timestamps=True)
    connection = _Connection(tts._opts, MagicMock())
    ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
    ws.closed = False
    ws.close_code = 1000
    ws.receive = AsyncMock(
        side_effect=[
            _audio_frame("s1", "Hi "),
            _audio_frame("s1", "there", start=0.3),
            _audio_frame("s1", "", audio_end=True),
            _audio_frame("s1", "", terminated=True),
            aiohttp.WSMessage(aiohttp.WSMsgType.CLOSE, 1000, ""),
        ]
    )
    connection._ws = ws

    waiter = asyncio.get_running_loop().create_future()
    connection.register_stream("s1", _RecordingEmitter(), waiter, opts=tts._opts)  # type: ignore[arg-type]
    try:
        await connection._recv_loop()
        # The entry is gone by now, so answering from it would say 0.0 and the
        # next stream of the reply would place its first word back at the
        # emitter's duration, before this stream's last word ended.
        assert connection.unregister_stream("s1") == pytest.approx(0.78)
    finally:
        waiter.cancel()
        await connection.aclose()
        if connection._close_task is not None:
            await connection._close_task
        await tts.aclose()


@pytest.mark.parametrize(
    "failure",
    [
        _audio_frame("s2", "", terminated=True),
        aiohttp.WSMessage(
            aiohttp.WSMsgType.TEXT,
            json.dumps({"stream_id": "s2", "error_code": 500, "error_message": "boom"}),
            None,
        ),
    ],
    ids=["terminated-without-audio-end", "error-code"],
)
async def test_a_failed_stream_does_not_report_where_the_previous_one_reached(failure) -> None:
    """A failed stream is replayed, so it must not inherit the last stream's end."""
    tts = soniox.TTS(api_key="test-key", word_timestamps=True)
    connection = _Connection(tts._opts, MagicMock())
    ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
    ws.closed = False
    ws.close_code = 1000
    ws.receive = AsyncMock(
        side_effect=[
            _audio_frame("s1", "Hi "),
            _audio_frame("s1", "there", start=0.3),
            _audio_frame("s1", "", audio_end=True),
            _audio_frame("s1", "", terminated=True),
            failure,
            aiohttp.WSMessage(aiohttp.WSMsgType.CLOSE, 1000, ""),
        ]
    )
    connection._ws = ws

    loop = asyncio.get_running_loop()
    waiters = [loop.create_future(), loop.create_future()]
    connection.register_stream("s1", _RecordingEmitter(), waiters[0], opts=tts._opts)  # type: ignore[arg-type]
    connection.register_stream("s2", _RecordingEmitter(), waiters[1], opts=tts._opts)  # type: ignore[arg-type]
    try:
        await connection._recv_loop()
        assert connection.unregister_stream("s1") == pytest.approx(0.78)
        assert connection.unregister_stream("s2") == 0.0
    finally:
        for waiter in waiters:
            if waiter.done() and not waiter.cancelled():
                waiter.exception()
            waiter.cancel()
        await connection.aclose()
        if connection._close_task is not None:
            await connection._close_task
        await tts.aclose()


async def test_a_stream_terminated_without_audio_end_does_not_report_its_last_word() -> None:
    emitter = await _drive_recv_loop(
        [
            _audio_frame("s1", "Hi there"),
            # The server aborted: no audio_end, which the loop reports as an
            # error so the framework replays the sentence.
            _audio_frame("s1", "", terminated=True),
        ],
        word_timestamps=True,
    )

    # The word held back stays unsaid, because the reply it belongs to is about
    # to be spoken again; flushing it would report speech twice.
    assert [str(word) for word in emitter.words] == ["Hi "]


@pytest.mark.parametrize(
    "spoken",
    [
        "Hello, world. Bye!",  # English with punctuation
        "好的，OK。",  # CJK whose punctuation a word-only split would drop
        "First point.\nSecond point.",  # a newline is a boundary, not glue
        "Two  spaces  survive",  # repeated spaces are not collapsed
        "สวัสดีครับ",  # Thai, split per character
    ],
)
async def test_timed_words_join_back_into_the_spoken_text(spoken: str) -> None:
    emitter = await _drive_recv_loop(
        [_audio_frame("s1", spoken), _audio_frame("s1", "", audio_end=True)],
        word_timestamps=True,
    )

    # What the words say is what the transcript and Soniox agent text are built
    # from, so anything dropped or collapsed here is wrong downstream.
    assert "".join(str(word) for word in emitter.words) == spoken


async def test_timed_words_keep_the_separator_a_rotated_stream_starts_with() -> None:
    # The sentence tokenizer hands the next stream its text with the separator
    # already on the front, so dropping it would glue the sentences together.
    emitter = await _drive_recv_loop(
        [
            _audio_frame("s1", " Then, after a pause"),
            _audio_frame("s1", "", audio_end=True),
        ],
        word_timestamps=True,
    )

    assert "".join(str(word) for word in emitter.words) == " Then, after a pause"
    assert str(emitter.words[0]) == " Then, "


async def test_timed_word_times_do_not_go_backwards() -> None:
    emitter = await _drive_recv_loop(
        [
            _audio_frame("s1", "One two "),
            _audio_frame("s1", "three four", start=0.8),
            _audio_frame("s1", "", audio_end=True),
        ],
        word_timestamps=True,
    )

    times = [(word.start_time, word.end_time) for word in emitter.words]
    assert times == sorted(times)
    assert all(start < end for start, end in times)


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        ("今日は", ["今", "日", "は"]),  # Japanese
        ("你好吗", ["你", "好", "吗"]),  # Chinese
        ("สวัสดี", ["ส", "ว", "ั", "ส", "ด", "ี"]),  # Thai, which is not East Asian
        ("مرحبا بك", ["مرحبا ", "بك"]),  # Arabic: spaced, and right-to-left
    ],
)
async def test_timed_words_of_a_script_without_spaces_stand_alone(
    spoken: str, expected: list[str]
) -> None:
    from livekit.plugins.soniox.tts import _flush_timed_word, _push_timed_words

    stream = _timed_stream()

    _push_timed_words(stream, _timed(spoken))
    _flush_timed_word(stream)

    words = [call.args[0] for call in stream.emitter.push_timed_transcript.call_args_list]
    # One annotation per character for an unspaced script, so playback has
    # something to pace against; a spaced one splits on its spaces, in the
    # logical order the text is stored in whichever way it is displayed.
    assert [str(word) for word in words] == expected
    assert "".join(str(word) for word in words) == spoken
    assert words[0].start_time == pytest.approx(0.0)
