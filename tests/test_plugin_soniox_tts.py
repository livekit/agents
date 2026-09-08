"""Unit tests for Soniox TTS sentence buffering, stream rotation and alignment.

The plugin buffers LLM chunks into complete sentences, feeds them into one
shared Soniox stream, and rotates to a fresh ``stream_id`` when input goes
idle for ``stream_idle_timeout`` or after a transient failure (batch replay).
These tests drive the real ``SynthesizeStream`` against a fake ``_Connection``.

The alignment tests cover ``return_timestamps``: Soniox times characters from
each stream's own first sample, so a reply spread over several rotated streams
only stays on one timeline if every timestamp is offset by the audio already
emitted.

Idle-timeout tests use ``virtual_time`` so timers advance deterministically.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest

from livekit.agents import APIStatusError, utils
from livekit.agents.tts import AudioEmitter
from livekit.agents.types import USERDATA_TIMED_TRANSCRIPT
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TimedString
from livekit.plugins import soniox
from livekit.plugins.soniox import tts as soniox_tts

pytestmark = [
    pytest.mark.plugin("soniox"),
    pytest.mark.virtual_time,
    pytest.mark.no_concurrent,
]

# 10 ms of mono s16le silence at 24 kHz
_SILENCE_PCM = b"\x00\x00" * 240
# One character of speech per silence chunk, so the fake server's audio and its
# character timestamps describe the same amount of time.
_SECONDS_PER_CHAR = 0.01

SENTENCES = [
    "Hello there, this is the first sentence. ",
    "And here comes a second sentence! ",
    "Finally a third one?",
]


def _character_timestamps(text: str, start: float) -> dict[str, list[Any]]:
    """Soniox-shaped timings for *text*, starting *start* seconds into the stream."""
    starts = [start + i * _SECONDS_PER_CHAR for i in range(len(text))]
    return {
        "characters": list(text),
        "character_start_times_seconds": starts,
        "character_end_times_seconds": [t + _SECONDS_PER_CHAR for t in starts],
    }


@dataclass
class _StreamSlot:
    data: soniox_tts._StreamData
    cursor: float = 0.0  # seconds of speech this stream has already timed

    @property
    def emitter(self) -> AudioEmitter:
        return self.data.emitter

    @property
    def waiter(self) -> asyncio.Future[None]:
        return self.data.waiter


class _FakeConnection:
    """Stand-in for ``_Connection``: text produces audio, ``text_end`` terminates.

    Audio and character timestamps are kept consistent at ``_SECONDS_PER_CHAR``
    per character, and, like the real server, timestamps restart at zero on
    every new ``stream_id`` and are reported against preprocessed text: the
    leading whitespace of a stream's first text is normalised away, while
    whitespace inside the stream survives.
    """

    def __init__(
        self,
        *,
        fail_on_send: int | None = None,
        timestamps: bool = False,
        emit_before_failure: bool = False,
    ) -> None:
        self.is_current = True
        self.closed = False
        self.registered_ids: list[str] = []
        self.send_calls: list[tuple[str, str, bool]] = []
        self.time_offsets: list[float] = []
        self.max_open_streams = 0
        self._streams: dict[str, _StreamSlot] = {}
        self._sends = 0
        self._fail_on_send = fail_on_send
        self._timestamps = timestamps
        self._emit_before_failure = emit_before_failure

    def register_stream(
        self,
        stream_id: str,
        emitter: AudioEmitter,
        waiter: asyncio.Future[None],
        *,
        opts: Any,
        time_offset: float = 0.0,
        timeline: soniox_tts._Timeline | None = None,
    ) -> soniox_tts._StreamData:
        self.registered_ids.append(stream_id)
        self.time_offsets.append(time_offset)
        data = soniox_tts._StreamData(
            emitter=emitter,
            waiter=waiter,
            opts=opts,
            time_offset=time_offset,
            timeline=timeline if timeline is not None else soniox_tts._Timeline(),
        )
        self._streams[stream_id] = _StreamSlot(data)
        self.max_open_streams = max(self.max_open_streams, len(self._streams))
        return data

    def unregister_stream(self, stream_id: str) -> None:
        self._streams.pop(stream_id, None)

    def _push_audio(self, slot: _StreamSlot, audio: bytes) -> None:
        # mirrors _recv_loop: audio handed over is audio the user will hear
        slot.data.produced_output = True
        slot.emitter.push(audio)

    def send_text(self, stream_id: str, text: str, *, text_end: bool = False) -> None:
        self.send_calls.append((stream_id, text, text_end))
        slot = self._streams.get(stream_id)
        if slot is None:
            return
        if text_end and not text:
            # server: final audio + audio_end + terminated
            if self._timestamps:
                soniox_tts._emit_timed_words(slot.data, flush=True)
            if not slot.waiter.done():
                slot.waiter.set_result(None)
            return
        self._sends += 1
        if self._sends == self._fail_on_send:
            self._fail_on_send = None
            if self._emit_before_failure:
                # One 10ms chunk fits entirely in the tail frame the emitter
                # holds back, so pushed_duration() still reads the value it had
                # when the stream opened, even though the audio is spoken for.
                slot.data.sent_text += text
                if self._timestamps:
                    soniox_tts._accumulate_timestamps(
                        slot.data, _character_timestamps(text.lstrip()[:20], 0.0)
                    )
                self._push_audio(slot, _SILENCE_PCM)
            if not slot.waiter.done():
                slot.waiter.set_exception(
                    APIStatusError("transient", status_code=429, retryable=True)
                )
            return
        if not self._timestamps:
            slot.data.sent_text += text
            self._push_audio(slot, _SILENCE_PCM)
            return

        # the server strips the separator that opens a stream, keeps the rest
        spoken = text.lstrip() if not slot.data.sent_text else text
        slot.data.sent_text += text

        soniox_tts._accumulate_timestamps(slot.data, _character_timestamps(spoken, slot.cursor))
        slot.cursor += len(spoken) * _SECONDS_PER_CHAR
        self._push_audio(slot, _SILENCE_PCM * len(spoken))

    def cancel_stream(self, stream_id: str) -> None:
        slot = self._streams.get(stream_id)
        if slot is not None and not slot.waiter.done():
            slot.waiter.set_result(None)


async def _synthesize(
    fake: _FakeConnection,
    *,
    chunk_delays: list[float] | None = None,
    timed_words: list[TimedString] | None = None,
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
    try:
        async for ev in stream:
            frames += 1
            if timed_words is not None:
                timed_words += ev.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, [])
    finally:
        # the stream may raise, so tear down either way
        push_t.cancel()
        with contextlib.suppress(asyncio.CancelledError):
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


class _FakeWebSocket:
    """Minimal ``aiohttp`` websocket: records what was sent, replays what was set."""

    def __init__(self, messages: list[object] | None = None) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self.close_code = 1000
        self._messages = list(messages or [])

    async def send_str(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def receive(self) -> object:
        if self._messages:
            return self._messages.pop(0)
        return SimpleNamespace(type=aiohttp.WSMsgType.CLOSE, data="", extra="")

    async def close(self) -> None:
        self.closed = True


class _RecordingEmitter:
    """Counts pushed audio the way ``AudioEmitter.pushed_duration`` does."""

    def __init__(self) -> None:
        self.audio: list[bytes] = []
        self.timed_words: list[TimedString] = []

    def push(self, data: bytes) -> None:
        self.audio.append(data)

    def pushed_duration(self, idx: int = -1) -> float:
        return sum(len(chunk) for chunk in self.audio) / (2 * 24000)

    def push_timed_transcript(self, delta_text: TimedString | list[TimedString]) -> None:
        if isinstance(delta_text, list):
            self.timed_words += delta_text
        else:
            self.timed_words.append(delta_text)


def _span(word: TimedString) -> tuple[float, float]:
    """The word's (start, end) in seconds, asserting the timings are present."""
    assert is_given(word.start_time) and is_given(word.end_time)
    return word.start_time, word.end_time


def _text_message(payload: dict[str, Any]) -> object:
    return SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(payload))


async def _sent_start_config(**tts_kwargs: Any) -> dict[str, Any]:
    """Run the real send loop over a single start config and return what it sent."""
    tts = soniox.TTS(api_key="fake-key", **tts_kwargs)
    conn = soniox_tts._Connection(tts._opts, session=None)  # type: ignore[arg-type]
    ws = _FakeWebSocket()
    conn._ws = ws  # type: ignore[assignment]
    conn._input_queue.send_nowait(soniox_tts._StartConfig(stream_id="s1", opts=tts._opts))
    conn._input_queue.close()

    await conn._send_loop()
    await conn.aclose()

    assert len(ws.sent) == 1
    return ws.sent[0]


async def test_capabilities_track_return_timestamps() -> None:
    assert soniox.TTS(api_key="fake-key").capabilities.aligned_transcript is True
    assert (
        soniox.TTS(api_key="fake-key", return_timestamps=False).capabilities.aligned_transcript
        is False
    )


async def test_start_config_requests_timestamps() -> None:
    config = await _sent_start_config()
    assert config["return_timestamps"] is True


async def test_start_config_omits_timestamps_when_disabled() -> None:
    config = await _sent_start_config(return_timestamps=False)
    assert "return_timestamps" not in config


async def test_recv_loop_maps_character_timestamps() -> None:
    """Characters arriving over two frames become words on the emitter's timeline."""
    stream_id = "stream-1"
    emitter = _RecordingEmitter()
    tts = soniox.TTS(api_key="fake-key")
    conn = soniox_tts._Connection(tts._opts, session=None)  # type: ignore[arg-type]
    waiter: asyncio.Future[None] = asyncio.get_event_loop().create_future()
    timeline = soniox_tts._Timeline(end=1.5)
    conn._streams[stream_id] = soniox_tts._StreamData(
        emitter=emitter,  # type: ignore[arg-type]
        waiter=waiter,
        opts=tts._opts,
        time_offset=1.5,  # this stream started 1.5s into the segment
        timeline=timeline,
    )
    conn._ws = _FakeWebSocket(  # type: ignore[assignment]
        [
            _text_message(
                {
                    "stream_id": stream_id,
                    "audio": base64.b64encode(_SILENCE_PCM).decode("ascii"),
                    "timestamps": _character_timestamps("Hi the", 0.0),
                }
            ),
            _text_message(
                {
                    "stream_id": stream_id,
                    "audio": base64.b64encode(_SILENCE_PCM).decode("ascii"),
                    "timestamps": _character_timestamps("re", 6 * _SECONDS_PER_CHAR),
                    "audio_end": True,
                }
            ),
            _text_message({"stream_id": stream_id, "terminated": True}),
        ]
    )

    await conn._recv_loop()
    await conn.aclose()

    assert [str(w) for w in emitter.timed_words] == ["Hi ", "there"]
    hi, there = emitter.timed_words
    # every timestamp shifted onto the segment timeline by time_offset
    assert hi.start_time == pytest.approx(1.50)
    assert hi.end_time == pytest.approx(1.52)
    assert there.start_time == pytest.approx(1.53)
    assert there.end_time == pytest.approx(1.58)
    assert len(emitter.audio) == 2
    assert waiter.done() and waiter.result() is None
    # the timeline advanced to the last character, so the next stream after a
    # rotation starts from here rather than from a lagging pushed_duration()
    assert timeline.end == pytest.approx(1.58)


async def test_aligned_transcript_reaches_frames_on_one_timeline() -> None:
    fake = _FakeConnection(timestamps=True)
    words: list[TimedString] = []
    frames = await _synthesize(fake, timed_words=words)

    assert frames > 0
    assert len(fake.registered_ids) == 1
    spoken = " ".join(str(w) for w in words)
    assert "Hello" in spoken and "third" in spoken
    spans = [_span(w) for w in words]
    starts = [start for start, _ in spans]
    assert starts == sorted(starts)
    assert all(end >= start for start, end in spans)


async def test_aligned_transcript_offset_across_stream_rotation() -> None:
    """A reply split over two streams stays on one non-decreasing timeline.

    Soniox restarts character times at zero for every ``stream_id``, so without
    the open-time offset the second stream's words would jump back to ~0.
    """
    fake = _FakeConnection(timestamps=True)
    words: list[TimedString] = []
    await _synthesize(
        fake,
        timed_words=words,
        chunk_delays=[3.0, 0.01, 0.01],
        stream_idle_timeout=1.0,
    )

    assert len(fake.registered_ids) == 2
    # the second stream opened onto audio already emitted, not at zero
    assert fake.time_offsets[0] == 0.0
    assert fake.time_offsets[1] > 0.0

    starts = [start for start, _ in (_span(w) for w in words)]
    assert starts == sorted(starts)

    # "And" opens the second sentence, i.e. the second stream; sentence one is
    # ~40 characters of speech, so its words cannot start back near zero.
    first_sentence_duration = len(SENTENCES[0]) * _SECONDS_PER_CHAR
    second_stream_start, _ = _span(next(w for w in words if str(w).strip() == "And"))
    assert second_stream_start > first_sentence_duration * 0.8


async def test_partial_timestamps_are_ignored() -> None:
    """A frame whose timing arrays disagree is dropped rather than mis-aligned."""
    emitter = _RecordingEmitter()
    tts = soniox.TTS(api_key="fake-key")
    stream = soniox_tts._StreamData(
        emitter=emitter,  # type: ignore[arg-type]
        waiter=asyncio.get_event_loop().create_future(),
        opts=tts._opts,
    )

    soniox_tts._accumulate_timestamps(stream, {"characters": ["a", "b"]})
    soniox_tts._accumulate_timestamps(
        stream,
        {
            "characters": ["a", "b"],
            "character_start_times_seconds": [0.0, 0.1],
            "character_end_times_seconds": [0.1],
        },
    )

    assert emitter.timed_words == []
    assert stream.char_text == ""


def test_to_timed_words_holds_back_the_trailing_word() -> None:
    text = "one two"
    starts = [i * 0.1 for i in range(len(text))]
    ends = [s + 0.1 for s in starts]

    timed, remaining = soniox_tts._to_timed_words(text, starts, ends)
    # the separator rides with the word before it, so the chunks concatenate
    assert [str(w) for w in timed] == ["one "]
    assert remaining == "two"

    timed, remaining = soniox_tts._to_timed_words(text, starts, ends, flush=True)
    assert [str(w) for w in timed] == ["one ", "two"]
    assert remaining == ""
    # timings describe the word, not the separator: "one" ends at 0.3, not 0.4
    assert timed[0].start_time == pytest.approx(0.0)
    assert timed[0].end_time == pytest.approx(0.3)
    assert timed[1].start_time == pytest.approx(0.4)
    assert timed[1].end_time == pytest.approx(0.7)


async def test_timed_words_reconstruct_the_spoken_text() -> None:
    """The SDK concatenates these chunks into chat history, so they must tile the text.

    ``perform_text_forwarding`` builds the reply with ``out.text += delta``; a
    chunk set that dropped the separators would store "Hellothere" as the text
    the agent spoke.
    """
    spoken = "Hello there, this is what the agent said."
    stream_id = "stream-1"
    emitter = _RecordingEmitter()
    tts = soniox.TTS(api_key="fake-key")
    conn = soniox_tts._Connection(tts._opts, session=None)  # type: ignore[arg-type]
    conn._streams[stream_id] = soniox_tts._StreamData(
        emitter=emitter,  # type: ignore[arg-type]
        waiter=asyncio.get_event_loop().create_future(),
        opts=tts._opts,
    )
    # split mid-word, the way audio frames land
    conn._ws = _FakeWebSocket(  # type: ignore[assignment]
        [
            _text_message(
                {"stream_id": stream_id, "timestamps": _character_timestamps(spoken[:17], 0.0)}
            ),
            _text_message(
                {
                    "stream_id": stream_id,
                    "timestamps": _character_timestamps(spoken[17:], 17 * _SECONDS_PER_CHAR),
                    "audio_end": True,
                }
            ),
            _text_message({"stream_id": stream_id, "terminated": True}),
        ]
    )

    await conn._recv_loop()
    await conn.aclose()

    assert "".join(str(w) for w in emitter.timed_words) == spoken


async def test_rotation_offset_outruns_a_lagging_pushed_duration() -> None:
    """A rotated stream starts from the timeline, not from ``pushed_duration()``.

    ``pushed_duration()`` counts neither audio still queued in the emitter nor
    the tail frame it holds back, so at rotation time it can sit behind words
    that were already emitted. Taking it alone would start the new stream early
    and overlap them.
    """
    fake = _FakeConnection(timestamps=True)
    tts = soniox.TTS(api_key="fake-key")

    async def _fake_current_connection(*, timeout: float) -> tuple[Any, float, bool]:
        return fake, 0.0, True

    tts._current_connection = _fake_current_connection  # type: ignore[method-assign]

    stream = tts.stream()
    try:
        # words emitted so far reach 2.134s, but the emitter only admits to 2.090s
        stream._timeline.end = 2.134
        emitter = SimpleNamespace(pushed_duration=lambda: 2.090)
        active = await stream._open_stream(emitter, "req-1")  # type: ignore[arg-type]

        assert fake.time_offsets[-1] == pytest.approx(2.134)
        # replay is gated on what the stream handed the emitter, not on either
        # clock, so a fresh stream starts replayable
        assert active.data.produced_output is False
    finally:
        await stream.aclose()
        await tts.aclose()


def test_to_timed_words_keeps_a_leading_separator() -> None:
    """Text before the first word still reaches the transcript."""
    text = " hi"
    starts = [i * 0.1 for i in range(len(text))]
    ends = [s + 0.1 for s in starts]

    timed, remaining = soniox_tts._to_timed_words(text, starts, ends, flush=True)
    assert "".join(str(w) for w in timed) + remaining == text


async def test_timed_words_rebuild_the_reply_across_a_rotation() -> None:
    """The reply survives a stream rotation intact, separator included.

    The sentence tokenizer hands each stream its leading separator and Soniox
    normalises it away, so without restoring it the chunks concatenate to
    "one sentence.And here comes" in chat history.
    """
    fake = _FakeConnection(timestamps=True)
    words: list[TimedString] = []
    await _synthesize(
        fake,
        timed_words=words,
        chunk_delays=[3.0, 0.01, 0.01],
        stream_idle_timeout=1.0,
    )

    assert len(fake.registered_ids) == 2
    assert "".join(str(w) for w in words) == "".join(SENTENCES)


async def test_timed_words_rebuild_the_reply_on_one_stream() -> None:
    fake = _FakeConnection(timestamps=True)
    words: list[TimedString] = []
    await _synthesize(fake, timed_words=words)

    assert len(fake.registered_ids) == 1
    assert "".join(str(w) for w in words) == "".join(SENTENCES)


async def test_a_stream_that_published_words_is_never_replayed() -> None:
    """A stream that handed the emitter anything is not replayed in place.

    ``push_timed_transcript`` cannot be taken back, so replaying such a stream
    onto the same emitter would leave the failed attempt's words queued in front
    of the replay's and say everything twice. The emitter's own duration does
    not notice in time: one 10ms chunk sits entirely in the tail frame it holds
    back, so ``pushed_duration()`` still reads the value it had at open.
    """
    fake = _FakeConnection(timestamps=True, fail_on_send=1, emit_before_failure=True)
    words: list[TimedString] = []
    with pytest.raises(APIStatusError):
        await _synthesize(fake, timed_words=words)

    # exactly one stream: the failure was not replayed
    assert len(fake.registered_ids) == 1
    spoken = "".join(str(w) for w in words)
    assert spoken and spoken in "".join(SENTENCES)


async def test_words_for_unreplayable_audio_are_released() -> None:
    """Words held for audio the emitter accepted are not dropped with the stream.

    Once the emitter counts a stream's audio the stream can no longer be
    replayed, so the words held back for it describe speech the user hears.
    Discarding them with the stream would drop that text from the aligned
    transcript, and from the history of a turn interrupted right there.
    """
    fake = _FakeConnection(timestamps=True, fail_on_send=2)
    words: list[TimedString] = []
    with pytest.raises(APIStatusError):
        await _synthesize(fake, timed_words=words)

    # the first sentence was spoken before the failure, so it must be reported
    assert "".join(str(w) for w in words).strip() == SENTENCES[0].strip()


async def test_cancelling_a_stream_keeps_the_words_it_spoke() -> None:
    """An interruption publishes what the stream said, including the last word.

    Cancellation rules out a replay, so the characters buffered as a
    possibly-incomplete word are as complete as they will get and describe
    audio the emitter already has. Dropping them with the stream would leave
    that speech out of the interrupted turn's history.
    """
    emitter = _RecordingEmitter()
    tts = soniox.TTS(api_key="fake-key")
    fake = _FakeConnection(timestamps=True)
    stream = tts.stream()
    waiter: asyncio.Future[None] = asyncio.get_event_loop().create_future()

    try:
        data = fake.register_stream("s1", emitter, waiter, opts=tts._opts)  # type: ignore[arg-type]
        spoken = SENTENCES[0].strip()
        data.sent_text = SENTENCES[0]
        soniox_tts._accumulate_timestamps(data, _character_timestamps(spoken, 0.0))
        assert data.produced_output is True

        active = soniox_tts._ActiveStream(
            connection=fake,  # type: ignore[arg-type]
            stream_id="s1",
            waiter=waiter,
            data=data,
            opened_at=0.0,
        )
        stream._cancelled.set()
        waiter.set_result(None)

        assert await stream._settle_stream(active, emitter, "req-1") is None  # type: ignore[arg-type]
        assert "".join(str(w) for w in emitter.timed_words) == spoken
    finally:
        await stream.aclose()
        await tts.aclose()


async def test_a_stream_that_pushed_audio_is_never_replayed() -> None:
    """Audio alone makes a stream unreplayable, before any word is complete.

    A frame can carry too few characters to finish a word, so nothing is
    published yet - but the audio is already the emitter's, and replaying the
    stream would speak it twice.
    """
    fake = _FakeConnection(fail_on_send=1, emit_before_failure=True)
    with pytest.raises(APIStatusError):
        await _synthesize(fake)

    assert len(fake.registered_ids) == 1


async def test_recv_loop_marks_audio_as_produced_output() -> None:
    """Audio handed to the emitter marks the stream, before any word completes.

    This is what makes a stream unreplayable, and it has to be recorded on
    receipt: the emitter's own duration will not show the audio until it turns
    into frames, by which time a replay decision may already have been made.
    """
    stream_id = "stream-1"
    emitter = _RecordingEmitter()
    tts = soniox.TTS(api_key="fake-key")
    conn = soniox_tts._Connection(tts._opts, session=None)  # type: ignore[arg-type]
    data = soniox_tts._StreamData(
        emitter=emitter,  # type: ignore[arg-type]
        waiter=asyncio.get_event_loop().create_future(),
        opts=tts._opts,
    )
    conn._streams[stream_id] = data
    conn._ws = _FakeWebSocket(  # type: ignore[assignment]
        [
            _text_message(
                {
                    "stream_id": stream_id,
                    "audio": base64.b64encode(_SILENCE_PCM).decode("ascii"),
                }
            ),
            _text_message({"stream_id": stream_id, "terminated": True}),
        ]
    )

    await conn._recv_loop()
    await conn.aclose()

    assert emitter.audio == [_SILENCE_PCM]
    assert data.produced_output is True
    assert emitter.timed_words == []


async def test_interrupting_a_reply_keeps_the_last_spoken_word() -> None:
    """An interruption publishes the word the buffer was still holding.

    ``_to_timed_words`` holds the trailing word back in case more characters
    complete it. ``aclose`` cancels ``_run`` outright, so the stream never
    settles - and without a flush on that path the last word the user heard
    never reaches the transcript.
    """
    fake = _FakeConnection(timestamps=True)
    tts = soniox.TTS(api_key="fake-key")

    async def _fake_current_connection(*, timeout: float) -> tuple[Any, float, bool]:
        return fake, 0.0, True

    tts._current_connection = _fake_current_connection  # type: ignore[method-assign]

    stream = tts.stream()
    words: list[TimedString] = []

    async def _drain() -> None:
        async for ev in stream:
            words.extend(ev.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, []))

    drain_t = asyncio.create_task(_drain())
    try:
        stream.push_text(SENTENCES[0])  # no end_input: the reply is cut short
        await asyncio.sleep(0.1)
        await stream.aclose()
        await drain_t
    finally:
        await utils.aio.gracefully_cancel(drain_t)
        await tts.aclose()

    assert "".join(str(w) for w in words).strip() == SENTENCES[0].strip()


async def test_only_published_words_advance_the_timeline() -> None:
    """Buffered characters move nothing until they are spoken for.

    The timeline is what a rotated stream starts from, so advancing it on
    characters that finish no word would hand a replacement an offset past
    audio that was never produced - and would make a stream that emitted
    nothing look unreplayable.
    """
    emitter = _RecordingEmitter()
    tts = soniox.TTS(api_key="fake-key")
    data = soniox_tts._StreamData(
        emitter=emitter,  # type: ignore[arg-type]
        waiter=asyncio.get_event_loop().create_future(),
        opts=tts._opts,
    )

    # "Hel" finishes no word: nothing published, nothing spent
    soniox_tts._accumulate_timestamps(data, _character_timestamps("Hel", 0.0))
    assert emitter.timed_words == []
    assert data.timeline.end == 0.0
    assert data.produced_output is False

    # "lo wo" completes "Hello ", which is published and does move the timeline
    soniox_tts._accumulate_timestamps(data, _character_timestamps("lo wo", 3 * _SECONDS_PER_CHAR))
    assert [str(w) for w in emitter.timed_words] == ["Hello "]
    assert data.timeline.end == pytest.approx(0.05)
    assert data.produced_output is True


async def test_a_timestamp_only_partial_word_stays_replayable() -> None:
    """Buffering characters is reversible, so it must not cost a replay.

    A frame can carry timestamps that finish no word. Nothing reaches the
    emitter, so a transient failure after it is still safe to replay - and
    aborting the reply instead would cut the agent off for nothing.
    """
    fake = _FakeConnection(timestamps=True)
    tts = soniox.TTS(api_key="fake-key")

    async def _fake_current_connection(*, timeout: float) -> tuple[Any, float, bool]:
        return fake, 0.0, True

    tts._current_connection = _fake_current_connection  # type: ignore[method-assign]

    stream = tts.stream()
    emitter = _RecordingEmitter()
    try:
        active = await stream._open_stream(emitter, "req-1")  # type: ignore[arg-type]
        soniox_tts._accumulate_timestamps(active.data, _character_timestamps("Hel", 0.0))

        assert emitter.timed_words == []
        assert active.data.produced_output is False

        active.waiter.set_exception(APIStatusError("transient", status_code=429, retryable=True))
        replacement = await stream._settle_stream(active, emitter, "req-1")  # type: ignore[arg-type]

        assert replacement is not None
        assert replacement.stream_id != active.stream_id
        # the replacement inherits no advance from the attempt that emitted nothing
        assert fake.time_offsets[-1] == 0.0
    finally:
        await stream.aclose()
        await tts.aclose()


async def test_a_malformed_frame_does_not_splice_across_the_gap() -> None:
    """Skipping a bad frame must not join the characters on either side of it.

    The buffer holds the word that frame was going to finish. Letting the next
    frame land on it turns "bro" + "ps" into "brops" - a word nobody spoke.
    Publishing either side alone is no better: "ps" is not a word either, so
    characters are dropped until a boundary.
    """
    emitter = _RecordingEmitter()
    tts = soniox.TTS(api_key="fake-key")
    data = soniox_tts._StreamData(
        emitter=emitter,  # type: ignore[arg-type]
        waiter=asyncio.get_event_loop().create_future(),
        opts=tts._opts,
    )

    spoken = "The quick brown fox jumps high"
    soniox_tts._accumulate_timestamps(data, _character_timestamps("The quick bro", 0.0))
    # "wn fox jum" arrives with timings that do not line up, so it is skipped
    soniox_tts._accumulate_timestamps(
        data,
        {
            "characters": list("wn fox jum"),
            "character_start_times_seconds": [0.13, 0.14],
            "character_end_times_seconds": [0.14, 0.15],
        },
    )
    soniox_tts._accumulate_timestamps(data, _character_timestamps("ps high", 0.23))

    published = "".join(str(w) for w in emitter.timed_words)
    # every published word is a whole word the agent said. Substring checks are
    # too weak here: "ps" is inside "jumps" but was never a word.
    spoken_words = set(spoken.split())
    for word in published.split():
        assert word in spoken_words, f"invented {word!r}"


async def test_resync_waits_for_a_boundary_across_frames() -> None:
    """A frame that is still inside the broken word publishes nothing."""
    emitter = _RecordingEmitter()
    tts = soniox.TTS(api_key="fake-key")
    data = soniox_tts._StreamData(
        emitter=emitter,  # type: ignore[arg-type]
        waiter=asyncio.get_event_loop().create_future(),
        opts=tts._opts,
    )

    soniox_tts._accumulate_timestamps(data, _character_timestamps("The quick bro", 0.0))
    soniox_tts._accumulate_timestamps(data, {"characters": list("wn fox ju")})  # malformed
    assert data.resync_pending is True

    # "mps" is the tail of the broken word, with no boundary in sight
    soniox_tts._accumulate_timestamps(data, _character_timestamps("mps", 0.22))
    assert data.resync_pending is True
    assert "".join(str(w) for w in emitter.timed_words) == "The quick "

    # the boundary arrives with the next frame, and alignment resumes after it
    soniox_tts._accumulate_timestamps(data, _character_timestamps(" high up", 0.25))
    assert data.resync_pending is False
    soniox_tts._emit_timed_words(data, flush=True)
    assert "".join(str(w) for w in emitter.timed_words) == "The quick brown fox jumps high up"


async def test_a_gap_costs_only_the_frame_in_a_spaceless_script() -> None:
    """CJK and Thai recover immediately: every character is already a word.

    Waiting for whitespace never resolves in these scripts, so the rest of the
    reply's transcript and timeline would be discarded entirely.
    """
    emitter = _RecordingEmitter()
    tts = soniox.TTS(api_key="fake-key")
    data = soniox_tts._StreamData(
        emitter=emitter,  # type: ignore[arg-type]
        waiter=asyncio.get_event_loop().create_future(),
        opts=tts._opts,
    )

    soniox_tts._accumulate_timestamps(data, _character_timestamps("\u4eca\u5929\u5929", 0.0))
    soniox_tts._accumulate_timestamps(data, {"characters": ["\u6c14"]})  # malformed
    soniox_tts._accumulate_timestamps(data, _character_timestamps("\u5f88\u597d", 0.04))
    soniox_tts._emit_timed_words(data, flush=True)

    published = "".join(str(w) for w in emitter.timed_words)
    assert data.resync_pending is False
    # nothing is lost: the skipped frame's characters come back untimed
    assert published == "\u4eca\u5929\u5929\u6c14\u5f88\u597d"


def test_word_boundary_follows_the_tokenizer() -> None:
    assert soniox_tts._is_word_boundary(" ") is True
    assert soniox_tts._is_word_boundary("\u4eca") is True  # CJK: a word on its own
    assert soniox_tts._is_word_boundary("\u0e2a") is True  # Thai
    assert soniox_tts._is_word_boundary("p") is False
    assert soniox_tts._is_word_boundary(",") is False  # punctuation stays in the word


async def test_a_gap_ending_on_a_boundary_costs_no_extra_word() -> None:
    """When the frame after a gap opens on a boundary, no whole word is dropped.

    Only the separator is consumed. Where the gap ends mid-word instead, the
    tail has to go: nothing in the stream says whether those characters finish
    the broken word or start a new one.
    """
    emitter = _RecordingEmitter()
    tts = soniox.TTS(api_key="fake-key")
    data = soniox_tts._StreamData(
        emitter=emitter,  # type: ignore[arg-type]
        waiter=asyncio.get_event_loop().create_future(),
        opts=tts._opts,
    )

    soniox_tts._accumulate_timestamps(data, _character_timestamps("The quick bro", 0.0))
    soniox_tts._accumulate_timestamps(data, {"characters": list("wn fox jumps")})  # malformed
    # the next frame opens on a separator, so the gap ended at a word boundary
    soniox_tts._accumulate_timestamps(data, _character_timestamps(" high up", 0.25))
    soniox_tts._emit_timed_words(data, flush=True)

    assert "".join(str(w) for w in emitter.timed_words) == "The quick brown fox jumps high up"


async def test_text_from_a_skipped_frame_is_published_without_timings() -> None:
    """A frame's timings can be unusable while its characters were still spoken.

    Dropping them would leave the turn's history missing words the agent said.
    They are published untimed instead, which the synchronizer estimates
    across - the old behaviour, confined to the stretch that lost its timings.
    """
    emitter = _RecordingEmitter()
    tts = soniox.TTS(api_key="fake-key")
    data = soniox_tts._StreamData(
        emitter=emitter,  # type: ignore[arg-type]
        waiter=asyncio.get_event_loop().create_future(),
        opts=tts._opts,
    )

    soniox_tts._accumulate_timestamps(data, _character_timestamps("The quick bro", 0.0))
    soniox_tts._accumulate_timestamps(data, {"characters": list("wn fox jum")})  # malformed
    soniox_tts._accumulate_timestamps(data, _character_timestamps("ps high", 0.23))
    soniox_tts._emit_timed_words(data, flush=True)

    assert "".join(str(w) for w in emitter.timed_words) == "The quick brown fox jumps high"
    # the gap's words arrive as one chunk spanning the region, not as words
    assert "brown fox jumps " in [str(w) for w in emitter.timed_words]


async def test_untimed_text_survives_a_stream_that_ends_inside_a_gap() -> None:
    """A stream can end before a boundary closes the untimed region.

    The held text is the tail of the reply, so losing it here would cut the
    transcript short exactly where an interruption is most likely.
    """
    emitter = _RecordingEmitter()
    tts = soniox.TTS(api_key="fake-key")
    data = soniox_tts._StreamData(
        emitter=emitter,  # type: ignore[arg-type]
        waiter=asyncio.get_event_loop().create_future(),
        opts=tts._opts,
    )

    soniox_tts._accumulate_timestamps(data, _character_timestamps("The quick bro", 0.0))
    soniox_tts._accumulate_timestamps(data, {"characters": list("wn fox")})  # malformed
    soniox_tts._emit_timed_words(data, flush=True)

    assert "".join(str(w) for w in emitter.timed_words) == "The quick brown fox"


async def test_untimed_text_first_keeps_the_separator_in_place() -> None:
    """When the gap opens a stream, the untimed text carries its separator.

    The separator belongs to whatever the stream publishes first. Leaving it for
    the first timed word instead moves the whitespace: the reply loses it here
    and grows a second one further along.
    """
    emitter = _RecordingEmitter()
    tts = soniox.TTS(api_key="fake-key")
    data = soniox_tts._StreamData(
        emitter=emitter,  # type: ignore[arg-type]
        waiter=asyncio.get_event_loop().create_future(),
        opts=tts._opts,
    )
    # a rotated stream is handed its leading separator by the tokenizer
    data.sent_text = " Then, after all"

    soniox_tts._accumulate_timestamps(data, {"characters": list("Then,")})  # malformed
    soniox_tts._accumulate_timestamps(data, _character_timestamps(" after all", 0.05))
    soniox_tts._emit_timed_words(data, flush=True)

    assert "".join(str(w) for w in emitter.timed_words) == " Then, after all"


async def test_an_untimed_tail_is_bounded_so_captions_reach_it() -> None:
    """The last untimed region is closed off, since nothing timed follows it.

    The synchronizer builds its rate curve from timed points and reveals text
    only as far as that curve goes. A tail left unbounded would sit past the
    end of it, and the caption would stop short of words the agent spoke.
    """
    emitter = _RecordingEmitter()
    tts = soniox.TTS(api_key="fake-key")
    data = soniox_tts._StreamData(
        emitter=emitter,  # type: ignore[arg-type]
        waiter=asyncio.get_event_loop().create_future(),
        opts=tts._opts,
    )

    soniox_tts._accumulate_timestamps(data, _character_timestamps("The quick bro", 0.0))
    emitter.push(_SILENCE_PCM * 50)  # 500ms of audio has reached the emitter
    soniox_tts._accumulate_timestamps(data, {"characters": list("wn fox")})  # malformed
    soniox_tts._emit_timed_words(data, flush=True)

    tail = emitter.timed_words[-1]
    assert str(tail) == "brown fox"
    # opens where the last timed word ended, closes on the audio produced
    assert tail.start_time == pytest.approx(0.09)
    assert tail.end_time == pytest.approx(0.5)


async def test_an_untimed_region_mid_reply_opens_where_timing_stopped() -> None:
    """A region followed by timed words needs no end: the next word closes it."""
    emitter = _RecordingEmitter()
    tts = soniox.TTS(api_key="fake-key")
    data = soniox_tts._StreamData(
        emitter=emitter,  # type: ignore[arg-type]
        waiter=asyncio.get_event_loop().create_future(),
        opts=tts._opts,
    )

    soniox_tts._accumulate_timestamps(data, _character_timestamps("The quick bro", 0.0))
    soniox_tts._accumulate_timestamps(data, {"characters": list("wn fox jum")})  # malformed
    soniox_tts._accumulate_timestamps(data, _character_timestamps("ps high up", 0.23))

    untimed = next(w for w in emitter.timed_words if str(w) == "brown fox jumps ")
    assert untimed.start_time == pytest.approx(0.09)
    assert not is_given(untimed.end_time)
