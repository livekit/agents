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
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest

from livekit.agents import APIStatusError
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
    every new ``stream_id``.
    """

    def __init__(self, *, fail_on_send: int | None = None, timestamps: bool = False) -> None:
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
        self._streams[stream_id] = _StreamSlot(
            soniox_tts._StreamData(
                emitter=emitter, waiter=waiter, opts=opts, time_offset=time_offset
            )
        )
        self.max_open_streams = max(self.max_open_streams, len(self._streams))

    def unregister_stream(self, stream_id: str) -> None:
        self._streams.pop(stream_id, None)

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
            if not slot.waiter.done():
                slot.waiter.set_exception(
                    APIStatusError("transient", status_code=429, retryable=True)
                )
            return
        if not self._timestamps:
            slot.emitter.push(_SILENCE_PCM)
            return

        soniox_tts._accumulate_timestamps(slot.data, _character_timestamps(text, slot.cursor))
        slot.cursor += len(text) * _SECONDS_PER_CHAR
        slot.emitter.push(_SILENCE_PCM * len(text))

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
    async for ev in stream:
        frames += 1
        if timed_words is not None:
            timed_words += ev.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, [])
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
    def __init__(self) -> None:
        self.audio: list[bytes] = []
        self.timed_words: list[TimedString] = []

    def push(self, data: bytes) -> None:
        self.audio.append(data)

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
    conn._streams[stream_id] = soniox_tts._StreamData(
        emitter=emitter,  # type: ignore[arg-type]
        waiter=waiter,
        opts=tts._opts,
        time_offset=1.5,  # this stream started 1.5s into the segment
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

    assert [str(w) for w in emitter.timed_words] == ["Hi", "there"]
    hi, there = emitter.timed_words
    # every timestamp shifted onto the segment timeline by time_offset
    assert hi.start_time == pytest.approx(1.50)
    assert hi.end_time == pytest.approx(1.52)
    assert there.start_time == pytest.approx(1.53)
    assert there.end_time == pytest.approx(1.58)
    assert len(emitter.audio) == 2
    assert waiter.done() and waiter.result() is None


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
    second_stream_start, _ = _span(next(w for w in words if str(w) == "And"))
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
    assert [str(w) for w in timed] == ["one"]
    # the separator stays with the remainder so its characters keep their timings
    assert remaining == " two"

    timed, remaining = soniox_tts._to_timed_words(text, starts, ends, flush=True)
    assert [str(w) for w in timed] == ["one", "two"]
    assert remaining == ""
    assert timed[1].start_time == pytest.approx(0.4)
    assert timed[1].end_time == pytest.approx(0.7)
