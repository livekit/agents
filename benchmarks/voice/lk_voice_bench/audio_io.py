"""Audio I/O that lets an in-process ``AgentSession`` talk to a benchmark instead of a room.

``CallerAudioInput`` is the agent's microphone: the benchmark pushes the caller's PCM into it.

``SpeakerOutput`` is the agent's loudspeaker. It plays the agent's audio out one 20 ms frame at a
time, so playback position, interruptions and the transcript synchronizer behave exactly as they
would against a telephony sink. Its clock is either its own real-time loop (``start()``) or the
benchmark's, which calls ``play_frame()`` once per 20 ms of benchmark time. Whatever it "played"
is collected for the benchmark with ``drain_played()``.

``SpokenTextOutput`` receives the agent's transcript from LiveKit's ``TranscriptSynchronizer``,
so text arrives as it is spoken, and attributes each fragment to the segment playing at the time.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from dataclasses import dataclass, field

from livekit import rtc
from livekit.agents.utils import shortuuid
from livekit.agents.voice import io

FRAME_MS = 20


class CallerAudioInput(io.AudioInput):
    """The agent's microphone, fed by the benchmark with mono int16 PCM."""

    def __init__(self, *, sample_rate: int = 16000) -> None:
        super().__init__(label="bench-caller")
        self._sample_rate = sample_rate
        self._queue: asyncio.Queue[rtc.AudioFrame | None] = asyncio.Queue()
        self._resamplers: dict[int, rtc.AudioResampler] = {}

    def push_pcm(self, pcm: bytes, sample_rate: int) -> None:
        """Queue caller audio; must be called on the session's event loop."""
        if not pcm:
            return
        frame = rtc.AudioFrame(
            data=pcm,
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=len(pcm) // 2,
        )
        if sample_rate == self._sample_rate:
            self._queue.put_nowait(frame)
            return
        resampler = self._resamplers.get(sample_rate)
        if resampler is None:
            resampler = self._resamplers[sample_rate] = rtc.AudioResampler(
                input_rate=sample_rate, output_rate=self._sample_rate, num_channels=1
            )
        for f in resampler.push(frame):
            self._queue.put_nowait(f)

    def close(self) -> None:
        self._queue.put_nowait(None)

    async def __anext__(self) -> rtc.AudioFrame:
        frame = await self._queue.get()
        if frame is None:
            raise StopAsyncIteration
        return frame


@dataclass
class PlayedChunk:
    """A stretch of agent audio the virtual speaker has played."""

    pcm: bytes
    """Mono int16 PCM at the speaker's sample rate."""
    segment_id: str
    played_at: float
    """Wall-clock time the chunk started playing."""


@dataclass
class _Segment:
    id: str
    buf: bytearray = field(default_factory=bytearray)
    flushed: bool = False
    interrupted: bool = False
    finished: bool = False
    started: bool = False
    played_bytes: int = 0
    dropped_bytes: int = 0


@dataclass
class Interruption:
    """A segment cut short by ``clear_buffer``."""

    segment_id: str
    at: float
    played_s: float
    dropped_bytes: int


class SpeakerOutput(io.AudioOutput):
    """A virtual loudspeaker that plays the agent's audio in real time."""

    def __init__(self, *, sample_rate: int = 8000, frame_ms: int = FRAME_MS) -> None:
        super().__init__(
            label="bench-speaker",
            capabilities=io.AudioOutputCapabilities(pause=True),
            sample_rate=sample_rate,
        )
        self._frame_bytes = sample_rate * frame_ms // 1000 * 2
        self._frame_s = frame_ms / 1000
        self._bytes_per_s = sample_rate * 2
        self._segments: deque[_Segment] = deque()
        self._capturing: _Segment | None = None
        self._played: list[PlayedChunk] = []
        self._interruptions: list[Interruption] = []
        self._last_played_segment: str | None = None
        self._paused = False
        self._playout_atask: asyncio.Task[None] | None = None

    # -- benchmark side -----------------------------------------------------------------------

    def start(self) -> None:
        """Play out on a real-time clock of the speaker's own."""
        if self._playout_atask is None:
            self._playout_atask = asyncio.create_task(self._playout_loop(), name="bench-speaker")

    async def aclose(self) -> None:
        if self._playout_atask is not None:
            self._playout_atask.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._playout_atask
            self._playout_atask = None

    def play_frame(self) -> None:
        """Play the next frame now; for a benchmark that keeps the clock itself."""
        assert self._playout_atask is None, "the speaker is already on its own clock"
        self._play_one_frame()

    def drain_played(self) -> list[PlayedChunk]:
        """Audio played since the last call."""
        played, self._played = self._played, []
        return played

    def drain_interruptions(self) -> list[Interruption]:
        interruptions, self._interruptions = self._interruptions, []
        return interruptions

    @property
    def speaking_segment_id(self) -> str | None:
        """The segment playing now, else the one that played last."""
        for seg in self._segments:
            if seg.started and not seg.finished:
                return seg.id
        return self._last_played_segment

    @property
    def is_playing(self) -> bool:
        return any(seg.started and not seg.finished for seg in self._segments)

    # -- AudioOutput --------------------------------------------------------------------------

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        if self._capturing is None:
            self._capturing = _Segment(id=f"seg_{shortuuid()}")
            self._segments.append(self._capturing)
        if not self._capturing.interrupted:
            self._capturing.buf.extend(frame.data.cast("B"))

    def flush(self) -> None:
        super().flush()
        seg, self._capturing = self._capturing, None
        if seg is None:
            return
        seg.flushed = True
        if seg.interrupted or not seg.buf and not seg.started:
            self._finish(seg)

    def clear_buffer(self) -> None:
        now = time.time()
        for seg in list(self._segments):
            if seg.finished:
                continue
            if not seg.interrupted:
                seg.interrupted = True
                seg.dropped_bytes = len(seg.buf)
                seg.buf.clear()
                self._interruptions.append(
                    Interruption(
                        segment_id=seg.id,
                        at=now,
                        played_s=seg.played_bytes / self._bytes_per_s,
                        dropped_bytes=seg.dropped_bytes,
                    )
                )
            if seg.flushed:
                self._finish(seg)

    def pause(self) -> None:
        # the session pauses on speech it is not yet sure is an interruption, and resumes if
        # it was not one (resume_false_interruption); nothing plays while paused
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    # -- playout ------------------------------------------------------------------------------

    def _finish(self, seg: _Segment) -> None:
        if seg.finished:
            return
        seg.finished = True
        with contextlib.suppress(ValueError):
            self._segments.remove(seg)
        self.on_playback_finished(
            playback_position=seg.played_bytes / self._bytes_per_s,
            interrupted=seg.interrupted,
        )

    def _play_one_frame(self) -> None:
        if self._paused:
            return
        while self._segments:
            seg = self._segments[0]
            if seg.finished:
                self._segments.popleft()
                continue
            if not seg.buf:
                if seg.flushed:
                    self._finish(seg)
                    continue
                return  # waiting on TTS: underrun, nothing plays this frame

            chunk = bytes(seg.buf[: self._frame_bytes])
            del seg.buf[: len(chunk)]
            now = time.time()
            if not seg.started:
                seg.started = True
                self.on_playback_started(created_at=now)
            seg.played_bytes += len(chunk)
            self._last_played_segment = seg.id
            self._played.append(PlayedChunk(pcm=chunk, segment_id=seg.id, played_at=now))
            if not seg.buf and seg.flushed:
                self._finish(seg)
            return

    async def _playout_loop(self) -> None:
        loop = asyncio.get_running_loop()
        next_at = loop.time()
        while True:
            self._play_one_frame()
            next_at += self._frame_s
            delay = next_at - loop.time()
            if delay < -0.5:  # fell far behind (stalled loop): resync rather than burst
                next_at = loop.time()
                delay = 0
            await asyncio.sleep(max(delay, 0))


@dataclass
class SpokenText:
    text: str
    segment_id: str | None
    at: float


class SpokenTextOutput(io.TextOutput):
    """Collects the agent's synchronized transcript, tagged with the segment being played."""

    def __init__(self, speaker: SpeakerOutput) -> None:
        super().__init__(label="bench-spoken-text", next_in_chain=None)
        self._speaker = speaker
        self._pending: list[SpokenText] = []

    async def capture_text(self, text: str) -> None:
        if text:
            self._pending.append(
                SpokenText(text=text, segment_id=self._speaker.speaking_segment_id, at=time.time())
            )

    def flush(self) -> None:
        pass

    def drain(self) -> list[SpokenText]:
        pending, self._pending = self._pending, []
        return pending
