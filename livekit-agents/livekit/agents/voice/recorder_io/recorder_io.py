from __future__ import annotations

import asyncio
import contextlib
import queue
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import av
import numpy as np

from livekit import rtc

from ... import utils
from ...log import logger
from .. import io

if TYPE_CHECKING:
    from ..agent_session import AgentSession

# Both channels sit on one absolute timeline: the user's audio where it arrived, the agent's
# where the device reports it played. Silence is whatever nothing was written over.

WRITE_INTERVAL = 2.5
FFMPEG_STRICT_LEVEL = "experimental"

# how long the writer waits on a source that stopped delivering before taking the silence as real
INPUT_STALL_TIMEOUT = 1.0

# a run continues while its own clock stays this close to the timestamps coming in; re-anchoring
# beyond it keeps a drifting capture clock from sliding the channel
RESYNC_TOLERANCE = 0.1


@dataclass
class _Captured:
    channel: int
    started_at: float
    frame: rtc.AudioFrame


@dataclass
class _Flush:
    until: float


class _Track:
    """One channel of the recording, holding runs of audio placed on the absolute timeline."""

    def __init__(self, *, sample_rate: int, t0: float) -> None:
        self._sample_rate = sample_rate
        self._t0 = t0
        self._placed: list[tuple[int, np.ndarray]] = []  # (start sample, mono float32)
        self._resampler: rtc.AudioResampler | None = None
        self._source_rate: int | None = None
        self._run_start: float | None = None
        self._run_samples = 0
        self.dropped_samples = 0

    def _resample(self, frame: rtc.AudioFrame) -> list[rtc.AudioFrame]:
        """Mono frames at the recording rate, of which the resampler may still hold some back."""
        data = np.frombuffer(frame.data, dtype=np.int16).reshape(-1, frame.num_channels)
        mono = data.mean(axis=1).astype(np.int16) if frame.num_channels > 1 else data[:, 0]
        mono_frame = rtc.AudioFrame(
            data=mono.tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=1,
            samples_per_channel=len(mono),
        )
        if frame.sample_rate == self._sample_rate:
            return [mono_frame]

        if self._resampler is None or self._source_rate != frame.sample_rate:
            self._source_rate = frame.sample_rate
            self._resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate, output_rate=self._sample_rate, num_channels=1
            )

        return self._resampler.push(mono_frame)

    def push(self, started_at: float, frame: rtc.AudioFrame) -> None:
        """Add audio that began at ``started_at``, extending the open run where it fits."""

        def _place(frames: list[rtc.AudioFrame]) -> None:
            if not frames:
                return

            assert self._run_start is not None
            joined = np.concatenate([np.frombuffer(f.data, dtype=np.int16) for f in frames])
            samples = joined.astype(np.float32) / 32768.0
            start = round((self._run_start - self._t0) * self._sample_rate) + self._run_samples
            self._placed.append((start, samples))
            self._run_samples += len(samples)

        expected = (
            None
            if self._run_start is None
            else self._run_start + self._run_samples / self._sample_rate
        )
        if expected is None or abs(started_at - expected) > RESYNC_TOLERANCE:
            if self._resampler is not None:
                # whatever the resampler still holds is the tail of the run that just ended
                _place(self._resampler.flush())
            self._run_start, self._run_samples = started_at, 0

        _place(self._resample(frame))

    def take(self, start: int, end: int) -> np.ndarray:
        """The channel over ``[start, end)``, silent wherever nothing was placed."""
        block = np.zeros(max(0, end - start), dtype=np.float32)
        keep: list[tuple[int, np.ndarray]] = []
        for pos, samples in self._placed:
            stop = pos + len(samples)
            if stop <= start:
                self.dropped_samples += len(samples)
                continue
            if pos >= end:
                keep.append((pos, samples))
                continue

            lo, hi = max(pos, start), min(stop, end)
            block[lo - start : hi - start] += samples[lo - pos : hi - pos]
            if stop > end:
                keep.append((end, samples[end - pos :]))
        self._placed = keep
        return block


class RecorderIO:
    def __init__(
        self,
        *,
        agent_session: AgentSession,
        sample_rate: int = 48000,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._in_record: RecorderAudioInput | None = None
        self._out_record: RecorderAudioOutput | None = None

        self._q: queue.Queue[_Captured | _Flush | None] = queue.Queue()
        self._session = agent_session
        self._sample_rate = sample_rate
        self._started = False
        self._loop = loop or asyncio.get_event_loop()
        self._lock = asyncio.Lock()
        self._close_fut: asyncio.Future[None] = self._loop.create_future()
        self._output_path: Path | None = None
        self._write_atask: asyncio.Task[None] | None = None

        self._t0: float | None = None
        self._input_settled: float = 0.0

    async def start(self, *, output_path: str | Path) -> None:
        async with self._lock:
            if self._started:
                return

            if not self._in_record or not self._out_record:
                raise RuntimeError(
                    "RecorderIO not properly initialized: both `record_input()` and "
                    "`record_output()` must be called before starting the recorder."
                )

            self._output_path = Path(output_path)
            self._started = True
            self._close_fut = self._loop.create_future()
            self._t0 = self._input_settled = time.time()
            self._write_atask = asyncio.create_task(self._write_task())

            thread = threading.Thread(
                target=self._encode_thread, daemon=True, name="recorder_io_encode_thread"
            )
            thread.start()

    async def aclose(self) -> None:
        async with self._lock:
            if not self._started:
                return

            if self._write_atask is not None:
                await utils.aio.cancel_and_wait(self._write_atask)
                self._write_atask = None

            self._q.put_nowait(_Flush(until=time.time()))
            self._q.put_nowait(None)
            await asyncio.shield(self._close_fut)
            self._started = False

    def record_input(self, audio_input: io.AudioInput) -> RecorderAudioInput:

        def on_frame(started_at: float, frame: rtc.AudioFrame) -> None:
            # a contiguous stream, so what has arrived is exactly what is settled
            self._input_settled = started_at + frame.duration
            self._q.put_nowait(_Captured(channel=0, started_at=started_at, frame=frame))

        self._in_record = RecorderAudioInput(
            recording_io=self, source=audio_input, on_frame=on_frame
        )
        return self._in_record

    def record_output(self, audio_output: io.AudioOutput) -> RecorderAudioOutput:

        def on_played(started_at: float, frame: rtc.AudioFrame) -> None:
            self._q.put_nowait(_Captured(channel=1, started_at=started_at, frame=frame))

        self._out_record = RecorderAudioOutput(
            recording_io=self, audio_output=audio_output, on_played=on_played
        )
        return self._out_record

    @property
    def recording(self) -> bool:
        return self._started

    @property
    def output_path(self) -> Path | None:
        return self._output_path

    @property
    def recording_started_at(self) -> float | None:
        return self._t0

    async def _write_task(self) -> None:
        assert self._out_record is not None

        while True:
            await asyncio.sleep(WRITE_INTERVAL)

            # a source gone quiet would hold the writer forever, so it is only waited on so long
            settled = max(self._input_settled, time.time() - INPUT_STALL_TIMEOUT)
            if pending := self._out_record.pending_since:
                # a segment in flight has not said where its audio went
                settled = min(settled, pending)

            self._q.put_nowait(_Flush(until=settled))

    def _encode_thread(self) -> None:
        assert self._output_path is not None and self._t0 is not None
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        container = av.open(self._output_path, mode="w", format="ogg")

        # prefer libopus; fallback to native opus only if necessary
        try:
            av.Codec("libopus", "w")
            codec_name = "libopus"
        except av.codec.codec.UnknownCodecError:
            logger.trace("libopus codec is not available, using opus")
            codec_name = "opus"

        stream: av.AudioStream = container.add_stream(  # type: ignore
            codec_name,
            rate=self._sample_rate,
            layout="stereo",
        )

        # native ffmpeg opus encoder is experimental
        if codec_name == "opus":
            stream.codec_context.options["strict"] = FFMPEG_STRICT_LEVEL

        tracks = [_Track(sample_rate=self._sample_rate, t0=self._t0) for _ in range(2)]
        cursor = 0

        try:
            with container:
                while (item := self._q.get()) is not None:
                    if isinstance(item, _Captured):
                        tracks[item.channel].push(item.started_at, item.frame)
                        continue

                    end = round((item.until - self._t0) * self._sample_rate)
                    if end <= cursor:
                        continue

                    block = np.stack([t.take(cursor, end) for t in tracks])
                    cursor = end

                    av_frame = av.AudioFrame.from_ndarray(block, format="fltp", layout="stereo")
                    av_frame.sample_rate = self._sample_rate
                    for packet in stream.encode(av_frame):
                        container.mux(packet)

                for packet in stream.encode(None):
                    container.mux(packet)
        except Exception:
            logger.exception("recorder encode thread failed; recording may be incomplete")
        finally:
            for label, track in zip(("input", "output"), tracks, strict=True):
                if track.dropped_samples:
                    logger.warning(
                        "recorder dropped audio that reached it after its place in the timeline "
                        "had been written",
                        extra={"channel": label, "samples": track.dropped_samples},
                    )

            def resolve_close_fut() -> None:
                if not self._close_fut.done():
                    self._close_fut.set_result(None)

            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(resolve_close_fut)


class RecorderAudioInput(io.AudioInput):
    def __init__(
        self,
        *,
        recording_io: RecorderIO,
        source: io.AudioInput,
        on_frame: Callable[[float, rtc.AudioFrame], None],
    ) -> None:
        super().__init__(label="RecorderIO", source=source)
        self.__audio_input = source
        self.__recording_io = recording_io
        self.__on_frame = on_frame

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame]:
        return self

    async def __anext__(self) -> rtc.AudioFrame:
        frame = await self.__audio_input.__anext__()

        if self.__recording_io.recording:
            # frames carry no capture timestamp, so arrival is the clock
            self.__on_frame(time.time() - frame.duration, frame)

        return frame


class RecorderAudioOutput(io.AudioOutput):
    def __init__(
        self,
        *,
        recording_io: RecorderIO,
        audio_output: io.AudioOutput | None = None,
        on_played: Callable[[float, rtc.AudioFrame], None],
    ) -> None:
        super().__init__(
            label="RecorderIO",
            next_in_chain=audio_output,
            capabilities=io.AudioOutputCapabilities(pause=True),
        )
        self.__recording_io = recording_io
        self.__on_played = on_played
        self.__acc: list[rtc.AudioFrame] = []
        self.__pcm: np.ndarray | None = None  # the segment's frames, joined on first slice
        self.__segment_since: float | None = None
        self.__started_at: float | None = None
        self.__reported = False

    @property
    def sample_rate(self) -> int | None:
        if self._sample_rate is not None:
            return self._sample_rate
        return self.next_in_chain.sample_rate if self.next_in_chain else None

    @property
    def pending_since(self) -> float | None:
        """Wall time from which the agent channel is unsettled, while a segment is in flight."""
        return self.__segment_since

    def on_playback_started(self, *, created_at: float) -> None:
        super().on_playback_started(created_at=created_at)
        if self.__started_at is None:
            self.__started_at = created_at

    def on_playback_progressed(self, *, started_at: float, offset: float, duration: float) -> None:
        super().on_playback_progressed(started_at=started_at, offset=offset, duration=duration)

        if self.__recording_io.recording:
            self.__reported = True
            self.__place(started_at=started_at, offset=offset, duration=duration)

    def on_playback_finished(
        self,
        *,
        playback_position: float,
        interrupted: bool,
        synchronized_transcript: str | None = None,
    ) -> None:
        super().on_playback_finished(
            playback_position=playback_position,
            interrupted=interrupted,
            synchronized_transcript=synchronized_transcript,
        )

        if self.__recording_io.recording and not self.__reported and self.__acc:
            # the sink reports nothing of its own, so its endpoints describe the segment
            self.__place(
                started_at=self.__started_at
                if self.__started_at is not None
                else time.time() - playback_position,
                offset=0.0,
                duration=playback_position,
            )

        self.__acc = []
        self.__pcm = None
        self.__segment_since = None
        self.__started_at = None
        self.__reported = False

    def __place(self, *, started_at: float, offset: float, duration: float) -> None:
        """Hand the recorder the captured audio a report covers, at the time it played."""
        if not self.__acc or duration <= 0:
            return

        rate, channels = self.__acc[0].sample_rate, self.__acc[0].num_channels
        if self.__pcm is None:
            self.__pcm = np.concatenate([np.frombuffer(f.data, dtype=np.int16) for f in self.__acc])

        lo = round(offset * rate) * channels
        hi = min(round((offset + duration) * rate) * channels, len(self.__pcm))
        if hi <= lo:
            return

        chunk = self.__pcm[lo:hi]
        self.__on_played(
            started_at,
            rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=rate,
                num_channels=channels,
                samples_per_channel=len(chunk) // channels,
            ),
        )

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        if self.next_in_chain:
            await self.next_in_chain.capture_frame(frame)

        await super().capture_frame(frame)

        if self.__recording_io.recording:
            if not self.__acc:
                self.__segment_since = time.time()
            self.__acc.append(frame)
            self.__pcm = None

    def flush(self) -> None:
        super().flush()

        if self.next_in_chain:
            self.next_in_chain.flush()

    def clear_buffer(self) -> None:
        if self.next_in_chain:
            self.next_in_chain.clear_buffer()
