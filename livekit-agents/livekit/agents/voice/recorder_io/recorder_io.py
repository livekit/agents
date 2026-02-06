from __future__ import annotations

import asyncio
import contextlib
import ctypes
import queue
import threading
import time
from collections import deque
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import av
import numpy as np

from livekit import rtc

from ...log import logger
from .. import io

if TYPE_CHECKING:
    from ..agent_session import AgentSession

# the recorder currently assume the input is a continous uninterrupted audio stream


WRITE_INTERVAL = 2.5


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

        self._in_q: queue.Queue[list[rtc.AudioFrame] | None] = queue.Queue()
        self._out_q: queue.Queue[list[rtc.AudioFrame] | None] = queue.Queue()
        self._session = agent_session
        self._sample_rate = sample_rate
        self._started = False
        self._loop = loop or asyncio.get_event_loop()
        self._lock = asyncio.Lock()
        self._close_fut: asyncio.Future[None] = self._loop.create_future()
        self._output_path: Path | None = None

        self._skip_padding_warning = False

    async def start(self, *, output_path: str | Path) -> None:
        async with self._lock:
            if self._started:
                return

            if not self._in_record or not self._out_record:
                raise RuntimeError(
                    "RecorderIO not properly initialized: both `record_input()` and `record_output()` "
                    "must be called before starting the recorder."
                )

            self._output_path = Path(output_path)
            self._started = True
            self._skip_padding_warning = False
            self._close_fut = self._loop.create_future()
            self._forward_atask = asyncio.create_task(self._forward_task())

            thread = threading.Thread(
                target=self._encode_thread, daemon=True, name="recorder_io_encode_thread"
            )
            thread.start()

    async def aclose(self) -> None:
        async with self._lock:
            if not self._started:
                return

            self._in_q.put_nowait(None)
            self._out_q.put_nowait(None)
            await asyncio.shield(self._close_fut)
            self._started = False

    def record_input(self, audio_input: io.AudioInput) -> RecorderAudioInput:
        self._in_record = RecorderAudioInput(recording_io=self, source=audio_input)
        return self._in_record

    def record_output(self, audio_output: io.AudioOutput) -> RecorderAudioOutput:
        self._out_record = RecorderAudioOutput(
            recording_io=self, audio_output=audio_output, write_fnc=self._write_cb
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
        in_t = self._in_record.started_wall_time if self._in_record else None
        out_t = self._out_record.started_wall_time if self._out_record else None

        if in_t is None:
            return out_t

        if out_t is None:
            return in_t

        return min(in_t, out_t)

    def _write_cb(self, buf: list[rtc.AudioFrame]) -> None:
        assert self._in_record is not None

        input_buf = self._in_record.take_buf(
            pad_since=self._out_record._last_speech_end_time if self._out_record else None
        )
        self._in_q.put_nowait(input_buf)
        self._out_q.put_nowait(buf)

    async def _forward_task(self) -> None:
        assert self._in_record is not None
        assert self._out_record is not None

        # Forward the input audio to the encoder every WRITE_INTERVAL seconds.
        while True:
            await asyncio.sleep(WRITE_INTERVAL)
            if self._out_record.has_pending_data:
                # if the output is currenetly playing audio, wait for it to stay in sync
                continue  # always wait for the complete output

            input_buf = self._in_record.take_buf(pad_since=self._out_record._last_speech_end_time)
            self._in_q.put_nowait(input_buf)
            self._out_q.put_nowait([])

    def _encode_thread(self) -> None:
        GROW_FACTOR = 1.5
        INV_INT16 = 1.0 / 32768.0

        assert self._output_path is not None
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        container = av.open(self._output_path, mode="w", format="ogg")
        stream: av.AudioStream = container.add_stream(
            "opus", rate=self._sample_rate, layout="stereo"
        )  # type: ignore

        in_resampler: rtc.AudioResampler | None = None
        out_resampler: rtc.AudioResampler | None = None

        capacity = self._sample_rate * 6  # 6s, 1ch
        stereo_buf = np.zeros((2, capacity), dtype=np.float32)

        def remix_and_resample(frames: list[rtc.AudioFrame], channel_idx: int) -> int:
            total_samples = sum(f.samples_per_channel * f.num_channels for f in frames)

            nonlocal capacity, stereo_buf
            if total_samples > capacity:
                while capacity < total_samples:
                    capacity = int(capacity * GROW_FACTOR)

                stereo_buf.resize((2, capacity), refcheck=False)

            pos = 0
            dest = stereo_buf[channel_idx]
            for f in frames:
                count = f.samples_per_channel * f.num_channels
                arr_i16 = np.frombuffer(f.data, dtype=np.int16, count=count).reshape(
                    -1, f.num_channels
                )
                slice_ = dest[pos : pos + f.samples_per_channel]
                np.sum(arr_i16, axis=1, dtype=np.float32, out=slice_)
                slice_ *= INV_INT16 / f.num_channels
                pos += f.samples_per_channel

            return pos

        with container:
            while True:
                input_buf = self._in_q.get()
                output_buf = self._out_q.get()

                if input_buf is None or output_buf is None:
                    break

                # lazy creation of the resamplers
                if in_resampler is None and len(input_buf):
                    input_rate, num_channels = input_buf[0].sample_rate, input_buf[0].num_channels
                    in_resampler = rtc.AudioResampler(
                        input_rate=input_rate,
                        output_rate=self._sample_rate,
                        num_channels=num_channels,
                    )

                if out_resampler is None and len(output_buf):
                    input_rate, num_channels = output_buf[0].sample_rate, output_buf[0].num_channels
                    out_resampler = rtc.AudioResampler(
                        input_rate=input_rate,
                        output_rate=self._sample_rate,
                        num_channels=num_channels,
                    )

                input_resampled = []
                for frame in input_buf:
                    assert in_resampler is not None
                    input_resampled.extend(in_resampler.push(frame))

                output_resampled = []
                for frame in output_buf:
                    assert out_resampler is not None
                    output_resampled.extend(out_resampler.push(frame))

                if output_buf:
                    assert out_resampler is not None
                    # the output is sent per-segment. Always flush when the playback is done
                    output_resampled.extend(out_resampler.flush())

                len_left = remix_and_resample(input_resampled, 0)
                len_right = remix_and_resample(output_resampled, 1)

                if len_left != len_right:
                    diff = abs(len_right - len_left)
                    if len_left < len_right:
                        if not self._skip_padding_warning:
                            logger.warning(
                                f"Input is shorter by {diff} samples; silence has been prepended to "
                                "align the input channel. The resulting recording may not accurately "
                                "reflect the original audio. This is expected if the input device "
                                "or audio input is disabled. This warning will only be shown once."
                            )
                            self._skip_padding_warning = True

                        stereo_buf[0, diff : diff + len_left] = stereo_buf[0, :len_left]
                        stereo_buf[0, :diff] = 0.0
                        len_left = len_right
                    else:
                        stereo_buf[1, diff : diff + len_right] = stereo_buf[1, :len_right]
                        stereo_buf[1, :diff] = 0.0
                        len_right = len_left

                max_len = max(len_left, len_right)
                if max_len <= 0:
                    continue

                stereo_slice = stereo_buf[:, :max_len]
                av_frame = av.AudioFrame.from_ndarray(stereo_slice, format="fltp", layout="stereo")
                av_frame.sample_rate = self._sample_rate

                for packet in stream.encode(av_frame):
                    container.mux(packet)

            for packet in stream.encode(None):
                container.mux(packet)

        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._close_fut.set_result, None)


class RecorderAudioInput(io.AudioInput):
    def __init__(self, *, recording_io: RecorderIO, source: io.AudioInput) -> None:
        super().__init__(label="RecorderIO", source=source)
        self.__audio_input = source
        self.__recording_io = recording_io
        self.__acc_frames: list[rtc.AudioFrame] = []
        self.__started_time: None | float = None
        self.__padded: bool = False

    @property
    def started_wall_time(self) -> float | None:
        return self.__started_time

    def take_buf(self, pad_since: float | None = None) -> list[rtc.AudioFrame]:
        frames = self.__acc_frames
        self.__acc_frames = []
        if (
            pad_since
            and self.__started_time
            and (padding := self.__started_time - pad_since) > 0
            and not self.__padded
            and len(frames) > 0
        ):
            logger.warning(
                "input speech started after last agent speech ended",
                extra={
                    "last_agent_speech_time": pad_since,
                    "input_started_time": self.__started_time,
                },
            )
            self.__padded = True
            frames = [
                _create_silence_frame(padding, frames[0].sample_rate, frames[0].num_channels),
                *frames,
            ]
        # we could pad with silence here with some fixed SR and channels,
        # but it's better for the user to know that this is happening
        elif pad_since and self.__started_time is None and not self.__padded and not frames:
            logger.warning(
                "input speech hasn't started yet, skipping silence padding, "
                "recording may be inaccurate until the speech starts"
            )

        return frames

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame]:
        return self

    async def __anext__(self) -> rtc.AudioFrame:
        frame = await self.__audio_input.__anext__()

        if self.__recording_io.recording:
            if self.__started_time is None:
                self.__started_time = time.time()

            self.__acc_frames.append(frame)

        return frame


class RecorderAudioOutput(io.AudioOutput):
    def __init__(
        self,
        *,
        recording_io: RecorderIO,
        audio_output: io.AudioOutput | None = None,
        write_fnc: Callable[[list[rtc.AudioFrame]], Any],
    ) -> None:
        super().__init__(
            label="RecorderIO",
            next_in_chain=audio_output,
            sample_rate=audio_output.sample_rate if audio_output else None,
            # TODO: support pause
            capabilities=io.AudioOutputCapabilities(pause=True),  # depends on the next_in_chain
        )
        self.__recording_io = recording_io
        self.__write = write_fnc
        self.__acc_frames: list[rtc.AudioFrame] = []
        self.__started_time: None | float = None
        self._last_speech_end_time: None | float = None
        self._last_speech_start_time: None | float = None

        # pause tracking
        self.__current_pause_start: float | None = None
        self.__pause_wall_times: list[tuple[float, float]] = []

    @property
    def started_wall_time(self) -> float | None:
        return self.__started_time

    @property
    def recorder_io(self) -> RecorderIO:
        return self.__recording_io

    @property
    def has_pending_data(self) -> bool:
        return len(self.__acc_frames) > 0

    def pause(self) -> None:
        """Pause playback and record the wall time."""
        if self.__current_pause_start is None and self.__recording_io.recording:
            self.__current_pause_start = time.time()

        if self.next_in_chain:
            self.next_in_chain.pause()

    def resume(self) -> None:
        """Resume playback and record the pause interval."""
        if self.__current_pause_start is not None and self.__recording_io.recording:
            self.__pause_wall_times.append((self.__current_pause_start, time.time()))
            self.__current_pause_start = None

        if self.next_in_chain:
            self.next_in_chain.resume()

    def _reset_pause_state(self) -> None:
        """Reset all pause tracking state."""
        self.__current_pause_start = None
        self.__pause_wall_times = []

    def on_playback_finished(
        self,
        *,
        playback_position: float,
        interrupted: bool,
        synchronized_transcript: str | None = None,
    ) -> None:
        finish_time = self.__current_pause_start or time.time()
        trailing_silence_duration = max(0.0, time.time() - finish_time)

        if self._last_speech_start_time is None:
            logger.warning(
                "playback finished before speech started",
                extra={
                    "finish_time": finish_time,
                    "playback_position": playback_position,
                    "interrupted": interrupted,
                },
            )
            playback_position = 0.0

        playback_position = max(
            0.0,
            min(
                finish_time - (self._last_speech_start_time or 0.0),
                playback_position,
            ),
        )

        super().on_playback_finished(
            playback_position=playback_position,
            interrupted=interrupted,
            synchronized_transcript=synchronized_transcript,
        )

        if not self.__recording_io.recording:
            return

        if self.__current_pause_start is not None:
            self.__pause_wall_times.append((self.__current_pause_start, finish_time))
            self.__current_pause_start = None

        if not self.__acc_frames:
            self._reset_pause_state()
            self._last_speech_end_time = time.time()
            self._last_speech_start_time = None
            return

        pause_events: deque[tuple[float, float]] = deque()  # (position, duration)
        playback_start_time = finish_time - playback_position
        if self.__pause_wall_times:
            total_pause_duration = sum(end - start for start, end in self.__pause_wall_times)
            playback_start_time = finish_time - playback_position - total_pause_duration

            accumulated_pause = 0.0
            for pause_start, pause_end in self.__pause_wall_times:
                position = (pause_start - playback_start_time) - accumulated_pause
                duration = pause_end - pause_start
                position = max(0.0, min(position, playback_position))
                pause_events.append((position, duration))
                accumulated_pause += duration

        buf: list[rtc.AudioFrame] = []
        acc_dur = 0.0
        sample_rate = self.__acc_frames[0].sample_rate
        num_channels = self.__acc_frames[0].num_channels

        should_break = False
        for frame in self.__acc_frames:
            if frame.duration + acc_dur > playback_position:
                frame, _ = _split_frame(frame, playback_position - acc_dur)
                should_break = True

            # process any pauses before this frame starts
            while pause_events and pause_events[0][0] <= acc_dur:
                pause_pos, pause_dur = pause_events.popleft()
                buf.append(_create_silence_frame(pause_dur, sample_rate, num_channels))

            # process any pauses within this frame
            while pause_events and pause_events[0][0] < acc_dur + frame.duration:
                pause_pos, pause_dur = pause_events.popleft()
                left, frame = _split_frame(frame, pause_pos - acc_dur)
                buf.append(left)
                acc_dur += left.duration
                buf.append(_create_silence_frame(pause_dur, sample_rate, num_channels))

            buf.append(frame)
            acc_dur += frame.duration

            if should_break:
                break

        while pause_events:
            pause_pos, pause_dur = pause_events.popleft()
            if pause_pos <= playback_position:
                buf.append(_create_silence_frame(pause_dur, sample_rate, num_channels))

        if buf:
            if trailing_silence_duration > 0.0:
                buf.append(
                    _create_silence_frame(trailing_silence_duration, sample_rate, num_channels)
                )
            self.__write(buf)

        self.__acc_frames = []
        self._reset_pause_state()
        self._last_speech_end_time = time.time()
        self._last_speech_start_time = None

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        if self.next_in_chain:
            await self.next_in_chain.capture_frame(frame)

        await super().capture_frame(frame)

        if self.__recording_io.recording:
            self.__acc_frames.append(frame)

        if self.__started_time is None:
            self.__started_time = time.time()

        if self._last_speech_start_time is None:
            self._last_speech_start_time = time.time()

    def flush(self) -> None:
        super().flush()

        if self.next_in_chain:
            self.next_in_chain.flush()

    def clear_buffer(self) -> None:
        if self.next_in_chain:
            self.next_in_chain.clear_buffer()


def _create_silence_frame(duration: float, sample_rate: int, num_channels: int) -> rtc.AudioFrame:
    samples = int(duration * sample_rate)
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples * num_channels,
        num_channels=num_channels,
        samples_per_channel=samples,
        sample_rate=sample_rate,
    )


def _split_frame(frame: rtc.AudioFrame, position: float) -> tuple[rtc.AudioFrame, rtc.AudioFrame]:
    if position <= 0.0:
        return rtc.AudioFrame(
            data=b"",
            num_channels=frame.num_channels,
            samples_per_channel=0,
            sample_rate=frame.sample_rate,
        ), frame

    if position >= frame.duration:
        return frame, rtc.AudioFrame(
            data=b"",
            num_channels=frame.num_channels,
            samples_per_channel=0,
            sample_rate=frame.sample_rate,
        )

    samples_needed = int(position * frame.sample_rate)
    bytes_per_sample = frame.num_channels * ctypes.sizeof(ctypes.c_int16)

    data_x, data_y = (
        frame.data[: samples_needed * bytes_per_sample],
        frame.data[samples_needed * bytes_per_sample :],
    )

    return (
        rtc.AudioFrame(
            data=data_x,
            num_channels=frame.num_channels,
            samples_per_channel=len(data_x) // bytes_per_sample,
            sample_rate=frame.sample_rate,
        ),
        rtc.AudioFrame(
            data=data_y,
            num_channels=frame.num_channels,
            samples_per_channel=len(data_y) // bytes_per_sample,
            sample_rate=frame.sample_rate,
        ),
    )
