from __future__ import annotations

import asyncio
import contextlib
import math
import threading
import time
from collections import deque
from typing import Callable

import numpy as np

from livekit import rtc
from livekit.agents import utils
from livekit.agents.voice.agent_session import AgentSession

from ...log import logger
from .. import io
from .recorder_io import RecorderIO

WRITE_INTERVAL = 0.1


class StereoAudioRecorder(RecorderIO):
    def __init__(
        self,
        *,
        agent_session: AgentSession,
        sample_rate: int = 16000,
        capacity: int = 30,
        inference_callback: Callable[[np.ndarray], None] | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        super().__init__(agent_session=agent_session, sample_rate=sample_rate, loop=loop)
        self._in_record: UserAudioInput | None = None  # type: ignore[assignment]
        self._out_record: AgentAudioOutput | None = None  # type: ignore[assignment]

        self._thread_lock = threading.Lock()
        # number of agent speech samples collected for the current turn
        self._agent_speech_written: int = 0
        self._capacity = sample_rate * capacity
        self._inference_callback = inference_callback

    @property
    def started(self) -> bool:
        return self._started

    def start_agent_speech(self, start_time: float) -> None:
        if self._out_record is None:
            return
        self._out_record.speech_started_at = start_time
        self._out_record._speech_taken = 0.0

    def end_agent_speech(self) -> None:
        if self._out_record is None:
            return
        self._out_record.speech_started_at = None
        self._out_record._speech_taken = 0.0
        self.reset()

    def record_input(self, audio_input: io.AudioInput) -> UserAudioInput:  # type: ignore[override]
        self._in_record = UserAudioInput(audio_recorder=self, source=audio_input)
        return self._in_record

    def record_output(self, audio_output: io.AudioOutput) -> AgentAudioOutput:  # type: ignore[override]
        self._out_record = AgentAudioOutput(audio_recorder=self, audio_output=audio_output)
        return self._out_record

    @utils.log_exceptions(logger=logger)
    async def _forward_task(self) -> None:
        assert self._in_record is not None
        assert self._out_record is not None

        while True:
            await asyncio.sleep(WRITE_INTERVAL)
            delta = self._out_record.playback_delta(speed_factor=1.0)
            input_buf = self._in_record.take_buf()
            output_buf = self._out_record.take_buf(delta)
            self._in_q.put_nowait(input_buf)
            self._out_q.put_nowait(output_buf)

    @utils.log_exceptions(logger=logger)
    def _encode_thread(self) -> None:
        INV_INT16 = 1.0 / 32768.0

        in_resampler: rtc.AudioResampler | None = None
        out_resampler: rtc.AudioResampler | None = None

        capacity = self._capacity
        stereo_buf = np.zeros((2, capacity), dtype=np.float32)

        def remix_and_resample(frames: list[rtc.AudioFrame], channel_idx: int) -> int:
            total_output_samples = sum(f.samples_per_channel for f in frames)

            nonlocal capacity, stereo_buf
            dest = stereo_buf[channel_idx]

            if total_output_samples >= capacity:
                stop_pos = 0
            else:
                stop_pos = capacity - total_output_samples
                # shift the existing buffer to the left to make room for the new audio
                # this makes sure the audio is continuous
                dest[:stop_pos] = dest[-stop_pos:]

            written = 0
            end_pos = len(dest)
            # process the audio frames in reverse order to ensure channel alignment
            # |--------silence--------|----agent speech----|
            # |----------------------------|--user speech--|
            for f in frames[::-1]:
                count = f.samples_per_channel * f.num_channels
                arr_i16 = np.frombuffer(f.data, dtype=np.int16, count=count).reshape(
                    -1, f.num_channels
                )
                start_pos = max(stop_pos, end_pos - f.samples_per_channel)
                if (chunk_size := end_pos - start_pos) <= 0:
                    break
                slice_ = dest[start_pos:end_pos]
                written += chunk_size
                np.sum(arr_i16[-chunk_size:, :], axis=1, dtype=np.float32, out=slice_)
                slice_ *= INV_INT16 / f.num_channels
                end_pos = start_pos

            return written

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

            with self._thread_lock:
                # reset buffer it the agent just starts speaking
                if self._agent_speech_written == 0 and output_buf:
                    stereo_buf[:, :] = 0.0

                # agent speech on channel 0, user speech on channel 1
                speech_sample_written = remix_and_resample(output_resampled, 0)
                _ = remix_and_resample(input_resampled, 1)
                self._agent_speech_written += speech_sample_written
                # trim silence from both channels based on agent speech
                if self._agent_speech_written < capacity:
                    inp = stereo_buf[:, -self._agent_speech_written :]
                else:
                    inp = stereo_buf

            if speech_sample_written > 0 and self._inference_callback is not None:
                self._loop.call_soon_threadsafe(self._inference_callback, inp)

        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._close_fut.set_result, None)

    def reset(self) -> None:
        with self._thread_lock:
            self._agent_speech_written = 0


class UserAudioInput(io.AudioInput):
    def __init__(self, *, audio_recorder: StereoAudioRecorder, source: io.AudioInput) -> None:
        super().__init__(
            label="UserAudioInput",
            source=source,
        )
        self.__audio_input = source
        self.__recorder = audio_recorder
        self.__acc_frames: list[rtc.AudioFrame] = []
        self._started_at: float | None = None

    def take_buf(self) -> list[rtc.AudioFrame]:
        frames = self.__acc_frames
        self.__acc_frames = []
        return frames

    async def __anext__(self) -> rtc.AudioFrame:
        frame = await self.__audio_input.__anext__()
        if self._started_at is None:
            # This is used for transcript synchronization
            self._started_at = time.time()

        if self.__recorder.started:
            self.__acc_frames.append(frame)

        return frame

    def on_attached(self) -> None:
        self.__audio_input.on_attached()

    def on_detached(self) -> None:
        self.__audio_input.on_detached()


class AgentAudioOutput(io.AudioOutput):
    def __init__(
        self,
        *,
        audio_recorder: StereoAudioRecorder,
        audio_output: io.AudioOutput | None = None,
    ) -> None:
        super().__init__(
            label="AgentAudioOutput",
            next_in_chain=audio_output,
            sample_rate=audio_output.sample_rate if audio_output is not None else None,
            capabilities=io.AudioOutputCapabilities(pause=True),
        )
        self.__recorder = audio_recorder
        self.__acc_frames: deque[rtc.AudioFrame] = deque()

        self._speech_started_at: float | None = None
        self._speech_taken: float = 0

    @property
    def speech_started_at(self) -> float | None:
        return self._speech_started_at

    @speech_started_at.setter
    def speech_started_at(self, value: float | None) -> None:
        self._speech_started_at = value

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)

        if self.__recorder.started:
            self.__acc_frames.append(frame)

        if self.next_in_chain:
            await self.next_in_chain.capture_frame(frame)

    @utils.log_exceptions(logger=logger)
    def take_buf(self, duration: float) -> list[rtc.AudioFrame]:
        """Take agent audio frames from the buffer up to the given duration."""
        buf = []
        acc_dur = 0.0
        while self.__acc_frames:
            frame = self.__acc_frames.popleft()
            if (duration_needed := duration - acc_dur) > 0:
                samples_needed = math.ceil(duration_needed * frame.sample_rate * frame.num_channels)
                left, right = self._split_frame(frame, samples_needed)
                acc_dur += left.duration
                self._speech_taken += left.duration
                buf.append(left)
                if right is not None and len(right.data) > 0:
                    self.__acc_frames.appendleft(right)
                break

            acc_dur += frame.duration
            self._speech_taken += frame.duration
            buf.append(frame)

        return buf

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

        if not self.__recorder.started:
            return

        self._reset()

    def _reset(self) -> None:
        self.__acc_frames.clear()
        self._speech_started_at = None
        self._speech_taken = 0.0

    @staticmethod
    def _split_frame(
        frame: rtc.AudioFrame, split_at: int
    ) -> tuple[rtc.AudioFrame, rtc.AudioFrame | None]:
        """Split the frame at the given sample position."""
        split_at -= split_at % frame.num_channels
        if len(frame.data) <= split_at:
            return frame, None

        return rtc.AudioFrame(
            data=frame.data[:split_at],
            num_channels=frame.num_channels,
            samples_per_channel=split_at // frame.num_channels,
            sample_rate=frame.sample_rate,
        ), rtc.AudioFrame(
            data=frame.data[split_at:],
            num_channels=frame.num_channels,
            samples_per_channel=len(frame.data[split_at:]) // frame.num_channels,
            sample_rate=frame.sample_rate,
        )

    def playback_delta(self, speed_factor: float) -> float:
        if self.speech_started_at is None:
            return 0.0

        return (time.time() - self.speech_started_at - self._speech_taken) * speed_factor

    def flush(self) -> None:
        super().flush()

        if self.next_in_chain:
            self.next_in_chain.flush()

    def clear_buffer(self) -> None:
        if self.next_in_chain:
            self.next_in_chain.clear_buffer()
