from collections.abc import AsyncIterator
from typing import Callable, Any
from livekit import rtc
import av
from livekit.agents.voice.agent_session import AgentSession
import numpy as np
import threading
import asyncio
import ctypes
import time
import queue

from ...utils import aio
from .. import io
from ...log import logger

# the recorder currently assume the input is a continous uninterrupted audio stream


class RecorderIO:
    def __init__(
        self,
        *,
        agent_session: AgentSession,
        sample_rate: int = 48000,
    ) -> None:
        self._in_q = queue.Queue[list[rtc.AudioFrame] | None]()
        self._out_q = queue.Queue[list[rtc.AudioFrame] | None]()
        self._session = agent_session

        self._sample_rate = sample_rate

    async def start(self) -> None:
        self._forward_atask = asyncio.create_task(self._forward_task())

        thread = threading.Thread(target=self._encode_thread)
        thread.start()

        # TODO: remove hacky
        @self._session.on("close")
        def test():
            self._in_q.put_nowait(None)
            self._out_q.put_nowait(None)

            thread.join()

    async def aclose(self) -> None:
        pass

    def record_input(self, audio_input: io.AudioInput) -> "RecorderAudioInput":
        self._in_record = RecorderAudioInput(audio_input)
        return self._in_record

    def record_output(self, audio_output: io.AudioOutput) -> "RecorderAudioOutput":
        self._out_record = RecorderAudioOutput(
            next_in_chain=audio_output, sample_rate=None, write_fnc=self._on_output_write
        )
        return self._out_record

    def _on_output_write(self, buf: list[rtc.AudioFrame]):
        input_buf = self._in_record.take_buf()
        self._in_q.put_nowait(input_buf)
        self._out_q.put_nowait(buf)

    async def _forward_task(self) -> None:
        while True:
            await asyncio.sleep(5.0)
            if self._out_record.has_pending_data:
                continue  # always wait for the complete output

            input_buf = self._in_record.take_buf()
            self._in_q.put_nowait(input_buf)
            self._out_q.put_nowait([])

    def _encode_thread(self) -> None:
        GROW_FACTOR = 1.5
        INV_INT16 = 1.0 / 32768.0

        container = av.open("test.ogg", mode="w", format="ogg")
        stream: av.AudioStream = container.add_stream(
            "opus", rate=self._sample_rate, layout="stereo"
        )  # type: ignore

        in_resampler: rtc.AudioResampler | None = None
        out_resampler: rtc.AudioResampler | None = None

        capacity = self._sample_rate * 6  # 6s, 1ch
        stereo_buf = np.zeros((2, capacity), dtype=np.float32)

        def remix_and_resample(frames: list[rtc.AudioFrame], channel_idx: int):
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
                        logger.warning(
                            f"Input is shorter by {diff} samples; silence has been prepended to "
                            "align the input channel. The resulting recording may not accurately "
                            "reflect the original audio."
                        )
                        stereo_buf[0, diff : diff + len_left] = stereo_buf[0, :len_left]
                        stereo_buf[0, :diff] = 0.0
                        len_left = len_right
                    else:
                        stereo_buf[1, diff : diff + len_right] = stereo_buf[1, :len_right]
                        stereo_buf[1, :diff] = 0.0
                        len_right = len_left

                max_len = max(len_left, len_right)
                stereo_slice = stereo_buf[:, :max_len]

                frame = av.AudioFrame.from_ndarray(stereo_slice, format="fltp", layout="stereo")
                frame.sample_rate = self._sample_rate

                for packet in stream.encode(frame):
                    container.mux(packet)

            for packet in stream.encode(None):
                container.mux(packet)


class RecorderAudioInput(io.AudioInput):
    def __init__(self, audio_input: io.AudioInput) -> None:
        super().__init__()
        self._audio_input = audio_input
        self._lock = threading.Lock()
        self.__acc_frames: list[rtc.AudioFrame] = []

    def take_buf(self) -> list[rtc.AudioFrame]:
        frames = self.__acc_frames
        self.__acc_frames = []
        return frames

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame]:
        return self

    async def __anext__(self) -> rtc.AudioFrame:
        frame = await self._audio_input.__anext__()
        self.__acc_frames.append(frame)
        return frame

    def on_attached(self) -> None: ...

    def on_detached(self) -> None: ...


class RecorderAudioOutput(io.AudioOutput):
    def __init__(
        self,
        *,
        next_in_chain: io.AudioOutput | None = None,
        sample_rate: int | None = None,
        write_fnc: Callable[[list[rtc.AudioFrame]], Any],
    ) -> None:
        super().__init__(next_in_chain=next_in_chain, sample_rate=sample_rate)
        self.__write = write_fnc
        self.__acc_frames: list[rtc.AudioFrame] = []

    @property
    def has_pending_data(self) -> bool:
        return len(self.__acc_frames) > 0

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

        buf = []
        acc_dur = 0.0
        for frame in self.__acc_frames:
            if frame.duration + acc_dur > playback_position:
                duration_needed = playback_position - acc_dur
                samples_needed = int(duration_needed * frame.sample_rate) * frame.num_channels
                truncated_frame = rtc.AudioFrame(
                    data=frame.data[:samples_needed],
                    num_channels=frame.num_channels,
                    samples_per_channel=samples_needed,
                    sample_rate=frame.sample_rate,
                )
                buf.append(truncated_frame)
                break

            acc_dur += frame.duration
            buf.append(frame)

        if buf:
            self.__write(buf)

        self.__acc_frames = []

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        self.__acc_frames.append(frame)

        if self._next_in_chain:
            await self._next_in_chain.capture_frame(frame)

    def flush(self) -> None:
        super().flush()

        if self._next_in_chain:
            self._next_in_chain.flush()

    def clear_buffer(self) -> None:
        super().clear_buffer()

        if self._next_in_chain:
            self._next_in_chain.clear_buffer()
