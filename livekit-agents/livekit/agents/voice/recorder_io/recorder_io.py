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

# the recorder currently assume the input is a continous uninterrupted audio stream


class RecorderIO:
    def __init__(
        self,
        agent_session: AgentSession,
    ) -> None:
        self._in_q = queue.Queue[list[rtc.AudioFrame] | None]()
        self._out_q = queue.Queue[list[rtc.AudioFrame] | None]()

        self._session = agent_session

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

    def record_output(
        self, next_in_chain: io.AudioOutput | None = None, sample_rate: int | None = None
    ) -> "RecorderAudioOutput":
        self._out_record = RecorderAudioOutput(
            next_in_chain=next_in_chain, sample_rate=sample_rate, write_fnc=self._on_output_write
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
        f = open("test.ogg", "wb")
        container = av.open(f, mode="w", format="ogg")
        stream: av.AudioStream = container.add_stream("opus", rate=48000, layout="stereo")  # type: ignore

        in_resampler: rtc.AudioResampler | None = None
        out_resampler: rtc.AudioResampler | None = None

        while True:
            input_buf = self._in_q.get()
            output_buf = self._out_q.get()

            if input_buf is None or output_buf is None:
                break

            # resample at the same sample_rate of the file output
            if in_resampler is None and len(input_buf):
                in_resampler = rtc.AudioResampler(
                    input_rate=input_buf[0].sample_rate,
                    output_rate=48000,
                    num_channels=input_buf[0].num_channels,
                )

            if out_resampler is None and len(output_buf):
                out_resampler = rtc.AudioResampler(
                    input_rate=output_buf[0].sample_rate,
                    output_rate=48000,
                    num_channels=output_buf[0].num_channels,
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
                output_resampled.extend(out_resampler.flush())

            def frames_to_mono_array(frames: list[rtc.AudioFrame]) -> np.ndarray:
                arrays: list[np.ndarray] = []
                for f in frames:
                    arr = np.frombuffer(f.data, dtype=np.int16)
                    arr = arr.reshape(-1, f.num_channels)
                    mono = arr.mean(axis=1).astype(np.int16)
                    arrays.append(mono)
                return np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.int16)

            left = frames_to_mono_array(input_resampled)
            right = frames_to_mono_array(output_resampled)

            max_len = max(left.shape[0], right.shape[0])
            if left.shape[0] < max_len:
                left = np.pad(left, (max_len - left.shape[0], 0))
            if right.shape[0] < max_len:
                right = np.pad(right, (max_len - right.shape[0], 0))

            stereo = np.stack((left, right), axis=0)

            frame = av.AudioFrame.from_ndarray(stereo, format="s16p", layout="stereo")
            frame.sample_rate = 48000
            for packet in stream.encode(frame):
                container.mux(packet)
                f.flush()

        for packet in stream.encode(None):
            container.mux(packet)
            f.flush()

        container.close()
        f.close()


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
