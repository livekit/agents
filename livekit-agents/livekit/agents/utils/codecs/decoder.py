# Copyright 2024 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import contextlib
import io
import struct
import threading
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import av
import av.container

from livekit import rtc
from livekit.agents.log import logger
from livekit.agents.utils import aio


class StreamBuffer:
    """
    A thread-safe buffer that behaves like an IO stream.
    Allows writing from one thread and reading from another.
    """

    def __init__(self):
        self._buffer = io.BytesIO()
        self._lock = threading.Lock()
        self._data_available = threading.Condition(self._lock)
        self._eof = False

    def write(self, data: bytes):
        """Write data to the buffer from a writer thread."""
        with self._data_available:
            self._buffer.seek(0, io.SEEK_END)
            self._buffer.write(data)
            self._data_available.notify_all()

    def read(self, size: int = -1) -> bytes:
        """Read data from the buffer in a reader thread."""

        if self._buffer.closed:
            return b""

        with self._data_available:
            while True:
                if self._buffer.closed:
                    return b""
                # always read from beginning
                self._buffer.seek(0)
                data = self._buffer.read(size)

                if data:
                    # shrink the buffer to remove already-read data
                    remaining = self._buffer.read()
                    self._buffer = io.BytesIO(remaining)
                    return data

                if self._eof:
                    return b""

                self._data_available.wait()

    def end_input(self):
        """Signal that no more data will be written."""
        with self._data_available:
            self._eof = True
            self._data_available.notify_all()

    def close(self):
        self._buffer.close()


class AudioStreamDecoder:
    """A class that can be used to decode audio stream into PCM AudioFrames.

    Decoders are stateful, and it should not be reused across multiple streams. Each decoder
    is designed to decode a single stream.
    """

    _max_workers: int = 10
    _executor: Optional[ThreadPoolExecutor] = None

    def __init__(
        self, *, sample_rate: int = 48000, num_channels: int = 1, format: Optional[str] = None
    ):
        self._sample_rate = sample_rate
        self._layout = "mono"
        if num_channels == 2:
            self._layout = "stereo"
        elif num_channels != 1:
            raise ValueError(f"Invalid number of channels: {num_channels}")
        self._format = format.lower() if format else None

        self._output_ch = aio.Chan[rtc.AudioFrame]()
        self._closed = False
        self._started = False
        self._input_buf = StreamBuffer()
        self._loop = asyncio.get_event_loop()

        if self.__class__._executor is None:
            # each decoder instance will submit jobs to the shared pool
            self.__class__._executor = ThreadPoolExecutor(max_workers=self.__class__._max_workers)

    def push(self, chunk: bytes):
        self._input_buf.write(chunk)
        if not self._started:
            self._started = True
            # choose decode loop based on format
            if self._format == "wav":
                target = self._decode_wav_loop
            else:
                target = self._decode_loop
            self._loop.run_in_executor(self.__class__._executor, target)

    def end_input(self):
        self._input_buf.end_input()
        if not self._started:
            # if no data was pushed, close the output channel
            self._output_ch.close()

    def _decode_loop(self):
        container: av.container.InputContainer | None = None
        resampler: av.AudioResampler | None = None
        try:
            # open container in low-latency streaming mode
            container = av.open(
                self._input_buf,
                mode="r",
                buffer_size=1024,
                options={
                    "fflags": "nobuffer+flush_packets",
                    "probesize": "32",
                    "analyzeduration": "0",
                    "max_delay": "0",
                },
            )
            # explicitly disable internal buffering flags on the FFmpeg container
            container.flags |= (
                av.container.Flags.no_buffer.value | av.container.Flags.flush_packets.value
            )
            if len(container.streams.audio) == 0:
                raise ValueError("no audio stream found")

            audio_stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="s16", layout=self._layout, rate=self._sample_rate)

            for frame in container.decode(audio_stream):
                if self._closed:
                    return

                for resampled_frame in resampler.resample(frame):
                    nchannels = len(resampled_frame.layout.channels)
                    self._loop.call_soon_threadsafe(
                        self._output_ch.send_nowait,
                        rtc.AudioFrame(
                            data=resampled_frame.to_ndarray().tobytes(),
                            num_channels=nchannels,
                            sample_rate=int(resampled_frame.sample_rate),
                            samples_per_channel=int(resampled_frame.samples / nchannels),
                        ),
                    )

        except Exception:
            logger.exception("error decoding audio")
        finally:
            self._loop.call_soon_threadsafe(self._output_ch.close)
            if container:
                container.close()

    def _decode_wav_loop(self):
        """Decode wav data from the buffer without ffmpeg, parse header and emit PCM frames.

        This can be much faster than using ffmpeg, as we are emitting frames as quickly as possible.
        """

        try:
            from livekit.agents.utils.audio import AudioByteStream

            # parse RIFF header
            header = b""
            while len(header) < 12:
                chunk = self._input_buf.read(12 - len(header))
                if not chunk:
                    raise ValueError("Invalid WAV file: incomplete header")
                header += chunk
            if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                raise ValueError(f"Invalid WAV file: missing RIFF/WAVE: {header}")

            # parse fmt chunk
            while True:
                sub_header = self._input_buf.read(8)
                if len(sub_header) < 8:
                    raise ValueError("Invalid WAV file: incomplete fmt chunk header")
                chunk_id, chunk_size = struct.unpack("<4sI", sub_header)
                data = b""
                remaining = chunk_size
                while remaining > 0:
                    part = self._input_buf.read(min(1024, remaining))
                    if not part:
                        raise ValueError("Invalid WAV file: incomplete fmt chunk data")
                    data += part
                    remaining -= len(part)
                if chunk_id == b"fmt ":
                    audio_format, wave_channels, wave_rate, _, _, bits_per_sample = struct.unpack(
                        "<HHIIHH", data[:16]
                    )
                    if audio_format != 1:
                        raise ValueError(f"Unsupported WAV audio format: {audio_format}")
                    break

            # parse data chunk
            while True:
                sub_header = self._input_buf.read(8)
                if len(sub_header) < 8:
                    raise ValueError("Invalid WAV file: incomplete data chunk header")
                chunk_id, chunk_size = struct.unpack("<4sI", sub_header)
                if chunk_id == b"data":
                    break

                # skip chunk data
                to_skip = chunk_size
                while to_skip > 0:
                    skipped = self._input_buf.read(min(1024, to_skip))
                    if not skipped:
                        raise ValueError("Invalid WAV file: incomplete chunk while seeking data")
                    to_skip -= len(skipped)

            # now ready to decode
            bstream = AudioByteStream(sample_rate=wave_rate, num_channels=wave_channels)
            resampler = rtc.AudioResampler(
                input_rate=wave_rate, output_rate=self._sample_rate, num_channels=wave_channels
            )

            def resample_and_push(frame: rtc.AudioFrame):
                for resampled_frame in resampler.push(frame):
                    self._loop.call_soon_threadsafe(
                        self._output_ch.send_nowait,
                        resampled_frame,
                    )

            while True:
                chunk = self._input_buf.read(1024)
                if not chunk:
                    break
                frames = bstream.push(chunk)
                for rtc_frame in frames:
                    resample_and_push(rtc_frame)

            for rtc_frame in bstream.flush():
                resample_and_push(rtc_frame)
        except Exception:
            logger.exception("error decoding wav")
        finally:
            self._loop.call_soon_threadsafe(self._output_ch.close)

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame]:
        return self

    async def __anext__(self) -> rtc.AudioFrame:
        return await self._output_ch.__anext__()

    async def aclose(self):
        if self._closed:
            return

        self.end_input()
        self._closed = True
        self._input_buf.close()
        # wait for decode loop to finish, only if anything's been pushed
        with contextlib.suppress(aio.ChanClosed):
            if self._started:
                await self._output_ch.recv()
