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
import io
from typing import AsyncIterator

from livekit.agents.utils import aio

try:
    # preload to ensure faster startup
    import av  # noqa
except ImportError:
    pass
import threading

from livekit import rtc


class StreamBuffer:
    """
    A thread-safe buffer that behaves like an IO stream.
    Allows writing from one thread and reading from another.
    """

    def __init__(self):
        self._buffer = io.BytesIO()
        self._lock = threading.Lock()
        self._data_available = threading.Condition(self._lock)
        self._eof = False  # EOF flag to signal no more writes

    def write(self, data: bytes):
        """Write data to the buffer from a writer thread."""
        with self._data_available:  # Lock and notify readers
            self._buffer.seek(0, io.SEEK_END)  # Move to the end
            self._buffer.write(data)
            self._data_available.notify_all()  # Notify waiting readers

    def read(self, size: int = -1) -> bytes:
        """Read data from the buffer in a reader thread."""

        if self._buffer.closed:
            return b""

        with self._data_available:
            while True:
                self._buffer.seek(0)  # Rewind for reading
                data = self._buffer.read(size)

                # If data is available, return it
                if data:
                    # Shrink the buffer to remove already-read data
                    remaining = self._buffer.read()
                    self._buffer = io.BytesIO(remaining)
                    return data

                # If EOF is signaled and no data remains, return EOF
                if self._eof:
                    return b""

                # Wait for more data
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

    def __init__(self):
        try:
            import av  # noqa
        except ImportError:
            raise ImportError(
                "You haven't included the 'codecs' optional dependencies. Please install the 'codecs' extra by running `pip install livekit-agents[codecs]`"
            )

        self._output_ch = aio.Chan[rtc.AudioFrame]()
        self._closed = False
        self._started = False
        self._output_finished = False
        self._input_buf = StreamBuffer()
        self._loop = asyncio.get_event_loop()

    def push(self, chunk: bytes):
        self._input_buf.write(chunk)
        if not self._started:
            self._started = True
            self._loop.run_in_executor(None, self._decode_loop)

    def end_input(self):
        self._input_buf.end_input()

    def _decode_loop(self):
        container = av.open(self._input_buf)
        audio_stream = next(s for s in container.streams if s.type == "audio")
        resampler = av.AudioResampler(
            # convert to signed 16-bit little endian
            format="s16",
            layout="mono",
            rate=audio_stream.rate,
        )
        try:
            # TODO: handle error where audio stream isn't found
            if not audio_stream:
                return
            for frame in container.decode(audio_stream):
                if self._closed:
                    return
                for resampled_frame in resampler.resample(frame):
                    nchannels = len(resampled_frame.layout.channels)
                    data = resampled_frame.to_ndarray().tobytes()
                    self._output_ch.send_nowait(
                        rtc.AudioFrame(
                            data=data,
                            num_channels=nchannels,
                            sample_rate=resampled_frame.sample_rate,
                            samples_per_channel=resampled_frame.samples / nchannels,
                        )
                    )
        finally:
            self._output_finished = True

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame]:
        return self

    async def __anext__(self) -> rtc.AudioFrame:
        if self._output_finished and self._output_ch.empty():
            raise StopAsyncIteration
        return await self._output_ch.__anext__()

    async def aclose(self):
        if self._closed:
            return
        self._closed = True
        self._input_buf.close()
        self._output_ch.close()
