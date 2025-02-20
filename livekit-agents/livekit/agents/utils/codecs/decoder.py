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

"""
Audio stream decoding utilities for real-time processing pipelines.
Provides chunk-based decoding of various audio formats into PCM frames.

Features:
- Thread-safe buffering for producer/consumer patterns
- Async/Await compatible output
- Automatic audio resampling to 16-bit mono PCM
- PyAV (FFmpeg) backend for broad format support

Typical use cases:
- WebSocket audio stream processing
- Real-time transcription services
- Voice command processing
"""

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

    Usage:
        buffer = StreamBuffer()
        # Writer thread
        buffer.write(b"audio data")
        # Reader thread
        data = buffer.read(1024)
    """

    def __init__(self):
        self._buffer = io.BytesIO()
        self._lock = threading.Lock()
        self._data_available = threading.Condition(self._lock)
        self._eof = False  # EOF flag to signal no more writes

    def write(self, data: bytes):
        """Write data to the buffer from a writer thread.
        
        Args:
            data: Bytes to append to the buffer
        """
        with self._data_available:  # Lock and notify readers
            self._buffer.seek(0, io.SEEK_END)  # Move to the end
            self._buffer.write(data)
            self._data_available.notify_all()  # Notify waiting readers

    def read(self, size: int = -1) -> bytes:
        """Read data from the buffer in a reader thread.
        
        Args:
            size: Number of bytes to read (-1 for all available)
            
        Returns:
            bytes: Read data chunk, empty bytes on EOF
        """
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
        """Close the buffer and release resources."""
        self._buffer.close()


class AudioStreamDecoder:
    """Real-time audio stream decoder with async output.
    
    Handles continuous audio streams with chunked input and PCM output.
    
    Usage:
        decoder = AudioStreamDecoder()
        
        # Producer thread
        decoder.push(audio_chunk)
        
        # Consumer async loop
        async for frame in decoder:
            process_frame(frame)
            
        # Signal end of input
        decoder.end_input()
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
        """Add new audio data to the decoding pipeline.
        
        Args:
            chunk: Raw audio bytes in supported format (e.g. MP3, AAC)
            
        Note: 
            Automatically starts decoding thread on first call
        """
        self._input_buf.write(chunk)
        if not self._started:
            self._started = True
            self._loop.run_in_executor(None, self._decode_loop)

    def end_input(self):
        """Signal end of input stream and flush remaining data."""
        self._input_buf.end_input()

    def _decode_loop(self):
        """Main decoding loop running in background thread."""
        container = av.open(self._input_buf)
        audio_stream = next(s for s in container.streams if s.type == "audio")
        resampler = av.AudioResampler(
            # Convert to LiveKit-compatible format:
            format="s16",  # Signed 16-bit PCM
            layout="mono", # Mono audio
            rate=audio_stream.rate,  # Original sample rate
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
        """Async iterator interface for consuming decoded frames."""
        return self

    async def __anext__(self) -> rtc.AudioFrame:
        """Get next decoded audio frame."""
        if self._output_finished and self._output_ch.empty():
            raise StopAsyncIteration
        return await self._output_ch.__anext__()

    async def aclose(self):
        """Clean up resources and stop decoding."""
        if self._closed:
            return
        self._closed = True
        self._input_buf.close()
        self._output_ch.close()
