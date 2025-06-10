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

from __future__ import annotations

import asyncio
import io
import struct
import threading
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import av
import av.container

from livekit import rtc

from ...log import logger
from .. import aio
from ..audio import AudioByteStream


def _mime_to_av_format(mime: str | None) -> str | None:
    """Return the libav *container* short‑name for a given MIME‑type.

    If *mime* is *None* or not recognised, return *None* so that PyAV will
    fall back to auto‑detection.
    """

    if not mime:
        return None

    mime = mime.lower()
    _TABLE: dict[str, str] = {
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/x-mpeg": "mp3",
        "audio/aac": "aac",
        "audio/x-aac": "aac",
        "audio/flac": "flac",
        "audio/x-flac": "flac",
        "audio/wav": "wav",
        "audio/wave": "wav",
        "audio/x-wav": "wav",
        "audio/opus": "ogg",
        "audio/ogg": "ogg",
        "audio/webm": "webm",
        "audio/mp4": "mp4",
    }
    return _TABLE.get(mime)


class StreamBuffer:
    """
    A thread-safe buffer that behaves like an IO stream.
    Allows writing from one thread and reading from another.
    """

    def __init__(self) -> None:
        self._buffer = io.BytesIO()
        self._lock = threading.Lock()
        self._data_available = threading.Condition(self._lock)
        self._eof = False

    def write(self, data: bytes) -> None:
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

    def end_input(self) -> None:
        """Signal that no more data will be written."""
        with self._data_available:
            self._eof = True
            self._data_available.notify_all()

    def close(self) -> None:
        self._buffer.close()


class AudioStreamDecoder:
    """A class that can be used to decode audio stream into PCM AudioFrames.

    Decoders are stateful, and it should not be reused across multiple streams. Each decoder
    is designed to decode a single stream.
    """

    _max_workers: int = 10
    _executor: ThreadPoolExecutor | None = None

    def __init__(
        self,
        *,
        sample_rate: int | None = 48000,
        num_channels: int | None = 1,
        format: str | None = None,
    ):
        self._sample_rate = sample_rate

        self._layout = "mono"
        if num_channels == 2:
            self._layout = "stereo"

        self._mime_type = format.lower() if format else None
        self._av_format = _mime_to_av_format(self._mime_type)

        self._output_ch = aio.Chan[rtc.AudioFrame]()
        self._closed = False
        self._started = False
        self._input_buf = StreamBuffer()
        self._loop = asyncio.get_event_loop()

        if self.__class__._executor is None:
            # each decoder instance will submit jobs to the shared pool
            self.__class__._executor = ThreadPoolExecutor(max_workers=self.__class__._max_workers)

    def push(self, chunk: bytes) -> None:
        self._input_buf.write(chunk)
        if not self._started:
            self._started = True
            target = self._decode_wav_loop if self._av_format == "wav" else self._decode_loop
            self._loop.run_in_executor(self.__class__._executor, target)

    def end_input(self) -> None:
        self._input_buf.end_input()
        if not self._started:
            # if no data was pushed, close the output channel
            self._output_ch.close()

    def _decode_loop(self) -> None:
        container: av.container.InputContainer | None = None
        resampler: av.AudioResampler | None = None
        try:
            # open container in low-latency streaming mode
            container = av.open(
                self._input_buf,
                mode="r",
                format=self._av_format,
                buffer_size=256,
                options={
                    "probesize": "32",
                    "analyzeduration": "0",
                    "fflags": "nobuffer+flush_packets",
                    "flags": "low_delay",
                    "reorder_queue_size": "0",
                    "max_delay": "0",
                    "avioflags": "direct",
                },
            )
            # explicitly disable internal buffering flags on the FFmpeg container
            container.flags |= cast(
                int, av.container.Flags.no_buffer.value | av.container.Flags.flush_packets.value
            )

            if len(container.streams.audio) == 0:
                raise ValueError("no audio stream found")

            audio_stream = container.streams.audio[0]

            # Set up resampler only if needed
            if self._sample_rate is not None or self._layout is not None:
                resampler = av.AudioResampler(
                    format="s16", layout=self._layout, rate=self._sample_rate
                )

            for frame in container.decode(audio_stream):
                if self._closed:
                    return

                if resampler:
                    frames = resampler.resample(frame)
                else:
                    frames = [frame]

                for f in frames:
                    nchannels = len(f.layout.channels)
                    self._loop.call_soon_threadsafe(
                        self._output_ch.send_nowait,
                        rtc.AudioFrame(
                            data=f.to_ndarray().tobytes(),
                            num_channels=nchannels,
                            sample_rate=int(f.sample_rate),
                            samples_per_channel=int(f.samples / nchannels),
                        ),
                    )

        except Exception:
            logger.exception("error decoding audio")
        finally:
            self._loop.call_soon_threadsafe(self._output_ch.close)
            if container:
                container.close()

    def _decode_wav_loop(self) -> None:
        """Decode wav data from the buffer without ffmpeg, parse header and emit PCM frames.

        This can be much faster than using ffmpeg, as we are emitting frames as quickly as possible.
        """

        try:
            # parse RIFF header
            header = b""
            while len(header) < 12:
                chunk = self._input_buf.read(12 - len(header))
                if not chunk:
                    raise ValueError("Invalid WAV file: incomplete header")
                header += chunk
            if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                raise ValueError(f"Invalid WAV file: missing RIFF/WAVE: {header!r}")

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
            resampler = (
                rtc.AudioResampler(
                    input_rate=wave_rate, output_rate=self._sample_rate, num_channels=wave_channels
                )
                if self._sample_rate is not None
                else None
            )

            def resample_and_push(frame: rtc.AudioFrame) -> None:
                if not resampler:
                    self._loop.call_soon_threadsafe(self._output_ch.send_nowait, frame)
                    return

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

    async def aclose(self) -> None:
        if self._closed:
            return

        self.end_input()
        self._closed = True
        self._input_buf.close()

        if not self._started:
            return

        async for _ in self._output_ch:
            pass
