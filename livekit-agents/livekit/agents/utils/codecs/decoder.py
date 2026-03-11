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
import enum
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

    _COMPACT_THRESHOLD = 5 * 1024 * 1024  # compact after 5MB consumed

    def __init__(self) -> None:
        self._bio = io.BytesIO()
        self._lock = threading.Lock()
        self._data_available = threading.Condition(self._lock)
        self._eof = False
        self._closed = False
        self._write_pos = 0
        self._read_pos = 0

    def write(self, data: bytes) -> None:
        """Write data to the buffer from a writer thread."""
        with self._data_available:
            self._bio.seek(self._write_pos)
            self._bio.write(data)
            self._write_pos = self._bio.tell()
            self._data_available.notify_all()

    def read(self, size: int = -1) -> bytes:
        """Read data from the buffer in a reader thread."""
        if size == 0:
            return b""

        with self._data_available:
            while True:
                if self._closed:
                    return b""

                available = self._write_pos - self._read_pos
                if available > 0:
                    self._bio.seek(self._read_pos)
                    if size < 0:
                        data = self._bio.read(available)
                    else:
                        data = self._bio.read(min(size, available))
                    self._read_pos = self._bio.tell()

                    if self._read_pos >= self._COMPACT_THRESHOLD:
                        remaining = self._bio.read()
                        self._bio = io.BytesIO(remaining)
                        self._bio.seek(0, io.SEEK_END)
                        self._write_pos = self._bio.tell()
                        self._read_pos = 0

                    return data if data else b""

                if self._eof:
                    return b""

                self._data_available.wait()

    def end_input(self) -> None:
        """Signal that no more data will be written."""
        with self._data_available:
            self._eof = True
            self._data_available.notify_all()

    def close(self) -> None:
        with self._data_available:
            self._closed = True
            self._data_available.notify_all()
            self._bio.close()


class _WavState(enum.IntEnum):
    RIFF_HEADER = 0
    CHUNK_HEADER = 1
    FMT_DATA = 2
    SKIP_CHUNK_DATA = 3
    STREAMING = 4


class _WavInlineDecoder:
    """Incremental WAV decoder that runs entirely on the event loop (no thread).

    Processes WAV bytes via a state machine:
    RIFF_HEADER → CHUNK_HEADER → FMT_DATA/SKIP_CHUNK_DATA → STREAMING.
    Once in STREAMING state, subsequent push() calls feed bytes directly to
    AudioByteStream → optional resampler → output channel.

    Each push() may contain a complete WAV file (with its own headers). When a
    new RIFF header is detected while already streaming, the current stream is
    flushed and the state machine resets to parse the new file's headers.
    """

    _RIFF_MAGIC = b"RIFF"

    def __init__(
        self,
        output_ch: aio.Chan[rtc.AudioFrame],
        sample_rate: int | None,
    ) -> None:
        self._output_ch = output_ch
        self._sample_rate = sample_rate

        self._state = _WavState.RIFF_HEADER
        self._hdr_buf = bytearray()
        self._need = 12  # bytes needed for current header state
        self._skip_remaining = 0
        self._chunk_size = 0

        # set after fmt is parsed
        self._bstream: AudioByteStream | None = None
        self._resampler: rtc.AudioResampler | None = None
        self._wave_channels = 0
        self._wave_rate = 0

    def push(self, data: bytes) -> None:
        if self._state == _WavState.STREAMING:
            if len(data) >= 4 and data[:4] == self._RIFF_MAGIC:
                self._flush_current()
                self._reset_state()
            else:
                self._push_pcm(data)
                return

        buf = memoryview(data)
        pos = 0
        while pos < len(buf):
            if self._state == _WavState.RIFF_HEADER:
                pos = self._consume_riff(buf, pos)
            elif self._state == _WavState.CHUNK_HEADER:
                pos = self._consume_chunk_header(buf, pos)
            elif self._state == _WavState.FMT_DATA:
                pos = self._consume_fmt_data(buf, pos)
            elif self._state == _WavState.SKIP_CHUNK_DATA:
                pos = self._consume_skip(buf, pos)
            elif self._state == _WavState.STREAMING:
                # remainder after headers goes straight to PCM path
                self._push_pcm(bytes(buf[pos:]))
                return

    def flush(self) -> None:
        self._flush_current()

    def _flush_current(self) -> None:
        """Flush AudioByteStream and resampler for the current WAV segment."""
        if self._bstream is not None:
            remaining = self._bstream.flush()
            if self._resampler is not None:
                for frame in remaining:
                    for resampled in self._resampler.push(frame):
                        self._output_ch.send_nowait(resampled)
                for frame in self._resampler.flush():
                    if frame.samples_per_channel > 0:
                        self._output_ch.send_nowait(frame)
            else:
                for frame in remaining:
                    self._output_ch.send_nowait(frame)

    def _reset_state(self) -> None:
        """Reset the state machine to parse a new WAV file."""
        self._state = _WavState.RIFF_HEADER
        self._hdr_buf.clear()
        self._need = 12
        self._skip_remaining = 0
        self._chunk_size = 0
        self._bstream = None
        self._resampler = None
        self._wave_channels = 0
        self._wave_rate = 0

    # -- state handlers -------------------------------------------------------

    def _consume_riff(self, buf: memoryview, pos: int) -> int:
        take = min(self._need - len(self._hdr_buf), len(buf) - pos)
        self._hdr_buf.extend(buf[pos : pos + take])
        pos += take
        if len(self._hdr_buf) < self._need:
            return pos

        if self._hdr_buf[:4] != b"RIFF" or self._hdr_buf[8:12] != b"WAVE":
            raise ValueError(f"Invalid WAV file: missing RIFF/WAVE: {bytes(self._hdr_buf)!r}")
        self._hdr_buf.clear()
        self._need = 8
        self._state = _WavState.CHUNK_HEADER
        return pos

    def _consume_chunk_header(self, buf: memoryview, pos: int) -> int:
        take = min(self._need - len(self._hdr_buf), len(buf) - pos)
        self._hdr_buf.extend(buf[pos : pos + take])
        pos += take
        if len(self._hdr_buf) < self._need:
            return pos

        chunk_id, chunk_size = struct.unpack("<4sI", bytes(self._hdr_buf[:8]))
        self._hdr_buf.clear()
        self._chunk_size = chunk_size

        if chunk_id == b"fmt ":
            self._need = chunk_size
            self._state = _WavState.FMT_DATA
        elif chunk_id == b"data":
            self._init_streaming()
            self._state = _WavState.STREAMING
        else:
            self._skip_remaining = chunk_size
            self._state = _WavState.SKIP_CHUNK_DATA
        return pos

    def _consume_fmt_data(self, buf: memoryview, pos: int) -> int:
        take = min(self._need - len(self._hdr_buf), len(buf) - pos)
        self._hdr_buf.extend(buf[pos : pos + take])
        pos += take
        if len(self._hdr_buf) < self._need:
            return pos

        fmt = bytes(self._hdr_buf[: self._chunk_size])
        audio_format, channels, rate = struct.unpack("<HHI", fmt[:8])
        if len(fmt) >= 16:
            bits_per_sample = struct.unpack("<H", fmt[14:16])[0]
            if bits_per_sample != 16:
                raise ValueError(
                    f"Unsupported WAV bits per sample: {bits_per_sample}"
                    " (only 16-bit PCM supported)"
                )
        if audio_format != 1:
            raise ValueError(f"Unsupported WAV audio format: {audio_format}")

        self._wave_channels = channels
        self._wave_rate = rate
        self._hdr_buf.clear()
        self._need = 8
        self._state = _WavState.CHUNK_HEADER
        return pos

    def _consume_skip(self, buf: memoryview, pos: int) -> int:
        take = min(self._skip_remaining, len(buf) - pos)
        self._skip_remaining -= take
        pos += take
        if self._skip_remaining == 0:
            self._hdr_buf.clear()
            self._need = 8
            self._state = _WavState.CHUNK_HEADER
        return pos

    # -- streaming helpers ----------------------------------------------------

    def _init_streaming(self) -> None:
        if self._wave_rate == 0:
            raise ValueError("Invalid WAV file: data chunk before fmt chunk")

        self._bstream = AudioByteStream(
            sample_rate=self._wave_rate, num_channels=self._wave_channels
        )
        if self._sample_rate is not None and self._sample_rate != self._wave_rate:
            self._resampler = rtc.AudioResampler(
                input_rate=self._wave_rate,
                output_rate=self._sample_rate,
                num_channels=self._wave_channels,
            )

    def _push_pcm(self, data: bytes) -> None:
        assert self._bstream is not None
        if self._resampler is not None:
            for frame in self._bstream.push(data):
                for resampled in self._resampler.push(frame):
                    self._output_ch.send_nowait(resampled)
        else:
            for frame in self._bstream.push(data):
                self._output_ch.send_nowait(frame)


class AudioStreamDecoder:
    """A class that can be used to decode audio stream into PCM AudioFrames.

    Decoders are stateful, and it should not be reused across multiple streams. Each decoder
    is designed to decode a single stream.
    """

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
        self._is_wav = self._av_format == "wav"

        self._output_ch = aio.Chan[rtc.AudioFrame]()
        self._closed = False
        self._started = False
        self._loop = asyncio.get_event_loop()

        # lazy-initialized only for non-WAV codecs
        self._input_buf: StreamBuffer | None = None
        self._executor: ThreadPoolExecutor | None = None

        # lazy-initialized only for WAV
        self._wav_decoder: _WavInlineDecoder | None = None

    def push(self, chunk: bytes) -> None:
        if self._is_wav:
            if self._wav_decoder is None:
                self._wav_decoder = _WavInlineDecoder(self._output_ch, self._sample_rate)
            try:
                self._wav_decoder.push(chunk)
            except Exception:
                if not self._closed:
                    logger.exception("error decoding WAV audio")
                    self._output_ch.close()
                    self._closed = True
                return
            self._started = True
            return

        if self._input_buf is None:
            self._input_buf = StreamBuffer()
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="AudioDecoder")
        self._input_buf.write(chunk)
        if not self._started:
            self._started = True
            self._loop.run_in_executor(self._executor, self._decode_loop)

    def end_input(self) -> None:
        if self._is_wav:
            if self._wav_decoder is not None and not self._closed:
                try:
                    self._wav_decoder.flush()
                except Exception:
                    logger.exception("error flushing WAV audio")
            if not self._closed:
                self._output_ch.close()
            return

        if self._input_buf is not None:
            self._input_buf.end_input()
        if not self._started:
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
                    self._emit_av_frame(f)

            # flush the resampler to get any remaining buffered samples
            if resampler and not self._closed:
                for f in resampler.resample(None):
                    self._emit_av_frame(f)

        except Exception:
            logger.exception("error decoding audio")
        finally:
            self._loop.call_soon_threadsafe(self._output_ch.close)
            if container:
                container.close()

    def _emit_av_frame(self, f: av.AudioFrame) -> None:
        self._loop.call_soon_threadsafe(
            self._output_ch.send_nowait,
            rtc.AudioFrame(
                data=f.to_ndarray().tobytes(),
                num_channels=len(f.layout.channels),
                sample_rate=int(f.sample_rate),
                samples_per_channel=f.samples,
            ),
        )

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame]:
        return self

    async def __anext__(self) -> rtc.AudioFrame:
        return await self._output_ch.__anext__()

    async def aclose(self) -> None:
        if self._closed:
            return

        self.end_input()
        self._closed = True

        if self._input_buf is not None:
            self._input_buf.close()

        if not self._started:
            return

        async for _ in self._output_ch:
            pass

        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
