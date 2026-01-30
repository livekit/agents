# Copyright 2026 LiveKit, Inc.
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
import threading
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import av
import numpy as np

from livekit import rtc

from ...log import logger
from .. import aio

AudioEncoding = Literal["pcm", "opus", "mp3"]

# Opus only supports specific sample rates
OPUS_SUPPORTED_SAMPLE_RATES = [8000, 12000, 16000, 24000, 48000]


@dataclass
class EncodedAudioChunk:
    """A chunk of encoded audio data."""

    data: bytes
    """The encoded audio bytes."""

    duration: float
    """Duration of this chunk in seconds."""


class FrameBuffer:
    """
    A thread-safe buffer for audio frames.
    Allows writing from one thread and reading from another.
    """

    def __init__(self) -> None:
        self._frames: list[rtc.AudioFrame] = []
        self._lock = threading.Lock()
        self._data_available = threading.Condition(self._lock)
        self._eof = False

    def write(self, frame: rtc.AudioFrame) -> None:
        """Write a frame to the buffer from a writer thread."""
        with self._data_available:
            self._frames.append(frame)
            self._data_available.notify_all()

    def read_all(self) -> list[rtc.AudioFrame]:
        """Read all available frames from the buffer."""
        with self._data_available:
            while True:
                if self._frames:
                    frames = self._frames
                    self._frames = []
                    return frames

                if self._eof:
                    return []

                self._data_available.wait()

    def end_input(self) -> None:
        """Signal that no more data will be written."""
        with self._data_available:
            self._eof = True
            self._data_available.notify_all()

    @property
    def is_eof(self) -> bool:
        with self._lock:
            return self._eof and not self._frames


class AudioStreamEncoder:
    """A class that can be used to encode PCM AudioFrames into various formats.

    Encoders are stateful, and should not be reused across multiple streams. Each encoder
    is designed to encode a single stream.

    Supported formats:
        - "pcm": Raw 16-bit signed PCM (passthrough, no encoding)
        - "opus": Opus codec in OGG container
        - "mp3": MP3 format

    Example:
        ```python
        encoder = AudioStreamEncoder(
            sample_rate=48000,
            num_channels=1,
            format="opus",
            bitrate=64000,
        )

        # Push audio frames
        encoder.push(audio_frame1)
        encoder.push(audio_frame2)
        encoder.end_input()

        # Consume encoded chunks
        async for chunk in encoder:
            # chunk.data contains the encoded bytes
            # chunk.duration contains the chunk duration
            pass

        await encoder.aclose()
        ```
    """

    def __init__(
        self,
        *,
        sample_rate: int = 48000,
        num_channels: int = 1,
        format: AudioEncoding = "pcm",
        bitrate: int | None = None,
    ):
        """Initialize an AudioStreamEncoder.

        Args:
            sample_rate: Target sample rate for encoding. For opus, will be adjusted
                to the nearest supported rate if not already supported.
            num_channels: Number of audio channels (1 for mono, 2 for stereo).
            format: Output format - "pcm", "opus", or "mp3".
            bitrate: Target bitrate in bits/second. Only used for opus and mp3.
                Defaults to 64000 for opus, 128000 for mp3.
        """
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._format = format

        if bitrate is None:
            if format == "opus":
                bitrate = 64000
            elif format == "mp3":
                bitrate = 128000

        self._bitrate = bitrate

        self._output_ch = aio.Chan[EncodedAudioChunk]()
        self._closed = False
        self._started = False
        self._input_buf = FrameBuffer()
        self._loop = asyncio.get_event_loop()

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="AudioEncoder")

    @property
    def sample_rate(self) -> int:
        """The target sample rate for encoding."""
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        """The number of audio channels."""
        return self._num_channels

    @property
    def format(self) -> AudioEncoding:
        """The output encoding format."""
        return self._format

    @property
    def bitrate(self) -> int | None:
        """The target bitrate for encoding (opus/mp3 only)."""
        return self._bitrate

    def push(self, frame: rtc.AudioFrame) -> None:
        """Push an audio frame to be encoded.

        Args:
            frame: An rtc.AudioFrame containing PCM audio data.
        """
        if self._closed:
            raise RuntimeError("Cannot push to a closed encoder")

        self._input_buf.write(frame)
        if not self._started:
            self._started = True
            if self._format == "pcm":
                self._loop.run_in_executor(self._executor, self._encode_pcm_loop)
            else:
                self._loop.run_in_executor(self._executor, self._encode_loop)

    def end_input(self) -> None:
        """Signal that no more frames will be pushed.

        This must be called to properly finalize the encoded stream.
        """
        self._input_buf.end_input()
        if not self._started:
            # if no data was pushed, close the output channel
            self._output_ch.close()

    def _encode_pcm_loop(self) -> None:
        """Simple passthrough for PCM - just emit frames as-is."""
        try:
            while True:
                frames = self._input_buf.read_all()
                if not frames:
                    if self._input_buf.is_eof:
                        break
                    continue

                for frame in frames:
                    if self._closed:
                        return

                    chunk = EncodedAudioChunk(
                        data=bytes(frame.data),
                        duration=frame.samples_per_channel / frame.sample_rate,
                    )
                    self._loop.call_soon_threadsafe(self._output_ch.send_nowait, chunk)

        except Exception:
            logger.exception("error in PCM encoding loop")
        finally:
            self._loop.call_soon_threadsafe(self._output_ch.close)

    def _encode_loop(self) -> None:
        """Encode audio frames using PyAV (FFmpeg)."""
        container: av.container.OutputContainer | None = None
        resampler: rtc.AudioResampler | None = None

        try:
            if self._format == "opus":
                container_format = "ogg"
                codec = "libopus"
                encode_rate = self._get_opus_sample_rate(self._sample_rate)
            elif self._format == "mp3":
                container_format = "mp3"
                codec = "libmp3lame"
                encode_rate = self._sample_rate
            else:
                raise ValueError(f"Unsupported format: {self._format}")

            output_buffer = io.BytesIO()
            container = av.open(output_buffer, mode="w", format=container_format)

            layout = "mono" if self._num_channels == 1 else "stereo"
            stream: av.AudioStream = container.add_stream(codec, rate=encode_rate)
            if self._bitrate:
                stream.bit_rate = self._bitrate
            stream.layout = layout

            # Track position for incremental reads
            last_read_pos = 0
            pts = 0

            def emit_encoded_data(duration: float) -> None:
                """Read and emit any new encoded data from the buffer."""
                nonlocal last_read_pos

                current_pos = output_buffer.tell()
                if current_pos > last_read_pos:
                    output_buffer.seek(last_read_pos)
                    data = output_buffer.read(current_pos - last_read_pos)
                    last_read_pos = current_pos
                    output_buffer.seek(current_pos)

                    if data:
                        chunk = EncodedAudioChunk(data=data, duration=duration)
                        self._loop.call_soon_threadsafe(self._output_ch.send_nowait, chunk)

            while True:
                frames = self._input_buf.read_all()
                if not frames:
                    if self._input_buf.is_eof:
                        break
                    continue

                for frame in frames:
                    if self._closed:
                        return

                    frames_to_encode: list[rtc.AudioFrame] = [frame]
                    if frame.sample_rate != encode_rate or frame.num_channels != self._num_channels:
                        if resampler is None:
                            resampler = rtc.AudioResampler(
                                input_rate=frame.sample_rate,
                                output_rate=encode_rate,
                                num_channels=frame.num_channels,
                            )
                        frames_to_encode = list(resampler.push(frame))

                    for f in frames_to_encode:
                        # PyAV s16 format expects shape (1, N*channels) with interleaved data
                        pcm_array = np.frombuffer(f.data, dtype=np.int16).reshape(1, -1)

                        av_frame = av.AudioFrame.from_ndarray(
                            pcm_array, format="s16", layout=layout
                        )
                        av_frame.sample_rate = encode_rate
                        av_frame.pts = pts
                        pts += f.samples_per_channel

                        for packet in stream.encode(av_frame):
                            container.mux(packet)

                        emit_encoded_data(f.samples_per_channel / f.sample_rate)

            if resampler:
                for f in resampler.flush():
                    pcm_array = np.frombuffer(f.data, dtype=np.int16).reshape(1, -1)

                    av_frame = av.AudioFrame.from_ndarray(pcm_array, format="s16", layout=layout)
                    av_frame.sample_rate = encode_rate
                    av_frame.pts = pts
                    pts += f.samples_per_channel

                    for packet in stream.encode(av_frame):
                        container.mux(packet)

                    emit_encoded_data(f.samples_per_channel / f.sample_rate)

            for packet in stream.encode(None):
                container.mux(packet)

            container.close()
            container = None

            current_pos = output_buffer.tell()
            if current_pos > last_read_pos:
                output_buffer.seek(last_read_pos)
                data = output_buffer.read()
                if data:
                    chunk = EncodedAudioChunk(data=data, duration=0.0)
                    self._loop.call_soon_threadsafe(self._output_ch.send_nowait, chunk)

        except Exception:
            logger.exception("error encoding audio")
        finally:
            self._loop.call_soon_threadsafe(self._output_ch.close)
            if container:
                container.close()

    def _get_opus_sample_rate(self, sample_rate: int) -> int:
        """Get the nearest Opus-supported sample rate."""
        if sample_rate in OPUS_SUPPORTED_SAMPLE_RATES:
            return sample_rate
        return min(OPUS_SUPPORTED_SAMPLE_RATES, key=lambda x: abs(x - sample_rate))

    def __aiter__(self) -> AsyncIterator[EncodedAudioChunk]:
        return self

    async def __anext__(self) -> EncodedAudioChunk:
        return await self._output_ch.__anext__()

    async def aclose(self) -> None:
        """Close the encoder and clean up resources.

        This will wait for any pending encoding to complete.
        """
        if self._closed:
            return

        self.end_input()
        self._closed = True

        if not self._started:
            return

        async for _ in self._output_ch:
            pass

        self._executor.shutdown(wait=False, cancel_futures=True)
