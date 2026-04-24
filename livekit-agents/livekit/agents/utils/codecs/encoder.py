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
import queue
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import av
import av.audio
import av.container

from livekit import rtc

from ...log import logger
from .. import aio


@dataclass
class EncodedAudioData:
    """Data returned by the encoder."""

    data: bytes
    """The encoded audio data."""
    num_samples: int
    """The number of samples in the encoded audio data,
    useful if the receiver needs to buffer audio data based on duration without decoding."""

    def as_tuple(self) -> tuple[bytes, int]:
        return (self.data, self.num_samples)


_CODEC_TABLE: dict[str, tuple[str, str]] = {
    "opus": ("libopus", "ogg"),
    "mp3": ("libmp3lame", "mp3"),
    "pcm": ("pcm_s16le", "wav"),
}

_SUPPORTED_CODECS = Literal["opus", "mp3", "pcm"]

_CLOSE = object()
_FLUSH = object()


def _resolve_codec(codec: str) -> tuple[str, str]:
    """Return (av_encoder_name, container_format) for a public codec name."""
    if codec not in _CODEC_TABLE:
        raise ValueError(f"unsupported codec for streaming encode: {codec!r}")
    return _CODEC_TABLE[codec]


class AudioStreamEncoder:
    """Encode PCM AudioFrames into a compressed audio byte stream.

    Encoding is performed on a dedicated background thread.  Call
    ``push`` to queue frames (non-blocking), and iterate with
    ``async for`` to receive ``EncodedAudioData`` pages.

    A ``None`` value yielded by the iterator indicates a flush point
    (see ``flush``).
    """

    def __init__(
        self,
        *,
        codec: _SUPPORTED_CODECS = "opus",
        sample_rate: int = 48000,
        num_channels: int = 1,
        bit_rate: int = 24000,
        codec_options: dict[str, str] | None = None,
    ) -> None:
        """Create an encoder.

        ``codec_options`` is an optional mapping of libav private codec AVOptions
        (e.g. ``{"application": "lowdelay", "frame_duration": "60",
        "compression_level": "0", "vbr": "on"}`` for a low-latency libopus profile).
        Values must be strings.
        """
        self._codec = codec
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._bit_rate = bit_rate
        self._codec_options = codec_options

        self._output_ch = aio.Chan[EncodedAudioData | None]()
        self._input_q: queue.Queue[rtc.AudioFrame | object] = queue.Queue()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="AudioEncoder")
        self._started = False
        self._closed = False
        self._loop = asyncio.get_event_loop()

    def push(self, frame: rtc.AudioFrame) -> None:
        """Queue a PCM frame for encoding.  Returns immediately."""
        if self._closed:
            raise RuntimeError("encoder is closed")

        self._input_q.put(frame)
        if not self._started:
            self._started = True
            self._loop.run_in_executor(self._executor, self._encode_loop)

    def flush(self) -> None:
        """Insert a flush marker.

        The iterator will yield ``None`` once all frames pushed before
        this call have been encoded and emitted.
        """
        self._input_q.put(_FLUSH)

    def end_input(self) -> None:
        """Signal that no more frames will be pushed."""
        self._input_q.put(_CLOSE)
        if not self._started:
            self._output_ch.close()

    def _encode_loop(self) -> None:
        layout = "mono" if self._num_channels == 1 else "stereo"
        av_codec, container_format = _resolve_codec(self._codec)

        output_buf = _OutputBuffer()
        container: av.container.OutputContainer = av.open(
            output_buf, mode="w", format=container_format
        )
        stream: av.audio.AudioStream = container.add_stream(  # type: ignore[assignment]
            av_codec, rate=self._sample_rate, layout=layout, options=self._codec_options
        )
        stream.bit_rate = self._bit_rate
        pending_samples = 0

        def _emit() -> None:
            nonlocal pending_samples
            data = output_buf.drain()
            if data:
                enc = EncodedAudioData(data=data, num_samples=pending_samples)
                pending_samples = 0
                self._loop.call_soon_threadsafe(self._output_ch.send_nowait, enc)

        try:
            while True:
                item = self._input_q.get()
                if item is _CLOSE:
                    break
                if item is _FLUSH:
                    _emit()
                    self._loop.call_soon_threadsafe(self._output_ch.send_nowait, None)
                    continue

                frame: rtc.AudioFrame = item  # type: ignore[assignment]
                av_frame = av.AudioFrame(
                    format="s16", layout=layout, samples=frame.samples_per_channel
                )
                av_frame.rate = self._sample_rate
                av_frame.planes[0].update(bytes(frame.data))

                for packet in stream.encode(av_frame):
                    container.mux(packet)
                pending_samples += frame.samples_per_channel
                _emit()

            for packet in stream.encode(None):
                container.mux(packet)
            container.close()
            _emit()
        except Exception:
            logger.exception("error encoding audio")
        finally:
            self._loop.call_soon_threadsafe(self._output_ch.close)

    def __aiter__(self) -> AsyncIterator[EncodedAudioData | None]:
        return self

    async def __anext__(self) -> EncodedAudioData | None:
        return await self._output_ch.__anext__()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.end_input()

        async for _ in self._output_ch:
            pass

        self._executor.shutdown(wait=False, cancel_futures=True)


class _OutputBuffer(io.RawIOBase):
    """Capture buffer for PyAV's muxer output.

    PyAV writes encoded bytes here via ``write()``, and the encode loop
    drains them with ``drain()``.  All access is from the encode thread.
    """

    _COMPACT_THRESHOLD = 5 * 1024 * 1024

    def __init__(self) -> None:
        super().__init__()
        self._buf = bytearray()
        self._read_pos = 0

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def write(self, b) -> int:  # type: ignore[no-untyped-def]
        self._buf.extend(b)
        return len(b)

    def drain(self) -> bytes:
        if self._read_pos >= len(self._buf):
            return b""
        data = bytes(self._buf[self._read_pos :])
        self._read_pos = len(self._buf)
        if self._read_pos >= self._COMPACT_THRESHOLD:
            self._buf.clear()
            self._read_pos = 0
        return data
