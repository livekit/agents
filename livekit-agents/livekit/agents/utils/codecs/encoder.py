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

import io
from dataclasses import dataclass
from typing import Literal

import av
import av.audio
import av.container

from livekit import rtc


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


def _resolve_codec(codec: str) -> tuple[str, str]:
    """Return (av_encoder_name, container_format) for a public codec name."""
    if codec not in _CODEC_TABLE:
        raise ValueError(f"unsupported codec for streaming encode: {codec!r}")
    return _CODEC_TABLE[codec]


class _CompactableBuffer(io.RawIOBase):
    _COMPACT_THRESHOLD = 5 * 1024 * 1024  # reclaim consumed prefix once it exceeds 5MB

    def __init__(self) -> None:
        super().__init__()
        self._buf = bytearray()
        self._read_pos = 0

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        """disable back-patching container headers"""
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


class AudioStreamEncoder:
    """Encode PCM AudioFrames into a compressed audio byte stream."""

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
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._layout = "mono" if num_channels == 1 else "stereo"

        av_codec, container_format = _resolve_codec(codec)

        self._output_buf = _CompactableBuffer()
        self._container: av.container.OutputContainer = av.open(
            self._output_buf,
            mode="w",
            format=container_format,
        )
        self._stream: av.audio.AudioStream = self._container.add_stream(  # type: ignore[assignment]
            av_codec,
            rate=sample_rate,
            layout=self._layout,
            options=codec_options,
        )
        self._stream.bit_rate = bit_rate

        self._closed = False
        self._pending_samples = 0

    def _flush(self) -> EncodedAudioData:
        data = self._output_buf.drain()
        if not data:
            return EncodedAudioData(data=b"", num_samples=0)

        num_samples = self._pending_samples
        self._pending_samples = 0
        return EncodedAudioData(data=data, num_samples=num_samples)

    def push(self, frame: rtc.AudioFrame) -> EncodedAudioData:
        """Encode a PCM audio frame.

        Returns any new encoded bytes produced by the muxer (may be empty when
        the container hasn't flushed a full page yet).  The very first call
        includes container headers (e.g. OGG OpusHead / OpusTags) if not empty.
        """
        if self._closed:
            raise RuntimeError("encoder is closed")

        av_frame = av.AudioFrame(
            format="s16",
            layout=self._layout,
            samples=frame.samples_per_channel,
        )
        av_frame.rate = self._sample_rate
        av_frame.planes[0].update(bytes(frame.data))

        for packet in self._stream.encode(av_frame):
            self._container.mux(packet)

        self._pending_samples += frame.samples_per_channel

        return self._flush()

    def close(self) -> EncodedAudioData:
        """Finalize the container and return any remaining bytes (e.g. OGG EOS).

        The encoder must not be used after calling this method.
        """
        if self._closed:
            return EncodedAudioData(data=b"", num_samples=0)

        self._closed = True
        for packet in self._stream.encode(None):
            self._container.mux(packet)
        self._container.close()
        return self._flush()
