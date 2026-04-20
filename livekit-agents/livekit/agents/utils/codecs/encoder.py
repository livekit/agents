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

        self._output_buf = io.BytesIO()
        self._container: av.container.OutputContainer = av.open(
            self._output_buf,
            mode="w",
            format=container_format,
        )
        self._stream: av.audio.AudioStream = self._container.add_stream(
            av_codec,
            rate=sample_rate,
            layout=self._layout,
            options=codec_options,
        )
        self._stream.bit_rate = bit_rate

        self._last_read_pos = 0
        self._closed = False

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

        num_samples = 0
        for packet in self._stream.encode(av_frame):
            if packet.duration is not None:
                num_samples += packet.duration
            self._container.mux(packet)

        return EncodedAudioData(data=self._drain(), num_samples=num_samples)

    def flush(self) -> EncodedAudioData:
        """Flush the codec's internal buffers without closing the container."""
        if self._closed:
            return EncodedAudioData(data=b"", num_samples=0)

        num_samples = 0
        for packet in self._stream.encode(None):
            if packet.duration is not None:
                num_samples += packet.duration
            self._container.mux(packet)
        return EncodedAudioData(data=self._drain(), num_samples=num_samples)

    def close(self) -> EncodedAudioData:
        """Finalize the container and return any remaining bytes (e.g. OGG EOS).

        The encoder must not be used after calling this method.
        """
        if self._closed:
            return EncodedAudioData(data=b"", num_samples=0)

        encoded_data = self.flush()
        self._closed = True
        self._container.close()
        final = self._drain()
        return EncodedAudioData(
            data=encoded_data.data + final, num_samples=encoded_data.num_samples
        )

    def _drain(self) -> bytes:
        """Read all new bytes written to the output buffer since the last drain."""
        current_pos = self._output_buf.tell()
        if current_pos <= self._last_read_pos:
            return b""

        self._output_buf.seek(self._last_read_pos)
        new_bytes = self._output_buf.read(current_pos - self._last_read_pos)
        self._last_read_pos = current_pos
        self._output_buf.seek(current_pos)
        return new_bytes
