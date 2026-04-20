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

import av
import av.audio
import av.container

from livekit import rtc


def _codec_to_container(codec: str) -> str:
    """Map a codec name to the appropriate streaming container format."""
    _TABLE: dict[str, str] = {
        "opus": "ogg",
        "libopus": "ogg",
        "mp3": "mp3",
        "libmp3lame": "mp3",
        "aac": "adts",
        "flac": "flac",
        "pcm_s16le": "wav",
    }
    fmt = _TABLE.get(codec)
    if fmt is None:
        raise ValueError(f"unsupported codec for streaming encode: {codec}")
    return fmt


class AudioStreamEncoder:
    """Encode PCM AudioFrames into a compressed audio byte stream.

    This is the encoding counterpart of :class:`AudioStreamDecoder`.  While the
    decoder needs a background thread (``container.decode()`` blocks on reads),
    encoding is non-blocking so this class uses a synchronous ``push() -> bytes, int``
    API instead.

    Encoders are stateful and must not be reused across multiple streams.
    """

    def __init__(
        self,
        *,
        codec: str = "opus",
        sample_rate: int = 48000,
        num_channels: int = 1,
        bit_rate: int = 24000,
    ) -> None:
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._layout = "mono" if num_channels == 1 else "stereo"

        self._output_buf = io.BytesIO()
        self._container: av.container.OutputContainer = av.open(
            self._output_buf,
            mode="w",
            format=_codec_to_container(codec),
        )
        self._stream: av.audio.AudioStream = self._container.add_stream(
            codec, rate=sample_rate, layout=self._layout
        )  # type: ignore[assignment]
        self._stream.bit_rate = bit_rate

        self._last_read_pos = 0
        self._closed = False

    def push(self, frame: rtc.AudioFrame) -> tuple[bytes, int]:
        """Encode a PCM audio frame.

        Returns any new encoded bytes produced by the muxer (may be empty when
        the container hasn't flushed a full page yet).  The very first call
        includes container headers (e.g. OGG OpusHead / OpusTags).
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

        return self._drain(), num_samples

    def flush(self) -> tuple[bytes, int]:
        """Flush the codec's internal buffers without closing the container."""
        if self._closed:
            return b"", 0

        num_samples = 0
        for packet in self._stream.encode(None):
            if packet.duration is not None:
                num_samples += packet.duration
            self._container.mux(packet)
        return self._drain(), num_samples

    def close(self) -> tuple[bytes, int]:
        """Finalize the container and return any remaining bytes (e.g. OGG EOS).

        The encoder must not be used after calling this method.
        """
        if self._closed:
            return b"", 0
        self._closed = True

        remaining, num_samples = self.flush()
        self._container.close()
        final = self._drain()
        return remaining + final, num_samples

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
