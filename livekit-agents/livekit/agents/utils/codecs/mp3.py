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

import ctypes
import logging
from typing import List

try:
    import av  # noqa
except ImportError:
    pass
from livekit import rtc


class Mp3StreamDecoder:
    """A class that can be used to stream arbitrary MP3 data (i.e. from an HTTP chunk) and decode it into PCM audio.
    This class is meant to be ephemeral. When you're done sending data, call close() to flush
    the decoder and create a new instance of this class if you need to decode more data.
    """

    def __init__(self):
        try:
            import av
        except ImportError:
            raise ImportError(
                "You haven't included the 'codecs' optional dependencies. Please install the 'codecs' extra by running `pip install livekit-agents[codecs]`"
            )
        self._codec = av.CodecContext.create("mp3", "r")  # noqa

    def decode_chunk(self, chunk: bytes) -> List[rtc.AudioFrame]:
        # Skip ID3v2 header if present
        if chunk.startswith(b"ID3"):
            # ID3v2 header is 10 bytes long
            # The size is encoded in the next 4 bytes (bytes 6-9)
            # Each byte only uses 7 bits (most significant bit is always 0)
            if len(chunk) >= 10:
                size = (
                    ((chunk[6] & 0x7F) << 21)
                    | ((chunk[7] & 0x7F) << 14)
                    | ((chunk[8] & 0x7F) << 7)
                    | (chunk[9] & 0x7F)
                )
                chunk = chunk[10 + size :]

        packets = self._codec.parse(chunk)
        result: List[rtc.AudioFrame] = []
        for packet in packets:
            try:
                decoded = self._codec.decode(packet)
            except Exception as e:
                logging.warning(f"Error decoding packet, skipping: {e}")
                continue
            for frame in decoded:
                nchannels = len(frame.layout.channels)
                if frame.format.is_planar and nchannels > 1:
                    logging.warning(
                        "TODO: planar audio has not yet been considered, skipping frame"
                    )
                    continue
                plane = frame.planes[0]
                ptr = plane.buffer_ptr
                size = plane.buffer_size
                byte_array_pointer = ctypes.cast(
                    ptr, ctypes.POINTER(ctypes.c_char * size)
                )
                result.append(
                    rtc.AudioFrame(
                        data=bytes(byte_array_pointer.contents),
                        num_channels=nchannels,
                        sample_rate=frame.sample_rate,
                        samples_per_channel=frame.samples,
                    )
                )
        return result
