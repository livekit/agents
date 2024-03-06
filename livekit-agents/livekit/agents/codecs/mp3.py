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
import ctypes
import logging
from importlib import import_module
from typing import List

from livekit import rtc


class Mp3StreamDecoder:
    """A class that can be used to stream arbitrary MP3 data (i.e. from an HTTP chunk) and decode it into PCM audio.
    This class is meant to be ephemeral. When you're done sending data, call close() to flush
    the decoder and create a new instance of this class if you need to decode more data.
    """

    def __init__(self):
        try:
            globals()["av"] = import_module("av")
        except ImportError:
            raise ImportError(
                "You haven't included the decoder_utils optional dependencies. Please install the decoder_utils extra by running `pip install livekit-agents[decoder_utils]`"
            )
        self._closed = False
        self._input_queue = asyncio.Queue()
        self._output_queue = asyncio.Queue()

        self._codec = av.CodecContext.create("mp3", "r")  # noqa
        self._run_task = asyncio.create_task(self._run())

    def close(self):
        self._closed = True
        self._input_queue.put_nowait(None)

    def push_chunk(self, chunk: bytes):
        if self._closed:
            raise ValueError("Cannot push chunk to closed decoder")
        self._input_queue.put_nowait(chunk)

    async def _run(self):
        while True:
            input = await self._input_queue.get()
            if input is None:
                self._output_queue.put_nowait(None)
                break

            result_frames = await asyncio.to_thread(self._decode_input, input)
            for frame in result_frames:
                await self._output_queue.put(frame)

    def _decode_input(self, input: bytes):
        packets = self._codec.parse(input)
        result_frames: List[rtc.AudioFrame] = []
        for packet in packets:
            try:
                decoded = self._codec.decode(packet)
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
                    result_frames.append(
                        rtc.AudioFrame(
                            data=bytes(byte_array_pointer.contents),
                            num_channels=nchannels,
                            sample_rate=frame.sample_rate,
                            samples_per_channel=frame.samples,
                        )
                    )
            except Exception as e:
                logging.error(f"Error decoding chunk: {e}")
                continue

        return result_frames

    def __aiter__(self):
        return self

    async def __anext__(self):
        packet = await self._output_queue.get()
        if packet is None:
            raise StopAsyncIteration
        return packet
