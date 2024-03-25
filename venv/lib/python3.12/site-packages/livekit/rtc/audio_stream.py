# Copyright 2023 LiveKit, Inc.
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
from typing import Optional

from ._ffi_client import FfiHandle, FfiClient
from ._proto import audio_frame_pb2 as proto_audio_frame
from ._proto import ffi_pb2 as proto_ffi
from ._utils import RingQueue, task_done_logger
from .audio_frame import AudioFrame
from .track import Track
from dataclasses import dataclass


@dataclass
class AudioFrameEvent:
    frame: AudioFrame


class AudioStream:
    """AudioStream is a stream of audio frames received from a RemoteTrack."""

    def __init__(
        self,
        track: Track,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        capacity: int = 0,
    ) -> None:
        self._track = track
        self._loop = loop or asyncio.get_event_loop()
        self._ffi_queue = FfiClient.instance.queue.subscribe(self._loop)
        self._queue: RingQueue[AudioFrameEvent] = RingQueue(capacity)

        req = proto_ffi.FfiRequest()
        new_audio_stream = req.new_audio_stream
        new_audio_stream.track_handle = track._ffi_handle.handle
        new_audio_stream.type = proto_audio_frame.AudioStreamType.AUDIO_STREAM_NATIVE
        resp = FfiClient.instance.request(req)

        stream_info = resp.new_audio_stream.stream
        self._ffi_handle = FfiHandle(stream_info.handle.id)
        self._info = stream_info

        self._task = self._loop.create_task(self._run())
        self._task.add_done_callback(task_done_logger)

    def __del__(self) -> None:
        FfiClient.instance.queue.unsubscribe(self._ffi_queue)

    async def _run(self):
        while True:
            event = await self._ffi_queue.wait_for(self._is_event)
            audio_event = event.audio_stream_event

            if audio_event.HasField("frame_received"):
                owned_buffer_info = audio_event.frame_received.frame
                frame = AudioFrame._from_owned_info(owned_buffer_info)
                event = AudioFrameEvent(frame)
                self._queue.put(event)
            elif audio_event.HasField("eos"):
                break

        FfiClient.instance.queue.unsubscribe(self._ffi_queue)

    async def aclose(self) -> None:
        self._ffi_handle.dispose()
        await self._task

    def __aiter__(self) -> "AudioStream":
        return self

    def _is_event(self, e: proto_ffi.FfiEvent) -> bool:
        return e.audio_stream_event.stream_handle == self._ffi_handle.handle

    async def __anext__(self) -> AudioFrameEvent:
        if self._task.done():
            raise StopAsyncIteration
        return await self._queue.get()
