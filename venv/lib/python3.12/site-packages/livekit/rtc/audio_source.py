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

from ._ffi_client import FfiHandle, FfiClient
from ._proto import audio_frame_pb2 as proto_audio_frame
from ._proto import ffi_pb2 as proto_ffi
from .audio_frame import AudioFrame


class AudioSource:
    def __init__(self, sample_rate: int, num_channels: int) -> None:
        req = proto_ffi.FfiRequest()
        req.new_audio_source.type = (
            proto_audio_frame.AudioSourceType.AUDIO_SOURCE_NATIVE
        )
        req.new_audio_source.sample_rate = sample_rate
        req.new_audio_source.num_channels = num_channels

        resp = FfiClient.instance.request(req)
        self._info = resp.new_audio_source.source
        self._ffi_handle = FfiHandle(self._info.handle.id)

    async def capture_frame(self, frame: AudioFrame) -> None:
        """Captures an AudioFrame.

        Used to push new audio data into the published Track. Audio data will
        be pushed in chunks of 10ms. It'll return only when all of the data in
        the buffer has been pushed.
        """
        req = proto_ffi.FfiRequest()

        req.capture_audio_frame.source_handle = self._ffi_handle.handle
        req.capture_audio_frame.buffer.CopyFrom(frame._proto_info())

        queue = FfiClient.instance.queue.subscribe()
        try:
            resp = FfiClient.instance.request(req)
            cb = await queue.wait_for(
                lambda e: e.capture_audio_frame.async_id
                == resp.capture_audio_frame.async_id
            )
        finally:
            FfiClient.instance.queue.unsubscribe(queue)

        if cb.capture_audio_frame.error:
            raise Exception(cb.capture_audio_frame.error)
