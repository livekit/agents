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

import ctypes
from ._ffi_client import FfiHandle, FfiClient
from ._proto import audio_frame_pb2 as proto_audio
from ._proto import ffi_pb2 as proto_ffi
from ._utils import get_address
from typing import Union


class AudioFrame:
    def __init__(
        self,
        data: Union[bytes, bytearray, memoryview],
        sample_rate: int,
        num_channels: int,
        samples_per_channel: int,
    ) -> None:
        if len(data) < num_channels * samples_per_channel * ctypes.sizeof(
            ctypes.c_int16
        ):
            raise ValueError(
                "data length must be >= num_channels * samples_per_channel * sizeof(int16)"
            )

        self._data = bytearray(data)
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._samples_per_channel = samples_per_channel

    @staticmethod
    def create(
        sample_rate: int, num_channels: int, samples_per_channel: int
    ) -> "AudioFrame":
        size = num_channels * samples_per_channel * ctypes.sizeof(ctypes.c_int16)
        data = bytearray(size)
        return AudioFrame(data, sample_rate, num_channels, samples_per_channel)

    @staticmethod
    def _from_owned_info(owned_info: proto_audio.OwnedAudioFrameBuffer) -> "AudioFrame":
        info = owned_info.info
        size = info.num_channels * info.samples_per_channel
        cdata = (ctypes.c_int16 * size).from_address(info.data_ptr)
        data = bytearray(cdata)
        FfiHandle(owned_info.handle.id)
        return AudioFrame(
            data, info.sample_rate, info.num_channels, info.samples_per_channel
        )

    def remix_and_resample(self, sample_rate: int, num_channels: int) -> "AudioFrame":
        """Resample the audio frame to the given sample rate and number of channels."""

        req = proto_ffi.FfiRequest()
        req.new_audio_resampler.CopyFrom(proto_audio.NewAudioResamplerRequest())

        resp = FfiClient.instance.request(req)
        resampler_handle = FfiHandle(resp.new_audio_resampler.resampler.handle.id)

        resample_req = proto_ffi.FfiRequest()
        resample_req.remix_and_resample.resampler_handle = resampler_handle.handle
        resample_req.remix_and_resample.buffer.CopyFrom(self._proto_info())
        resample_req.remix_and_resample.sample_rate = sample_rate
        resample_req.remix_and_resample.num_channels = num_channels

        resp = FfiClient.instance.request(resample_req)
        return AudioFrame._from_owned_info(resp.remix_and_resample.buffer)

    def _proto_info(self) -> proto_audio.AudioFrameBufferInfo:
        audio_info = proto_audio.AudioFrameBufferInfo()
        audio_info.data_ptr = get_address(memoryview(self._data))
        audio_info.sample_rate = self.sample_rate
        audio_info.num_channels = self.num_channels
        audio_info.samples_per_channel = self.samples_per_channel
        return audio_info

    @property
    def data(self) -> memoryview:
        return memoryview(self._data).cast("h")

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @property
    def samples_per_channel(self) -> int:
        return self._samples_per_channel
