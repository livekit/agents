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

from typing import Optional

from ._ffi_client import FfiHandle, FfiClient
from ._proto import e2ee_pb2 as proto_e2ee
from ._proto import ffi_pb2 as proto_ffi
from ._proto import track_pb2 as proto_track
from .track import Track


class TrackPublication:
    def __init__(self, owned_info: proto_track.OwnedTrackPublication):
        self._info = owned_info.info
        self.track: Optional[Track] = None
        self._ffi_handle = FfiHandle(owned_info.handle.id)

    @property
    def sid(self) -> str:
        return self._info.sid

    @property
    def name(self) -> str:
        return self._info.name

    @property
    def kind(self) -> proto_track.TrackKind.ValueType:
        return self._info.kind

    @property
    def source(self) -> proto_track.TrackSource.ValueType:
        return self._info.source

    @property
    def simulcasted(self) -> bool:
        return self._info.simulcasted

    @property
    def width(self) -> int:
        return self._info.width

    @property
    def height(self) -> int:
        return self._info.height

    @property
    def mime_type(self) -> str:
        return self._info.mime_type

    @property
    def muted(self) -> bool:
        return self._info.muted

    @property
    def encryption_type(self) -> proto_e2ee.EncryptionType.ValueType:
        return self._info.encryption_type


class LocalTrackPublication(TrackPublication):
    def __init__(self, owned_info: proto_track.OwnedTrackPublication):
        super().__init__(owned_info)


class RemoteTrackPublication(TrackPublication):
    def __init__(self, owned_info: proto_track.OwnedTrackPublication):
        super().__init__(owned_info)
        self.subscribed = False

    def set_subscribed(self, subscribed: bool):
        req = proto_ffi.FfiRequest()
        req.set_subscribed.subscribe = subscribed
        req.set_subscribed.publication_handle = self._ffi_handle.handle
        FfiClient.instance.request(req)
