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
from typing import List, Union

from ._ffi_client import FfiHandle, FfiClient
from ._proto import ffi_pb2 as proto_ffi
from ._proto import participant_pb2 as proto_participant
from ._proto.room_pb2 import DataPacketKind, TrackPublishOptions
from ._utils import BroadcastQueue
from .track import LocalTrack
from .track_publication import (
    LocalTrackPublication,
    RemoteTrackPublication,
    TrackPublication,
)


class PublishTrackError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message


class UnpublishTrackError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message


class PublishDataError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message


class Participant:
    def __init__(self, owned_info: proto_participant.OwnedParticipant) -> None:
        self._info = owned_info.info
        self._ffi_handle = FfiHandle(owned_info.handle.id)
        self.tracks: dict[str, TrackPublication] = {}

    @property
    def sid(self) -> str:
        return self._info.sid

    @property
    def name(self) -> str:
        return self._info.name

    @property
    def identity(self) -> str:
        return self._info.identity

    @property
    def metadata(self) -> str:
        return self._info.metadata


class LocalParticipant(Participant):
    def __init__(
        self,
        room_queue: BroadcastQueue[proto_ffi.FfiEvent],
        owned_info: proto_participant.OwnedParticipant,
    ) -> None:
        super().__init__(owned_info)
        self._room_queue = room_queue
        self.tracks: dict[str, LocalTrackPublication] = {}  # type: ignore

    async def publish_data(
        self,
        payload: Union[bytes, str],
        kind: DataPacketKind.ValueType = DataPacketKind.KIND_RELIABLE,
        destination_sids: List[Union[str, "RemoteParticipant"]] = [],
        topic: str = "",
    ) -> None:
        if isinstance(payload, str):
            payload = payload.encode("utf-8")

        data_len = len(payload)
        cdata = (ctypes.c_byte * data_len)(*payload)

        req = proto_ffi.FfiRequest()
        req.publish_data.local_participant_handle = self._ffi_handle.handle
        req.publish_data.data_ptr = ctypes.addressof(cdata)
        req.publish_data.data_len = data_len
        req.publish_data.kind = kind
        req.publish_data.topic = topic

        sids = []
        for p in destination_sids:
            if isinstance(p, RemoteParticipant):
                sids.append(p.sid)
            else:
                sids.append(p)

        req.publish_data.destination_sids.extend(sids)

        queue = FfiClient.instance.queue.subscribe()
        try:
            resp = FfiClient.instance.request(req)
            cb = await queue.wait_for(
                lambda e: e.publish_data.async_id == resp.publish_data.async_id
            )
        finally:
            FfiClient.instance.queue.unsubscribe(queue)

        if cb.publish_data.error:
            raise PublishDataError(cb.publish_data.error)

    async def update_metadata(self, metadata: str) -> None:
        req = proto_ffi.FfiRequest()
        req.update_local_metadata.local_participant_handle = self._ffi_handle.handle
        req.update_local_metadata.metadata = metadata

        queue = FfiClient.instance.queue.subscribe()
        try:
            resp = FfiClient.instance.request(req)
            await queue.wait_for(
                lambda e: e.update_local_metadata.async_id
                == resp.update_local_metadata.async_id
            )
        finally:
            FfiClient.instance.queue.unsubscribe(queue)

    async def update_name(self, name: str) -> None:
        req = proto_ffi.FfiRequest()
        req.update_local_name.local_participant_handle = self._ffi_handle.handle
        req.update_local_name.name = name

        queue = FfiClient.instance.queue.subscribe()
        try:
            resp = FfiClient.instance.request(req)
            await queue.wait_for(
                lambda e: e.update_local_name.async_id
                == resp.update_local_name.async_id
            )
        finally:
            FfiClient.instance.queue.unsubscribe(queue)

    async def publish_track(
        self, track: LocalTrack, options: TrackPublishOptions = TrackPublishOptions()
    ) -> LocalTrackPublication:
        req = proto_ffi.FfiRequest()
        req.publish_track.track_handle = track._ffi_handle.handle
        req.publish_track.local_participant_handle = self._ffi_handle.handle
        req.publish_track.options.CopyFrom(options)

        queue = self._room_queue.subscribe()
        try:
            resp = FfiClient.instance.request(req)
            cb = await queue.wait_for(
                lambda e: e.publish_track.async_id == resp.publish_track.async_id
            )

            if cb.publish_track.error:
                raise PublishTrackError(cb.publish_track.error)

            track_publication = LocalTrackPublication(cb.publish_track.publication)
            track_publication.track = track
            track._info.sid = track_publication.sid
            self.tracks[track_publication.sid] = track_publication

            queue.task_done()
            return track_publication
        finally:
            self._room_queue.unsubscribe(queue)

    async def unpublish_track(self, track_sid: str) -> None:
        req = proto_ffi.FfiRequest()
        req.unpublish_track.local_participant_handle = self._ffi_handle.handle
        req.unpublish_track.track_sid = track_sid

        queue = self._room_queue.subscribe()
        try:
            resp = FfiClient.instance.request(req)
            cb = await queue.wait_for(
                lambda e: e.unpublish_track.async_id == resp.unpublish_track.async_id
            )

            if cb.unpublish_track.error:
                raise UnpublishTrackError(cb.unpublish_track.error)

            publication = self.tracks.pop(track_sid)
            publication.track = None
            queue.task_done()
        finally:
            self._room_queue.unsubscribe(queue)


class RemoteParticipant(Participant):
    def __init__(self, owned_info: proto_participant.OwnedParticipant) -> None:
        super().__init__(owned_info)
        self.tracks: dict[str, RemoteTrackPublication] = {}  # type: ignore
