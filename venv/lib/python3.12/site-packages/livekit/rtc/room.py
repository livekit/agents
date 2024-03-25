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
import ctypes
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Literal
from ._ffi_client import FfiHandle, FfiClient
from ._proto import ffi_pb2 as proto_ffi
from ._proto import participant_pb2 as proto_participant
from ._proto import room_pb2 as proto_room
from ._proto.room_pb2 import ConnectionState
from ._proto.track_pb2 import TrackKind
from ._utils import BroadcastQueue
from ._event_emitter import EventEmitter
from .e2ee import E2EEManager, E2EEOptions
from .participant import LocalParticipant, Participant, RemoteParticipant
from .track import RemoteAudioTrack, RemoteVideoTrack
from .track_publication import RemoteTrackPublication

EventTypes = Literal[
    "participant_connected",
    "participant_disconnected",
    "local_track_published",
    "local_track_unpublished",
    "track_published",
    "track_unpublished",
    "track_subscribed",
    "track_unsubscribed",
    "track_subscription_failed",
    "track_muted",
    "track_unmuted",
    "active_speakers_changed",
    "room_metadata_changed",
    "participant_metadata_changed",
    "participant_name_changed",
    "connection_quality_changed",
    "data_received",
    "e2ee_state_changed",
    "connection_state_changed",
    "connected",
    "disconnected",
    "reconnecting",
    "reconnected",
]


@dataclass
class RtcConfiguration:
    ice_transport_type: proto_room.IceTransportType.ValueType = (
        proto_room.IceTransportType.TRANSPORT_ALL
    )
    continual_gathering_policy: proto_room.ContinualGatheringPolicy.ValueType = (
        proto_room.ContinualGatheringPolicy.GATHER_CONTINUALLY
    )
    ice_servers: list[proto_room.IceServer] = field(default_factory=list)


@dataclass
class RoomOptions:
    auto_subscribe: bool = True
    dynacast: bool = False
    e2ee: Optional[E2EEOptions] = None
    rtc_config: Optional[RtcConfiguration] = None


@dataclass
class DataPacket:
    data: bytes
    kind: proto_room.DataPacketKind.ValueType
    participant: Optional[
        RemoteParticipant
    ] = None  # None when the data has been sent by a server SDK
    topic: Optional[str] = None


class ConnectError(Exception):
    def __init__(self, message: str):
        self.message = message


class Room(EventEmitter[EventTypes]):
    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        super().__init__()

        self._ffi_handle: Optional[FfiHandle] = None
        self._loop = loop or asyncio.get_event_loop()
        self._room_queue = BroadcastQueue[proto_ffi.FfiEvent]()
        self._info = proto_room.RoomInfo()

        self.participants: Dict[str, RemoteParticipant] = {}
        self.connection_state = ConnectionState.CONN_DISCONNECTED

    def __del__(self) -> None:
        if self._ffi_handle is not None:
            FfiClient.instance.queue.unsubscribe(self._ffi_queue)

    @property
    def sid(self) -> str:
        return self._info.sid

    @property
    def name(self) -> str:
        return self._info.name

    @property
    def metadata(self) -> str:
        return self._info.metadata

    @property
    def e2ee_manager(self) -> E2EEManager:
        return self._e2ee_manager

    def isconnected(self) -> bool:
        return (
            self._ffi_handle is not None
            and self.connection_state != ConnectionState.CONN_DISCONNECTED
        )

    async def connect(
        self, url: str, token: str, options: RoomOptions = RoomOptions()
    ) -> None:
        req = proto_ffi.FfiRequest()
        req.connect.url = url
        req.connect.token = token

        # options
        req.connect.options.auto_subscribe = options.auto_subscribe
        req.connect.options.dynacast = options.dynacast

        if options.e2ee:
            req.connect.options.e2ee.encryption_type = options.e2ee.encryption_type
            req.connect.options.e2ee.key_provider_options.shared_key = (
                options.e2ee.key_provider_options.shared_key  # type: ignore
            )
            req.connect.options.e2ee.key_provider_options.ratchet_salt = (
                options.e2ee.key_provider_options.ratchet_salt
            )
            req.connect.options.e2ee.key_provider_options.failure_tolerance = (
                options.e2ee.key_provider_options.failure_tolerance
            )
            req.connect.options.e2ee.key_provider_options.ratchet_window_size = (
                options.e2ee.key_provider_options.ratchet_window_size
            )

        if options.rtc_config:
            req.connect.options.rtc_config.ice_transport_type = (
                options.rtc_config.ice_transport_type
            )  # type: ignore
            req.connect.options.rtc_config.continual_gathering_policy = (
                options.rtc_config.continual_gathering_policy
            )  # type: ignore
            req.connect.options.rtc_config.ice_servers.extend(
                options.rtc_config.ice_servers
            )

        # subscribe before connecting so we don't miss any events
        self._ffi_queue = FfiClient.instance.queue.subscribe(self._loop)

        queue = FfiClient.instance.queue.subscribe()
        try:
            resp = FfiClient.instance.request(req)
            cb = await queue.wait_for(
                lambda e: e.connect.async_id == resp.connect.async_id
            )
        finally:
            FfiClient.instance.queue.unsubscribe(queue)

        if cb.connect.error:
            FfiClient.instance.queue.unsubscribe(self._ffi_queue)
            raise ConnectError(cb.connect.error)

        self._ffi_handle = FfiHandle(cb.connect.room.handle.id)

        self._e2ee_manager = E2EEManager(self._ffi_handle.handle, options.e2ee)

        self._info = cb.connect.room.info
        self.connection_state = ConnectionState.CONN_CONNECTED

        self.local_participant = LocalParticipant(
            self._room_queue, cb.connect.local_participant
        )

        for pt in cb.connect.participants:
            rp = self._create_remote_participant(pt.participant)

            # add the initial remote participant tracks
            for owned_publication_info in pt.publications:
                publication = RemoteTrackPublication(owned_publication_info)
                rp.tracks[publication.sid] = publication

        # start listening to room events
        self._task = self._loop.create_task(self._listen_task())

    async def disconnect(self) -> None:
        if not self.isconnected():
            return

        req = proto_ffi.FfiRequest()
        req.disconnect.room_handle = self._ffi_handle.handle  # type: ignore

        queue = FfiClient.instance.queue.subscribe()
        try:
            resp = FfiClient.instance.request(req)
            await queue.wait_for(
                lambda e: e.disconnect.async_id == resp.disconnect.async_id
            )
        finally:
            FfiClient.instance.queue.unsubscribe(queue)

        await self._task
        FfiClient.instance.queue.unsubscribe(self._ffi_queue)

    async def _listen_task(self) -> None:
        # listen to incoming room events
        while True:
            event = await self._ffi_queue.get()
            if event.room_event.room_handle == self._ffi_handle.handle:  # type: ignore
                if event.room_event.HasField("eos"):
                    break

                try:
                    self._on_room_event(event.room_event)
                except Exception:
                    logging.exception(
                        "error running user callback for %s: %s",
                        event.room_event.WhichOneof("message"),
                        event.room_event,
                    )

            # wait for the subscribers to process the event
            # before processing the next one
            self._room_queue.put_nowait(event)
            await self._room_queue.join()

    def _on_room_event(self, event: proto_room.RoomEvent):
        which = event.WhichOneof("message")
        if which == "participant_connected":
            rparticipant = self._create_remote_participant(
                event.participant_connected.info
            )
            self.emit("participant_connected", rparticipant)
        elif which == "participant_disconnected":
            sid = event.participant_disconnected.participant_sid
            rparticipant = self.participants.pop(sid)
            self.emit("participant_disconnected", rparticipant)
        elif which == "local_track_published":
            sid = event.local_track_published.track_sid
            lpublication = self.local_participant.tracks[sid]
            track = lpublication.track
            self.emit("local_track_published", lpublication, track)
        elif which == "local_track_unpublished":
            sid = event.local_track_unpublished.publication_sid
            lpublication = self.local_participant.tracks[sid]
            self.emit("local_track_unpublished", lpublication)
        elif which == "track_published":
            rparticipant = self.participants[event.track_published.participant_sid]
            rpublication = RemoteTrackPublication(event.track_published.publication)
            rparticipant.tracks[rpublication.sid] = rpublication
            self.emit("track_published", rpublication, rparticipant)
        elif which == "track_unpublished":
            rparticipant = self.participants[event.track_unpublished.participant_sid]
            rpublication = rparticipant.tracks.pop(
                event.track_unpublished.publication_sid
            )
            self.emit("track_unpublished", rpublication, rparticipant)
        elif which == "track_subscribed":
            owned_track_info = event.track_subscribed.track
            track_info = owned_track_info.info
            rparticipant = self.participants[event.track_subscribed.participant_sid]
            rpublication = rparticipant.tracks[track_info.sid]
            rpublication.subscribed = True
            if track_info.kind == TrackKind.KIND_VIDEO:
                remote_video_track = RemoteVideoTrack(owned_track_info)
                rpublication.track = remote_video_track
                self.emit(
                    "track_subscribed", remote_video_track, rpublication, rparticipant
                )
            elif track_info.kind == TrackKind.KIND_AUDIO:
                remote_audio_track = RemoteAudioTrack(owned_track_info)
                rpublication.track = remote_audio_track
                self.emit(
                    "track_subscribed", remote_audio_track, rpublication, rparticipant
                )
        elif which == "track_unsubscribed":
            sid = event.track_unsubscribed.participant_sid
            rparticipant = self.participants[sid]
            rpublication = rparticipant.tracks[event.track_unsubscribed.track_sid]
            track = rpublication.track
            rpublication.track = None
            rpublication.subscribed = False
            self.emit("track_unsubscribed", track, rpublication, rparticipant)
        elif which == "track_subscription_failed":
            sid = event.track_subscription_failed.participant_sid
            rparticipant = self.participants[sid]
            error = event.track_subscription_failed.error
            self.emit(
                "track_subscription_failed",
                rparticipant,
                event.track_subscription_failed.track_sid,
                error,
            )
        elif which == "track_muted":
            sid = event.track_muted.participant_sid
            participant = self._retrieve_participant(sid)
            publication = participant.tracks[event.track_muted.track_sid]
            publication._info.muted = True
            if publication.track:
                publication.track._info.muted = True

            self.emit("track_muted", participant, publication)
        elif which == "track_unmuted":
            sid = event.track_unmuted.participant_sid
            participant = self._retrieve_participant(sid)
            publication = participant.tracks[event.track_unmuted.track_sid]
            publication._info.muted = False
            if publication.track:
                publication.track._info.muted = False

            self.emit("track_unmuted", participant, publication)
        elif which == "active_speakers_changed":
            speakers: list[Participant] = []
            for sid in event.active_speakers_changed.participant_sids:
                speakers.append(self._retrieve_participant(sid))

            self.emit("active_speakers_changed", speakers)
        elif which == "room_metadata_changed":
            old_metadata = self.metadata
            self._info.metadata = event.room_metadata_changed.metadata
            self.emit("room_metadata_changed", old_metadata, self.metadata)
        elif which == "participant_metadata_changed":
            sid = event.participant_metadata_changed.participant_sid
            participant = self._retrieve_participant(sid)
            old_metadata = participant.metadata
            participant._info.metadata = event.participant_metadata_changed.metadata
            self.emit(
                "participant_metadata_changed",
                participant,
                old_metadata,
                participant.metadata,
            )
        elif which == "participant_name_changed":
            sid = event.participant_name_changed.participant_sid
            participant = self._retrieve_participant(sid)
            old_name = participant.name
            participant._info.name = event.participant_name_changed.name
            self.emit(
                "participant_name_changed", participant, old_name, participant.name
            )
        elif which == "connection_quality_changed":
            sid = event.connection_quality_changed.participant_sid
            participant = self._retrieve_participant(sid)
            self.emit(
                "connection_quality_changed",
                participant,
                event.connection_quality_changed.quality,
            )
        elif which == "data_received":
            owned_buffer_info = event.data_received.data
            buffer_info = owned_buffer_info.data
            native_data = ctypes.cast(
                buffer_info.data_ptr,
                ctypes.POINTER(ctypes.c_byte * buffer_info.data_len),
            ).contents

            data = bytes(native_data)
            FfiHandle(owned_buffer_info.handle.id)
            rparticipant = None
            if event.data_received.participant_sid:
                rparticipant = self.participants[event.data_received.participant_sid]
            self.emit(
                "data_received",
                DataPacket(
                    data=data,
                    kind=event.data_received.kind,
                    participant=rparticipant,
                    topic=event.data_received.topic,
                ),
            )
        elif which == "e2ee_state_changed":
            sid = event.e2ee_state_changed.participant_sid
            e2ee_state = event.e2ee_state_changed.state
            self.emit("e2ee_state_changed", self._retrieve_participant(sid), e2ee_state)
        elif which == "connection_state_changed":
            connection_state = event.connection_state_changed.state
            self.connection_state = connection_state
            self.emit("connection_state_changed", connection_state)
        elif which == "connected":
            self.emit("connected")
        elif which == "disconnected":
            self.emit("disconnected")
        elif which == "reconnecting":
            self.emit("reconnecting")
        elif which == "reconnected":
            self.emit("reconnected")

    def _retrieve_participant(self, sid: str) -> Participant:
        """Retrieve a participant by sid, returns the LocalParticipant
        if sid matches"""
        if sid == self.local_participant.sid:
            return self.local_participant
        else:
            return self.participants[sid]

    def _create_remote_participant(
        self, owned_info: proto_participant.OwnedParticipant
    ) -> RemoteParticipant:
        if owned_info.info.sid in self.participants:
            raise Exception("participant already exists")

        participant = RemoteParticipant(owned_info)
        self.participants[participant.sid] = participant
        return participant
