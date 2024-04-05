from . import models as _models
from . import egress as _egress
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CreateRoomRequest(_message.Message):
    __slots__ = ("name", "empty_timeout", "max_participants", "node_id", "metadata", "egress", "min_playout_delay", "max_playout_delay", "sync_streams")
    NAME_FIELD_NUMBER: _ClassVar[int]
    EMPTY_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    MAX_PARTICIPANTS_FIELD_NUMBER: _ClassVar[int]
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    EGRESS_FIELD_NUMBER: _ClassVar[int]
    MIN_PLAYOUT_DELAY_FIELD_NUMBER: _ClassVar[int]
    MAX_PLAYOUT_DELAY_FIELD_NUMBER: _ClassVar[int]
    SYNC_STREAMS_FIELD_NUMBER: _ClassVar[int]
    name: str
    empty_timeout: int
    max_participants: int
    node_id: str
    metadata: str
    egress: RoomEgress
    min_playout_delay: int
    max_playout_delay: int
    sync_streams: bool
    def __init__(self, name: _Optional[str] = ..., empty_timeout: _Optional[int] = ..., max_participants: _Optional[int] = ..., node_id: _Optional[str] = ..., metadata: _Optional[str] = ..., egress: _Optional[_Union[RoomEgress, _Mapping]] = ..., min_playout_delay: _Optional[int] = ..., max_playout_delay: _Optional[int] = ..., sync_streams: bool = ...) -> None: ...

class RoomEgress(_message.Message):
    __slots__ = ("room", "participant", "tracks")
    ROOM_FIELD_NUMBER: _ClassVar[int]
    PARTICIPANT_FIELD_NUMBER: _ClassVar[int]
    TRACKS_FIELD_NUMBER: _ClassVar[int]
    room: _egress.RoomCompositeEgressRequest
    participant: _egress.AutoParticipantEgress
    tracks: _egress.AutoTrackEgress
    def __init__(self, room: _Optional[_Union[_egress.RoomCompositeEgressRequest, _Mapping]] = ..., participant: _Optional[_Union[_egress.AutoParticipantEgress, _Mapping]] = ..., tracks: _Optional[_Union[_egress.AutoTrackEgress, _Mapping]] = ...) -> None: ...

class ListRoomsRequest(_message.Message):
    __slots__ = ("names",)
    NAMES_FIELD_NUMBER: _ClassVar[int]
    names: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, names: _Optional[_Iterable[str]] = ...) -> None: ...

class ListRoomsResponse(_message.Message):
    __slots__ = ("rooms",)
    ROOMS_FIELD_NUMBER: _ClassVar[int]
    rooms: _containers.RepeatedCompositeFieldContainer[_models.Room]
    def __init__(self, rooms: _Optional[_Iterable[_Union[_models.Room, _Mapping]]] = ...) -> None: ...

class DeleteRoomRequest(_message.Message):
    __slots__ = ("room",)
    ROOM_FIELD_NUMBER: _ClassVar[int]
    room: str
    def __init__(self, room: _Optional[str] = ...) -> None: ...

class DeleteRoomResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListParticipantsRequest(_message.Message):
    __slots__ = ("room",)
    ROOM_FIELD_NUMBER: _ClassVar[int]
    room: str
    def __init__(self, room: _Optional[str] = ...) -> None: ...

class ListParticipantsResponse(_message.Message):
    __slots__ = ("participants",)
    PARTICIPANTS_FIELD_NUMBER: _ClassVar[int]
    participants: _containers.RepeatedCompositeFieldContainer[_models.ParticipantInfo]
    def __init__(self, participants: _Optional[_Iterable[_Union[_models.ParticipantInfo, _Mapping]]] = ...) -> None: ...

class RoomParticipantIdentity(_message.Message):
    __slots__ = ("room", "identity")
    ROOM_FIELD_NUMBER: _ClassVar[int]
    IDENTITY_FIELD_NUMBER: _ClassVar[int]
    room: str
    identity: str
    def __init__(self, room: _Optional[str] = ..., identity: _Optional[str] = ...) -> None: ...

class RemoveParticipantResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class MuteRoomTrackRequest(_message.Message):
    __slots__ = ("room", "identity", "track_sid", "muted")
    ROOM_FIELD_NUMBER: _ClassVar[int]
    IDENTITY_FIELD_NUMBER: _ClassVar[int]
    TRACK_SID_FIELD_NUMBER: _ClassVar[int]
    MUTED_FIELD_NUMBER: _ClassVar[int]
    room: str
    identity: str
    track_sid: str
    muted: bool
    def __init__(self, room: _Optional[str] = ..., identity: _Optional[str] = ..., track_sid: _Optional[str] = ..., muted: bool = ...) -> None: ...

class MuteRoomTrackResponse(_message.Message):
    __slots__ = ("track",)
    TRACK_FIELD_NUMBER: _ClassVar[int]
    track: _models.TrackInfo
    def __init__(self, track: _Optional[_Union[_models.TrackInfo, _Mapping]] = ...) -> None: ...

class UpdateParticipantRequest(_message.Message):
    __slots__ = ("room", "identity", "metadata", "permission", "name")
    ROOM_FIELD_NUMBER: _ClassVar[int]
    IDENTITY_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    PERMISSION_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    room: str
    identity: str
    metadata: str
    permission: _models.ParticipantPermission
    name: str
    def __init__(self, room: _Optional[str] = ..., identity: _Optional[str] = ..., metadata: _Optional[str] = ..., permission: _Optional[_Union[_models.ParticipantPermission, _Mapping]] = ..., name: _Optional[str] = ...) -> None: ...

class UpdateSubscriptionsRequest(_message.Message):
    __slots__ = ("room", "identity", "track_sids", "subscribe", "participant_tracks")
    ROOM_FIELD_NUMBER: _ClassVar[int]
    IDENTITY_FIELD_NUMBER: _ClassVar[int]
    TRACK_SIDS_FIELD_NUMBER: _ClassVar[int]
    SUBSCRIBE_FIELD_NUMBER: _ClassVar[int]
    PARTICIPANT_TRACKS_FIELD_NUMBER: _ClassVar[int]
    room: str
    identity: str
    track_sids: _containers.RepeatedScalarFieldContainer[str]
    subscribe: bool
    participant_tracks: _containers.RepeatedCompositeFieldContainer[_models.ParticipantTracks]
    def __init__(self, room: _Optional[str] = ..., identity: _Optional[str] = ..., track_sids: _Optional[_Iterable[str]] = ..., subscribe: bool = ..., participant_tracks: _Optional[_Iterable[_Union[_models.ParticipantTracks, _Mapping]]] = ...) -> None: ...

class UpdateSubscriptionsResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SendDataRequest(_message.Message):
    __slots__ = ("room", "data", "kind", "destination_sids", "destination_identities", "topic")
    ROOM_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    DESTINATION_SIDS_FIELD_NUMBER: _ClassVar[int]
    DESTINATION_IDENTITIES_FIELD_NUMBER: _ClassVar[int]
    TOPIC_FIELD_NUMBER: _ClassVar[int]
    room: str
    data: bytes
    kind: _models.DataPacket.Kind
    destination_sids: _containers.RepeatedScalarFieldContainer[str]
    destination_identities: _containers.RepeatedScalarFieldContainer[str]
    topic: str
    def __init__(self, room: _Optional[str] = ..., data: _Optional[bytes] = ..., kind: _Optional[_Union[_models.DataPacket.Kind, str]] = ..., destination_sids: _Optional[_Iterable[str]] = ..., destination_identities: _Optional[_Iterable[str]] = ..., topic: _Optional[str] = ...) -> None: ...

class SendDataResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class UpdateRoomMetadataRequest(_message.Message):
    __slots__ = ("room", "metadata")
    ROOM_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    room: str
    metadata: str
    def __init__(self, room: _Optional[str] = ..., metadata: _Optional[str] = ...) -> None: ...
