from . import models as _models
from . import egress as _egress
from . import ingress as _ingress
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class WebhookEvent(_message.Message):
    __slots__ = ("event", "room", "participant", "egress_info", "ingress_info", "track", "id", "created_at", "num_dropped")
    EVENT_FIELD_NUMBER: _ClassVar[int]
    ROOM_FIELD_NUMBER: _ClassVar[int]
    PARTICIPANT_FIELD_NUMBER: _ClassVar[int]
    EGRESS_INFO_FIELD_NUMBER: _ClassVar[int]
    INGRESS_INFO_FIELD_NUMBER: _ClassVar[int]
    TRACK_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    NUM_DROPPED_FIELD_NUMBER: _ClassVar[int]
    event: str
    room: _models.Room
    participant: _models.ParticipantInfo
    egress_info: _egress.EgressInfo
    ingress_info: _ingress.IngressInfo
    track: _models.TrackInfo
    id: str
    created_at: int
    num_dropped: int
    def __init__(self, event: _Optional[str] = ..., room: _Optional[_Union[_models.Room, _Mapping]] = ..., participant: _Optional[_Union[_models.ParticipantInfo, _Mapping]] = ..., egress_info: _Optional[_Union[_egress.EgressInfo, _Mapping]] = ..., ingress_info: _Optional[_Union[_ingress.IngressInfo, _Mapping]] = ..., track: _Optional[_Union[_models.TrackInfo, _Mapping]] = ..., id: _Optional[str] = ..., created_at: _Optional[int] = ..., num_dropped: _Optional[int] = ...) -> None: ...
