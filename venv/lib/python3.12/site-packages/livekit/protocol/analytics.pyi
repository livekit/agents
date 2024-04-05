from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from . import models as _models
from . import egress as _egress
from . import ingress as _ingress
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class StreamType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UPSTREAM: _ClassVar[StreamType]
    DOWNSTREAM: _ClassVar[StreamType]

class AnalyticsEventType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ROOM_CREATED: _ClassVar[AnalyticsEventType]
    ROOM_ENDED: _ClassVar[AnalyticsEventType]
    PARTICIPANT_JOINED: _ClassVar[AnalyticsEventType]
    PARTICIPANT_LEFT: _ClassVar[AnalyticsEventType]
    TRACK_PUBLISHED: _ClassVar[AnalyticsEventType]
    TRACK_PUBLISH_REQUESTED: _ClassVar[AnalyticsEventType]
    TRACK_UNPUBLISHED: _ClassVar[AnalyticsEventType]
    TRACK_SUBSCRIBED: _ClassVar[AnalyticsEventType]
    TRACK_SUBSCRIBE_REQUESTED: _ClassVar[AnalyticsEventType]
    TRACK_SUBSCRIBE_FAILED: _ClassVar[AnalyticsEventType]
    TRACK_UNSUBSCRIBED: _ClassVar[AnalyticsEventType]
    TRACK_PUBLISHED_UPDATE: _ClassVar[AnalyticsEventType]
    TRACK_MUTED: _ClassVar[AnalyticsEventType]
    TRACK_UNMUTED: _ClassVar[AnalyticsEventType]
    TRACK_PUBLISH_STATS: _ClassVar[AnalyticsEventType]
    TRACK_SUBSCRIBE_STATS: _ClassVar[AnalyticsEventType]
    PARTICIPANT_ACTIVE: _ClassVar[AnalyticsEventType]
    PARTICIPANT_RESUMED: _ClassVar[AnalyticsEventType]
    EGRESS_STARTED: _ClassVar[AnalyticsEventType]
    EGRESS_ENDED: _ClassVar[AnalyticsEventType]
    EGRESS_UPDATED: _ClassVar[AnalyticsEventType]
    TRACK_MAX_SUBSCRIBED_VIDEO_QUALITY: _ClassVar[AnalyticsEventType]
    RECONNECTED: _ClassVar[AnalyticsEventType]
    INGRESS_CREATED: _ClassVar[AnalyticsEventType]
    INGRESS_DELETED: _ClassVar[AnalyticsEventType]
    INGRESS_STARTED: _ClassVar[AnalyticsEventType]
    INGRESS_ENDED: _ClassVar[AnalyticsEventType]
    INGRESS_UPDATED: _ClassVar[AnalyticsEventType]
UPSTREAM: StreamType
DOWNSTREAM: StreamType
ROOM_CREATED: AnalyticsEventType
ROOM_ENDED: AnalyticsEventType
PARTICIPANT_JOINED: AnalyticsEventType
PARTICIPANT_LEFT: AnalyticsEventType
TRACK_PUBLISHED: AnalyticsEventType
TRACK_PUBLISH_REQUESTED: AnalyticsEventType
TRACK_UNPUBLISHED: AnalyticsEventType
TRACK_SUBSCRIBED: AnalyticsEventType
TRACK_SUBSCRIBE_REQUESTED: AnalyticsEventType
TRACK_SUBSCRIBE_FAILED: AnalyticsEventType
TRACK_UNSUBSCRIBED: AnalyticsEventType
TRACK_PUBLISHED_UPDATE: AnalyticsEventType
TRACK_MUTED: AnalyticsEventType
TRACK_UNMUTED: AnalyticsEventType
TRACK_PUBLISH_STATS: AnalyticsEventType
TRACK_SUBSCRIBE_STATS: AnalyticsEventType
PARTICIPANT_ACTIVE: AnalyticsEventType
PARTICIPANT_RESUMED: AnalyticsEventType
EGRESS_STARTED: AnalyticsEventType
EGRESS_ENDED: AnalyticsEventType
EGRESS_UPDATED: AnalyticsEventType
TRACK_MAX_SUBSCRIBED_VIDEO_QUALITY: AnalyticsEventType
RECONNECTED: AnalyticsEventType
INGRESS_CREATED: AnalyticsEventType
INGRESS_DELETED: AnalyticsEventType
INGRESS_STARTED: AnalyticsEventType
INGRESS_ENDED: AnalyticsEventType
INGRESS_UPDATED: AnalyticsEventType

class AnalyticsVideoLayer(_message.Message):
    __slots__ = ("layer", "packets", "bytes", "frames")
    LAYER_FIELD_NUMBER: _ClassVar[int]
    PACKETS_FIELD_NUMBER: _ClassVar[int]
    BYTES_FIELD_NUMBER: _ClassVar[int]
    FRAMES_FIELD_NUMBER: _ClassVar[int]
    layer: int
    packets: int
    bytes: int
    frames: int
    def __init__(self, layer: _Optional[int] = ..., packets: _Optional[int] = ..., bytes: _Optional[int] = ..., frames: _Optional[int] = ...) -> None: ...

class AnalyticsStream(_message.Message):
    __slots__ = ("ssrc", "primary_packets", "primary_bytes", "retransmit_packets", "retransmit_bytes", "padding_packets", "padding_bytes", "packets_lost", "frames", "rtt", "jitter", "nacks", "plis", "firs", "video_layers")
    SSRC_FIELD_NUMBER: _ClassVar[int]
    PRIMARY_PACKETS_FIELD_NUMBER: _ClassVar[int]
    PRIMARY_BYTES_FIELD_NUMBER: _ClassVar[int]
    RETRANSMIT_PACKETS_FIELD_NUMBER: _ClassVar[int]
    RETRANSMIT_BYTES_FIELD_NUMBER: _ClassVar[int]
    PADDING_PACKETS_FIELD_NUMBER: _ClassVar[int]
    PADDING_BYTES_FIELD_NUMBER: _ClassVar[int]
    PACKETS_LOST_FIELD_NUMBER: _ClassVar[int]
    FRAMES_FIELD_NUMBER: _ClassVar[int]
    RTT_FIELD_NUMBER: _ClassVar[int]
    JITTER_FIELD_NUMBER: _ClassVar[int]
    NACKS_FIELD_NUMBER: _ClassVar[int]
    PLIS_FIELD_NUMBER: _ClassVar[int]
    FIRS_FIELD_NUMBER: _ClassVar[int]
    VIDEO_LAYERS_FIELD_NUMBER: _ClassVar[int]
    ssrc: int
    primary_packets: int
    primary_bytes: int
    retransmit_packets: int
    retransmit_bytes: int
    padding_packets: int
    padding_bytes: int
    packets_lost: int
    frames: int
    rtt: int
    jitter: int
    nacks: int
    plis: int
    firs: int
    video_layers: _containers.RepeatedCompositeFieldContainer[AnalyticsVideoLayer]
    def __init__(self, ssrc: _Optional[int] = ..., primary_packets: _Optional[int] = ..., primary_bytes: _Optional[int] = ..., retransmit_packets: _Optional[int] = ..., retransmit_bytes: _Optional[int] = ..., padding_packets: _Optional[int] = ..., padding_bytes: _Optional[int] = ..., packets_lost: _Optional[int] = ..., frames: _Optional[int] = ..., rtt: _Optional[int] = ..., jitter: _Optional[int] = ..., nacks: _Optional[int] = ..., plis: _Optional[int] = ..., firs: _Optional[int] = ..., video_layers: _Optional[_Iterable[_Union[AnalyticsVideoLayer, _Mapping]]] = ...) -> None: ...

class AnalyticsStat(_message.Message):
    __slots__ = ("analytics_key", "kind", "time_stamp", "node", "room_id", "room_name", "participant_id", "track_id", "score", "streams", "mime", "min_score", "median_score", "project_id")
    ANALYTICS_KEY_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    TIME_STAMP_FIELD_NUMBER: _ClassVar[int]
    NODE_FIELD_NUMBER: _ClassVar[int]
    ROOM_ID_FIELD_NUMBER: _ClassVar[int]
    ROOM_NAME_FIELD_NUMBER: _ClassVar[int]
    PARTICIPANT_ID_FIELD_NUMBER: _ClassVar[int]
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    STREAMS_FIELD_NUMBER: _ClassVar[int]
    MIME_FIELD_NUMBER: _ClassVar[int]
    MIN_SCORE_FIELD_NUMBER: _ClassVar[int]
    MEDIAN_SCORE_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    analytics_key: str
    kind: StreamType
    time_stamp: _timestamp_pb2.Timestamp
    node: str
    room_id: str
    room_name: str
    participant_id: str
    track_id: str
    score: float
    streams: _containers.RepeatedCompositeFieldContainer[AnalyticsStream]
    mime: str
    min_score: float
    median_score: float
    project_id: str
    def __init__(self, analytics_key: _Optional[str] = ..., kind: _Optional[_Union[StreamType, str]] = ..., time_stamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., node: _Optional[str] = ..., room_id: _Optional[str] = ..., room_name: _Optional[str] = ..., participant_id: _Optional[str] = ..., track_id: _Optional[str] = ..., score: _Optional[float] = ..., streams: _Optional[_Iterable[_Union[AnalyticsStream, _Mapping]]] = ..., mime: _Optional[str] = ..., min_score: _Optional[float] = ..., median_score: _Optional[float] = ..., project_id: _Optional[str] = ...) -> None: ...

class AnalyticsStats(_message.Message):
    __slots__ = ("stats",)
    STATS_FIELD_NUMBER: _ClassVar[int]
    stats: _containers.RepeatedCompositeFieldContainer[AnalyticsStat]
    def __init__(self, stats: _Optional[_Iterable[_Union[AnalyticsStat, _Mapping]]] = ...) -> None: ...

class AnalyticsClientMeta(_message.Message):
    __slots__ = ("region", "node", "client_addr", "client_connect_time", "connection_type", "reconnect_reason")
    REGION_FIELD_NUMBER: _ClassVar[int]
    NODE_FIELD_NUMBER: _ClassVar[int]
    CLIENT_ADDR_FIELD_NUMBER: _ClassVar[int]
    CLIENT_CONNECT_TIME_FIELD_NUMBER: _ClassVar[int]
    CONNECTION_TYPE_FIELD_NUMBER: _ClassVar[int]
    RECONNECT_REASON_FIELD_NUMBER: _ClassVar[int]
    region: str
    node: str
    client_addr: str
    client_connect_time: int
    connection_type: str
    reconnect_reason: _models.ReconnectReason
    def __init__(self, region: _Optional[str] = ..., node: _Optional[str] = ..., client_addr: _Optional[str] = ..., client_connect_time: _Optional[int] = ..., connection_type: _Optional[str] = ..., reconnect_reason: _Optional[_Union[_models.ReconnectReason, str]] = ...) -> None: ...

class AnalyticsEvent(_message.Message):
    __slots__ = ("type", "timestamp", "room_id", "room", "participant_id", "participant", "track_id", "track", "analytics_key", "client_info", "client_meta", "egress_id", "ingress_id", "max_subscribed_video_quality", "publisher", "mime", "egress", "ingress", "error", "rtp_stats", "video_layer", "project_id")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    ROOM_ID_FIELD_NUMBER: _ClassVar[int]
    ROOM_FIELD_NUMBER: _ClassVar[int]
    PARTICIPANT_ID_FIELD_NUMBER: _ClassVar[int]
    PARTICIPANT_FIELD_NUMBER: _ClassVar[int]
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    TRACK_FIELD_NUMBER: _ClassVar[int]
    ANALYTICS_KEY_FIELD_NUMBER: _ClassVar[int]
    CLIENT_INFO_FIELD_NUMBER: _ClassVar[int]
    CLIENT_META_FIELD_NUMBER: _ClassVar[int]
    EGRESS_ID_FIELD_NUMBER: _ClassVar[int]
    INGRESS_ID_FIELD_NUMBER: _ClassVar[int]
    MAX_SUBSCRIBED_VIDEO_QUALITY_FIELD_NUMBER: _ClassVar[int]
    PUBLISHER_FIELD_NUMBER: _ClassVar[int]
    MIME_FIELD_NUMBER: _ClassVar[int]
    EGRESS_FIELD_NUMBER: _ClassVar[int]
    INGRESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    RTP_STATS_FIELD_NUMBER: _ClassVar[int]
    VIDEO_LAYER_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    type: AnalyticsEventType
    timestamp: _timestamp_pb2.Timestamp
    room_id: str
    room: _models.Room
    participant_id: str
    participant: _models.ParticipantInfo
    track_id: str
    track: _models.TrackInfo
    analytics_key: str
    client_info: _models.ClientInfo
    client_meta: AnalyticsClientMeta
    egress_id: str
    ingress_id: str
    max_subscribed_video_quality: _models.VideoQuality
    publisher: _models.ParticipantInfo
    mime: str
    egress: _egress.EgressInfo
    ingress: _ingress.IngressInfo
    error: str
    rtp_stats: _models.RTPStats
    video_layer: int
    project_id: str
    def __init__(self, type: _Optional[_Union[AnalyticsEventType, str]] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., room_id: _Optional[str] = ..., room: _Optional[_Union[_models.Room, _Mapping]] = ..., participant_id: _Optional[str] = ..., participant: _Optional[_Union[_models.ParticipantInfo, _Mapping]] = ..., track_id: _Optional[str] = ..., track: _Optional[_Union[_models.TrackInfo, _Mapping]] = ..., analytics_key: _Optional[str] = ..., client_info: _Optional[_Union[_models.ClientInfo, _Mapping]] = ..., client_meta: _Optional[_Union[AnalyticsClientMeta, _Mapping]] = ..., egress_id: _Optional[str] = ..., ingress_id: _Optional[str] = ..., max_subscribed_video_quality: _Optional[_Union[_models.VideoQuality, str]] = ..., publisher: _Optional[_Union[_models.ParticipantInfo, _Mapping]] = ..., mime: _Optional[str] = ..., egress: _Optional[_Union[_egress.EgressInfo, _Mapping]] = ..., ingress: _Optional[_Union[_ingress.IngressInfo, _Mapping]] = ..., error: _Optional[str] = ..., rtp_stats: _Optional[_Union[_models.RTPStats, _Mapping]] = ..., video_layer: _Optional[int] = ..., project_id: _Optional[str] = ...) -> None: ...

class AnalyticsEvents(_message.Message):
    __slots__ = ("events",)
    EVENTS_FIELD_NUMBER: _ClassVar[int]
    events: _containers.RepeatedCompositeFieldContainer[AnalyticsEvent]
    def __init__(self, events: _Optional[_Iterable[_Union[AnalyticsEvent, _Mapping]]] = ...) -> None: ...

class AnalyticsRoomParticipant(_message.Message):
    __slots__ = ("id", "identity", "name", "state", "joined_at")
    ID_FIELD_NUMBER: _ClassVar[int]
    IDENTITY_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    JOINED_AT_FIELD_NUMBER: _ClassVar[int]
    id: str
    identity: str
    name: str
    state: _models.ParticipantInfo.State
    joined_at: _timestamp_pb2.Timestamp
    def __init__(self, id: _Optional[str] = ..., identity: _Optional[str] = ..., name: _Optional[str] = ..., state: _Optional[_Union[_models.ParticipantInfo.State, str]] = ..., joined_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class AnalyticsRoom(_message.Message):
    __slots__ = ("id", "name", "created_at", "participants")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    PARTICIPANTS_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    created_at: _timestamp_pb2.Timestamp
    participants: _containers.RepeatedCompositeFieldContainer[AnalyticsRoomParticipant]
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., created_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., participants: _Optional[_Iterable[_Union[AnalyticsRoomParticipant, _Mapping]]] = ...) -> None: ...

class AnalyticsNodeRooms(_message.Message):
    __slots__ = ("node_id", "sequence_number", "timestamp", "rooms")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    ROOMS_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    sequence_number: int
    timestamp: _timestamp_pb2.Timestamp
    rooms: _containers.RepeatedCompositeFieldContainer[AnalyticsRoom]
    def __init__(self, node_id: _Optional[str] = ..., sequence_number: _Optional[int] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., rooms: _Optional[_Iterable[_Union[AnalyticsRoom, _Mapping]]] = ...) -> None: ...
