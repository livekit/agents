from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AudioCodec(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DEFAULT_AC: _ClassVar[AudioCodec]
    OPUS: _ClassVar[AudioCodec]
    AAC: _ClassVar[AudioCodec]

class VideoCodec(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DEFAULT_VC: _ClassVar[VideoCodec]
    H264_BASELINE: _ClassVar[VideoCodec]
    H264_MAIN: _ClassVar[VideoCodec]
    H264_HIGH: _ClassVar[VideoCodec]
    VP8: _ClassVar[VideoCodec]

class ImageCodec(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    IC_DEFAULT: _ClassVar[ImageCodec]
    IC_JPEG: _ClassVar[ImageCodec]

class TrackType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    AUDIO: _ClassVar[TrackType]
    VIDEO: _ClassVar[TrackType]
    DATA: _ClassVar[TrackType]

class TrackSource(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UNKNOWN: _ClassVar[TrackSource]
    CAMERA: _ClassVar[TrackSource]
    MICROPHONE: _ClassVar[TrackSource]
    SCREEN_SHARE: _ClassVar[TrackSource]
    SCREEN_SHARE_AUDIO: _ClassVar[TrackSource]

class VideoQuality(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LOW: _ClassVar[VideoQuality]
    MEDIUM: _ClassVar[VideoQuality]
    HIGH: _ClassVar[VideoQuality]
    OFF: _ClassVar[VideoQuality]

class ConnectionQuality(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    POOR: _ClassVar[ConnectionQuality]
    GOOD: _ClassVar[ConnectionQuality]
    EXCELLENT: _ClassVar[ConnectionQuality]
    LOST: _ClassVar[ConnectionQuality]

class ClientConfigSetting(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UNSET: _ClassVar[ClientConfigSetting]
    DISABLED: _ClassVar[ClientConfigSetting]
    ENABLED: _ClassVar[ClientConfigSetting]

class DisconnectReason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UNKNOWN_REASON: _ClassVar[DisconnectReason]
    CLIENT_INITIATED: _ClassVar[DisconnectReason]
    DUPLICATE_IDENTITY: _ClassVar[DisconnectReason]
    SERVER_SHUTDOWN: _ClassVar[DisconnectReason]
    PARTICIPANT_REMOVED: _ClassVar[DisconnectReason]
    ROOM_DELETED: _ClassVar[DisconnectReason]
    STATE_MISMATCH: _ClassVar[DisconnectReason]
    JOIN_FAILURE: _ClassVar[DisconnectReason]

class ReconnectReason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RR_UNKNOWN: _ClassVar[ReconnectReason]
    RR_SIGNAL_DISCONNECTED: _ClassVar[ReconnectReason]
    RR_PUBLISHER_FAILED: _ClassVar[ReconnectReason]
    RR_SUBSCRIBER_FAILED: _ClassVar[ReconnectReason]
    RR_SWITCH_CANDIDATE: _ClassVar[ReconnectReason]

class SubscriptionError(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SE_UNKNOWN: _ClassVar[SubscriptionError]
    SE_CODEC_UNSUPPORTED: _ClassVar[SubscriptionError]
    SE_TRACK_NOTFOUND: _ClassVar[SubscriptionError]
DEFAULT_AC: AudioCodec
OPUS: AudioCodec
AAC: AudioCodec
DEFAULT_VC: VideoCodec
H264_BASELINE: VideoCodec
H264_MAIN: VideoCodec
H264_HIGH: VideoCodec
VP8: VideoCodec
IC_DEFAULT: ImageCodec
IC_JPEG: ImageCodec
AUDIO: TrackType
VIDEO: TrackType
DATA: TrackType
UNKNOWN: TrackSource
CAMERA: TrackSource
MICROPHONE: TrackSource
SCREEN_SHARE: TrackSource
SCREEN_SHARE_AUDIO: TrackSource
LOW: VideoQuality
MEDIUM: VideoQuality
HIGH: VideoQuality
OFF: VideoQuality
POOR: ConnectionQuality
GOOD: ConnectionQuality
EXCELLENT: ConnectionQuality
LOST: ConnectionQuality
UNSET: ClientConfigSetting
DISABLED: ClientConfigSetting
ENABLED: ClientConfigSetting
UNKNOWN_REASON: DisconnectReason
CLIENT_INITIATED: DisconnectReason
DUPLICATE_IDENTITY: DisconnectReason
SERVER_SHUTDOWN: DisconnectReason
PARTICIPANT_REMOVED: DisconnectReason
ROOM_DELETED: DisconnectReason
STATE_MISMATCH: DisconnectReason
JOIN_FAILURE: DisconnectReason
RR_UNKNOWN: ReconnectReason
RR_SIGNAL_DISCONNECTED: ReconnectReason
RR_PUBLISHER_FAILED: ReconnectReason
RR_SUBSCRIBER_FAILED: ReconnectReason
RR_SWITCH_CANDIDATE: ReconnectReason
SE_UNKNOWN: SubscriptionError
SE_CODEC_UNSUPPORTED: SubscriptionError
SE_TRACK_NOTFOUND: SubscriptionError

class Room(_message.Message):
    __slots__ = ("sid", "name", "empty_timeout", "max_participants", "creation_time", "turn_password", "enabled_codecs", "metadata", "num_participants", "num_publishers", "active_recording", "version")
    SID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    EMPTY_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    MAX_PARTICIPANTS_FIELD_NUMBER: _ClassVar[int]
    CREATION_TIME_FIELD_NUMBER: _ClassVar[int]
    TURN_PASSWORD_FIELD_NUMBER: _ClassVar[int]
    ENABLED_CODECS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    NUM_PARTICIPANTS_FIELD_NUMBER: _ClassVar[int]
    NUM_PUBLISHERS_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_RECORDING_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    sid: str
    name: str
    empty_timeout: int
    max_participants: int
    creation_time: int
    turn_password: str
    enabled_codecs: _containers.RepeatedCompositeFieldContainer[Codec]
    metadata: str
    num_participants: int
    num_publishers: int
    active_recording: bool
    version: TimedVersion
    def __init__(self, sid: _Optional[str] = ..., name: _Optional[str] = ..., empty_timeout: _Optional[int] = ..., max_participants: _Optional[int] = ..., creation_time: _Optional[int] = ..., turn_password: _Optional[str] = ..., enabled_codecs: _Optional[_Iterable[_Union[Codec, _Mapping]]] = ..., metadata: _Optional[str] = ..., num_participants: _Optional[int] = ..., num_publishers: _Optional[int] = ..., active_recording: bool = ..., version: _Optional[_Union[TimedVersion, _Mapping]] = ...) -> None: ...

class Codec(_message.Message):
    __slots__ = ("mime", "fmtp_line")
    MIME_FIELD_NUMBER: _ClassVar[int]
    FMTP_LINE_FIELD_NUMBER: _ClassVar[int]
    mime: str
    fmtp_line: str
    def __init__(self, mime: _Optional[str] = ..., fmtp_line: _Optional[str] = ...) -> None: ...

class PlayoutDelay(_message.Message):
    __slots__ = ("enabled", "min", "max")
    ENABLED_FIELD_NUMBER: _ClassVar[int]
    MIN_FIELD_NUMBER: _ClassVar[int]
    MAX_FIELD_NUMBER: _ClassVar[int]
    enabled: bool
    min: int
    max: int
    def __init__(self, enabled: bool = ..., min: _Optional[int] = ..., max: _Optional[int] = ...) -> None: ...

class ParticipantPermission(_message.Message):
    __slots__ = ("can_subscribe", "can_publish", "can_publish_data", "can_publish_sources", "hidden", "recorder", "can_update_metadata", "agent")
    CAN_SUBSCRIBE_FIELD_NUMBER: _ClassVar[int]
    CAN_PUBLISH_FIELD_NUMBER: _ClassVar[int]
    CAN_PUBLISH_DATA_FIELD_NUMBER: _ClassVar[int]
    CAN_PUBLISH_SOURCES_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_FIELD_NUMBER: _ClassVar[int]
    RECORDER_FIELD_NUMBER: _ClassVar[int]
    CAN_UPDATE_METADATA_FIELD_NUMBER: _ClassVar[int]
    AGENT_FIELD_NUMBER: _ClassVar[int]
    can_subscribe: bool
    can_publish: bool
    can_publish_data: bool
    can_publish_sources: _containers.RepeatedScalarFieldContainer[TrackSource]
    hidden: bool
    recorder: bool
    can_update_metadata: bool
    agent: bool
    def __init__(self, can_subscribe: bool = ..., can_publish: bool = ..., can_publish_data: bool = ..., can_publish_sources: _Optional[_Iterable[_Union[TrackSource, str]]] = ..., hidden: bool = ..., recorder: bool = ..., can_update_metadata: bool = ..., agent: bool = ...) -> None: ...

class ParticipantInfo(_message.Message):
    __slots__ = ("sid", "identity", "state", "tracks", "metadata", "joined_at", "name", "version", "permission", "region", "is_publisher", "kind")
    class State(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        JOINING: _ClassVar[ParticipantInfo.State]
        JOINED: _ClassVar[ParticipantInfo.State]
        ACTIVE: _ClassVar[ParticipantInfo.State]
        DISCONNECTED: _ClassVar[ParticipantInfo.State]
    JOINING: ParticipantInfo.State
    JOINED: ParticipantInfo.State
    ACTIVE: ParticipantInfo.State
    DISCONNECTED: ParticipantInfo.State
    class Kind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STANDARD: _ClassVar[ParticipantInfo.Kind]
        INGRESS: _ClassVar[ParticipantInfo.Kind]
        EGRESS: _ClassVar[ParticipantInfo.Kind]
        SIP: _ClassVar[ParticipantInfo.Kind]
        AGENT: _ClassVar[ParticipantInfo.Kind]
    STANDARD: ParticipantInfo.Kind
    INGRESS: ParticipantInfo.Kind
    EGRESS: ParticipantInfo.Kind
    SIP: ParticipantInfo.Kind
    AGENT: ParticipantInfo.Kind
    SID_FIELD_NUMBER: _ClassVar[int]
    IDENTITY_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    TRACKS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    JOINED_AT_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    PERMISSION_FIELD_NUMBER: _ClassVar[int]
    REGION_FIELD_NUMBER: _ClassVar[int]
    IS_PUBLISHER_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    sid: str
    identity: str
    state: ParticipantInfo.State
    tracks: _containers.RepeatedCompositeFieldContainer[TrackInfo]
    metadata: str
    joined_at: int
    name: str
    version: int
    permission: ParticipantPermission
    region: str
    is_publisher: bool
    kind: ParticipantInfo.Kind
    def __init__(self, sid: _Optional[str] = ..., identity: _Optional[str] = ..., state: _Optional[_Union[ParticipantInfo.State, str]] = ..., tracks: _Optional[_Iterable[_Union[TrackInfo, _Mapping]]] = ..., metadata: _Optional[str] = ..., joined_at: _Optional[int] = ..., name: _Optional[str] = ..., version: _Optional[int] = ..., permission: _Optional[_Union[ParticipantPermission, _Mapping]] = ..., region: _Optional[str] = ..., is_publisher: bool = ..., kind: _Optional[_Union[ParticipantInfo.Kind, str]] = ...) -> None: ...

class Encryption(_message.Message):
    __slots__ = ()
    class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        NONE: _ClassVar[Encryption.Type]
        GCM: _ClassVar[Encryption.Type]
        CUSTOM: _ClassVar[Encryption.Type]
    NONE: Encryption.Type
    GCM: Encryption.Type
    CUSTOM: Encryption.Type
    def __init__(self) -> None: ...

class SimulcastCodecInfo(_message.Message):
    __slots__ = ("mime_type", "mid", "cid", "layers")
    MIME_TYPE_FIELD_NUMBER: _ClassVar[int]
    MID_FIELD_NUMBER: _ClassVar[int]
    CID_FIELD_NUMBER: _ClassVar[int]
    LAYERS_FIELD_NUMBER: _ClassVar[int]
    mime_type: str
    mid: str
    cid: str
    layers: _containers.RepeatedCompositeFieldContainer[VideoLayer]
    def __init__(self, mime_type: _Optional[str] = ..., mid: _Optional[str] = ..., cid: _Optional[str] = ..., layers: _Optional[_Iterable[_Union[VideoLayer, _Mapping]]] = ...) -> None: ...

class TrackInfo(_message.Message):
    __slots__ = ("sid", "type", "name", "muted", "width", "height", "simulcast", "disable_dtx", "source", "layers", "mime_type", "mid", "codecs", "stereo", "disable_red", "encryption", "stream", "version")
    SID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    MUTED_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    SIMULCAST_FIELD_NUMBER: _ClassVar[int]
    DISABLE_DTX_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    LAYERS_FIELD_NUMBER: _ClassVar[int]
    MIME_TYPE_FIELD_NUMBER: _ClassVar[int]
    MID_FIELD_NUMBER: _ClassVar[int]
    CODECS_FIELD_NUMBER: _ClassVar[int]
    STEREO_FIELD_NUMBER: _ClassVar[int]
    DISABLE_RED_FIELD_NUMBER: _ClassVar[int]
    ENCRYPTION_FIELD_NUMBER: _ClassVar[int]
    STREAM_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    sid: str
    type: TrackType
    name: str
    muted: bool
    width: int
    height: int
    simulcast: bool
    disable_dtx: bool
    source: TrackSource
    layers: _containers.RepeatedCompositeFieldContainer[VideoLayer]
    mime_type: str
    mid: str
    codecs: _containers.RepeatedCompositeFieldContainer[SimulcastCodecInfo]
    stereo: bool
    disable_red: bool
    encryption: Encryption.Type
    stream: str
    version: TimedVersion
    def __init__(self, sid: _Optional[str] = ..., type: _Optional[_Union[TrackType, str]] = ..., name: _Optional[str] = ..., muted: bool = ..., width: _Optional[int] = ..., height: _Optional[int] = ..., simulcast: bool = ..., disable_dtx: bool = ..., source: _Optional[_Union[TrackSource, str]] = ..., layers: _Optional[_Iterable[_Union[VideoLayer, _Mapping]]] = ..., mime_type: _Optional[str] = ..., mid: _Optional[str] = ..., codecs: _Optional[_Iterable[_Union[SimulcastCodecInfo, _Mapping]]] = ..., stereo: bool = ..., disable_red: bool = ..., encryption: _Optional[_Union[Encryption.Type, str]] = ..., stream: _Optional[str] = ..., version: _Optional[_Union[TimedVersion, _Mapping]] = ...) -> None: ...

class VideoLayer(_message.Message):
    __slots__ = ("quality", "width", "height", "bitrate", "ssrc")
    QUALITY_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    BITRATE_FIELD_NUMBER: _ClassVar[int]
    SSRC_FIELD_NUMBER: _ClassVar[int]
    quality: VideoQuality
    width: int
    height: int
    bitrate: int
    ssrc: int
    def __init__(self, quality: _Optional[_Union[VideoQuality, str]] = ..., width: _Optional[int] = ..., height: _Optional[int] = ..., bitrate: _Optional[int] = ..., ssrc: _Optional[int] = ...) -> None: ...

class DataPacket(_message.Message):
    __slots__ = ("kind", "user", "speaker")
    class Kind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        RELIABLE: _ClassVar[DataPacket.Kind]
        LOSSY: _ClassVar[DataPacket.Kind]
    RELIABLE: DataPacket.Kind
    LOSSY: DataPacket.Kind
    KIND_FIELD_NUMBER: _ClassVar[int]
    USER_FIELD_NUMBER: _ClassVar[int]
    SPEAKER_FIELD_NUMBER: _ClassVar[int]
    kind: DataPacket.Kind
    user: UserPacket
    speaker: ActiveSpeakerUpdate
    def __init__(self, kind: _Optional[_Union[DataPacket.Kind, str]] = ..., user: _Optional[_Union[UserPacket, _Mapping]] = ..., speaker: _Optional[_Union[ActiveSpeakerUpdate, _Mapping]] = ...) -> None: ...

class ActiveSpeakerUpdate(_message.Message):
    __slots__ = ("speakers",)
    SPEAKERS_FIELD_NUMBER: _ClassVar[int]
    speakers: _containers.RepeatedCompositeFieldContainer[SpeakerInfo]
    def __init__(self, speakers: _Optional[_Iterable[_Union[SpeakerInfo, _Mapping]]] = ...) -> None: ...

class SpeakerInfo(_message.Message):
    __slots__ = ("sid", "level", "active")
    SID_FIELD_NUMBER: _ClassVar[int]
    LEVEL_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_FIELD_NUMBER: _ClassVar[int]
    sid: str
    level: float
    active: bool
    def __init__(self, sid: _Optional[str] = ..., level: _Optional[float] = ..., active: bool = ...) -> None: ...

class UserPacket(_message.Message):
    __slots__ = ("participant_sid", "participant_identity", "payload", "destination_sids", "destination_identities", "topic")
    PARTICIPANT_SID_FIELD_NUMBER: _ClassVar[int]
    PARTICIPANT_IDENTITY_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    DESTINATION_SIDS_FIELD_NUMBER: _ClassVar[int]
    DESTINATION_IDENTITIES_FIELD_NUMBER: _ClassVar[int]
    TOPIC_FIELD_NUMBER: _ClassVar[int]
    participant_sid: str
    participant_identity: str
    payload: bytes
    destination_sids: _containers.RepeatedScalarFieldContainer[str]
    destination_identities: _containers.RepeatedScalarFieldContainer[str]
    topic: str
    def __init__(self, participant_sid: _Optional[str] = ..., participant_identity: _Optional[str] = ..., payload: _Optional[bytes] = ..., destination_sids: _Optional[_Iterable[str]] = ..., destination_identities: _Optional[_Iterable[str]] = ..., topic: _Optional[str] = ...) -> None: ...

class ParticipantTracks(_message.Message):
    __slots__ = ("participant_sid", "track_sids")
    PARTICIPANT_SID_FIELD_NUMBER: _ClassVar[int]
    TRACK_SIDS_FIELD_NUMBER: _ClassVar[int]
    participant_sid: str
    track_sids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, participant_sid: _Optional[str] = ..., track_sids: _Optional[_Iterable[str]] = ...) -> None: ...

class ServerInfo(_message.Message):
    __slots__ = ("edition", "version", "protocol", "region", "node_id", "debug_info")
    class Edition(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        Standard: _ClassVar[ServerInfo.Edition]
        Cloud: _ClassVar[ServerInfo.Edition]
    Standard: ServerInfo.Edition
    Cloud: ServerInfo.Edition
    EDITION_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_FIELD_NUMBER: _ClassVar[int]
    REGION_FIELD_NUMBER: _ClassVar[int]
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    DEBUG_INFO_FIELD_NUMBER: _ClassVar[int]
    edition: ServerInfo.Edition
    version: str
    protocol: int
    region: str
    node_id: str
    debug_info: str
    def __init__(self, edition: _Optional[_Union[ServerInfo.Edition, str]] = ..., version: _Optional[str] = ..., protocol: _Optional[int] = ..., region: _Optional[str] = ..., node_id: _Optional[str] = ..., debug_info: _Optional[str] = ...) -> None: ...

class ClientInfo(_message.Message):
    __slots__ = ("sdk", "version", "protocol", "os", "os_version", "device_model", "browser", "browser_version", "address", "network")
    class SDK(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        UNKNOWN: _ClassVar[ClientInfo.SDK]
        JS: _ClassVar[ClientInfo.SDK]
        SWIFT: _ClassVar[ClientInfo.SDK]
        ANDROID: _ClassVar[ClientInfo.SDK]
        FLUTTER: _ClassVar[ClientInfo.SDK]
        GO: _ClassVar[ClientInfo.SDK]
        UNITY: _ClassVar[ClientInfo.SDK]
        REACT_NATIVE: _ClassVar[ClientInfo.SDK]
        RUST: _ClassVar[ClientInfo.SDK]
        PYTHON: _ClassVar[ClientInfo.SDK]
        CPP: _ClassVar[ClientInfo.SDK]
    UNKNOWN: ClientInfo.SDK
    JS: ClientInfo.SDK
    SWIFT: ClientInfo.SDK
    ANDROID: ClientInfo.SDK
    FLUTTER: ClientInfo.SDK
    GO: ClientInfo.SDK
    UNITY: ClientInfo.SDK
    REACT_NATIVE: ClientInfo.SDK
    RUST: ClientInfo.SDK
    PYTHON: ClientInfo.SDK
    CPP: ClientInfo.SDK
    SDK_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_FIELD_NUMBER: _ClassVar[int]
    OS_FIELD_NUMBER: _ClassVar[int]
    OS_VERSION_FIELD_NUMBER: _ClassVar[int]
    DEVICE_MODEL_FIELD_NUMBER: _ClassVar[int]
    BROWSER_FIELD_NUMBER: _ClassVar[int]
    BROWSER_VERSION_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    NETWORK_FIELD_NUMBER: _ClassVar[int]
    sdk: ClientInfo.SDK
    version: str
    protocol: int
    os: str
    os_version: str
    device_model: str
    browser: str
    browser_version: str
    address: str
    network: str
    def __init__(self, sdk: _Optional[_Union[ClientInfo.SDK, str]] = ..., version: _Optional[str] = ..., protocol: _Optional[int] = ..., os: _Optional[str] = ..., os_version: _Optional[str] = ..., device_model: _Optional[str] = ..., browser: _Optional[str] = ..., browser_version: _Optional[str] = ..., address: _Optional[str] = ..., network: _Optional[str] = ...) -> None: ...

class ClientConfiguration(_message.Message):
    __slots__ = ("video", "screen", "resume_connection", "disabled_codecs", "force_relay")
    VIDEO_FIELD_NUMBER: _ClassVar[int]
    SCREEN_FIELD_NUMBER: _ClassVar[int]
    RESUME_CONNECTION_FIELD_NUMBER: _ClassVar[int]
    DISABLED_CODECS_FIELD_NUMBER: _ClassVar[int]
    FORCE_RELAY_FIELD_NUMBER: _ClassVar[int]
    video: VideoConfiguration
    screen: VideoConfiguration
    resume_connection: ClientConfigSetting
    disabled_codecs: DisabledCodecs
    force_relay: ClientConfigSetting
    def __init__(self, video: _Optional[_Union[VideoConfiguration, _Mapping]] = ..., screen: _Optional[_Union[VideoConfiguration, _Mapping]] = ..., resume_connection: _Optional[_Union[ClientConfigSetting, str]] = ..., disabled_codecs: _Optional[_Union[DisabledCodecs, _Mapping]] = ..., force_relay: _Optional[_Union[ClientConfigSetting, str]] = ...) -> None: ...

class VideoConfiguration(_message.Message):
    __slots__ = ("hardware_encoder",)
    HARDWARE_ENCODER_FIELD_NUMBER: _ClassVar[int]
    hardware_encoder: ClientConfigSetting
    def __init__(self, hardware_encoder: _Optional[_Union[ClientConfigSetting, str]] = ...) -> None: ...

class DisabledCodecs(_message.Message):
    __slots__ = ("codecs", "publish")
    CODECS_FIELD_NUMBER: _ClassVar[int]
    PUBLISH_FIELD_NUMBER: _ClassVar[int]
    codecs: _containers.RepeatedCompositeFieldContainer[Codec]
    publish: _containers.RepeatedCompositeFieldContainer[Codec]
    def __init__(self, codecs: _Optional[_Iterable[_Union[Codec, _Mapping]]] = ..., publish: _Optional[_Iterable[_Union[Codec, _Mapping]]] = ...) -> None: ...

class RTPDrift(_message.Message):
    __slots__ = ("start_time", "end_time", "duration", "start_timestamp", "end_timestamp", "rtp_clock_ticks", "drift_samples", "drift_ms", "clock_rate")
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    DURATION_FIELD_NUMBER: _ClassVar[int]
    START_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    END_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RTP_CLOCK_TICKS_FIELD_NUMBER: _ClassVar[int]
    DRIFT_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    DRIFT_MS_FIELD_NUMBER: _ClassVar[int]
    CLOCK_RATE_FIELD_NUMBER: _ClassVar[int]
    start_time: _timestamp_pb2.Timestamp
    end_time: _timestamp_pb2.Timestamp
    duration: float
    start_timestamp: int
    end_timestamp: int
    rtp_clock_ticks: int
    drift_samples: int
    drift_ms: float
    clock_rate: float
    def __init__(self, start_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., end_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., duration: _Optional[float] = ..., start_timestamp: _Optional[int] = ..., end_timestamp: _Optional[int] = ..., rtp_clock_ticks: _Optional[int] = ..., drift_samples: _Optional[int] = ..., drift_ms: _Optional[float] = ..., clock_rate: _Optional[float] = ...) -> None: ...

class RTPStats(_message.Message):
    __slots__ = ("start_time", "end_time", "duration", "packets", "packet_rate", "bytes", "header_bytes", "bitrate", "packets_lost", "packet_loss_rate", "packet_loss_percentage", "packets_duplicate", "packet_duplicate_rate", "bytes_duplicate", "header_bytes_duplicate", "bitrate_duplicate", "packets_padding", "packet_padding_rate", "bytes_padding", "header_bytes_padding", "bitrate_padding", "packets_out_of_order", "frames", "frame_rate", "jitter_current", "jitter_max", "gap_histogram", "nacks", "nack_acks", "nack_misses", "nack_repeated", "plis", "last_pli", "firs", "last_fir", "rtt_current", "rtt_max", "key_frames", "last_key_frame", "layer_lock_plis", "last_layer_lock_pli", "packet_drift", "report_drift")
    class GapHistogramEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: int
        def __init__(self, key: _Optional[int] = ..., value: _Optional[int] = ...) -> None: ...
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    DURATION_FIELD_NUMBER: _ClassVar[int]
    PACKETS_FIELD_NUMBER: _ClassVar[int]
    PACKET_RATE_FIELD_NUMBER: _ClassVar[int]
    BYTES_FIELD_NUMBER: _ClassVar[int]
    HEADER_BYTES_FIELD_NUMBER: _ClassVar[int]
    BITRATE_FIELD_NUMBER: _ClassVar[int]
    PACKETS_LOST_FIELD_NUMBER: _ClassVar[int]
    PACKET_LOSS_RATE_FIELD_NUMBER: _ClassVar[int]
    PACKET_LOSS_PERCENTAGE_FIELD_NUMBER: _ClassVar[int]
    PACKETS_DUPLICATE_FIELD_NUMBER: _ClassVar[int]
    PACKET_DUPLICATE_RATE_FIELD_NUMBER: _ClassVar[int]
    BYTES_DUPLICATE_FIELD_NUMBER: _ClassVar[int]
    HEADER_BYTES_DUPLICATE_FIELD_NUMBER: _ClassVar[int]
    BITRATE_DUPLICATE_FIELD_NUMBER: _ClassVar[int]
    PACKETS_PADDING_FIELD_NUMBER: _ClassVar[int]
    PACKET_PADDING_RATE_FIELD_NUMBER: _ClassVar[int]
    BYTES_PADDING_FIELD_NUMBER: _ClassVar[int]
    HEADER_BYTES_PADDING_FIELD_NUMBER: _ClassVar[int]
    BITRATE_PADDING_FIELD_NUMBER: _ClassVar[int]
    PACKETS_OUT_OF_ORDER_FIELD_NUMBER: _ClassVar[int]
    FRAMES_FIELD_NUMBER: _ClassVar[int]
    FRAME_RATE_FIELD_NUMBER: _ClassVar[int]
    JITTER_CURRENT_FIELD_NUMBER: _ClassVar[int]
    JITTER_MAX_FIELD_NUMBER: _ClassVar[int]
    GAP_HISTOGRAM_FIELD_NUMBER: _ClassVar[int]
    NACKS_FIELD_NUMBER: _ClassVar[int]
    NACK_ACKS_FIELD_NUMBER: _ClassVar[int]
    NACK_MISSES_FIELD_NUMBER: _ClassVar[int]
    NACK_REPEATED_FIELD_NUMBER: _ClassVar[int]
    PLIS_FIELD_NUMBER: _ClassVar[int]
    LAST_PLI_FIELD_NUMBER: _ClassVar[int]
    FIRS_FIELD_NUMBER: _ClassVar[int]
    LAST_FIR_FIELD_NUMBER: _ClassVar[int]
    RTT_CURRENT_FIELD_NUMBER: _ClassVar[int]
    RTT_MAX_FIELD_NUMBER: _ClassVar[int]
    KEY_FRAMES_FIELD_NUMBER: _ClassVar[int]
    LAST_KEY_FRAME_FIELD_NUMBER: _ClassVar[int]
    LAYER_LOCK_PLIS_FIELD_NUMBER: _ClassVar[int]
    LAST_LAYER_LOCK_PLI_FIELD_NUMBER: _ClassVar[int]
    PACKET_DRIFT_FIELD_NUMBER: _ClassVar[int]
    REPORT_DRIFT_FIELD_NUMBER: _ClassVar[int]
    start_time: _timestamp_pb2.Timestamp
    end_time: _timestamp_pb2.Timestamp
    duration: float
    packets: int
    packet_rate: float
    bytes: int
    header_bytes: int
    bitrate: float
    packets_lost: int
    packet_loss_rate: float
    packet_loss_percentage: float
    packets_duplicate: int
    packet_duplicate_rate: float
    bytes_duplicate: int
    header_bytes_duplicate: int
    bitrate_duplicate: float
    packets_padding: int
    packet_padding_rate: float
    bytes_padding: int
    header_bytes_padding: int
    bitrate_padding: float
    packets_out_of_order: int
    frames: int
    frame_rate: float
    jitter_current: float
    jitter_max: float
    gap_histogram: _containers.ScalarMap[int, int]
    nacks: int
    nack_acks: int
    nack_misses: int
    nack_repeated: int
    plis: int
    last_pli: _timestamp_pb2.Timestamp
    firs: int
    last_fir: _timestamp_pb2.Timestamp
    rtt_current: int
    rtt_max: int
    key_frames: int
    last_key_frame: _timestamp_pb2.Timestamp
    layer_lock_plis: int
    last_layer_lock_pli: _timestamp_pb2.Timestamp
    packet_drift: RTPDrift
    report_drift: RTPDrift
    def __init__(self, start_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., end_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., duration: _Optional[float] = ..., packets: _Optional[int] = ..., packet_rate: _Optional[float] = ..., bytes: _Optional[int] = ..., header_bytes: _Optional[int] = ..., bitrate: _Optional[float] = ..., packets_lost: _Optional[int] = ..., packet_loss_rate: _Optional[float] = ..., packet_loss_percentage: _Optional[float] = ..., packets_duplicate: _Optional[int] = ..., packet_duplicate_rate: _Optional[float] = ..., bytes_duplicate: _Optional[int] = ..., header_bytes_duplicate: _Optional[int] = ..., bitrate_duplicate: _Optional[float] = ..., packets_padding: _Optional[int] = ..., packet_padding_rate: _Optional[float] = ..., bytes_padding: _Optional[int] = ..., header_bytes_padding: _Optional[int] = ..., bitrate_padding: _Optional[float] = ..., packets_out_of_order: _Optional[int] = ..., frames: _Optional[int] = ..., frame_rate: _Optional[float] = ..., jitter_current: _Optional[float] = ..., jitter_max: _Optional[float] = ..., gap_histogram: _Optional[_Mapping[int, int]] = ..., nacks: _Optional[int] = ..., nack_acks: _Optional[int] = ..., nack_misses: _Optional[int] = ..., nack_repeated: _Optional[int] = ..., plis: _Optional[int] = ..., last_pli: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., firs: _Optional[int] = ..., last_fir: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., rtt_current: _Optional[int] = ..., rtt_max: _Optional[int] = ..., key_frames: _Optional[int] = ..., last_key_frame: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., layer_lock_plis: _Optional[int] = ..., last_layer_lock_pli: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., packet_drift: _Optional[_Union[RTPDrift, _Mapping]] = ..., report_drift: _Optional[_Union[RTPDrift, _Mapping]] = ...) -> None: ...

class TimedVersion(_message.Message):
    __slots__ = ("unix_micro", "ticks")
    UNIX_MICRO_FIELD_NUMBER: _ClassVar[int]
    TICKS_FIELD_NUMBER: _ClassVar[int]
    unix_micro: int
    ticks: int
    def __init__(self, unix_micro: _Optional[int] = ..., ticks: _Optional[int] = ...) -> None: ...
