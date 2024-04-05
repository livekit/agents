from . import models as _models
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class IngressInput(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RTMP_INPUT: _ClassVar[IngressInput]
    WHIP_INPUT: _ClassVar[IngressInput]
    URL_INPUT: _ClassVar[IngressInput]

class IngressAudioEncodingPreset(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OPUS_STEREO_96KBPS: _ClassVar[IngressAudioEncodingPreset]
    OPUS_MONO_64KBS: _ClassVar[IngressAudioEncodingPreset]

class IngressVideoEncodingPreset(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    H264_720P_30FPS_3_LAYERS: _ClassVar[IngressVideoEncodingPreset]
    H264_1080P_30FPS_3_LAYERS: _ClassVar[IngressVideoEncodingPreset]
    H264_540P_25FPS_2_LAYERS: _ClassVar[IngressVideoEncodingPreset]
    H264_720P_30FPS_1_LAYER: _ClassVar[IngressVideoEncodingPreset]
    H264_1080P_30FPS_1_LAYER: _ClassVar[IngressVideoEncodingPreset]
    H264_720P_30FPS_3_LAYERS_HIGH_MOTION: _ClassVar[IngressVideoEncodingPreset]
    H264_1080P_30FPS_3_LAYERS_HIGH_MOTION: _ClassVar[IngressVideoEncodingPreset]
    H264_540P_25FPS_2_LAYERS_HIGH_MOTION: _ClassVar[IngressVideoEncodingPreset]
    H264_720P_30FPS_1_LAYER_HIGH_MOTION: _ClassVar[IngressVideoEncodingPreset]
    H264_1080P_30FPS_1_LAYER_HIGH_MOTION: _ClassVar[IngressVideoEncodingPreset]
RTMP_INPUT: IngressInput
WHIP_INPUT: IngressInput
URL_INPUT: IngressInput
OPUS_STEREO_96KBPS: IngressAudioEncodingPreset
OPUS_MONO_64KBS: IngressAudioEncodingPreset
H264_720P_30FPS_3_LAYERS: IngressVideoEncodingPreset
H264_1080P_30FPS_3_LAYERS: IngressVideoEncodingPreset
H264_540P_25FPS_2_LAYERS: IngressVideoEncodingPreset
H264_720P_30FPS_1_LAYER: IngressVideoEncodingPreset
H264_1080P_30FPS_1_LAYER: IngressVideoEncodingPreset
H264_720P_30FPS_3_LAYERS_HIGH_MOTION: IngressVideoEncodingPreset
H264_1080P_30FPS_3_LAYERS_HIGH_MOTION: IngressVideoEncodingPreset
H264_540P_25FPS_2_LAYERS_HIGH_MOTION: IngressVideoEncodingPreset
H264_720P_30FPS_1_LAYER_HIGH_MOTION: IngressVideoEncodingPreset
H264_1080P_30FPS_1_LAYER_HIGH_MOTION: IngressVideoEncodingPreset

class CreateIngressRequest(_message.Message):
    __slots__ = ("input_type", "url", "name", "room_name", "participant_identity", "participant_name", "bypass_transcoding", "audio", "video")
    INPUT_TYPE_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ROOM_NAME_FIELD_NUMBER: _ClassVar[int]
    PARTICIPANT_IDENTITY_FIELD_NUMBER: _ClassVar[int]
    PARTICIPANT_NAME_FIELD_NUMBER: _ClassVar[int]
    BYPASS_TRANSCODING_FIELD_NUMBER: _ClassVar[int]
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    VIDEO_FIELD_NUMBER: _ClassVar[int]
    input_type: IngressInput
    url: str
    name: str
    room_name: str
    participant_identity: str
    participant_name: str
    bypass_transcoding: bool
    audio: IngressAudioOptions
    video: IngressVideoOptions
    def __init__(self, input_type: _Optional[_Union[IngressInput, str]] = ..., url: _Optional[str] = ..., name: _Optional[str] = ..., room_name: _Optional[str] = ..., participant_identity: _Optional[str] = ..., participant_name: _Optional[str] = ..., bypass_transcoding: bool = ..., audio: _Optional[_Union[IngressAudioOptions, _Mapping]] = ..., video: _Optional[_Union[IngressVideoOptions, _Mapping]] = ...) -> None: ...

class IngressAudioOptions(_message.Message):
    __slots__ = ("name", "source", "preset", "options")
    NAME_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    PRESET_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    name: str
    source: _models.TrackSource
    preset: IngressAudioEncodingPreset
    options: IngressAudioEncodingOptions
    def __init__(self, name: _Optional[str] = ..., source: _Optional[_Union[_models.TrackSource, str]] = ..., preset: _Optional[_Union[IngressAudioEncodingPreset, str]] = ..., options: _Optional[_Union[IngressAudioEncodingOptions, _Mapping]] = ...) -> None: ...

class IngressVideoOptions(_message.Message):
    __slots__ = ("name", "source", "preset", "options")
    NAME_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    PRESET_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    name: str
    source: _models.TrackSource
    preset: IngressVideoEncodingPreset
    options: IngressVideoEncodingOptions
    def __init__(self, name: _Optional[str] = ..., source: _Optional[_Union[_models.TrackSource, str]] = ..., preset: _Optional[_Union[IngressVideoEncodingPreset, str]] = ..., options: _Optional[_Union[IngressVideoEncodingOptions, _Mapping]] = ...) -> None: ...

class IngressAudioEncodingOptions(_message.Message):
    __slots__ = ("audio_codec", "bitrate", "disable_dtx", "channels")
    AUDIO_CODEC_FIELD_NUMBER: _ClassVar[int]
    BITRATE_FIELD_NUMBER: _ClassVar[int]
    DISABLE_DTX_FIELD_NUMBER: _ClassVar[int]
    CHANNELS_FIELD_NUMBER: _ClassVar[int]
    audio_codec: _models.AudioCodec
    bitrate: int
    disable_dtx: bool
    channels: int
    def __init__(self, audio_codec: _Optional[_Union[_models.AudioCodec, str]] = ..., bitrate: _Optional[int] = ..., disable_dtx: bool = ..., channels: _Optional[int] = ...) -> None: ...

class IngressVideoEncodingOptions(_message.Message):
    __slots__ = ("video_codec", "frame_rate", "layers")
    VIDEO_CODEC_FIELD_NUMBER: _ClassVar[int]
    FRAME_RATE_FIELD_NUMBER: _ClassVar[int]
    LAYERS_FIELD_NUMBER: _ClassVar[int]
    video_codec: _models.VideoCodec
    frame_rate: float
    layers: _containers.RepeatedCompositeFieldContainer[_models.VideoLayer]
    def __init__(self, video_codec: _Optional[_Union[_models.VideoCodec, str]] = ..., frame_rate: _Optional[float] = ..., layers: _Optional[_Iterable[_Union[_models.VideoLayer, _Mapping]]] = ...) -> None: ...

class IngressInfo(_message.Message):
    __slots__ = ("ingress_id", "name", "stream_key", "url", "input_type", "bypass_transcoding", "audio", "video", "room_name", "participant_identity", "participant_name", "reusable", "state")
    INGRESS_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    STREAM_KEY_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    INPUT_TYPE_FIELD_NUMBER: _ClassVar[int]
    BYPASS_TRANSCODING_FIELD_NUMBER: _ClassVar[int]
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    VIDEO_FIELD_NUMBER: _ClassVar[int]
    ROOM_NAME_FIELD_NUMBER: _ClassVar[int]
    PARTICIPANT_IDENTITY_FIELD_NUMBER: _ClassVar[int]
    PARTICIPANT_NAME_FIELD_NUMBER: _ClassVar[int]
    REUSABLE_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    ingress_id: str
    name: str
    stream_key: str
    url: str
    input_type: IngressInput
    bypass_transcoding: bool
    audio: IngressAudioOptions
    video: IngressVideoOptions
    room_name: str
    participant_identity: str
    participant_name: str
    reusable: bool
    state: IngressState
    def __init__(self, ingress_id: _Optional[str] = ..., name: _Optional[str] = ..., stream_key: _Optional[str] = ..., url: _Optional[str] = ..., input_type: _Optional[_Union[IngressInput, str]] = ..., bypass_transcoding: bool = ..., audio: _Optional[_Union[IngressAudioOptions, _Mapping]] = ..., video: _Optional[_Union[IngressVideoOptions, _Mapping]] = ..., room_name: _Optional[str] = ..., participant_identity: _Optional[str] = ..., participant_name: _Optional[str] = ..., reusable: bool = ..., state: _Optional[_Union[IngressState, _Mapping]] = ...) -> None: ...

class IngressState(_message.Message):
    __slots__ = ("status", "error", "video", "audio", "room_id", "started_at", "ended_at", "resource_id", "tracks")
    class Status(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        ENDPOINT_INACTIVE: _ClassVar[IngressState.Status]
        ENDPOINT_BUFFERING: _ClassVar[IngressState.Status]
        ENDPOINT_PUBLISHING: _ClassVar[IngressState.Status]
        ENDPOINT_ERROR: _ClassVar[IngressState.Status]
        ENDPOINT_COMPLETE: _ClassVar[IngressState.Status]
    ENDPOINT_INACTIVE: IngressState.Status
    ENDPOINT_BUFFERING: IngressState.Status
    ENDPOINT_PUBLISHING: IngressState.Status
    ENDPOINT_ERROR: IngressState.Status
    ENDPOINT_COMPLETE: IngressState.Status
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    VIDEO_FIELD_NUMBER: _ClassVar[int]
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    ROOM_ID_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    ENDED_AT_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    TRACKS_FIELD_NUMBER: _ClassVar[int]
    status: IngressState.Status
    error: str
    video: InputVideoState
    audio: InputAudioState
    room_id: str
    started_at: int
    ended_at: int
    resource_id: str
    tracks: _containers.RepeatedCompositeFieldContainer[_models.TrackInfo]
    def __init__(self, status: _Optional[_Union[IngressState.Status, str]] = ..., error: _Optional[str] = ..., video: _Optional[_Union[InputVideoState, _Mapping]] = ..., audio: _Optional[_Union[InputAudioState, _Mapping]] = ..., room_id: _Optional[str] = ..., started_at: _Optional[int] = ..., ended_at: _Optional[int] = ..., resource_id: _Optional[str] = ..., tracks: _Optional[_Iterable[_Union[_models.TrackInfo, _Mapping]]] = ...) -> None: ...

class InputVideoState(_message.Message):
    __slots__ = ("mime_type", "average_bitrate", "width", "height", "framerate")
    MIME_TYPE_FIELD_NUMBER: _ClassVar[int]
    AVERAGE_BITRATE_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    FRAMERATE_FIELD_NUMBER: _ClassVar[int]
    mime_type: str
    average_bitrate: int
    width: int
    height: int
    framerate: float
    def __init__(self, mime_type: _Optional[str] = ..., average_bitrate: _Optional[int] = ..., width: _Optional[int] = ..., height: _Optional[int] = ..., framerate: _Optional[float] = ...) -> None: ...

class InputAudioState(_message.Message):
    __slots__ = ("mime_type", "average_bitrate", "channels", "sample_rate")
    MIME_TYPE_FIELD_NUMBER: _ClassVar[int]
    AVERAGE_BITRATE_FIELD_NUMBER: _ClassVar[int]
    CHANNELS_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    mime_type: str
    average_bitrate: int
    channels: int
    sample_rate: int
    def __init__(self, mime_type: _Optional[str] = ..., average_bitrate: _Optional[int] = ..., channels: _Optional[int] = ..., sample_rate: _Optional[int] = ...) -> None: ...

class UpdateIngressRequest(_message.Message):
    __slots__ = ("ingress_id", "name", "room_name", "participant_identity", "participant_name", "bypass_transcoding", "audio", "video")
    INGRESS_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ROOM_NAME_FIELD_NUMBER: _ClassVar[int]
    PARTICIPANT_IDENTITY_FIELD_NUMBER: _ClassVar[int]
    PARTICIPANT_NAME_FIELD_NUMBER: _ClassVar[int]
    BYPASS_TRANSCODING_FIELD_NUMBER: _ClassVar[int]
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    VIDEO_FIELD_NUMBER: _ClassVar[int]
    ingress_id: str
    name: str
    room_name: str
    participant_identity: str
    participant_name: str
    bypass_transcoding: bool
    audio: IngressAudioOptions
    video: IngressVideoOptions
    def __init__(self, ingress_id: _Optional[str] = ..., name: _Optional[str] = ..., room_name: _Optional[str] = ..., participant_identity: _Optional[str] = ..., participant_name: _Optional[str] = ..., bypass_transcoding: bool = ..., audio: _Optional[_Union[IngressAudioOptions, _Mapping]] = ..., video: _Optional[_Union[IngressVideoOptions, _Mapping]] = ...) -> None: ...

class ListIngressRequest(_message.Message):
    __slots__ = ("room_name", "ingress_id")
    ROOM_NAME_FIELD_NUMBER: _ClassVar[int]
    INGRESS_ID_FIELD_NUMBER: _ClassVar[int]
    room_name: str
    ingress_id: str
    def __init__(self, room_name: _Optional[str] = ..., ingress_id: _Optional[str] = ...) -> None: ...

class ListIngressResponse(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[IngressInfo]
    def __init__(self, items: _Optional[_Iterable[_Union[IngressInfo, _Mapping]]] = ...) -> None: ...

class DeleteIngressRequest(_message.Message):
    __slots__ = ("ingress_id",)
    INGRESS_ID_FIELD_NUMBER: _ClassVar[int]
    ingress_id: str
    def __init__(self, ingress_id: _Optional[str] = ...) -> None: ...
