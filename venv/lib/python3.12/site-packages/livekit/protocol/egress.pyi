from . import models as _models
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class EncodedFileType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DEFAULT_FILETYPE: _ClassVar[EncodedFileType]
    MP4: _ClassVar[EncodedFileType]
    OGG: _ClassVar[EncodedFileType]

class SegmentedFileProtocol(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DEFAULT_SEGMENTED_FILE_PROTOCOL: _ClassVar[SegmentedFileProtocol]
    HLS_PROTOCOL: _ClassVar[SegmentedFileProtocol]

class SegmentedFileSuffix(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    INDEX: _ClassVar[SegmentedFileSuffix]
    TIMESTAMP: _ClassVar[SegmentedFileSuffix]

class ImageFileSuffix(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    IMAGE_SUFFIX_INDEX: _ClassVar[ImageFileSuffix]
    IMAGE_SUFFIX_TIMESTAMP: _ClassVar[ImageFileSuffix]

class StreamProtocol(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DEFAULT_PROTOCOL: _ClassVar[StreamProtocol]
    RTMP: _ClassVar[StreamProtocol]

class EncodingOptionsPreset(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    H264_720P_30: _ClassVar[EncodingOptionsPreset]
    H264_720P_60: _ClassVar[EncodingOptionsPreset]
    H264_1080P_30: _ClassVar[EncodingOptionsPreset]
    H264_1080P_60: _ClassVar[EncodingOptionsPreset]
    PORTRAIT_H264_720P_30: _ClassVar[EncodingOptionsPreset]
    PORTRAIT_H264_720P_60: _ClassVar[EncodingOptionsPreset]
    PORTRAIT_H264_1080P_30: _ClassVar[EncodingOptionsPreset]
    PORTRAIT_H264_1080P_60: _ClassVar[EncodingOptionsPreset]

class EgressStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EGRESS_STARTING: _ClassVar[EgressStatus]
    EGRESS_ACTIVE: _ClassVar[EgressStatus]
    EGRESS_ENDING: _ClassVar[EgressStatus]
    EGRESS_COMPLETE: _ClassVar[EgressStatus]
    EGRESS_FAILED: _ClassVar[EgressStatus]
    EGRESS_ABORTED: _ClassVar[EgressStatus]
    EGRESS_LIMIT_REACHED: _ClassVar[EgressStatus]
DEFAULT_FILETYPE: EncodedFileType
MP4: EncodedFileType
OGG: EncodedFileType
DEFAULT_SEGMENTED_FILE_PROTOCOL: SegmentedFileProtocol
HLS_PROTOCOL: SegmentedFileProtocol
INDEX: SegmentedFileSuffix
TIMESTAMP: SegmentedFileSuffix
IMAGE_SUFFIX_INDEX: ImageFileSuffix
IMAGE_SUFFIX_TIMESTAMP: ImageFileSuffix
DEFAULT_PROTOCOL: StreamProtocol
RTMP: StreamProtocol
H264_720P_30: EncodingOptionsPreset
H264_720P_60: EncodingOptionsPreset
H264_1080P_30: EncodingOptionsPreset
H264_1080P_60: EncodingOptionsPreset
PORTRAIT_H264_720P_30: EncodingOptionsPreset
PORTRAIT_H264_720P_60: EncodingOptionsPreset
PORTRAIT_H264_1080P_30: EncodingOptionsPreset
PORTRAIT_H264_1080P_60: EncodingOptionsPreset
EGRESS_STARTING: EgressStatus
EGRESS_ACTIVE: EgressStatus
EGRESS_ENDING: EgressStatus
EGRESS_COMPLETE: EgressStatus
EGRESS_FAILED: EgressStatus
EGRESS_ABORTED: EgressStatus
EGRESS_LIMIT_REACHED: EgressStatus

class RoomCompositeEgressRequest(_message.Message):
    __slots__ = ("room_name", "layout", "audio_only", "video_only", "custom_base_url", "file", "stream", "segments", "preset", "advanced", "file_outputs", "stream_outputs", "segment_outputs", "image_outputs")
    ROOM_NAME_FIELD_NUMBER: _ClassVar[int]
    LAYOUT_FIELD_NUMBER: _ClassVar[int]
    AUDIO_ONLY_FIELD_NUMBER: _ClassVar[int]
    VIDEO_ONLY_FIELD_NUMBER: _ClassVar[int]
    CUSTOM_BASE_URL_FIELD_NUMBER: _ClassVar[int]
    FILE_FIELD_NUMBER: _ClassVar[int]
    STREAM_FIELD_NUMBER: _ClassVar[int]
    SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    PRESET_FIELD_NUMBER: _ClassVar[int]
    ADVANCED_FIELD_NUMBER: _ClassVar[int]
    FILE_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    STREAM_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    room_name: str
    layout: str
    audio_only: bool
    video_only: bool
    custom_base_url: str
    file: EncodedFileOutput
    stream: StreamOutput
    segments: SegmentedFileOutput
    preset: EncodingOptionsPreset
    advanced: EncodingOptions
    file_outputs: _containers.RepeatedCompositeFieldContainer[EncodedFileOutput]
    stream_outputs: _containers.RepeatedCompositeFieldContainer[StreamOutput]
    segment_outputs: _containers.RepeatedCompositeFieldContainer[SegmentedFileOutput]
    image_outputs: _containers.RepeatedCompositeFieldContainer[ImageOutput]
    def __init__(self, room_name: _Optional[str] = ..., layout: _Optional[str] = ..., audio_only: bool = ..., video_only: bool = ..., custom_base_url: _Optional[str] = ..., file: _Optional[_Union[EncodedFileOutput, _Mapping]] = ..., stream: _Optional[_Union[StreamOutput, _Mapping]] = ..., segments: _Optional[_Union[SegmentedFileOutput, _Mapping]] = ..., preset: _Optional[_Union[EncodingOptionsPreset, str]] = ..., advanced: _Optional[_Union[EncodingOptions, _Mapping]] = ..., file_outputs: _Optional[_Iterable[_Union[EncodedFileOutput, _Mapping]]] = ..., stream_outputs: _Optional[_Iterable[_Union[StreamOutput, _Mapping]]] = ..., segment_outputs: _Optional[_Iterable[_Union[SegmentedFileOutput, _Mapping]]] = ..., image_outputs: _Optional[_Iterable[_Union[ImageOutput, _Mapping]]] = ...) -> None: ...

class WebEgressRequest(_message.Message):
    __slots__ = ("url", "audio_only", "video_only", "await_start_signal", "file", "stream", "segments", "preset", "advanced", "file_outputs", "stream_outputs", "segment_outputs", "image_outputs")
    URL_FIELD_NUMBER: _ClassVar[int]
    AUDIO_ONLY_FIELD_NUMBER: _ClassVar[int]
    VIDEO_ONLY_FIELD_NUMBER: _ClassVar[int]
    AWAIT_START_SIGNAL_FIELD_NUMBER: _ClassVar[int]
    FILE_FIELD_NUMBER: _ClassVar[int]
    STREAM_FIELD_NUMBER: _ClassVar[int]
    SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    PRESET_FIELD_NUMBER: _ClassVar[int]
    ADVANCED_FIELD_NUMBER: _ClassVar[int]
    FILE_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    STREAM_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    url: str
    audio_only: bool
    video_only: bool
    await_start_signal: bool
    file: EncodedFileOutput
    stream: StreamOutput
    segments: SegmentedFileOutput
    preset: EncodingOptionsPreset
    advanced: EncodingOptions
    file_outputs: _containers.RepeatedCompositeFieldContainer[EncodedFileOutput]
    stream_outputs: _containers.RepeatedCompositeFieldContainer[StreamOutput]
    segment_outputs: _containers.RepeatedCompositeFieldContainer[SegmentedFileOutput]
    image_outputs: _containers.RepeatedCompositeFieldContainer[ImageOutput]
    def __init__(self, url: _Optional[str] = ..., audio_only: bool = ..., video_only: bool = ..., await_start_signal: bool = ..., file: _Optional[_Union[EncodedFileOutput, _Mapping]] = ..., stream: _Optional[_Union[StreamOutput, _Mapping]] = ..., segments: _Optional[_Union[SegmentedFileOutput, _Mapping]] = ..., preset: _Optional[_Union[EncodingOptionsPreset, str]] = ..., advanced: _Optional[_Union[EncodingOptions, _Mapping]] = ..., file_outputs: _Optional[_Iterable[_Union[EncodedFileOutput, _Mapping]]] = ..., stream_outputs: _Optional[_Iterable[_Union[StreamOutput, _Mapping]]] = ..., segment_outputs: _Optional[_Iterable[_Union[SegmentedFileOutput, _Mapping]]] = ..., image_outputs: _Optional[_Iterable[_Union[ImageOutput, _Mapping]]] = ...) -> None: ...

class ParticipantEgressRequest(_message.Message):
    __slots__ = ("room_name", "identity", "screen_share", "preset", "advanced", "file_outputs", "stream_outputs", "segment_outputs", "image_outputs")
    ROOM_NAME_FIELD_NUMBER: _ClassVar[int]
    IDENTITY_FIELD_NUMBER: _ClassVar[int]
    SCREEN_SHARE_FIELD_NUMBER: _ClassVar[int]
    PRESET_FIELD_NUMBER: _ClassVar[int]
    ADVANCED_FIELD_NUMBER: _ClassVar[int]
    FILE_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    STREAM_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    room_name: str
    identity: str
    screen_share: bool
    preset: EncodingOptionsPreset
    advanced: EncodingOptions
    file_outputs: _containers.RepeatedCompositeFieldContainer[EncodedFileOutput]
    stream_outputs: _containers.RepeatedCompositeFieldContainer[StreamOutput]
    segment_outputs: _containers.RepeatedCompositeFieldContainer[SegmentedFileOutput]
    image_outputs: _containers.RepeatedCompositeFieldContainer[ImageOutput]
    def __init__(self, room_name: _Optional[str] = ..., identity: _Optional[str] = ..., screen_share: bool = ..., preset: _Optional[_Union[EncodingOptionsPreset, str]] = ..., advanced: _Optional[_Union[EncodingOptions, _Mapping]] = ..., file_outputs: _Optional[_Iterable[_Union[EncodedFileOutput, _Mapping]]] = ..., stream_outputs: _Optional[_Iterable[_Union[StreamOutput, _Mapping]]] = ..., segment_outputs: _Optional[_Iterable[_Union[SegmentedFileOutput, _Mapping]]] = ..., image_outputs: _Optional[_Iterable[_Union[ImageOutput, _Mapping]]] = ...) -> None: ...

class TrackCompositeEgressRequest(_message.Message):
    __slots__ = ("room_name", "audio_track_id", "video_track_id", "file", "stream", "segments", "preset", "advanced", "file_outputs", "stream_outputs", "segment_outputs", "image_outputs")
    ROOM_NAME_FIELD_NUMBER: _ClassVar[int]
    AUDIO_TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    VIDEO_TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    FILE_FIELD_NUMBER: _ClassVar[int]
    STREAM_FIELD_NUMBER: _ClassVar[int]
    SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    PRESET_FIELD_NUMBER: _ClassVar[int]
    ADVANCED_FIELD_NUMBER: _ClassVar[int]
    FILE_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    STREAM_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    room_name: str
    audio_track_id: str
    video_track_id: str
    file: EncodedFileOutput
    stream: StreamOutput
    segments: SegmentedFileOutput
    preset: EncodingOptionsPreset
    advanced: EncodingOptions
    file_outputs: _containers.RepeatedCompositeFieldContainer[EncodedFileOutput]
    stream_outputs: _containers.RepeatedCompositeFieldContainer[StreamOutput]
    segment_outputs: _containers.RepeatedCompositeFieldContainer[SegmentedFileOutput]
    image_outputs: _containers.RepeatedCompositeFieldContainer[ImageOutput]
    def __init__(self, room_name: _Optional[str] = ..., audio_track_id: _Optional[str] = ..., video_track_id: _Optional[str] = ..., file: _Optional[_Union[EncodedFileOutput, _Mapping]] = ..., stream: _Optional[_Union[StreamOutput, _Mapping]] = ..., segments: _Optional[_Union[SegmentedFileOutput, _Mapping]] = ..., preset: _Optional[_Union[EncodingOptionsPreset, str]] = ..., advanced: _Optional[_Union[EncodingOptions, _Mapping]] = ..., file_outputs: _Optional[_Iterable[_Union[EncodedFileOutput, _Mapping]]] = ..., stream_outputs: _Optional[_Iterable[_Union[StreamOutput, _Mapping]]] = ..., segment_outputs: _Optional[_Iterable[_Union[SegmentedFileOutput, _Mapping]]] = ..., image_outputs: _Optional[_Iterable[_Union[ImageOutput, _Mapping]]] = ...) -> None: ...

class TrackEgressRequest(_message.Message):
    __slots__ = ("room_name", "track_id", "file", "websocket_url")
    ROOM_NAME_FIELD_NUMBER: _ClassVar[int]
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    FILE_FIELD_NUMBER: _ClassVar[int]
    WEBSOCKET_URL_FIELD_NUMBER: _ClassVar[int]
    room_name: str
    track_id: str
    file: DirectFileOutput
    websocket_url: str
    def __init__(self, room_name: _Optional[str] = ..., track_id: _Optional[str] = ..., file: _Optional[_Union[DirectFileOutput, _Mapping]] = ..., websocket_url: _Optional[str] = ...) -> None: ...

class EncodedFileOutput(_message.Message):
    __slots__ = ("file_type", "filepath", "disable_manifest", "s3", "gcp", "azure", "aliOSS")
    FILE_TYPE_FIELD_NUMBER: _ClassVar[int]
    FILEPATH_FIELD_NUMBER: _ClassVar[int]
    DISABLE_MANIFEST_FIELD_NUMBER: _ClassVar[int]
    S3_FIELD_NUMBER: _ClassVar[int]
    GCP_FIELD_NUMBER: _ClassVar[int]
    AZURE_FIELD_NUMBER: _ClassVar[int]
    ALIOSS_FIELD_NUMBER: _ClassVar[int]
    file_type: EncodedFileType
    filepath: str
    disable_manifest: bool
    s3: S3Upload
    gcp: GCPUpload
    azure: AzureBlobUpload
    aliOSS: AliOSSUpload
    def __init__(self, file_type: _Optional[_Union[EncodedFileType, str]] = ..., filepath: _Optional[str] = ..., disable_manifest: bool = ..., s3: _Optional[_Union[S3Upload, _Mapping]] = ..., gcp: _Optional[_Union[GCPUpload, _Mapping]] = ..., azure: _Optional[_Union[AzureBlobUpload, _Mapping]] = ..., aliOSS: _Optional[_Union[AliOSSUpload, _Mapping]] = ...) -> None: ...

class SegmentedFileOutput(_message.Message):
    __slots__ = ("protocol", "filename_prefix", "playlist_name", "live_playlist_name", "segment_duration", "filename_suffix", "disable_manifest", "s3", "gcp", "azure", "aliOSS")
    PROTOCOL_FIELD_NUMBER: _ClassVar[int]
    FILENAME_PREFIX_FIELD_NUMBER: _ClassVar[int]
    PLAYLIST_NAME_FIELD_NUMBER: _ClassVar[int]
    LIVE_PLAYLIST_NAME_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_DURATION_FIELD_NUMBER: _ClassVar[int]
    FILENAME_SUFFIX_FIELD_NUMBER: _ClassVar[int]
    DISABLE_MANIFEST_FIELD_NUMBER: _ClassVar[int]
    S3_FIELD_NUMBER: _ClassVar[int]
    GCP_FIELD_NUMBER: _ClassVar[int]
    AZURE_FIELD_NUMBER: _ClassVar[int]
    ALIOSS_FIELD_NUMBER: _ClassVar[int]
    protocol: SegmentedFileProtocol
    filename_prefix: str
    playlist_name: str
    live_playlist_name: str
    segment_duration: int
    filename_suffix: SegmentedFileSuffix
    disable_manifest: bool
    s3: S3Upload
    gcp: GCPUpload
    azure: AzureBlobUpload
    aliOSS: AliOSSUpload
    def __init__(self, protocol: _Optional[_Union[SegmentedFileProtocol, str]] = ..., filename_prefix: _Optional[str] = ..., playlist_name: _Optional[str] = ..., live_playlist_name: _Optional[str] = ..., segment_duration: _Optional[int] = ..., filename_suffix: _Optional[_Union[SegmentedFileSuffix, str]] = ..., disable_manifest: bool = ..., s3: _Optional[_Union[S3Upload, _Mapping]] = ..., gcp: _Optional[_Union[GCPUpload, _Mapping]] = ..., azure: _Optional[_Union[AzureBlobUpload, _Mapping]] = ..., aliOSS: _Optional[_Union[AliOSSUpload, _Mapping]] = ...) -> None: ...

class DirectFileOutput(_message.Message):
    __slots__ = ("filepath", "disable_manifest", "s3", "gcp", "azure", "aliOSS")
    FILEPATH_FIELD_NUMBER: _ClassVar[int]
    DISABLE_MANIFEST_FIELD_NUMBER: _ClassVar[int]
    S3_FIELD_NUMBER: _ClassVar[int]
    GCP_FIELD_NUMBER: _ClassVar[int]
    AZURE_FIELD_NUMBER: _ClassVar[int]
    ALIOSS_FIELD_NUMBER: _ClassVar[int]
    filepath: str
    disable_manifest: bool
    s3: S3Upload
    gcp: GCPUpload
    azure: AzureBlobUpload
    aliOSS: AliOSSUpload
    def __init__(self, filepath: _Optional[str] = ..., disable_manifest: bool = ..., s3: _Optional[_Union[S3Upload, _Mapping]] = ..., gcp: _Optional[_Union[GCPUpload, _Mapping]] = ..., azure: _Optional[_Union[AzureBlobUpload, _Mapping]] = ..., aliOSS: _Optional[_Union[AliOSSUpload, _Mapping]] = ...) -> None: ...

class ImageOutput(_message.Message):
    __slots__ = ("capture_interval", "width", "height", "filename_prefix", "filename_suffix", "image_codec", "disable_manifest", "s3", "gcp", "azure", "aliOSS")
    CAPTURE_INTERVAL_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    FILENAME_PREFIX_FIELD_NUMBER: _ClassVar[int]
    FILENAME_SUFFIX_FIELD_NUMBER: _ClassVar[int]
    IMAGE_CODEC_FIELD_NUMBER: _ClassVar[int]
    DISABLE_MANIFEST_FIELD_NUMBER: _ClassVar[int]
    S3_FIELD_NUMBER: _ClassVar[int]
    GCP_FIELD_NUMBER: _ClassVar[int]
    AZURE_FIELD_NUMBER: _ClassVar[int]
    ALIOSS_FIELD_NUMBER: _ClassVar[int]
    capture_interval: int
    width: int
    height: int
    filename_prefix: str
    filename_suffix: ImageFileSuffix
    image_codec: _models.ImageCodec
    disable_manifest: bool
    s3: S3Upload
    gcp: GCPUpload
    azure: AzureBlobUpload
    aliOSS: AliOSSUpload
    def __init__(self, capture_interval: _Optional[int] = ..., width: _Optional[int] = ..., height: _Optional[int] = ..., filename_prefix: _Optional[str] = ..., filename_suffix: _Optional[_Union[ImageFileSuffix, str]] = ..., image_codec: _Optional[_Union[_models.ImageCodec, str]] = ..., disable_manifest: bool = ..., s3: _Optional[_Union[S3Upload, _Mapping]] = ..., gcp: _Optional[_Union[GCPUpload, _Mapping]] = ..., azure: _Optional[_Union[AzureBlobUpload, _Mapping]] = ..., aliOSS: _Optional[_Union[AliOSSUpload, _Mapping]] = ...) -> None: ...

class S3Upload(_message.Message):
    __slots__ = ("access_key", "secret", "region", "endpoint", "bucket", "force_path_style", "metadata", "tagging", "content_disposition")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ACCESS_KEY_FIELD_NUMBER: _ClassVar[int]
    SECRET_FIELD_NUMBER: _ClassVar[int]
    REGION_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    BUCKET_FIELD_NUMBER: _ClassVar[int]
    FORCE_PATH_STYLE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    TAGGING_FIELD_NUMBER: _ClassVar[int]
    CONTENT_DISPOSITION_FIELD_NUMBER: _ClassVar[int]
    access_key: str
    secret: str
    region: str
    endpoint: str
    bucket: str
    force_path_style: bool
    metadata: _containers.ScalarMap[str, str]
    tagging: str
    content_disposition: str
    def __init__(self, access_key: _Optional[str] = ..., secret: _Optional[str] = ..., region: _Optional[str] = ..., endpoint: _Optional[str] = ..., bucket: _Optional[str] = ..., force_path_style: bool = ..., metadata: _Optional[_Mapping[str, str]] = ..., tagging: _Optional[str] = ..., content_disposition: _Optional[str] = ...) -> None: ...

class GCPUpload(_message.Message):
    __slots__ = ("credentials", "bucket")
    CREDENTIALS_FIELD_NUMBER: _ClassVar[int]
    BUCKET_FIELD_NUMBER: _ClassVar[int]
    credentials: str
    bucket: str
    def __init__(self, credentials: _Optional[str] = ..., bucket: _Optional[str] = ...) -> None: ...

class AzureBlobUpload(_message.Message):
    __slots__ = ("account_name", "account_key", "container_name")
    ACCOUNT_NAME_FIELD_NUMBER: _ClassVar[int]
    ACCOUNT_KEY_FIELD_NUMBER: _ClassVar[int]
    CONTAINER_NAME_FIELD_NUMBER: _ClassVar[int]
    account_name: str
    account_key: str
    container_name: str
    def __init__(self, account_name: _Optional[str] = ..., account_key: _Optional[str] = ..., container_name: _Optional[str] = ...) -> None: ...

class AliOSSUpload(_message.Message):
    __slots__ = ("access_key", "secret", "region", "endpoint", "bucket")
    ACCESS_KEY_FIELD_NUMBER: _ClassVar[int]
    SECRET_FIELD_NUMBER: _ClassVar[int]
    REGION_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    BUCKET_FIELD_NUMBER: _ClassVar[int]
    access_key: str
    secret: str
    region: str
    endpoint: str
    bucket: str
    def __init__(self, access_key: _Optional[str] = ..., secret: _Optional[str] = ..., region: _Optional[str] = ..., endpoint: _Optional[str] = ..., bucket: _Optional[str] = ...) -> None: ...

class StreamOutput(_message.Message):
    __slots__ = ("protocol", "urls")
    PROTOCOL_FIELD_NUMBER: _ClassVar[int]
    URLS_FIELD_NUMBER: _ClassVar[int]
    protocol: StreamProtocol
    urls: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, protocol: _Optional[_Union[StreamProtocol, str]] = ..., urls: _Optional[_Iterable[str]] = ...) -> None: ...

class EncodingOptions(_message.Message):
    __slots__ = ("width", "height", "depth", "framerate", "audio_codec", "audio_bitrate", "audio_quality", "audio_frequency", "video_codec", "video_bitrate", "video_quality", "key_frame_interval")
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    DEPTH_FIELD_NUMBER: _ClassVar[int]
    FRAMERATE_FIELD_NUMBER: _ClassVar[int]
    AUDIO_CODEC_FIELD_NUMBER: _ClassVar[int]
    AUDIO_BITRATE_FIELD_NUMBER: _ClassVar[int]
    AUDIO_QUALITY_FIELD_NUMBER: _ClassVar[int]
    AUDIO_FREQUENCY_FIELD_NUMBER: _ClassVar[int]
    VIDEO_CODEC_FIELD_NUMBER: _ClassVar[int]
    VIDEO_BITRATE_FIELD_NUMBER: _ClassVar[int]
    VIDEO_QUALITY_FIELD_NUMBER: _ClassVar[int]
    KEY_FRAME_INTERVAL_FIELD_NUMBER: _ClassVar[int]
    width: int
    height: int
    depth: int
    framerate: int
    audio_codec: _models.AudioCodec
    audio_bitrate: int
    audio_quality: int
    audio_frequency: int
    video_codec: _models.VideoCodec
    video_bitrate: int
    video_quality: int
    key_frame_interval: float
    def __init__(self, width: _Optional[int] = ..., height: _Optional[int] = ..., depth: _Optional[int] = ..., framerate: _Optional[int] = ..., audio_codec: _Optional[_Union[_models.AudioCodec, str]] = ..., audio_bitrate: _Optional[int] = ..., audio_quality: _Optional[int] = ..., audio_frequency: _Optional[int] = ..., video_codec: _Optional[_Union[_models.VideoCodec, str]] = ..., video_bitrate: _Optional[int] = ..., video_quality: _Optional[int] = ..., key_frame_interval: _Optional[float] = ...) -> None: ...

class UpdateLayoutRequest(_message.Message):
    __slots__ = ("egress_id", "layout")
    EGRESS_ID_FIELD_NUMBER: _ClassVar[int]
    LAYOUT_FIELD_NUMBER: _ClassVar[int]
    egress_id: str
    layout: str
    def __init__(self, egress_id: _Optional[str] = ..., layout: _Optional[str] = ...) -> None: ...

class UpdateStreamRequest(_message.Message):
    __slots__ = ("egress_id", "add_output_urls", "remove_output_urls")
    EGRESS_ID_FIELD_NUMBER: _ClassVar[int]
    ADD_OUTPUT_URLS_FIELD_NUMBER: _ClassVar[int]
    REMOVE_OUTPUT_URLS_FIELD_NUMBER: _ClassVar[int]
    egress_id: str
    add_output_urls: _containers.RepeatedScalarFieldContainer[str]
    remove_output_urls: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, egress_id: _Optional[str] = ..., add_output_urls: _Optional[_Iterable[str]] = ..., remove_output_urls: _Optional[_Iterable[str]] = ...) -> None: ...

class ListEgressRequest(_message.Message):
    __slots__ = ("room_name", "egress_id", "active")
    ROOM_NAME_FIELD_NUMBER: _ClassVar[int]
    EGRESS_ID_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_FIELD_NUMBER: _ClassVar[int]
    room_name: str
    egress_id: str
    active: bool
    def __init__(self, room_name: _Optional[str] = ..., egress_id: _Optional[str] = ..., active: bool = ...) -> None: ...

class ListEgressResponse(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[EgressInfo]
    def __init__(self, items: _Optional[_Iterable[_Union[EgressInfo, _Mapping]]] = ...) -> None: ...

class StopEgressRequest(_message.Message):
    __slots__ = ("egress_id",)
    EGRESS_ID_FIELD_NUMBER: _ClassVar[int]
    egress_id: str
    def __init__(self, egress_id: _Optional[str] = ...) -> None: ...

class EgressInfo(_message.Message):
    __slots__ = ("egress_id", "room_id", "room_name", "status", "started_at", "ended_at", "updated_at", "error", "room_composite", "web", "participant", "track_composite", "track", "stream", "file", "segments", "stream_results", "file_results", "segment_results", "image_results")
    EGRESS_ID_FIELD_NUMBER: _ClassVar[int]
    ROOM_ID_FIELD_NUMBER: _ClassVar[int]
    ROOM_NAME_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    ENDED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    ROOM_COMPOSITE_FIELD_NUMBER: _ClassVar[int]
    WEB_FIELD_NUMBER: _ClassVar[int]
    PARTICIPANT_FIELD_NUMBER: _ClassVar[int]
    TRACK_COMPOSITE_FIELD_NUMBER: _ClassVar[int]
    TRACK_FIELD_NUMBER: _ClassVar[int]
    STREAM_FIELD_NUMBER: _ClassVar[int]
    FILE_FIELD_NUMBER: _ClassVar[int]
    SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    STREAM_RESULTS_FIELD_NUMBER: _ClassVar[int]
    FILE_RESULTS_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_RESULTS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_RESULTS_FIELD_NUMBER: _ClassVar[int]
    egress_id: str
    room_id: str
    room_name: str
    status: EgressStatus
    started_at: int
    ended_at: int
    updated_at: int
    error: str
    room_composite: RoomCompositeEgressRequest
    web: WebEgressRequest
    participant: ParticipantEgressRequest
    track_composite: TrackCompositeEgressRequest
    track: TrackEgressRequest
    stream: StreamInfoList
    file: FileInfo
    segments: SegmentsInfo
    stream_results: _containers.RepeatedCompositeFieldContainer[StreamInfo]
    file_results: _containers.RepeatedCompositeFieldContainer[FileInfo]
    segment_results: _containers.RepeatedCompositeFieldContainer[SegmentsInfo]
    image_results: _containers.RepeatedCompositeFieldContainer[ImagesInfo]
    def __init__(self, egress_id: _Optional[str] = ..., room_id: _Optional[str] = ..., room_name: _Optional[str] = ..., status: _Optional[_Union[EgressStatus, str]] = ..., started_at: _Optional[int] = ..., ended_at: _Optional[int] = ..., updated_at: _Optional[int] = ..., error: _Optional[str] = ..., room_composite: _Optional[_Union[RoomCompositeEgressRequest, _Mapping]] = ..., web: _Optional[_Union[WebEgressRequest, _Mapping]] = ..., participant: _Optional[_Union[ParticipantEgressRequest, _Mapping]] = ..., track_composite: _Optional[_Union[TrackCompositeEgressRequest, _Mapping]] = ..., track: _Optional[_Union[TrackEgressRequest, _Mapping]] = ..., stream: _Optional[_Union[StreamInfoList, _Mapping]] = ..., file: _Optional[_Union[FileInfo, _Mapping]] = ..., segments: _Optional[_Union[SegmentsInfo, _Mapping]] = ..., stream_results: _Optional[_Iterable[_Union[StreamInfo, _Mapping]]] = ..., file_results: _Optional[_Iterable[_Union[FileInfo, _Mapping]]] = ..., segment_results: _Optional[_Iterable[_Union[SegmentsInfo, _Mapping]]] = ..., image_results: _Optional[_Iterable[_Union[ImagesInfo, _Mapping]]] = ...) -> None: ...

class StreamInfoList(_message.Message):
    __slots__ = ("info",)
    INFO_FIELD_NUMBER: _ClassVar[int]
    info: _containers.RepeatedCompositeFieldContainer[StreamInfo]
    def __init__(self, info: _Optional[_Iterable[_Union[StreamInfo, _Mapping]]] = ...) -> None: ...

class StreamInfo(_message.Message):
    __slots__ = ("url", "started_at", "ended_at", "duration", "status", "error")
    class Status(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        ACTIVE: _ClassVar[StreamInfo.Status]
        FINISHED: _ClassVar[StreamInfo.Status]
        FAILED: _ClassVar[StreamInfo.Status]
    ACTIVE: StreamInfo.Status
    FINISHED: StreamInfo.Status
    FAILED: StreamInfo.Status
    URL_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    ENDED_AT_FIELD_NUMBER: _ClassVar[int]
    DURATION_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    url: str
    started_at: int
    ended_at: int
    duration: int
    status: StreamInfo.Status
    error: str
    def __init__(self, url: _Optional[str] = ..., started_at: _Optional[int] = ..., ended_at: _Optional[int] = ..., duration: _Optional[int] = ..., status: _Optional[_Union[StreamInfo.Status, str]] = ..., error: _Optional[str] = ...) -> None: ...

class FileInfo(_message.Message):
    __slots__ = ("filename", "started_at", "ended_at", "duration", "size", "location")
    FILENAME_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    ENDED_AT_FIELD_NUMBER: _ClassVar[int]
    DURATION_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    filename: str
    started_at: int
    ended_at: int
    duration: int
    size: int
    location: str
    def __init__(self, filename: _Optional[str] = ..., started_at: _Optional[int] = ..., ended_at: _Optional[int] = ..., duration: _Optional[int] = ..., size: _Optional[int] = ..., location: _Optional[str] = ...) -> None: ...

class SegmentsInfo(_message.Message):
    __slots__ = ("playlist_name", "live_playlist_name", "duration", "size", "playlist_location", "live_playlist_location", "segment_count", "started_at", "ended_at")
    PLAYLIST_NAME_FIELD_NUMBER: _ClassVar[int]
    LIVE_PLAYLIST_NAME_FIELD_NUMBER: _ClassVar[int]
    DURATION_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    PLAYLIST_LOCATION_FIELD_NUMBER: _ClassVar[int]
    LIVE_PLAYLIST_LOCATION_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_COUNT_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    ENDED_AT_FIELD_NUMBER: _ClassVar[int]
    playlist_name: str
    live_playlist_name: str
    duration: int
    size: int
    playlist_location: str
    live_playlist_location: str
    segment_count: int
    started_at: int
    ended_at: int
    def __init__(self, playlist_name: _Optional[str] = ..., live_playlist_name: _Optional[str] = ..., duration: _Optional[int] = ..., size: _Optional[int] = ..., playlist_location: _Optional[str] = ..., live_playlist_location: _Optional[str] = ..., segment_count: _Optional[int] = ..., started_at: _Optional[int] = ..., ended_at: _Optional[int] = ...) -> None: ...

class ImagesInfo(_message.Message):
    __slots__ = ("image_count", "started_at", "ended_at")
    IMAGE_COUNT_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    ENDED_AT_FIELD_NUMBER: _ClassVar[int]
    image_count: int
    started_at: int
    ended_at: int
    def __init__(self, image_count: _Optional[int] = ..., started_at: _Optional[int] = ..., ended_at: _Optional[int] = ...) -> None: ...

class AutoParticipantEgress(_message.Message):
    __slots__ = ("preset", "advanced", "file_outputs", "segment_outputs")
    PRESET_FIELD_NUMBER: _ClassVar[int]
    ADVANCED_FIELD_NUMBER: _ClassVar[int]
    FILE_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    preset: EncodingOptionsPreset
    advanced: EncodingOptions
    file_outputs: _containers.RepeatedCompositeFieldContainer[EncodedFileOutput]
    segment_outputs: _containers.RepeatedCompositeFieldContainer[SegmentedFileOutput]
    def __init__(self, preset: _Optional[_Union[EncodingOptionsPreset, str]] = ..., advanced: _Optional[_Union[EncodingOptions, _Mapping]] = ..., file_outputs: _Optional[_Iterable[_Union[EncodedFileOutput, _Mapping]]] = ..., segment_outputs: _Optional[_Iterable[_Union[SegmentedFileOutput, _Mapping]]] = ...) -> None: ...

class AutoTrackEgress(_message.Message):
    __slots__ = ("filepath", "disable_manifest", "s3", "gcp", "azure")
    FILEPATH_FIELD_NUMBER: _ClassVar[int]
    DISABLE_MANIFEST_FIELD_NUMBER: _ClassVar[int]
    S3_FIELD_NUMBER: _ClassVar[int]
    GCP_FIELD_NUMBER: _ClassVar[int]
    AZURE_FIELD_NUMBER: _ClassVar[int]
    filepath: str
    disable_manifest: bool
    s3: S3Upload
    gcp: GCPUpload
    azure: AzureBlobUpload
    def __init__(self, filepath: _Optional[str] = ..., disable_manifest: bool = ..., s3: _Optional[_Union[S3Upload, _Mapping]] = ..., gcp: _Optional[_Union[GCPUpload, _Mapping]] = ..., azure: _Optional[_Union[AzureBlobUpload, _Mapping]] = ...) -> None: ...
