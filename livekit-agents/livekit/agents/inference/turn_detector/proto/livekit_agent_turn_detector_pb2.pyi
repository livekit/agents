import datetime

from google.protobuf import duration_pb2 as _duration_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AudioEncoding(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    AUDIO_ENCODING_PCM_S16LE: _ClassVar[AudioEncoding]

class TdChatRole(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TD_CHAT_ROLE_USER: _ClassVar[TdChatRole]
    TD_CHAT_ROLE_ASSISTANT: _ClassVar[TdChatRole]
AUDIO_ENCODING_PCM_S16LE: AudioEncoding
TD_CHAT_ROLE_USER: TdChatRole
TD_CHAT_ROLE_ASSISTANT: TdChatRole

class TdChatMessage(_message.Message):
    __slots__ = ("role", "content")
    ROLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    role: TdChatRole
    content: str
    def __init__(self, role: _Optional[_Union[TdChatRole, str]] = ..., content: _Optional[str] = ...) -> None: ...

class SessionSettings(_message.Message):
    __slots__ = ("sample_rate", "encoding")
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    ENCODING_FIELD_NUMBER: _ClassVar[int]
    sample_rate: int
    encoding: AudioEncoding
    def __init__(self, sample_rate: _Optional[int] = ..., encoding: _Optional[_Union[AudioEncoding, str]] = ...) -> None: ...

class SessionCreate(_message.Message):
    __slots__ = ("settings", "model")
    SETTINGS_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    settings: SessionSettings
    model: str
    def __init__(self, settings: _Optional[_Union[SessionSettings, _Mapping]] = ..., model: _Optional[str] = ...) -> None: ...

class InputAudio(_message.Message):
    __slots__ = ("audio",)
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    audio: bytes
    def __init__(self, audio: _Optional[bytes] = ...) -> None: ...

class InputChatContext(_message.Message):
    __slots__ = ("messages",)
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    messages: _containers.RepeatedCompositeFieldContainer[TdChatMessage]
    def __init__(self, messages: _Optional[_Iterable[_Union[TdChatMessage, _Mapping]]] = ...) -> None: ...

class SessionFlush(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SessionFinalize(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class InferenceStart(_message.Message):
    __slots__ = ("request_id",)
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    def __init__(self, request_id: _Optional[str] = ...) -> None: ...

class InferenceStop(_message.Message):
    __slots__ = ("request_id",)
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    def __init__(self, request_id: _Optional[str] = ...) -> None: ...

class TurnDetectorClientMessage(_message.Message):
    __slots__ = ("session_create", "input_audio", "input_chat_context", "session_flush", "session_finalize", "inference_start", "inference_stop", "created_at")
    SESSION_CREATE_FIELD_NUMBER: _ClassVar[int]
    INPUT_AUDIO_FIELD_NUMBER: _ClassVar[int]
    INPUT_CHAT_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    SESSION_FLUSH_FIELD_NUMBER: _ClassVar[int]
    SESSION_FINALIZE_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_START_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_STOP_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    session_create: SessionCreate
    input_audio: InputAudio
    input_chat_context: InputChatContext
    session_flush: SessionFlush
    session_finalize: SessionFinalize
    inference_start: InferenceStart
    inference_stop: InferenceStop
    created_at: _timestamp_pb2.Timestamp
    def __init__(self, session_create: _Optional[_Union[SessionCreate, _Mapping]] = ..., input_audio: _Optional[_Union[InputAudio, _Mapping]] = ..., input_chat_context: _Optional[_Union[InputChatContext, _Mapping]] = ..., session_flush: _Optional[_Union[SessionFlush, _Mapping]] = ..., session_finalize: _Optional[_Union[SessionFinalize, _Mapping]] = ..., inference_start: _Optional[_Union[InferenceStart, _Mapping]] = ..., inference_stop: _Optional[_Union[InferenceStop, _Mapping]] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class SessionCreated(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ProcessingStats(_message.Message):
    __slots__ = ("earliest_client_created_at", "latest_client_created_at", "batching_wait_duration", "preprocessing_duration", "inference_duration")
    EARLIEST_CLIENT_CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    LATEST_CLIENT_CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    BATCHING_WAIT_DURATION_FIELD_NUMBER: _ClassVar[int]
    PREPROCESSING_DURATION_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_DURATION_FIELD_NUMBER: _ClassVar[int]
    earliest_client_created_at: _timestamp_pb2.Timestamp
    latest_client_created_at: _timestamp_pb2.Timestamp
    batching_wait_duration: _duration_pb2.Duration
    preprocessing_duration: _duration_pb2.Duration
    inference_duration: _duration_pb2.Duration
    def __init__(self, earliest_client_created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., latest_client_created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., batching_wait_duration: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., preprocessing_duration: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., inference_duration: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ...) -> None: ...

class EouPrediction(_message.Message):
    __slots__ = ("probability", "language", "processing_stats")
    PROBABILITY_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_STATS_FIELD_NUMBER: _ClassVar[int]
    probability: float
    language: str
    processing_stats: ProcessingStats
    def __init__(self, probability: _Optional[float] = ..., language: _Optional[str] = ..., processing_stats: _Optional[_Union[ProcessingStats, _Mapping]] = ...) -> None: ...

class InferenceStarted(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class InferenceStopped(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SessionFinalized(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SessionClosed(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class TurnDetectorError(_message.Message):
    __slots__ = ("message", "code")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CODE_FIELD_NUMBER: _ClassVar[int]
    message: str
    code: int
    def __init__(self, message: _Optional[str] = ..., code: _Optional[int] = ...) -> None: ...

class TurnDetectorServerMessage(_message.Message):
    __slots__ = ("session_created", "inference_started", "inference_stopped", "eou_prediction", "session_finalized", "session_closed", "error", "request_id", "server_created_at", "client_created_at")
    SESSION_CREATED_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_STARTED_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_STOPPED_FIELD_NUMBER: _ClassVar[int]
    EOU_PREDICTION_FIELD_NUMBER: _ClassVar[int]
    SESSION_FINALIZED_FIELD_NUMBER: _ClassVar[int]
    SESSION_CLOSED_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SERVER_CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    CLIENT_CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    session_created: SessionCreated
    inference_started: InferenceStarted
    inference_stopped: InferenceStopped
    eou_prediction: EouPrediction
    session_finalized: SessionFinalized
    session_closed: SessionClosed
    error: TurnDetectorError
    request_id: str
    server_created_at: _timestamp_pb2.Timestamp
    client_created_at: _timestamp_pb2.Timestamp
    def __init__(self, session_created: _Optional[_Union[SessionCreated, _Mapping]] = ..., inference_started: _Optional[_Union[InferenceStarted, _Mapping]] = ..., inference_stopped: _Optional[_Union[InferenceStopped, _Mapping]] = ..., eou_prediction: _Optional[_Union[EouPrediction, _Mapping]] = ..., session_finalized: _Optional[_Union[SessionFinalized, _Mapping]] = ..., session_closed: _Optional[_Union[SessionClosed, _Mapping]] = ..., error: _Optional[_Union[TurnDetectorError, _Mapping]] = ..., request_id: _Optional[str] = ..., server_created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., client_created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...
