import datetime

from . import livekit_agent_session_pb2 as _livekit_agent_session_pb2
from google.protobuf import duration_pb2 as _duration_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TdAudioEncoding(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TD_AUDIO_ENCODING_OPUS: _ClassVar[TdAudioEncoding]
TD_AUDIO_ENCODING_OPUS: TdAudioEncoding

class TdSessionSettings(_message.Message):
    __slots__ = ("sample_rate", "encoding")
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    ENCODING_FIELD_NUMBER: _ClassVar[int]
    sample_rate: int
    encoding: TdAudioEncoding
    def __init__(self, sample_rate: _Optional[int] = ..., encoding: _Optional[_Union[TdAudioEncoding, str]] = ...) -> None: ...

class TdInferenceStats(_message.Message):
    __slots__ = ("e2e_latency", "preprocessing_duration", "inference_duration")
    E2E_LATENCY_FIELD_NUMBER: _ClassVar[int]
    PREPROCESSING_DURATION_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_DURATION_FIELD_NUMBER: _ClassVar[int]
    e2e_latency: _duration_pb2.Duration
    preprocessing_duration: _duration_pb2.Duration
    inference_duration: _duration_pb2.Duration
    def __init__(self, e2e_latency: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., preprocessing_duration: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., inference_duration: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ...) -> None: ...

class TdError(_message.Message):
    __slots__ = ("message", "code")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CODE_FIELD_NUMBER: _ClassVar[int]
    message: str
    code: int
    def __init__(self, message: _Optional[str] = ..., code: _Optional[int] = ...) -> None: ...

class TdSessionCreate(_message.Message):
    __slots__ = ("settings",)
    SETTINGS_FIELD_NUMBER: _ClassVar[int]
    settings: TdSessionSettings
    def __init__(self, settings: _Optional[_Union[TdSessionSettings, _Mapping]] = ...) -> None: ...

class TdInputAudio(_message.Message):
    __slots__ = ("audio", "created_at", "num_samples")
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    NUM_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    audio: bytes
    created_at: _timestamp_pb2.Timestamp
    num_samples: int
    def __init__(self, audio: _Optional[bytes] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., num_samples: _Optional[int] = ...) -> None: ...

class TdInputChatContext(_message.Message):
    __slots__ = ("messages",)
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    messages: _containers.RepeatedCompositeFieldContainer[_livekit_agent_session_pb2.ChatMessage]
    def __init__(self, messages: _Optional[_Iterable[_Union[_livekit_agent_session_pb2.ChatMessage, _Mapping]]] = ...) -> None: ...

class TdSessionFlush(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class TdSessionClose(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class TdInferenceStart(_message.Message):
    __slots__ = ("request_id",)
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    def __init__(self, request_id: _Optional[str] = ...) -> None: ...

class TdInferenceStop(_message.Message):
    __slots__ = ("request_id",)
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    def __init__(self, request_id: _Optional[str] = ...) -> None: ...

class TdClientMessage(_message.Message):
    __slots__ = ("session_create", "input_audio", "input_chat_context", "session_flush", "session_close", "inference_start", "inference_stop", "created_at")
    SESSION_CREATE_FIELD_NUMBER: _ClassVar[int]
    INPUT_AUDIO_FIELD_NUMBER: _ClassVar[int]
    INPUT_CHAT_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    SESSION_FLUSH_FIELD_NUMBER: _ClassVar[int]
    SESSION_CLOSE_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_START_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_STOP_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    session_create: TdSessionCreate
    input_audio: TdInputAudio
    input_chat_context: TdInputChatContext
    session_flush: TdSessionFlush
    session_close: TdSessionClose
    inference_start: TdInferenceStart
    inference_stop: TdInferenceStop
    created_at: _timestamp_pb2.Timestamp
    def __init__(self, session_create: _Optional[_Union[TdSessionCreate, _Mapping]] = ..., input_audio: _Optional[_Union[TdInputAudio, _Mapping]] = ..., input_chat_context: _Optional[_Union[TdInputChatContext, _Mapping]] = ..., session_flush: _Optional[_Union[TdSessionFlush, _Mapping]] = ..., session_close: _Optional[_Union[TdSessionClose, _Mapping]] = ..., inference_start: _Optional[_Union[TdInferenceStart, _Mapping]] = ..., inference_stop: _Optional[_Union[TdInferenceStop, _Mapping]] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class TdInferenceRequest(_message.Message):
    __slots__ = ("audio", "assistant_text", "encoding", "sample_rate")
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    ASSISTANT_TEXT_FIELD_NUMBER: _ClassVar[int]
    ENCODING_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    audio: bytes
    assistant_text: str
    encoding: TdAudioEncoding
    sample_rate: int
    def __init__(self, audio: _Optional[bytes] = ..., assistant_text: _Optional[str] = ..., encoding: _Optional[_Union[TdAudioEncoding, str]] = ..., sample_rate: _Optional[int] = ...) -> None: ...

class TdInferenceResponse(_message.Message):
    __slots__ = ("probability", "stats")
    PROBABILITY_FIELD_NUMBER: _ClassVar[int]
    STATS_FIELD_NUMBER: _ClassVar[int]
    probability: float
    stats: TdInferenceStats
    def __init__(self, probability: _Optional[float] = ..., stats: _Optional[_Union[TdInferenceStats, _Mapping]] = ...) -> None: ...

class TdSessionCreated(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class TdProcessingStats(_message.Message):
    __slots__ = ("earliest_client_created_at", "latest_client_created_at", "e2e_latency", "inference_stats")
    EARLIEST_CLIENT_CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    LATEST_CLIENT_CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    E2E_LATENCY_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_STATS_FIELD_NUMBER: _ClassVar[int]
    earliest_client_created_at: _timestamp_pb2.Timestamp
    latest_client_created_at: _timestamp_pb2.Timestamp
    e2e_latency: _duration_pb2.Duration
    inference_stats: TdInferenceStats
    def __init__(self, earliest_client_created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., latest_client_created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., e2e_latency: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., inference_stats: _Optional[_Union[TdInferenceStats, _Mapping]] = ...) -> None: ...

class TdEouPrediction(_message.Message):
    __slots__ = ("probability", "processing_stats")
    PROBABILITY_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_STATS_FIELD_NUMBER: _ClassVar[int]
    probability: float
    processing_stats: TdProcessingStats
    def __init__(self, probability: _Optional[float] = ..., processing_stats: _Optional[_Union[TdProcessingStats, _Mapping]] = ...) -> None: ...

class TdInferenceStarted(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class TdInferenceStopped(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class TdSessionClosed(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class TdServerMessage(_message.Message):
    __slots__ = ("session_created", "inference_started", "inference_stopped", "eou_prediction", "session_closed", "error", "request_id", "server_created_at", "client_created_at")
    SESSION_CREATED_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_STARTED_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_STOPPED_FIELD_NUMBER: _ClassVar[int]
    EOU_PREDICTION_FIELD_NUMBER: _ClassVar[int]
    SESSION_CLOSED_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SERVER_CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    CLIENT_CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    session_created: TdSessionCreated
    inference_started: TdInferenceStarted
    inference_stopped: TdInferenceStopped
    eou_prediction: TdEouPrediction
    session_closed: TdSessionClosed
    error: TdError
    request_id: str
    server_created_at: _timestamp_pb2.Timestamp
    client_created_at: _timestamp_pb2.Timestamp
    def __init__(self, session_created: _Optional[_Union[TdSessionCreated, _Mapping]] = ..., inference_started: _Optional[_Union[TdInferenceStarted, _Mapping]] = ..., inference_stopped: _Optional[_Union[TdInferenceStopped, _Mapping]] = ..., eou_prediction: _Optional[_Union[TdEouPrediction, _Mapping]] = ..., session_closed: _Optional[_Union[TdSessionClosed, _Mapping]] = ..., error: _Optional[_Union[TdError, _Mapping]] = ..., request_id: _Optional[str] = ..., server_created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., client_created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...
