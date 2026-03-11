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

class TdChatContext(_message.Message):
    __slots__ = ("messages",)
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    messages: _containers.RepeatedCompositeFieldContainer[TdChatMessage]
    def __init__(self, messages: _Optional[_Iterable[_Union[TdChatMessage, _Mapping]]] = ...) -> None: ...

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
    __slots__ = ("chat_context",)
    CHAT_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    chat_context: TdChatContext
    def __init__(self, chat_context: _Optional[_Union[TdChatContext, _Mapping]] = ...) -> None: ...

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
    __slots__ = ()
    def __init__(self) -> None: ...

class TurnDetectorClientMessage(_message.Message):
    __slots__ = ("session_create", "input_audio", "input_chat_context", "session_flush", "session_finalize", "inference_start", "inference_stop")
    SESSION_CREATE_FIELD_NUMBER: _ClassVar[int]
    INPUT_AUDIO_FIELD_NUMBER: _ClassVar[int]
    INPUT_CHAT_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    SESSION_FLUSH_FIELD_NUMBER: _ClassVar[int]
    SESSION_FINALIZE_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_START_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_STOP_FIELD_NUMBER: _ClassVar[int]
    session_create: SessionCreate
    input_audio: InputAudio
    input_chat_context: InputChatContext
    session_flush: SessionFlush
    session_finalize: SessionFinalize
    inference_start: InferenceStart
    inference_stop: InferenceStop
    def __init__(self, session_create: _Optional[_Union[SessionCreate, _Mapping]] = ..., input_audio: _Optional[_Union[InputAudio, _Mapping]] = ..., input_chat_context: _Optional[_Union[InputChatContext, _Mapping]] = ..., session_flush: _Optional[_Union[SessionFlush, _Mapping]] = ..., session_finalize: _Optional[_Union[SessionFinalize, _Mapping]] = ..., inference_start: _Optional[_Union[InferenceStart, _Mapping]] = ..., inference_stop: _Optional[_Union[InferenceStop, _Mapping]] = ...) -> None: ...

class SessionCreated(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class EouPrediction(_message.Message):
    __slots__ = ("probability", "language", "request_id")
    PROBABILITY_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    probability: float
    language: str
    request_id: str
    def __init__(self, probability: _Optional[float] = ..., language: _Optional[str] = ..., request_id: _Optional[str] = ...) -> None: ...

class SessionFinalized(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SessionClosed(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class TurnDetectorError(_message.Message):
    __slots__ = ("message",)
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    message: str
    def __init__(self, message: _Optional[str] = ...) -> None: ...

class TurnDetectorServerMessage(_message.Message):
    __slots__ = ("session_created", "eou_prediction", "session_finalized", "session_closed", "error")
    SESSION_CREATED_FIELD_NUMBER: _ClassVar[int]
    EOU_PREDICTION_FIELD_NUMBER: _ClassVar[int]
    SESSION_FINALIZED_FIELD_NUMBER: _ClassVar[int]
    SESSION_CLOSED_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    session_created: SessionCreated
    eou_prediction: EouPrediction
    session_finalized: SessionFinalized
    session_closed: SessionClosed
    error: TurnDetectorError
    def __init__(self, session_created: _Optional[_Union[SessionCreated, _Mapping]] = ..., eou_prediction: _Optional[_Union[EouPrediction, _Mapping]] = ..., session_finalized: _Optional[_Union[SessionFinalized, _Mapping]] = ..., session_closed: _Optional[_Union[SessionClosed, _Mapping]] = ..., error: _Optional[_Union[TurnDetectorError, _Mapping]] = ...) -> None: ...

class PredictRequest(_message.Message):
    __slots__ = ("model", "audio", "settings", "chat_context", "request_id")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    SETTINGS_FIELD_NUMBER: _ClassVar[int]
    CHAT_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    model: str
    audio: bytes
    settings: SessionSettings
    chat_context: TdChatContext
    request_id: str
    def __init__(self, model: _Optional[str] = ..., audio: _Optional[bytes] = ..., settings: _Optional[_Union[SessionSettings, _Mapping]] = ..., chat_context: _Optional[_Union[TdChatContext, _Mapping]] = ..., request_id: _Optional[str] = ...) -> None: ...
