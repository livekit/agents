# ruff: noqa

from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SplitStrategy(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SPLIT_STRATEGY_UNSPECIFIED: _ClassVar[SplitStrategy]
    SPLIT_STRATEGY_SENTENCE: _ClassVar[SplitStrategy]
    SPLIT_STRATEGY_NONE: _ClassVar[SplitStrategy]

SPLIT_STRATEGY_UNSPECIFIED: SplitStrategy
SPLIT_STRATEGY_SENTENCE: SplitStrategy
SPLIT_STRATEGY_NONE: SplitStrategy

class SynthesisRequest(_message.Message):
    __slots__ = (
        "language",
        "speaker",
        "text",
        "audio_parameters",
        "split_strategy",
        "coda_parameters",
        "mist_parameters",
    )
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    SPEAKER_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    AUDIO_PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    SPLIT_STRATEGY_FIELD_NUMBER: _ClassVar[int]
    CODA_PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    MIST_PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    language: str
    speaker: str
    text: str
    audio_parameters: AudioParameters
    split_strategy: SplitStrategy
    coda_parameters: CodaParameters
    mist_parameters: MistParameters
    def __init__(
        self,
        language: _Optional[str] = ...,
        speaker: _Optional[str] = ...,
        text: _Optional[str] = ...,
        audio_parameters: _Optional[_Union[AudioParameters, _Mapping]] = ...,
        split_strategy: _Optional[_Union[SplitStrategy, str]] = ...,
        coda_parameters: _Optional[_Union[CodaParameters, _Mapping]] = ...,
        mist_parameters: _Optional[_Union[MistParameters, _Mapping]] = ...,
    ) -> None: ...

class AudioParameters(_message.Message):
    __slots__ = ("audio_format", "sampling_rate", "time_scale_factor")
    AUDIO_FORMAT_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_RATE_FIELD_NUMBER: _ClassVar[int]
    TIME_SCALE_FACTOR_FIELD_NUMBER: _ClassVar[int]
    audio_format: str
    sampling_rate: int
    time_scale_factor: float
    def __init__(
        self,
        audio_format: _Optional[str] = ...,
        sampling_rate: _Optional[int] = ...,
        time_scale_factor: _Optional[float] = ...,
    ) -> None: ...

class CodaParameters(_message.Message):
    __slots__ = ("text_lookahead_tokens",)
    TEXT_LOOKAHEAD_TOKENS_FIELD_NUMBER: _ClassVar[int]
    text_lookahead_tokens: int
    def __init__(self, text_lookahead_tokens: _Optional[int] = ...) -> None: ...

class MistParameters(_message.Message):
    __slots__ = (
        "pause_between_brackets",
        "phonemize_between_brackets",
        "inline_time_scale_factors",
        "save_oovs",
    )
    PAUSE_BETWEEN_BRACKETS_FIELD_NUMBER: _ClassVar[int]
    PHONEMIZE_BETWEEN_BRACKETS_FIELD_NUMBER: _ClassVar[int]
    INLINE_TIME_SCALE_FACTORS_FIELD_NUMBER: _ClassVar[int]
    SAVE_OOVS_FIELD_NUMBER: _ClassVar[int]
    pause_between_brackets: bool
    phonemize_between_brackets: bool
    inline_time_scale_factors: _containers.RepeatedScalarFieldContainer[float]
    save_oovs: bool
    def __init__(
        self,
        pause_between_brackets: _Optional[bool] = ...,
        phonemize_between_brackets: _Optional[bool] = ...,
        inline_time_scale_factors: _Optional[_Iterable[float]] = ...,
        save_oovs: _Optional[bool] = ...,
    ) -> None: ...

class WebSocketRequest(_message.Message):
    __slots__ = ("context_id", "config", "start", "text", "end", "cancel")
    CONTEXT_ID_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    START_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    CANCEL_FIELD_NUMBER: _ClassVar[int]
    context_id: str
    config: WebSocketConfig
    start: SynthesisRequest
    text: str
    end: WebSocketEnd
    cancel: WebSocketCancel
    def __init__(
        self,
        context_id: _Optional[str] = ...,
        config: _Optional[_Union[WebSocketConfig, _Mapping]] = ...,
        start: _Optional[_Union[SynthesisRequest, _Mapping]] = ...,
        text: _Optional[str] = ...,
        end: _Optional[_Union[WebSocketEnd, _Mapping]] = ...,
        cancel: _Optional[_Union[WebSocketCancel, _Mapping]] = ...,
    ) -> None: ...

class WebSocketConfig(_message.Message):
    __slots__ = ("authorization", "license", "defaults")
    AUTHORIZATION_FIELD_NUMBER: _ClassVar[int]
    LICENSE_FIELD_NUMBER: _ClassVar[int]
    DEFAULTS_FIELD_NUMBER: _ClassVar[int]
    authorization: str
    license: str
    defaults: SynthesisRequest
    def __init__(
        self,
        authorization: _Optional[str] = ...,
        license: _Optional[str] = ...,
        defaults: _Optional[_Union[SynthesisRequest, _Mapping]] = ...,
    ) -> None: ...

class WebSocketEnd(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class WebSocketCancel(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class WebSocketResponse(_message.Message):
    __slots__ = ("context_id", "ready", "started", "audio", "done", "cancelled", "error")
    CONTEXT_ID_FIELD_NUMBER: _ClassVar[int]
    READY_FIELD_NUMBER: _ClassVar[int]
    STARTED_FIELD_NUMBER: _ClassVar[int]
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    CANCELLED_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    context_id: str
    ready: WebSocketReady
    started: WebSocketStarted
    audio: bytes
    done: WebSocketDone
    cancelled: WebSocketCancelled
    error: WebSocketError
    def __init__(
        self,
        context_id: _Optional[str] = ...,
        ready: _Optional[_Union[WebSocketReady, _Mapping]] = ...,
        started: _Optional[_Union[WebSocketStarted, _Mapping]] = ...,
        audio: _Optional[bytes] = ...,
        done: _Optional[_Union[WebSocketDone, _Mapping]] = ...,
        cancelled: _Optional[_Union[WebSocketCancelled, _Mapping]] = ...,
        error: _Optional[_Union[WebSocketError, _Mapping]] = ...,
    ) -> None: ...

class WebSocketReady(_message.Message):
    __slots__ = ("protocol", "languages", "default_language")
    PROTOCOL_FIELD_NUMBER: _ClassVar[int]
    LANGUAGES_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    protocol: int
    languages: _containers.RepeatedScalarFieldContainer[str]
    default_language: str
    def __init__(
        self,
        protocol: _Optional[int] = ...,
        languages: _Optional[_Iterable[str]] = ...,
        default_language: _Optional[str] = ...,
    ) -> None: ...

class WebSocketStarted(_message.Message):
    __slots__ = ("request_id",)
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    def __init__(self, request_id: _Optional[str] = ...) -> None: ...

class WebSocketDone(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class WebSocketCancelled(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class WebSocketError(_message.Message):
    __slots__ = ("kind", "message", "request_id")
    KIND_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    kind: str
    message: str
    request_id: str
    def __init__(
        self,
        kind: _Optional[str] = ...,
        message: _Optional[str] = ...,
        request_id: _Optional[str] = ...,
    ) -> None: ...
