import datetime

from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from .logger import options_pb2 as _options_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ChatRole(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DEVELOPER: _ClassVar[ChatRole]
    SYSTEM: _ClassVar[ChatRole]
    USER: _ClassVar[ChatRole]
    ASSISTANT: _ClassVar[ChatRole]

class AgentState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    AS_INITIALIZING: _ClassVar[AgentState]
    AS_IDLE: _ClassVar[AgentState]
    AS_LISTENING: _ClassVar[AgentState]
    AS_THINKING: _ClassVar[AgentState]
    AS_SPEAKING: _ClassVar[AgentState]

class UserState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    US_SPEAKING: _ClassVar[UserState]
    US_LISTENING: _ClassVar[UserState]
    US_AWAY: _ClassVar[UserState]
DEVELOPER: ChatRole
SYSTEM: ChatRole
USER: ChatRole
ASSISTANT: ChatRole
AS_INITIALIZING: AgentState
AS_IDLE: AgentState
AS_LISTENING: AgentState
AS_THINKING: AgentState
AS_SPEAKING: AgentState
US_SPEAKING: UserState
US_LISTENING: UserState
US_AWAY: UserState

class MetricsReport(_message.Message):
    __slots__ = ("started_speaking_at", "stopped_speaking_at", "transcription_delay", "end_of_turn_delay", "on_user_turn_completed_delay", "llm_node_ttft", "tts_node_ttfb", "e2e_latency")
    STARTED_SPEAKING_AT_FIELD_NUMBER: _ClassVar[int]
    STOPPED_SPEAKING_AT_FIELD_NUMBER: _ClassVar[int]
    TRANSCRIPTION_DELAY_FIELD_NUMBER: _ClassVar[int]
    END_OF_TURN_DELAY_FIELD_NUMBER: _ClassVar[int]
    ON_USER_TURN_COMPLETED_DELAY_FIELD_NUMBER: _ClassVar[int]
    LLM_NODE_TTFT_FIELD_NUMBER: _ClassVar[int]
    TTS_NODE_TTFB_FIELD_NUMBER: _ClassVar[int]
    E2E_LATENCY_FIELD_NUMBER: _ClassVar[int]
    started_speaking_at: _timestamp_pb2.Timestamp
    stopped_speaking_at: _timestamp_pb2.Timestamp
    transcription_delay: float
    end_of_turn_delay: float
    on_user_turn_completed_delay: float
    llm_node_ttft: float
    tts_node_ttfb: float
    e2e_latency: float
    def __init__(self, started_speaking_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., stopped_speaking_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., transcription_delay: _Optional[float] = ..., end_of_turn_delay: _Optional[float] = ..., on_user_turn_completed_delay: _Optional[float] = ..., llm_node_ttft: _Optional[float] = ..., tts_node_ttfb: _Optional[float] = ..., e2e_latency: _Optional[float] = ...) -> None: ...

class ChatMessage(_message.Message):
    __slots__ = ("id", "role", "content", "interrupted", "transcript_confidence", "extra", "metrics", "created_at")
    class ChatContent(_message.Message):
        __slots__ = ("text",)
        TEXT_FIELD_NUMBER: _ClassVar[int]
        text: str
        def __init__(self, text: _Optional[str] = ...) -> None: ...
    class ExtraEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    INTERRUPTED_FIELD_NUMBER: _ClassVar[int]
    TRANSCRIPT_CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    EXTRA_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    id: str
    role: ChatRole
    content: _containers.RepeatedCompositeFieldContainer[ChatMessage.ChatContent]
    interrupted: bool
    transcript_confidence: float
    extra: _containers.ScalarMap[str, str]
    metrics: MetricsReport
    created_at: _timestamp_pb2.Timestamp
    def __init__(self, id: _Optional[str] = ..., role: _Optional[_Union[ChatRole, str]] = ..., content: _Optional[_Iterable[_Union[ChatMessage.ChatContent, _Mapping]]] = ..., interrupted: bool = ..., transcript_confidence: _Optional[float] = ..., extra: _Optional[_Mapping[str, str]] = ..., metrics: _Optional[_Union[MetricsReport, _Mapping]] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class FunctionCall(_message.Message):
    __slots__ = ("id", "call_id", "arguments", "name", "created_at")
    ID_FIELD_NUMBER: _ClassVar[int]
    CALL_ID_FIELD_NUMBER: _ClassVar[int]
    ARGUMENTS_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    id: str
    call_id: str
    arguments: str
    name: str
    created_at: _timestamp_pb2.Timestamp
    def __init__(self, id: _Optional[str] = ..., call_id: _Optional[str] = ..., arguments: _Optional[str] = ..., name: _Optional[str] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class FunctionCallOutput(_message.Message):
    __slots__ = ("id", "name", "call_id", "output", "is_error", "created_at")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    CALL_ID_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    IS_ERROR_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    call_id: str
    output: str
    is_error: bool
    created_at: _timestamp_pb2.Timestamp
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., call_id: _Optional[str] = ..., output: _Optional[str] = ..., is_error: bool = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class AgentHandoff(_message.Message):
    __slots__ = ("id", "old_agent_id", "new_agent_id", "created_at")
    ID_FIELD_NUMBER: _ClassVar[int]
    OLD_AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    NEW_AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    id: str
    old_agent_id: str
    new_agent_id: str
    created_at: _timestamp_pb2.Timestamp
    def __init__(self, id: _Optional[str] = ..., old_agent_id: _Optional[str] = ..., new_agent_id: _Optional[str] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class AgentConfigUpdate(_message.Message):
    __slots__ = ("id", "instructions", "tools_added", "tools_removed", "created_at")
    ID_FIELD_NUMBER: _ClassVar[int]
    INSTRUCTIONS_FIELD_NUMBER: _ClassVar[int]
    TOOLS_ADDED_FIELD_NUMBER: _ClassVar[int]
    TOOLS_REMOVED_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    id: str
    instructions: str
    tools_added: _containers.RepeatedScalarFieldContainer[str]
    tools_removed: _containers.RepeatedScalarFieldContainer[str]
    created_at: _timestamp_pb2.Timestamp
    def __init__(self, id: _Optional[str] = ..., instructions: _Optional[str] = ..., tools_added: _Optional[_Iterable[str]] = ..., tools_removed: _Optional[_Iterable[str]] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class ChatContext(_message.Message):
    __slots__ = ("items",)
    class ChatItem(_message.Message):
        __slots__ = ("message", "function_call", "function_call_output", "agent_handoff", "agent_config_update")
        MESSAGE_FIELD_NUMBER: _ClassVar[int]
        FUNCTION_CALL_FIELD_NUMBER: _ClassVar[int]
        FUNCTION_CALL_OUTPUT_FIELD_NUMBER: _ClassVar[int]
        AGENT_HANDOFF_FIELD_NUMBER: _ClassVar[int]
        AGENT_CONFIG_UPDATE_FIELD_NUMBER: _ClassVar[int]
        message: ChatMessage
        function_call: FunctionCall
        function_call_output: FunctionCallOutput
        agent_handoff: AgentHandoff
        agent_config_update: AgentConfigUpdate
        def __init__(self, message: _Optional[_Union[ChatMessage, _Mapping]] = ..., function_call: _Optional[_Union[FunctionCall, _Mapping]] = ..., function_call_output: _Optional[_Union[FunctionCallOutput, _Mapping]] = ..., agent_handoff: _Optional[_Union[AgentHandoff, _Mapping]] = ..., agent_config_update: _Optional[_Union[AgentConfigUpdate, _Mapping]] = ...) -> None: ...
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[ChatContext.ChatItem]
    def __init__(self, items: _Optional[_Iterable[_Union[ChatContext.ChatItem, _Mapping]]] = ...) -> None: ...

class LLMModelUsage(_message.Message):
    __slots__ = ("provider", "model", "input_tokens", "input_cached_tokens", "input_audio_tokens", "input_cached_audio_tokens", "input_text_tokens", "input_cached_text_tokens", "input_image_tokens", "input_cached_image_tokens", "output_tokens", "output_audio_tokens", "output_text_tokens", "session_duration")
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    INPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    INPUT_CACHED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    INPUT_AUDIO_TOKENS_FIELD_NUMBER: _ClassVar[int]
    INPUT_CACHED_AUDIO_TOKENS_FIELD_NUMBER: _ClassVar[int]
    INPUT_TEXT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    INPUT_CACHED_TEXT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    INPUT_IMAGE_TOKENS_FIELD_NUMBER: _ClassVar[int]
    INPUT_CACHED_IMAGE_TOKENS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_AUDIO_TOKENS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TEXT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    SESSION_DURATION_FIELD_NUMBER: _ClassVar[int]
    provider: str
    model: str
    input_tokens: int
    input_cached_tokens: int
    input_audio_tokens: int
    input_cached_audio_tokens: int
    input_text_tokens: int
    input_cached_text_tokens: int
    input_image_tokens: int
    input_cached_image_tokens: int
    output_tokens: int
    output_audio_tokens: int
    output_text_tokens: int
    session_duration: float
    def __init__(self, provider: _Optional[str] = ..., model: _Optional[str] = ..., input_tokens: _Optional[int] = ..., input_cached_tokens: _Optional[int] = ..., input_audio_tokens: _Optional[int] = ..., input_cached_audio_tokens: _Optional[int] = ..., input_text_tokens: _Optional[int] = ..., input_cached_text_tokens: _Optional[int] = ..., input_image_tokens: _Optional[int] = ..., input_cached_image_tokens: _Optional[int] = ..., output_tokens: _Optional[int] = ..., output_audio_tokens: _Optional[int] = ..., output_text_tokens: _Optional[int] = ..., session_duration: _Optional[float] = ...) -> None: ...

class TTSModelUsage(_message.Message):
    __slots__ = ("provider", "model", "input_tokens", "output_tokens", "characters_count", "audio_duration")
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    INPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    CHARACTERS_COUNT_FIELD_NUMBER: _ClassVar[int]
    AUDIO_DURATION_FIELD_NUMBER: _ClassVar[int]
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    characters_count: int
    audio_duration: float
    def __init__(self, provider: _Optional[str] = ..., model: _Optional[str] = ..., input_tokens: _Optional[int] = ..., output_tokens: _Optional[int] = ..., characters_count: _Optional[int] = ..., audio_duration: _Optional[float] = ...) -> None: ...

class STTModelUsage(_message.Message):
    __slots__ = ("provider", "model", "input_tokens", "output_tokens", "audio_duration")
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    INPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    AUDIO_DURATION_FIELD_NUMBER: _ClassVar[int]
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    audio_duration: float
    def __init__(self, provider: _Optional[str] = ..., model: _Optional[str] = ..., input_tokens: _Optional[int] = ..., output_tokens: _Optional[int] = ..., audio_duration: _Optional[float] = ...) -> None: ...

class InterruptionModelUsage(_message.Message):
    __slots__ = ("provider", "model", "total_requests")
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    TOTAL_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    provider: str
    model: str
    total_requests: int
    def __init__(self, provider: _Optional[str] = ..., model: _Optional[str] = ..., total_requests: _Optional[int] = ...) -> None: ...

class ModelUsage(_message.Message):
    __slots__ = ("llm", "tts", "stt", "interruption")
    LLM_FIELD_NUMBER: _ClassVar[int]
    TTS_FIELD_NUMBER: _ClassVar[int]
    STT_FIELD_NUMBER: _ClassVar[int]
    INTERRUPTION_FIELD_NUMBER: _ClassVar[int]
    llm: LLMModelUsage
    tts: TTSModelUsage
    stt: STTModelUsage
    interruption: InterruptionModelUsage
    def __init__(self, llm: _Optional[_Union[LLMModelUsage, _Mapping]] = ..., tts: _Optional[_Union[TTSModelUsage, _Mapping]] = ..., stt: _Optional[_Union[STTModelUsage, _Mapping]] = ..., interruption: _Optional[_Union[InterruptionModelUsage, _Mapping]] = ...) -> None: ...

class AgentSessionUsage(_message.Message):
    __slots__ = ("model_usage",)
    MODEL_USAGE_FIELD_NUMBER: _ClassVar[int]
    model_usage: _containers.RepeatedCompositeFieldContainer[ModelUsage]
    def __init__(self, model_usage: _Optional[_Iterable[_Union[ModelUsage, _Mapping]]] = ...) -> None: ...

class AgentSessionEvent(_message.Message):
    __slots__ = ("created_at", "agent_state_changed", "user_state_changed", "conversation_item_added", "user_input_transcribed", "function_tools_executed", "error", "overlapping_speech", "session_usage_updated")
    class AgentStateChanged(_message.Message):
        __slots__ = ("old_state", "new_state")
        OLD_STATE_FIELD_NUMBER: _ClassVar[int]
        NEW_STATE_FIELD_NUMBER: _ClassVar[int]
        old_state: AgentState
        new_state: AgentState
        def __init__(self, old_state: _Optional[_Union[AgentState, str]] = ..., new_state: _Optional[_Union[AgentState, str]] = ...) -> None: ...
    class UserStateChanged(_message.Message):
        __slots__ = ("old_state", "new_state")
        OLD_STATE_FIELD_NUMBER: _ClassVar[int]
        NEW_STATE_FIELD_NUMBER: _ClassVar[int]
        old_state: UserState
        new_state: UserState
        def __init__(self, old_state: _Optional[_Union[UserState, str]] = ..., new_state: _Optional[_Union[UserState, str]] = ...) -> None: ...
    class ConversationItemAdded(_message.Message):
        __slots__ = ("item",)
        ITEM_FIELD_NUMBER: _ClassVar[int]
        item: ChatContext.ChatItem
        def __init__(self, item: _Optional[_Union[ChatContext.ChatItem, _Mapping]] = ...) -> None: ...
    class UserInputTranscribed(_message.Message):
        __slots__ = ("transcript", "is_final", "language")
        TRANSCRIPT_FIELD_NUMBER: _ClassVar[int]
        IS_FINAL_FIELD_NUMBER: _ClassVar[int]
        LANGUAGE_FIELD_NUMBER: _ClassVar[int]
        transcript: str
        is_final: bool
        language: str
        def __init__(self, transcript: _Optional[str] = ..., is_final: bool = ..., language: _Optional[str] = ...) -> None: ...
    class FunctionToolsExecuted(_message.Message):
        __slots__ = ("function_calls", "function_call_outputs")
        FUNCTION_CALLS_FIELD_NUMBER: _ClassVar[int]
        FUNCTION_CALL_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
        function_calls: _containers.RepeatedCompositeFieldContainer[FunctionCall]
        function_call_outputs: _containers.RepeatedCompositeFieldContainer[FunctionCallOutput]
        def __init__(self, function_calls: _Optional[_Iterable[_Union[FunctionCall, _Mapping]]] = ..., function_call_outputs: _Optional[_Iterable[_Union[FunctionCallOutput, _Mapping]]] = ...) -> None: ...
    class Error(_message.Message):
        __slots__ = ("message",)
        MESSAGE_FIELD_NUMBER: _ClassVar[int]
        message: str
        def __init__(self, message: _Optional[str] = ...) -> None: ...
    class OverlappingSpeech(_message.Message):
        __slots__ = ("is_interruption", "overlap_started_at", "detection_delay", "detected_at")
        IS_INTERRUPTION_FIELD_NUMBER: _ClassVar[int]
        OVERLAP_STARTED_AT_FIELD_NUMBER: _ClassVar[int]
        DETECTION_DELAY_FIELD_NUMBER: _ClassVar[int]
        DETECTED_AT_FIELD_NUMBER: _ClassVar[int]
        is_interruption: bool
        overlap_started_at: _timestamp_pb2.Timestamp
        detection_delay: float
        detected_at: _timestamp_pb2.Timestamp
        def __init__(self, is_interruption: bool = ..., overlap_started_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., detection_delay: _Optional[float] = ..., detected_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...
    class SessionUsageUpdated(_message.Message):
        __slots__ = ("usage",)
        USAGE_FIELD_NUMBER: _ClassVar[int]
        usage: AgentSessionUsage
        def __init__(self, usage: _Optional[_Union[AgentSessionUsage, _Mapping]] = ...) -> None: ...
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    AGENT_STATE_CHANGED_FIELD_NUMBER: _ClassVar[int]
    USER_STATE_CHANGED_FIELD_NUMBER: _ClassVar[int]
    CONVERSATION_ITEM_ADDED_FIELD_NUMBER: _ClassVar[int]
    USER_INPUT_TRANSCRIBED_FIELD_NUMBER: _ClassVar[int]
    FUNCTION_TOOLS_EXECUTED_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    OVERLAPPING_SPEECH_FIELD_NUMBER: _ClassVar[int]
    SESSION_USAGE_UPDATED_FIELD_NUMBER: _ClassVar[int]
    created_at: _timestamp_pb2.Timestamp
    agent_state_changed: AgentSessionEvent.AgentStateChanged
    user_state_changed: AgentSessionEvent.UserStateChanged
    conversation_item_added: AgentSessionEvent.ConversationItemAdded
    user_input_transcribed: AgentSessionEvent.UserInputTranscribed
    function_tools_executed: AgentSessionEvent.FunctionToolsExecuted
    error: AgentSessionEvent.Error
    overlapping_speech: AgentSessionEvent.OverlappingSpeech
    session_usage_updated: AgentSessionEvent.SessionUsageUpdated
    def __init__(self, created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., agent_state_changed: _Optional[_Union[AgentSessionEvent.AgentStateChanged, _Mapping]] = ..., user_state_changed: _Optional[_Union[AgentSessionEvent.UserStateChanged, _Mapping]] = ..., conversation_item_added: _Optional[_Union[AgentSessionEvent.ConversationItemAdded, _Mapping]] = ..., user_input_transcribed: _Optional[_Union[AgentSessionEvent.UserInputTranscribed, _Mapping]] = ..., function_tools_executed: _Optional[_Union[AgentSessionEvent.FunctionToolsExecuted, _Mapping]] = ..., error: _Optional[_Union[AgentSessionEvent.Error, _Mapping]] = ..., overlapping_speech: _Optional[_Union[AgentSessionEvent.OverlappingSpeech, _Mapping]] = ..., session_usage_updated: _Optional[_Union[AgentSessionEvent.SessionUsageUpdated, _Mapping]] = ...) -> None: ...

class SessionRequest(_message.Message):
    __slots__ = ("request_id", "ping", "get_chat_history", "run_input", "get_agent_info", "get_session_state", "get_rtc_stats", "get_session_usage")
    class Ping(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class GetChatHistory(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class RunInput(_message.Message):
        __slots__ = ("text",)
        TEXT_FIELD_NUMBER: _ClassVar[int]
        text: str
        def __init__(self, text: _Optional[str] = ...) -> None: ...
    class GetAgentInfo(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class GetSessionState(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class GetRTCStats(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class GetSessionUsage(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PING_FIELD_NUMBER: _ClassVar[int]
    GET_CHAT_HISTORY_FIELD_NUMBER: _ClassVar[int]
    RUN_INPUT_FIELD_NUMBER: _ClassVar[int]
    GET_AGENT_INFO_FIELD_NUMBER: _ClassVar[int]
    GET_SESSION_STATE_FIELD_NUMBER: _ClassVar[int]
    GET_RTC_STATS_FIELD_NUMBER: _ClassVar[int]
    GET_SESSION_USAGE_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    ping: SessionRequest.Ping
    get_chat_history: SessionRequest.GetChatHistory
    run_input: SessionRequest.RunInput
    get_agent_info: SessionRequest.GetAgentInfo
    get_session_state: SessionRequest.GetSessionState
    get_rtc_stats: SessionRequest.GetRTCStats
    get_session_usage: SessionRequest.GetSessionUsage
    def __init__(self, request_id: _Optional[str] = ..., ping: _Optional[_Union[SessionRequest.Ping, _Mapping]] = ..., get_chat_history: _Optional[_Union[SessionRequest.GetChatHistory, _Mapping]] = ..., run_input: _Optional[_Union[SessionRequest.RunInput, _Mapping]] = ..., get_agent_info: _Optional[_Union[SessionRequest.GetAgentInfo, _Mapping]] = ..., get_session_state: _Optional[_Union[SessionRequest.GetSessionState, _Mapping]] = ..., get_rtc_stats: _Optional[_Union[SessionRequest.GetRTCStats, _Mapping]] = ..., get_session_usage: _Optional[_Union[SessionRequest.GetSessionUsage, _Mapping]] = ...) -> None: ...

class SessionResponse(_message.Message):
    __slots__ = ("request_id", "error", "pong", "get_chat_history", "run_input", "get_agent_info", "get_session_state", "get_rtc_stats", "get_session_usage")
    class Pong(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class GetChatHistoryResponse(_message.Message):
        __slots__ = ("items",)
        ITEMS_FIELD_NUMBER: _ClassVar[int]
        items: _containers.RepeatedCompositeFieldContainer[ChatContext.ChatItem]
        def __init__(self, items: _Optional[_Iterable[_Union[ChatContext.ChatItem, _Mapping]]] = ...) -> None: ...
    class GetAgentInfoResponse(_message.Message):
        __slots__ = ("id", "instructions", "tools", "chat_ctx")
        ID_FIELD_NUMBER: _ClassVar[int]
        INSTRUCTIONS_FIELD_NUMBER: _ClassVar[int]
        TOOLS_FIELD_NUMBER: _ClassVar[int]
        CHAT_CTX_FIELD_NUMBER: _ClassVar[int]
        id: str
        instructions: str
        tools: _containers.RepeatedScalarFieldContainer[str]
        chat_ctx: _containers.RepeatedCompositeFieldContainer[ChatContext.ChatItem]
        def __init__(self, id: _Optional[str] = ..., instructions: _Optional[str] = ..., tools: _Optional[_Iterable[str]] = ..., chat_ctx: _Optional[_Iterable[_Union[ChatContext.ChatItem, _Mapping]]] = ...) -> None: ...
    class RunInputResponse(_message.Message):
        __slots__ = ("items",)
        ITEMS_FIELD_NUMBER: _ClassVar[int]
        items: _containers.RepeatedCompositeFieldContainer[ChatContext.ChatItem]
        def __init__(self, items: _Optional[_Iterable[_Union[ChatContext.ChatItem, _Mapping]]] = ...) -> None: ...
    class GetSessionStateResponse(_message.Message):
        __slots__ = ("agent_state", "user_state", "agent_id", "options", "created_at")
        class OptionsEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: str
            def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
        AGENT_STATE_FIELD_NUMBER: _ClassVar[int]
        USER_STATE_FIELD_NUMBER: _ClassVar[int]
        AGENT_ID_FIELD_NUMBER: _ClassVar[int]
        OPTIONS_FIELD_NUMBER: _ClassVar[int]
        CREATED_AT_FIELD_NUMBER: _ClassVar[int]
        agent_state: AgentState
        user_state: UserState
        agent_id: str
        options: _containers.ScalarMap[str, str]
        created_at: _timestamp_pb2.Timestamp
        def __init__(self, agent_state: _Optional[_Union[AgentState, str]] = ..., user_state: _Optional[_Union[UserState, str]] = ..., agent_id: _Optional[str] = ..., options: _Optional[_Mapping[str, str]] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...
    class GetRTCStatsResponse(_message.Message):
        __slots__ = ("publisher_stats", "subscriber_stats")
        PUBLISHER_STATS_FIELD_NUMBER: _ClassVar[int]
        SUBSCRIBER_STATS_FIELD_NUMBER: _ClassVar[int]
        publisher_stats: _containers.RepeatedCompositeFieldContainer[_struct_pb2.Struct]
        subscriber_stats: _containers.RepeatedCompositeFieldContainer[_struct_pb2.Struct]
        def __init__(self, publisher_stats: _Optional[_Iterable[_Union[_struct_pb2.Struct, _Mapping]]] = ..., subscriber_stats: _Optional[_Iterable[_Union[_struct_pb2.Struct, _Mapping]]] = ...) -> None: ...
    class GetSessionUsageResponse(_message.Message):
        __slots__ = ("usage", "created_at")
        USAGE_FIELD_NUMBER: _ClassVar[int]
        CREATED_AT_FIELD_NUMBER: _ClassVar[int]
        usage: AgentSessionUsage
        created_at: _timestamp_pb2.Timestamp
        def __init__(self, usage: _Optional[_Union[AgentSessionUsage, _Mapping]] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    PONG_FIELD_NUMBER: _ClassVar[int]
    GET_CHAT_HISTORY_FIELD_NUMBER: _ClassVar[int]
    RUN_INPUT_FIELD_NUMBER: _ClassVar[int]
    GET_AGENT_INFO_FIELD_NUMBER: _ClassVar[int]
    GET_SESSION_STATE_FIELD_NUMBER: _ClassVar[int]
    GET_RTC_STATS_FIELD_NUMBER: _ClassVar[int]
    GET_SESSION_USAGE_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    error: str
    pong: SessionResponse.Pong
    get_chat_history: SessionResponse.GetChatHistoryResponse
    run_input: SessionResponse.RunInputResponse
    get_agent_info: SessionResponse.GetAgentInfoResponse
    get_session_state: SessionResponse.GetSessionStateResponse
    get_rtc_stats: SessionResponse.GetRTCStatsResponse
    get_session_usage: SessionResponse.GetSessionUsageResponse
    def __init__(self, request_id: _Optional[str] = ..., error: _Optional[str] = ..., pong: _Optional[_Union[SessionResponse.Pong, _Mapping]] = ..., get_chat_history: _Optional[_Union[SessionResponse.GetChatHistoryResponse, _Mapping]] = ..., run_input: _Optional[_Union[SessionResponse.RunInputResponse, _Mapping]] = ..., get_agent_info: _Optional[_Union[SessionResponse.GetAgentInfoResponse, _Mapping]] = ..., get_session_state: _Optional[_Union[SessionResponse.GetSessionStateResponse, _Mapping]] = ..., get_rtc_stats: _Optional[_Union[SessionResponse.GetRTCStatsResponse, _Mapping]] = ..., get_session_usage: _Optional[_Union[SessionResponse.GetSessionUsageResponse, _Mapping]] = ...) -> None: ...

class AgentSessionMessage(_message.Message):
    __slots__ = ("audio_input", "audio_output", "event", "request", "response", "audio_playback_flush", "audio_playback_clear", "audio_playback_finished")
    class ConsoleIO(_message.Message):
        __slots__ = ()
        class AudioFrame(_message.Message):
            __slots__ = ("data", "sample_rate", "num_channels", "samples_per_channel")
            DATA_FIELD_NUMBER: _ClassVar[int]
            SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
            NUM_CHANNELS_FIELD_NUMBER: _ClassVar[int]
            SAMPLES_PER_CHANNEL_FIELD_NUMBER: _ClassVar[int]
            data: bytes
            sample_rate: int
            num_channels: int
            samples_per_channel: int
            def __init__(self, data: _Optional[bytes] = ..., sample_rate: _Optional[int] = ..., num_channels: _Optional[int] = ..., samples_per_channel: _Optional[int] = ...) -> None: ...
        class AudioPlaybackFlush(_message.Message):
            __slots__ = ()
            def __init__(self) -> None: ...
        class AudioPlaybackClear(_message.Message):
            __slots__ = ()
            def __init__(self) -> None: ...
        class AudioPlaybackFinished(_message.Message):
            __slots__ = ()
            def __init__(self) -> None: ...
        def __init__(self) -> None: ...
    AUDIO_INPUT_FIELD_NUMBER: _ClassVar[int]
    AUDIO_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    EVENT_FIELD_NUMBER: _ClassVar[int]
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    AUDIO_PLAYBACK_FLUSH_FIELD_NUMBER: _ClassVar[int]
    AUDIO_PLAYBACK_CLEAR_FIELD_NUMBER: _ClassVar[int]
    AUDIO_PLAYBACK_FINISHED_FIELD_NUMBER: _ClassVar[int]
    audio_input: AgentSessionMessage.ConsoleIO.AudioFrame
    audio_output: AgentSessionMessage.ConsoleIO.AudioFrame
    event: AgentSessionEvent
    request: SessionRequest
    response: SessionResponse
    audio_playback_flush: AgentSessionMessage.ConsoleIO.AudioPlaybackFlush
    audio_playback_clear: AgentSessionMessage.ConsoleIO.AudioPlaybackClear
    audio_playback_finished: AgentSessionMessage.ConsoleIO.AudioPlaybackFinished
    def __init__(self, audio_input: _Optional[_Union[AgentSessionMessage.ConsoleIO.AudioFrame, _Mapping]] = ..., audio_output: _Optional[_Union[AgentSessionMessage.ConsoleIO.AudioFrame, _Mapping]] = ..., event: _Optional[_Union[AgentSessionEvent, _Mapping]] = ..., request: _Optional[_Union[SessionRequest, _Mapping]] = ..., response: _Optional[_Union[SessionResponse, _Mapping]] = ..., audio_playback_flush: _Optional[_Union[AgentSessionMessage.ConsoleIO.AudioPlaybackFlush, _Mapping]] = ..., audio_playback_clear: _Optional[_Union[AgentSessionMessage.ConsoleIO.AudioPlaybackClear, _Mapping]] = ..., audio_playback_finished: _Optional[_Union[AgentSessionMessage.ConsoleIO.AudioPlaybackFinished, _Mapping]] = ...) -> None: ...
