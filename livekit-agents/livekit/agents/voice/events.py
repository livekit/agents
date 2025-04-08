from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Literal, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field

from ..llm import LLM, ChatMessage, FunctionCall, FunctionCallOutput, LLMError
from ..metrics import AgentMetrics
from ..stt import STT, STTError
from ..tts import TTS, TTSError
from ..types import AgentState
from .speech_handle import SpeechHandle

if TYPE_CHECKING:
    from .agent_session import AgentSession


Userdata_T = TypeVar("Userdata_T")


class RunContext(Generic[Userdata_T]):
    # private ctor
    def __init__(
        self,
        *,
        session: AgentSession,
        speech_handle: SpeechHandle,
        function_call: FunctionCall,
    ) -> None:
        self._session = session
        self._speech_handle = speech_handle
        self._function_call = function_call

    @property
    def session(self) -> AgentSession[Userdata_T]:
        return self._session

    @property
    def speech_handle(self) -> SpeechHandle:
        return self._speech_handle

    @property
    def function_call(self) -> FunctionCall:
        return self._function_call

    @property
    def userdata(self) -> Userdata_T:
        return self.session.userdata


EventTypes = Literal[
    "user_started_speaking",
    "user_stopped_speaking",
    "user_input_transcribed",
    "agent_started_speaking",
    "agent_stopped_speaking",
    "agent_state_changed",
    "conversation_item_added",
    "function_tools_executed",
    "metrics_collected",
    "speech_created",
    "error",
    "session_close",
]


class UserStartedSpeakingEvent(BaseModel):
    type: Literal["user_started_speaking"] = "user_started_speaking"


class UserStoppedSpeakingEvent(BaseModel):
    type: Literal["user_stopped_speaking"] = "user_stopped_speaking"


class UserInputTranscribedEvent(BaseModel):
    type: Literal["user_input_transcribed"] = "user_input_transcribed"
    transcript: str
    is_final: bool


class AgentStartedSpeakingEvent(BaseModel):
    type: Literal["agent_started_speaking"] = "agent_started_speaking"


class AgentStoppedSpeakingEvent(BaseModel):
    type: Literal["agent_stopped_speaking"] = "agent_stopped_speaking"


class AgentStateChangedEvent(BaseModel):
    type: Literal["agent_state_changed"] = "agent_state_changed"
    state: AgentState


class MetricsCollectedEvent(BaseModel):
    type: Literal["metrics_collected"] = "metrics_collected"
    metrics: AgentMetrics


class ConversationItemAddedEvent(BaseModel):
    type: Literal["conversation_item_added"] = "conversation_item_added"
    message: ChatMessage


class FunctionToolsExecutedEvent(BaseModel):
    type: Literal["function_tools_executed"] = "function_tools_executed"
    function_calls: list[FunctionCall]
    function_call_outputs: list[FunctionCallOutput]


class SpeechCreatedEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["speech_created"] = "speech_created"
    user_initiated: bool
    """True if the speech was created using public methods like `say` or `generate_reply`"""
    source: Literal["say", "generate_reply", "tool_response"]
    """Source indicating how the speech handle was created"""
    speech_handle: SpeechHandle = Field(..., exclude=True)
    """The speech handle that was created"""


class ErrorEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["error"] = "error"
    error: LLMError | STTError | TTSError
    source: LLM | STT | TTS


class SessionCloseEvent(BaseModel):
    type: Literal["session_close"] = "session_close"
    cause: ErrorEvent | None = None


AgentEvent = Union[
    UserStartedSpeakingEvent,
    UserStoppedSpeakingEvent,
    UserInputTranscribedEvent,
    AgentStartedSpeakingEvent,
    AgentStoppedSpeakingEvent,
    AgentStateChangedEvent,
    MetricsCollectedEvent,
    ConversationItemAddedEvent,
    SpeechCreatedEvent,
    ErrorEvent,
    SessionCloseEvent,
]
