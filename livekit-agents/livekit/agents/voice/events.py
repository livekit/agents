from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Generic, Literal, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from ..llm import LLM, ChatMessage, FunctionCall, FunctionCallOutput, LLMError
from ..metrics import AgentMetrics
from ..stt import STT, STTError
from ..tts import TTS, TTSError
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
    "user_state_changed",
    "agent_state_changed",
    "user_input_transcribed",
    "conversation_item_added",
    "function_tools_executed",
    "metrics_collected",
    "speech_created",
    "error",
    "close",
]

UserState = Literal["speaking", "listening", "away"]
AgentState = Literal["initializing", "idle", "listening", "thinking", "speaking"]


class UserStateChangedEvent(BaseModel):
    type: Literal["user_state_changed"] = "user_state_changed"
    old_state: UserState
    new_state: UserState


class AgentStateChangedEvent(BaseModel):
    type: Literal["agent_state_changed"] = "agent_state_changed"
    old_state: AgentState
    new_state: AgentState


class UserInputTranscribedEvent(BaseModel):
    type: Literal["user_input_transcribed"] = "user_input_transcribed"
    transcript: str
    is_final: bool


class MetricsCollectedEvent(BaseModel):
    type: Literal["metrics_collected"] = "metrics_collected"
    metrics: AgentMetrics


class _TypeDiscriminator(BaseModel):
    type: Literal["unknown"] = "unknown"  # force user to use the type discriminator


class ConversationItemAddedEvent(BaseModel):
    type: Literal["conversation_item_added"] = "conversation_item_added"
    item: ChatMessage | _TypeDiscriminator


class FunctionToolsExecutedEvent(BaseModel):
    type: Literal["function_tools_executed"] = "function_tools_executed"
    function_calls: list[FunctionCall]
    function_call_outputs: list[FunctionCallOutput]

    def zipped(self) -> list[tuple[FunctionCall, FunctionCallOutput]]:
        return list(zip(self.function_calls, self.function_call_outputs))

    @model_validator(mode="after")
    def verify_lists_length(self) -> Self:
        if len(self.function_calls) != len(self.function_call_outputs):
            raise ValueError("The number of function_calls and function_call_outputs must match.")

        return self


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
    error: LLMError | STTError | TTSError | Any
    source: LLM | STT | TTS | Any


class CloseEvent(BaseModel):
    type: Literal["close"] = "close"
    error: LLMError | STTError | TTSError | None = None


AgentEvent = Annotated[
    Union[
        UserInputTranscribedEvent,
        UserStateChangedEvent,
        AgentStateChangedEvent,
        MetricsCollectedEvent,
        ConversationItemAddedEvent,
        SpeechCreatedEvent,
        ErrorEvent,
        CloseEvent,
    ],
    Field(discriminator="type"),
]
