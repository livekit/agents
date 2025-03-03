from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar, Literal, Union

from ..llm import FunctionCall, ChatMessage
from ..metrics import AgentMetrics
from ..types import AgentState

from pydantic import BaseModel

if TYPE_CHECKING:
    from .pipeline_agent import PipelineAgent
    from .speech_handle import SpeechHandle


Userdata_T = TypeVar("Userdata_T")


class CallContext(Generic[Userdata_T]):
    # private ctor
    def __init__(
        self,
        *,
        agent: PipelineAgent,
        speech_handle: SpeechHandle,
        function_call: FunctionCall,
    ) -> None:
        self._agent = agent
        self._speech_handle = speech_handle
        self._function_call = function_call

    @property
    def agent(self) -> PipelineAgent[Userdata_T]:
        return self._agent

    @property
    def speech_handle(self) -> SpeechHandle:
        return self._speech_handle

    @property
    def function_call(self) -> FunctionCall:
        return self._function_call

    @property
    def userdata(self) -> Userdata_T:
        return self.agent.userdata


EventTypes = Literal[
    "user_started_speaking",
    "user_stopped_speaking",
    "user_input_transcribed",
    "agent_started_speaking",
    "agent_stopped_speaking",
    "agent_state_changed",
    "conversation_item_added",
    "metrics_collected",
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


AgentEvent = Union[
    UserStartedSpeakingEvent,
    UserStoppedSpeakingEvent,
    UserInputTranscribedEvent,
    AgentStartedSpeakingEvent,
    AgentStoppedSpeakingEvent,
    AgentStateChangedEvent,
    MetricsCollectedEvent,
    ConversationItemAddedEvent,
]
