from .agent import Agent, InlineTask, ModelSettings
from .agent_session import AgentSession, VoiceActivityVideoSampler
from .chat_cli import ChatCLI
from .events import (
    AgentEvent,
    AgentStateChangedEvent,
    CloseEvent,
    CloseReason,
    ConversationItemAddedEvent,
    ErrorEvent,
    FunctionToolsExecutedEvent,
    MetricsCollectedEvent,
    RunContext,
    SpeechCreatedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
)
from .speech_handle import SpeechHandle

__all__ = [
    "ChatCLI",
    "AgentSession",
    "VoiceActivityVideoSampler",
    "Agent",
    "ModelSettings",
    "InlineTask",
    "SpeechHandle",
    "RunContext",
    "UserInputTranscribedEvent",
    "AgentEvent",
    "MetricsCollectedEvent",
    "ConversationItemAddedEvent",
    "SpeechCreatedEvent",
    "ErrorEvent",
    "CloseEvent",
    "CloseReason",
    "UserStateChangedEvent",
    "AgentStateChangedEvent",
    "FunctionToolsExecutedEvent",
]

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
