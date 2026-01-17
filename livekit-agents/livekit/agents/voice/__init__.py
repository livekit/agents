from . import io, run_result
from .agent import Agent, AgentTask, ModelSettings
from .agent_session import AgentSession, VoiceActivityVideoSampler
from .events import (
    AgentEvent,
    AgentFalseInterruptionEvent,
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
from .filter import InterruptionFilter
from .room_io import (
    _ParticipantAudioOutput,
    _ParticipantStreamTranscriptionOutput,
    _ParticipantTranscriptionOutput,
)
from .speech_handle import SpeechHandle
from .transcription import TranscriptSynchronizer

__all__ = [
    # Core agent
    "Agent",
    "AgentTask",
    "ModelSettings",
    "AgentSession",
    "VoiceActivityVideoSampler",
    # Speech / transcription
    "SpeechHandle",
    "TranscriptSynchronizer",
    # Interruption handling
    "InterruptionFilter",
    # Events
    "AgentEvent",
    "AgentFalseInterruptionEvent",
    "AgentStateChangedEvent",
    "ConversationItemAddedEvent",
    "SpeechCreatedEvent",
    "UserInputTranscribedEvent",
    "UserStateChangedEvent",
    "FunctionToolsExecutedEvent",
    "MetricsCollectedEvent",
    "ErrorEvent",
    "CloseEvent",
    "CloseReason",
    # IO / results
    "RunContext",
    "io",
    "run_result",
    # Internal outputs (intentionally exported)
    "_ParticipantAudioOutput",
    "_ParticipantTranscriptionOutput",
    "_ParticipantStreamTranscriptionOutput",
]

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__: dict[str, bool] = {}

for name in NOT_IN_ALL:
    __pdoc__[name] = False
