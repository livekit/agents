# IO / results
from . import io, run_result

# Core agent
from .agent import Agent, AgentTask, ModelSettings
from .agent_session import AgentSession, VoiceActivityVideoSampler

# Events
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

# Interruption handling
from .interruption_filter import InterruptionFilter

# Internal outputs (intentionally exported)
from .room_io import (
    _ParticipantAudioOutput,
    _ParticipantStreamTranscriptionOutput,
    _ParticipantTranscriptionOutput,
)

# Speech / transcription
from .speech_handle import SpeechHandle
from .transcription import TranscriptSynchronizer

__all__ = [
    # Core agent
    "Agent",
    "AgentTask",
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
    "ErrorEvent",
    "CloseEvent",
    "CloseReason",

    # IO / results
    "RunContext",
    "io",
    "run_result",

    # Internal outputs
    "_ParticipantAudioOutput",
    "_ParticipantTranscriptionOutput",
    "_ParticipantStreamTranscriptionOutput",
]

# Hide non-exported symbols from docs (pdoc)
__pdoc__ = {}

for name in list(globals()):
    if name not in __all__ and not name.startswith("_"):
        __pdoc__[name] = False
