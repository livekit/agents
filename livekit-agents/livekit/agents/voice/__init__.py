from ..tts._provider_format import (
    CONVERSATIONAL_EXPRESSIVE_PRESET,
    CUSTOMER_SERVICE_EXPRESSIVE_PRESET,
    HEALTHCARE_EXPRESSIVE_PRESET,
)
from . import io, run_result
from .agent import Agent, AgentTask, ModelSettings
from .agent_session import (
    AgentSession,
    ExpressiveOptions,
    RecordingOptions,
    VoiceActivityVideoSampler,
)
from .audio_recognition import AudioRecognition
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
    SessionUsageUpdatedEvent,
    SpeechCreatedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
    UserTurnExceededEvent,
)
from .remote_session import RemoteSession
from .room_io import (
    _ParticipantAudioOutput,
    _ParticipantStreamTranscriptionOutput,
    _ParticipantTranscriptionOutput,
)
from .speech_handle import SpeechHandle
from .transcription import TranscriptSynchronizer, text_transforms

__all__ = [
    "AgentSession",
    "RecordingOptions",
    "VoiceActivityVideoSampler",
    "Agent",
    "ModelSettings",
    "ExpressiveOptions",
    "CUSTOMER_SERVICE_EXPRESSIVE_PRESET",
    "HEALTHCARE_EXPRESSIVE_PRESET",
    "CONVERSATIONAL_EXPRESSIVE_PRESET",
    "AgentTask",
    "SpeechHandle",
    "RunContext",
    "UserInputTranscribedEvent",
    "AgentEvent",
    "MetricsCollectedEvent",
    "SessionUsageUpdatedEvent",
    "ConversationItemAddedEvent",
    "SpeechCreatedEvent",
    "ErrorEvent",
    "CloseEvent",
    "CloseReason",
    "UserStateChangedEvent",
    "AgentStateChangedEvent",
    "FunctionToolsExecutedEvent",
    "AgentFalseInterruptionEvent",
    "RemoteSession",
    "UserTurnExceededEvent",
    "TranscriptSynchronizer",
    "io",
    "room_io",
    "run_result",
    "_ParticipantAudioOutput",
    "_ParticipantTranscriptionOutput",
    "_ParticipantStreamTranscriptionOutput",
    "text_transforms",
    "AudioRecognition",
]

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
