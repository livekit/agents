# GENERATED FILE -- DO NOT EDIT.
# From ABI declarations and authored plugin bases:
#     rust/crates/lk-agents-abi/src/llm/realtime.rs

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

# Record declarations:
#     rust/crates/lk-agents-abi/src/llm/realtime_options.rs


RealtimeProvider: TypeAlias = Literal["openai", "xai", "inference", "voice_live"]
"""The realtime provider selected at configuration time, including gateway model prefixes."""


RealtimeModels: TypeAlias = Literal["gpt-realtime", "gpt-realtime-1.5", "gpt-realtime-2", "gpt-realtime-2025-08-28", "gpt-4o-realtime-preview"]
"""The OpenAI realtime models."""


@dataclass(kw_only=True, slots=True)
class AudioTranscription:
    """Transcription of the caller's audio."""

    model: str | None = None
    """The transcription model, e.g. ``"gpt-4o-mini-transcribe"``, ``"whisper-1"``, or Voice
    Live's ``"azure-speech"``.
    """
    language: str | None = None
    """The input language (ISO-639-1, or Voice Live's ``"en-US,fr-FR"``)."""
    prompt: str | None = None
    """Text guiding the transcription's style or vocabulary."""


NoiseReductionType: TypeAlias = Literal["near_field", "far_field"]
"""OpenAI's input noise reduction: a close-talking or a far-field microphone."""


@dataclass(kw_only=True, slots=True)
class AzureDeepNoiseSuppression:
    """Azure Voice Live's deep noise suppression of the caller's audio."""


@dataclass(kw_only=True, slots=True)
class ServerVad:
    """Server turn detection on the input's volume."""

    threshold: float | None = None
    """The activation threshold, 0.0 to 1.0; higher needs louder audio."""
    prefix_padding_ms: int | None = None
    """Audio kept before the detected speech, in milliseconds."""
    silence_duration_ms: int | None = None
    """The silence that ends speech, in milliseconds."""
    idle_timeout_ms: int | None = None
    """The silence after which the model replies on its own, in milliseconds."""
    create_response: bool | None = None
    """Reply when speech ends; the server's default is on."""
    interrupt_response: bool | None = None
    """Cut the reply when speech starts; the server's default is on."""


Eagerness: TypeAlias = Literal["low", "medium", "high", "auto"]
"""How readily semantic turn detection ends the caller's turn."""


@dataclass(kw_only=True, slots=True)
class SemanticVad:
    """Server turn detection on what the caller says."""

    eagerness: Eagerness | None = None
    """``"auto"`` (the server's default) is ``"medium"``."""
    create_response: bool | None = None
    interrupt_response: bool | None = None


AzureSemanticVadType: TypeAlias = Literal["azure_semantic_vad", "azure_semantic_vad_en", "azure_semantic_vad_multilingual"]
"""Azure Voice Live's semantic turn detection models."""


@dataclass(kw_only=True, slots=True)
class AzureSemanticVad:
    """Azure Voice Live's semantic turn detection."""

    type: AzureSemanticVadType = "azure_semantic_vad"
    threshold: float | None = None
    prefix_padding_ms: int | None = None
    silence_duration_ms: int | None = None
    remove_filler_words: bool | None = None
    """Ignore fillers ("um", "uh") when deciding the turn ended."""
    languages: list[str] | None = None
    """The languages ``azure_semantic_vad_multilingual`` listens for."""
    create_response: bool | None = None
    interrupt_response: bool | None = None


@dataclass(kw_only=True, slots=True)
class CustomVoice:
    """An OpenAI custom voice."""

    id: str


AzureVoiceType: TypeAlias = Literal["azure-standard", "azure-custom"]


@dataclass(kw_only=True, slots=True)
class AzureVoice:
    """An Azure Voice Live voice."""

    name: str
    """The voice, e.g. ``"en-US-AvaMultilingualNeural"``."""
    type: AzureVoiceType = "azure-standard"
    locale: str | None = None
    """The one language it speaks, e.g. ``"en-US"``."""
    endpoint_id: str | None = None
    """A custom voice's deployment."""
    temperature: float | None = None


@dataclass(kw_only=True, slots=True)
class Tracing:
    """OpenAI's traces of the session. Nothing set is the server's defaults (``"auto"``)."""

    workflow_name: str | None = None
    """The trace's name in the Traces Dashboard."""
    group_id: str | None = None
    metadata: object | None = None


TruncationStrategy: TypeAlias = Literal["auto", "disabled"]
"""How the server drops conversation over the input token limit: ``"auto"``, or never."""


@dataclass(kw_only=True, slots=True)
class RetentionRatio:
    """Truncation that keeps a fraction of the conversation, amortizing truncations over turns."""

    retention_ratio: float
    """The fraction of the post-instruction conversation kept, 0.0 to 1.0."""
    post_instructions_token_limit: int | None = None
    """The conversation tokens allowed after the instructions; the model's limit by default."""


ReasoningEffort: TypeAlias = Literal["minimal", "low", "medium", "high", "xhigh"]


@dataclass(kw_only=True, slots=True)
class Reasoning:
    """Reasoning of a reasoning model such as ``gpt-realtime-2``."""

    effort: ReasoningEffort | None = None


@dataclass(kw_only=True, slots=True)
class InferenceGateway:
    """The LiveKit Inference gateway's routing of a realtime session, set by `inference.RealtimeModel`."""

    api_secret: str | None = None
    provider: str | None = None
    """The provider the gateway routes to."""
    inference_class: str | None = None


NoiseReduction: TypeAlias = NoiseReductionType | AzureDeepNoiseSuppression
"""Input noise reduction: OpenAI's by microphone type, or Voice Live's."""


TurnDetection: TypeAlias = ServerVad | SemanticVad | AzureSemanticVad
"""Server turn detection."""


Voice: TypeAlias = str | CustomVoice | AzureVoice
"""The voice of the model's audio: a name, or a voice object."""


Truncation: TypeAlias = TruncationStrategy | RetentionRatio
"""Conversation truncation over the input token limit."""


AudioTranscription.__module__ = "livekit.agents.llm._realtime"
AzureDeepNoiseSuppression.__module__ = "livekit.agents.llm._realtime"
ServerVad.__module__ = "livekit.agents.llm._realtime"
SemanticVad.__module__ = "livekit.agents.llm._realtime"
AzureSemanticVad.__module__ = "livekit.agents.llm._realtime"
CustomVoice.__module__ = "livekit.agents.llm._realtime"
AzureVoice.__module__ = "livekit.agents.llm._realtime"
Tracing.__module__ = "livekit.agents.llm._realtime"
RetentionRatio.__module__ = "livekit.agents.llm._realtime"
Reasoning.__module__ = "livekit.agents.llm._realtime"


SAMPLE_RATE: Final = 24000
"""The OpenAI Realtime protocol's audio rate, in and out."""
NUM_CHANNELS: Final = 1
OPENAI_BASE_URL: Final = "https://api.openai.com/v1"
"""The OpenAI API's base URL, the realtime model's default."""
DEFAULT_VOICE: Final = "marin"
"""The OpenAI realtime model's default voice."""
