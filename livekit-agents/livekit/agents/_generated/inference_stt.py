# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/inference/fallback.rs
#     rust/crates/lk-agents-abi/src/inference/mod.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import Any, Literal, TypedDict

from typing_extensions import Required


class FallbackModel(TypedDict, total=False):
    """Inference Fallback Adapter: configuration for a fallback STT model that runs server-side in LiveKit Inference, providing automatic fallback between providers.

    Extra fields are passed through to the provider.

    Example:
        >>> FallbackModel(model="deepgram/nova-3", extra_kwargs={"keyterm": ["livekit"]})
    """

    model: Required[str]
    """Model name (e.g. "deepgram/nova-3", "assemblyai/universal-streaming", "cartesia/ink-whisper")."""
    extra_kwargs: dict[str, Any]
    """Extra configuration for the model."""


DeepgramModels = Literal[
    "deepgram/nova-3",
    "deepgram/nova-3-medical",
    "deepgram/nova-2",
    "deepgram/nova-2-medical",
    "deepgram/nova-2-conversationalai",
    "deepgram/nova-2-phonecall",
]


DeepgramFluxModels = Literal[
    "deepgram/flux-general", "deepgram/flux-general-en", "deepgram/flux-general-multi"
]


CartesiaModels = Literal["cartesia/ink-whisper", "cartesia/ink-2"]


AssemblyAIModels = Literal[
    "assemblyai/universal-streaming",
    "assemblyai/universal-streaming-multilingual",
    "assemblyai/u3-rt-pro",
    "assemblyai/universal-3-5-pro",
]


ElevenlabsModels = Literal["elevenlabs/scribe_v2_realtime"]


XaiModels = Literal["xai/stt-1"]


SpeechmaticsModels = Literal["speechmatics/enhanced", "speechmatics/standard"]


InworldModels = Literal["inworld/inworld-stt-1"]


class CartesiaOptions(TypedDict, total=False):
    min_volume: float
    max_silence_duration_secs: float
    turn_start_threshold: float
    turn_eager_end_threshold: float
    turn_end_threshold: float
    turn_end_timeout_ms: int
    keyterm: str | list[str]


class DeepgramOptions(TypedDict, total=False):
    filler_words: bool
    interim_results: bool
    endpointing: int
    punctuate: bool
    smart_format: bool
    keywords: list[tuple[str, float]]
    keyterm: str | list[str]
    profanity_filter: bool
    numerals: bool
    mip_opt_out: bool
    vad_events: bool
    diarize: bool
    dictation: bool
    detect_language: bool
    no_delay: bool
    utterance_end: bool
    redact: str | list[str]
    replace: str | list[str]
    search: str | list[str]
    tag: str | list[str]
    channels: int
    version: str
    callback: str
    callback_method: str
    extra: str


class DeepgramFluxOptions(TypedDict, total=False):
    eager_eot_threshold: float
    eot_threshold: float
    eot_timeout_ms: int
    keyterm: str | list[str]
    mip_opt_out: bool
    tag: str | list[str]
    detect_language: bool


class AssemblyaiOptions(TypedDict, total=False):
    format_turns: bool
    end_of_turn_confidence_threshold: float
    min_end_of_turn_silence_when_confident: int
    max_turn_silence: int
    keyterms_prompt: list[str]
    language_detection: bool
    inactivity_timeout: float
    prompt: str
    speaker_labels: bool
    agent_context: str
    previous_context_n_turns: int
    voice_focus: Literal["near-field", "far-field"]
    voice_focus_threshold: float
    mode: Literal["min_latency", "balanced", "max_accuracy"]


class ElevenlabsOptions(TypedDict, total=False):
    commit_strategy: Literal["manual", "vad"]
    include_timestamps: bool
    vad_silence_threshold_secs: float
    vad_threshold: float
    min_speech_duration_ms: int
    min_silence_duration_ms: int
    language_code: str


class SpeechmaticsOptions(TypedDict, total=False):
    domain: str
    output_locale: str
    max_delay: float
    max_delay_mode: str
    diarization: str
    speaker_sensitivity: float
    max_speakers: int
    prefer_current_speaker: bool
    enable_partials: bool
    enable_entities: bool
    punctuation_overrides: dict[str, Any]
    additional_vocab: list[dict[str, Any]]
    end_of_utterance_silence_trigger: float
    audio_filtering_config: dict[str, Any]
    transcript_filtering_config: dict[str, Any]


class XaiOptions(TypedDict, total=False):
    diarize: bool
    endpointing: int
    format: bool
    interim_results: bool


class InworldOptions(TypedDict, total=False):
    enable_voice_profile: bool
    voice_profile_top_n: int
    include_word_timestamps: bool
    audio_encoding: Literal["LINEAR16", "AUTO_DETECT"]
    inactivity_timeout_seconds: int
    end_of_turn_confidence_threshold: float
    min_end_of_turn_silence_when_confident: int
    prompts: list[str]
    vad_threshold: float


STTLanguages = Literal["multi", "en", "de", "es", "fr", "ja", "pt", "zh", "hi"]


FallbackModelType = FallbackModel | str


STTModels = (
    DeepgramModels
    | DeepgramFluxModels
    | CartesiaModels
    | AssemblyAIModels
    | ElevenlabsModels
    | XaiModels
    | SpeechmaticsModels
    | InworldModels
    | Literal["auto"]
)


STTEncoding = Literal["pcm_s16le"]


# --- the frozen public paths -----------------------------------------------------------------
#
# Assigned here rather than in each class body: pydantic resolves annotations against
# `sys.modules[cls.__module__]` as it builds the class, so setting it early sends it
# looking for `Literal` and every foreign type in the re-export module, which imports
# none of them. What this fixes is `repr`, `pickle` and the equivalence gate all
# reporting `_generated.*` for a name whose import path never moved.
FallbackModel.__module__ = "livekit.agents.inference.stt"
CartesiaOptions.__module__ = "livekit.agents.inference.stt"
DeepgramOptions.__module__ = "livekit.agents.inference.stt"
DeepgramFluxOptions.__module__ = "livekit.agents.inference.stt"
AssemblyaiOptions.__module__ = "livekit.agents.inference.stt"
ElevenlabsOptions.__module__ = "livekit.agents.inference.stt"
SpeechmaticsOptions.__module__ = "livekit.agents.inference.stt"
XaiOptions.__module__ = "livekit.agents.inference.stt"
InworldOptions.__module__ = "livekit.agents.inference.stt"
