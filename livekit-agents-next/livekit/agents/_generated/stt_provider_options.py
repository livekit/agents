# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/stt/inference.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from typing import Any, Literal


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
    voice_focus: Literal['near-field', 'far-field']
    voice_focus_threshold: float
    mode: Literal['min_latency', 'balanced', 'max_accuracy']


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
    audio_encoding: Literal['LINEAR16', 'AUTO_DETECT']
    inactivity_timeout_seconds: int
    end_of_turn_confidence_threshold: float
    min_end_of_turn_silence_when_confident: int
    prompts: list[str]
    vad_threshold: float


class GoogleOptions(TypedDict, total=False):
    language_codes: list[str]
    custom_vocabulary: list[str]


CartesiaOptions.__module__ = "livekit.agents.inference.stt"
DeepgramOptions.__module__ = "livekit.agents.inference.stt"
DeepgramFluxOptions.__module__ = "livekit.agents.inference.stt"
AssemblyaiOptions.__module__ = "livekit.agents.inference.stt"
SpeechmaticsOptions.__module__ = "livekit.agents.inference.stt"
XaiOptions.__module__ = "livekit.agents.inference.stt"
InworldOptions.__module__ = "livekit.agents.inference.stt"
GoogleOptions.__module__ = "livekit.agents.inference.stt"
