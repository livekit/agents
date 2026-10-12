# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/stt/stt.rs
#     rust/crates/lk-agents-abi/src/stt/events.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Literal

from .. import _bindings
from ..language import LanguageCode
from ..types import TimedString


@dataclass
class STTCapabilities:
    streaming: bool
    interim_results: bool
    diarization: bool = False
    aligned_transcript: Literal["word", "chunk", False] = False
    offline_recognize: bool = True
    """Whether the STT supports batch recognition via recognize() method"""
    keyterms: bool = False
    """Whether the STT supports keyterm prompting"""
    chat_context: bool = False
    """Whether the STT can natively consume conversation context (see STT._push_conversation_item)"""


@unique
class SpeechEventType(str, Enum):
    START_OF_SPEECH = "start_of_speech"
    """indicate the start of speech
    if the STT doesn't support this event, this will be emitted as the same time as the first INTERIM_TRANSCRIPT
    """
    INTERIM_TRANSCRIPT = "interim_transcript"
    """interim transcript, useful for real-time transcription"""
    PREFLIGHT_TRANSCRIPT = "preflight_transcript"
    """preflight transcript, emitted when the STT is confident enough that a certain
    portion of speech will not change. This is different from final transcript in that
    the same transcript may still be updated; but it is stable enough to be used for
    preemptive generation
    """
    FINAL_TRANSCRIPT = "final_transcript"
    """final transcript, emitted when the STT is confident enough that a certain
    portion of speech will not change
    """
    RECOGNITION_USAGE = "recognition_usage"
    """usage event, emitted periodically to indicate usage metrics"""
    END_OF_SPEECH = "end_of_speech"
    """indicate the end of speech, emitted when the user stops speaking"""


@dataclass
class SpeechData:
    language: LanguageCode
    text: str
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 0.0
    speaker_id: str | None = None
    is_primary_speaker: bool | None = None
    words: list[TimedString] | None = None
    source_languages: list[LanguageCode] | None = None
    """the source languages spoken by the user. populated by STT services that support translation,
    where `language` holds the target language and `source_languages` holds the original spoken language(s),
    or by multi-language detection services where `language` holds the dominant language and
    `source_languages` holds all detected languages sorted by prevalence.
    may contain multiple entries when a single utterance spans multiple source languages.
    """
    source_texts: list[str] | None = None
    """the original transcription segments in the source language(s), when translation is active.
    each entry corresponds to the same-indexed entry in `source_languages`.
    """
    target_languages: list[LanguageCode] | None = None
    """the target language(s) produced by a translation-capable STT service, one entry per
    consecutive same-language run, parallel to `target_texts`. mirrors `source_languages` /
    `source_texts` on the source side: `language` holds the dominant (first) target language
    and `target_languages` carries the fine-grained per-run breakdown.
    populated when translation is active; None otherwise.
    """
    target_texts: list[str] | None = None
    """the translated transcription segments in the target language(s).
    each entry corresponds to the same-indexed entry in `target_languages`.
    """
    metadata: dict[str, Any] | None = None
    """optional plugin-specific metadata (e.g. voice profile, provider diagnostics).
    plugins may populate this with provider-specific data that doesn't map to standard fields.
    """

    def __post_init__(self) -> None:
        _bindings.speech_data_post_init(self)


@dataclass
class RecognitionUsage:
    audio_duration: float
    """Incremental audio duration/usage in seconds"""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class SpeechEvent:
    type: SpeechEventType
    request_id: str = ""
    alternatives: list[SpeechData] = field(default_factory=list)
    recognition_usage: RecognitionUsage | None = None
    speech_start_time: float | None = None
    """server-reported wall-clock time of speech onset, when the provider sends
    a separate speech-start signal carrying onset timing.
    """


# --- the frozen public paths -----------------------------------------------------------------
#
# Assigned here rather than in each class body: pydantic resolves annotations against
# `sys.modules[cls.__module__]` as it builds the class, so setting it early sends it
# looking for `Literal` and every foreign type in the re-export module, which imports
# none of them. What this fixes is `repr`, `pickle` and the equivalence gate all
# reporting `_generated.*` for a name whose import path never moved.
STTCapabilities.__module__ = "livekit.agents.stt.stt"
SpeechEventType.__module__ = "livekit.agents.stt.stt"
SpeechData.__module__ = "livekit.agents.stt.stt"
RecognitionUsage.__module__ = "livekit.agents.stt.stt"
SpeechEvent.__module__ = "livekit.agents.stt.stt"
