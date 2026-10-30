//! The STT pipeline events (`stt/stt.py`). A `///` on a Python-targeted declaration is the
//! reference's docstring verbatim; notes for a Rust reader are `//`.

use crate::tokenize::language::LanguageCode;

use crate::llm::chat_context::Extra;
use crate::tts::timed::TimedString;

// `stt/stt.py`, a `(str, Enum)`: users write `ev.type == "final_transcript"`, so the wire strings
// are public API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, lk_agents_host_derive::Vocabulary)]
#[crate::host(record(module = "stt", path = "livekit.agents.stt", enum, extract, emit))]
#[crate::host(wire_strs(
    err = crate::llm::chat_context::UnknownLiteral,
    msg = crate::llm::chat_context::UnknownLiteral { kind: "SpeechEventType", value: s.to_owned() },
))]
pub enum SpeechEventType {
    /// indicate the start of speech
    /// if the STT doesn't support this event, this will be emitted as the same time as the first INTERIM_TRANSCRIPT
    #[host(rename = "start_of_speech")]
    StartOfSpeech,
    /// interim transcript, useful for real-time transcription
    #[host(rename = "interim_transcript")]
    InterimTranscript,
    /// preflight transcript, emitted when the STT is confident enough that a certain
    /// portion of speech will not change. This is different from final transcript in that
    /// the same transcript may still be updated; but it is stable enough to be used for
    /// preemptive generation
    #[host(rename = "preflight_transcript")]
    PreflightTranscript,
    /// final transcript, emitted when the STT is confident enough that a certain
    /// portion of speech will not change
    #[host(rename = "final_transcript")]
    FinalTranscript,
    /// usage event, emitted periodically to indicate usage metrics
    #[host(rename = "recognition_usage")]
    RecognitionUsage,
    /// indicate the end of speech, emitted when the user stops speaking
    #[host(rename = "end_of_speech")]
    EndOfSpeech,
}

// `stt/stt.py:57`. `None` and `[]` are kept apart on the `source_*`/`target_*` lists. The
// `LanguageCode` coercion in `__post_init__` stays host code.
#[crate::host(record(
    module = "stt",
    path = "livekit.agents.stt",
    dataclass,
    post_init = "speech_data_post_init",
    extract,
    emit,
))]
#[derive(Debug, Clone, PartialEq)]
pub struct SpeechData {
    // The target when translating; normalised through `LanguageCode::new` in both directions.
    #[host(annotation = ".language.LanguageCode")]
    pub language: LanguageCode,
    pub text: String,
    #[host(default = 0.0)]
    pub start_time: f64,
    #[host(default = 0.0)]
    pub end_time: f64,
    #[host(default = 0.0)]
    pub confidence: f64,
    #[host(default = None)]
    pub speaker_id: Option<String>,
    // `None` vs `[]` matters to `TranscriptSynchronizer`.
    #[host(default = None)]
    pub words: Option<Vec<TimedString>>,
    /// the source languages spoken by the user. populated by STT services that support translation,
    /// where `language` holds the target language and `source_languages` holds the original spoken language(s),
    /// or by multi-language detection services where `language` holds the dominant language and
    /// `source_languages` holds all detected languages sorted by prevalence.
    /// may contain multiple entries when a single utterance spans multiple source languages.
    #[host(annotation = "list[.language.LanguageCode] | None", default = None)]
    pub source_languages: Option<Vec<LanguageCode>>,
    /// the original transcription segments in the source language(s), when translation is active.
    /// each entry corresponds to the same-indexed entry in `source_languages`.
    #[host(default = None)]
    pub source_texts: Option<Vec<String>>,
    /// the target language(s) produced by a translation-capable STT service, one entry per
    /// consecutive same-language run, parallel to `target_texts`. mirrors `source_languages` /
    /// `source_texts` on the source side: `language` holds the dominant (first) target language
    /// and `target_languages` carries the fine-grained per-run breakdown.
    /// populated when translation is active; None otherwise.
    #[host(annotation = "list[.language.LanguageCode] | None", default = None)]
    pub target_languages: Option<Vec<LanguageCode>>,
    /// the translated transcription segments in the target language(s).
    /// each entry corresponds to the same-indexed entry in `target_languages`.
    #[host(default = None)]
    pub target_texts: Option<Vec<String>>,
    /// optional plugin-specific metadata (e.g. voice profile, provider diagnostics).
    /// plugins may populate this with provider-specific data that doesn't map to standard fields.
    // JSON, so the telemetry path can read it.
    #[host(default = None)]
    pub metadata: Option<Extra>,
}

impl SpeechData {
    pub fn new(language: &str, text: impl Into<String>) -> Self {
        SpeechData {
            language: LanguageCode::new(language),
            text: text.into(),
            start_time: 0.0,
            end_time: 0.0,
            confidence: 0.0,
            speaker_id: None,
            words: None,
            source_languages: None,
            source_texts: None,
            target_languages: None,
            target_texts: None,
            metadata: None,
        }
    }
}

// `stt/stt.py:106`.
#[crate::host(record(module = "stt", path = "livekit.agents.stt", dataclass, extract, emit))]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RecognitionUsage {
    /// Incremental audio duration/usage in seconds
    pub audio_duration: f64,
    #[host(default = 0)]
    pub input_tokens: u32,
    #[host(default = 0)]
    pub output_tokens: u32,
}

// `stt/stt.py:114`: one struct for all six kinds, as in the reference.
#[crate::host(record(module = "stt", path = "livekit.agents.stt", dataclass, extract, emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct SpeechEvent {
    #[host(rename = "type")]
    pub kind: SpeechEventType,
    #[host(default = "")]
    pub request_id: String,
    #[host(default_factory = "list")]
    pub alternatives: Vec<SpeechData>,
    #[host(default = None)]
    pub recognition_usage: Option<RecognitionUsage>,
    /// server-reported wall-clock time of speech onset, when the provider sends
    /// a separate speech-start signal carrying onset timing.
    #[host(default = None)]
    pub speech_start_time: Option<f64>,
    /// Wall-clock time when this event was created.
    // `None` for an event the core minted: the host stamps it as it reads it.
    #[host(annotation = "float", default_factory = "time.time")]
    pub created_at: Option<f64>,
    /// Wall-clock time when the recognized speech ended, when known.
    #[host(default = None)]
    pub speech_end_time: Option<f64>,
}

impl SpeechEvent {
    pub fn new(kind: SpeechEventType) -> Self {
        SpeechEvent {
            kind,
            request_id: String::new(),
            alternatives: Vec::new(),
            recognition_usage: None,
            speech_start_time: None,
            created_at: None,
            speech_end_time: None,
        }
    }

    /// The only alternative the pipeline reads.
    pub fn best_alternative(&self) -> Option<&SpeechData> {
        self.alternatives.first()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: dropping the `__post_init__` language normalisation ("eng", "english" -> "en").
    #[test]
    fn speech_data_normalises_the_language_on_construction() {
        assert_eq!(SpeechData::new("English", "hi").language, "en");
        assert_eq!(SpeechData::new("eng", "hi").language, "en");
        assert_eq!(SpeechData::new("en_us", "hi").language, "en-US");
    }
}
