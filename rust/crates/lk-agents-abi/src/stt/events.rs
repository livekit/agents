//! The STT pipeline events (`stt/stt.py`). A `///` on a Python-targeted declaration is the
//! reference's docstring verbatim; notes for a Rust reader are `//`.

use crate::tokenize::language::LanguageCode;

use crate::llm::chat_context::Extra;
use crate::tts::timed::TimedString;
use lk_agents_host_derive::Host;

// `stt/stt.py`, a `(str, Enum)`: users write `ev.type == "final_transcript"`, so the wire strings
// are public API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Host)]
#[python(enum, module = "stt", path = "livekit.agents.stt.stt", name = "SpeechEventType")]
#[node(enum)]
#[host(wire(
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
#[derive(Debug, Clone, PartialEq, Host)]
#[python(
    dataclass,
    zero_defaults,
    module = "stt",
    path = "livekit.agents.stt.stt",
    name = "SpeechData",
    post_init = "speech_data_post_init"
)]
#[node(interface)]
#[binding(emit, extract)]
pub struct SpeechData {
    // The target when translating; normalised through `LanguageCode::new` in both directions.
    #[python(ty = "LanguageCode", import = "..language:LanguageCode")]
    #[node(ty = "string")]
    pub language: LanguageCode,
    pub text: String,
    pub start_time: f64,
    pub end_time: f64,
    pub confidence: f64,
    pub speaker_id: Option<String>,
    pub is_primary_speaker: Option<bool>,
    // `None` vs `[]` matters to `TranscriptSynchronizer`.
    #[python(ty = "list[TimedString] | None", import = "..types:TimedString", default = "None")]
    #[node(ty = "string[] | null")]
    pub words: Option<Vec<TimedString>>,
    /// the source languages spoken by the user. populated by STT services that support translation,
    /// where `language` holds the target language and `source_languages` holds the original spoken language(s),
    /// or by multi-language detection services where `language` holds the dominant language and
    /// `source_languages` holds all detected languages sorted by prevalence.
    /// may contain multiple entries when a single utterance spans multiple source languages.
    #[python(ty = "list[LanguageCode] | None", import = "..language:LanguageCode", default = "None")]
    #[node(ty = "string[] | null")]
    pub source_languages: Option<Vec<LanguageCode>>,
    /// the original transcription segments in the source language(s), when translation is active.
    /// each entry corresponds to the same-indexed entry in `source_languages`.
    pub source_texts: Option<Vec<String>>,
    /// the target language(s) produced by a translation-capable STT service, one entry per
    /// consecutive same-language run, parallel to `target_texts`. mirrors `source_languages` /
    /// `source_texts` on the source side: `language` holds the dominant (first) target language
    /// and `target_languages` carries the fine-grained per-run breakdown.
    /// populated when translation is active; None otherwise.
    #[python(ty = "list[LanguageCode] | None", import = "..language:LanguageCode", default = "None")]
    #[node(ty = "string[] | null")]
    pub target_languages: Option<Vec<LanguageCode>>,
    /// the translated transcription segments in the target language(s).
    /// each entry corresponds to the same-indexed entry in `target_languages`.
    pub target_texts: Option<Vec<String>>,
    /// optional plugin-specific metadata (e.g. voice profile, provider diagnostics).
    /// plugins may populate this with provider-specific data that doesn't map to standard fields.
    // JSON, so the telemetry path can read it.
    #[python(ty = "dict[str, Any] | None", import = "typing:Any", default = "None")]
    #[node(ty = "Record<string, unknown> | null")]
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
            is_primary_speaker: None,
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
#[derive(Debug, Clone, Copy, Default, PartialEq, Host)]
#[python(dataclass, module = "stt", path = "livekit.agents.stt.stt", name = "RecognitionUsage")]
#[node(interface)]
#[binding(emit, extract)]
pub struct RecognitionUsage {
    /// Incremental audio duration/usage in seconds
    pub audio_duration: f64,
    #[python(default = "0")]
    pub input_tokens: u32,
    #[python(default = "0")]
    pub output_tokens: u32,
}

// `stt/stt.py:114`: one struct for all six kinds, as in the reference.
#[derive(Debug, Clone, PartialEq, Host)]
#[python(dataclass, zero_defaults, module = "stt", path = "livekit.agents.stt.stt", name = "SpeechEvent")]
#[node(interface)]
#[binding(emit, extract)]
pub struct SpeechEvent {
    #[host(rename = "type")]
    pub kind: SpeechEventType,
    #[python(default = "\"\"")]
    pub request_id: String,
    #[python(default_factory = "list")]
    pub alternatives: Vec<SpeechData>,
    pub recognition_usage: Option<RecognitionUsage>,
    /// server-reported wall-clock time of speech onset, when the provider sends
    /// a separate speech-start signal carrying onset timing.
    pub speech_start_time: Option<f64>,
}

impl SpeechEvent {
    pub fn new(kind: SpeechEventType) -> Self {
        SpeechEvent {
            kind,
            request_id: String::new(),
            alternatives: Vec::new(),
            recognition_usage: None,
            speech_start_time: None,
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

    // Catches: `best_alternative` returning something for an event without a transcript.
    #[test]
    fn an_event_with_no_alternatives_has_no_best_one() {
        let ev = SpeechEvent::new(SpeechEventType::StartOfSpeech);
        assert!(ev.best_alternative().is_none());

        let ev = SpeechEvent {
            alternatives: vec![SpeechData::new("en", "first"), SpeechData::new("en", "second")],
            ..SpeechEvent::new(SpeechEventType::FinalTranscript)
        };
        assert_eq!(ev.best_alternative().map(|a| a.text.as_str()), Some("first"));
    }
}
