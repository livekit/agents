//! Supplied speech transformations shared by native provider execution.

use super::stream_adapter::{SentenceSplitter, SentenceTextProcessor};
use crate::TimedString;

pub trait TranscriptProcessor: Send {
    fn push_alignment(&mut self, segment_id: String, tokens: Vec<TimedString>);
    fn take_audio_transcript(&mut self, segment_id: &str, is_final: bool) -> Vec<TimedString>;
}

/// The speech text processing a synthesis applies: the voice core's in a host process, lent by
/// the runtime without giving providers an inverse crate edge.
#[derive(Clone, Copy)]
pub struct SpeechTextProcessors {
    /// `None` selects the sentence adapter's default; a provider selects its gateway limits.
    pub sentence_splitter:
        fn(provider: Option<&str>, expressive: bool) -> Box<dyn SentenceSplitter>,
    pub markup: fn(provider: &str) -> Option<Box<dyn SentenceTextProcessor>>,
    pub transcript: fn(expressive: bool) -> Box<dyn TranscriptProcessor>,
}

impl SpeechTextProcessors {
    /// Text as given: one sentence per push, no markup and no aligned transcript.
    pub const VERBATIM: Self = Self {
        sentence_splitter: |_, _| Box::new(Verbatim),
        markup: |_| None,
        transcript: |_| Box::new(Verbatim),
    };
}

struct Verbatim;

impl SentenceSplitter for Verbatim {
    fn push(&mut self, text: &str) -> Vec<String> {
        vec![text.to_owned()]
    }

    fn flush(&mut self) -> Vec<String> {
        Vec::new()
    }
}

impl TranscriptProcessor for Verbatim {
    fn push_alignment(&mut self, _: String, _: Vec<TimedString>) {}

    fn take_audio_transcript(&mut self, _: &str, _: bool) -> Vec<TimedString> {
        Vec::new()
    }
}
