//! Supplied speech transformations shared by native provider execution.

use super::stream_adapter::{SentenceSplitter, SentenceTextProcessor};
use crate::TimedString;

pub trait TranscriptProcessor: Send {
    fn push_alignment(&mut self, segment_id: String, tokens: Vec<TimedString>);
    fn take_audio_transcript(&mut self, segment_id: &str, is_final: bool) -> Vec<TimedString>;
}

/// The runtime selects the voice implementation without giving providers an inverse crate edge.
pub trait TtsTextProcessorFactory: Clone + Send + Sync + 'static {
    /// `None` selects the sentence adapter's default; a provider selects its gateway limits.
    fn sentence_splitter(provider: Option<&str>, expressive: bool) -> Box<dyn SentenceSplitter>;
    fn markup(provider: &str) -> Option<Box<dyn SentenceTextProcessor>>;
    fn transcript(expressive: bool) -> Box<dyn TranscriptProcessor>;
}
