//! Wiring native synthesis to the runtime services and voice-owned speech transformations.

use lk_agents_abi::tts::stream_adapter::{SentenceSplitter, SentenceTextProcessor};
use lk_agents_abi::tts::text::{TranscriptProcessor, TtsTextProcessorFactory};
use lk_agents_abi::{TtsSink, TtsSource};
use lk_agents_voice::tts::processing::SpeechText;

use super::stream::TtsStream;

pub type NativeTts = lk_agents_inference::tts::NativeTts<TtsHost>;
pub type FallbackTts = lk_agents_inference::fallback::FallbackTts<TtsHost>;

pub type TtsHost = crate::process::Process;

impl TtsTextProcessorFactory for TtsHost {
    fn sentence_splitter(provider: Option<&str>, expressive: bool) -> Box<dyn SentenceSplitter> {
        SpeechText::sentence_splitter(provider, expressive)
    }
    fn markup(provider: &str) -> Option<Box<dyn SentenceTextProcessor>> {
        SpeechText::markup(provider)
    }
    fn transcript(expressive: bool) -> Box<dyn TranscriptProcessor> {
        SpeechText::transcript(expressive)
    }
}

impl lk_agents_inference::tts::TtsHost for TtsHost {
    type Stream = TtsStream;
    fn stream(sink: Option<TtsSink>, source: TtsSource) -> Self::Stream {
        TtsStream::new(sink, source)
    }
}

impl lk_agents_inference::evals::JudgeEnv for TtsHost {
    fn strip_assistant_markup(text: &str) -> String {
        lk_agents_voice::tts::markup::strip_expr_markup(text)
    }
}
