//! The sentence adapter's contracts (`tts/stream_adapter.py`): the speech text it is composed
//! with and its pre-split host entry. The adapter itself is `lk_agents_inference`'s.

use std::sync::Arc;

use crate::errors::ProviderError;
use crate::tts::tts::Tts;

/// No default implementation on purpose: segmentation boundaries are audible, and the reference's
/// default is blingfire's, not a hand-rolled punctuation splitter.
pub trait SentenceSplitter: Send {
    /// Partial text stays buffered.
    fn push(&mut self, text: &str) -> Vec<String>;
    /// Hard boundary: whatever is buffered, complete or not.
    fn flush(&mut self) -> Vec<String>;
}

/// Called once per stream with `xml_aware`: on only while markup flows, so a marker is never split
/// across two sentences (`StreamAdapter._tokenizer_for`).
pub type SplitterFactory = Arc<dyn Fn(bool) -> Box<dyn SentenceSplitter> + Send + Sync>;

/// Provider text transformations supplied by the owning speech core.
pub trait SentenceTextProcessor: Send {
    fn normalize(&self, text: &str) -> String;
    fn convert(&self, text: &str) -> String;
}

/// Select a supported provider dialect for one synthesis, or leave text unchanged.
pub type TextProcessorFactory =
    Arc<dyn Fn(&str) -> Option<Box<dyn SentenceTextProcessor>> + Send + Sync>;

/// `tts.StreamAdapter`'s pre-split host entry; runtime supplies the speech algorithms.
#[crate::host(functions)]
pub trait StreamAdapterFunctions {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(class(name = "TtsStream"))]
    type Stream;

    #[host(function)]
    async fn tts_adapter_stream(
        env: &Self::Env,
        provider: Arc<dyn Tts>,
        conn_options: crate::ApiConnectOptions,
    ) -> Result<Self::Stream, ProviderError>;
}
