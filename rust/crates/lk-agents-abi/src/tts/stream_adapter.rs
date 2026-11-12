use std::sync::Arc;

use crate::errors::ProviderError;
use crate::tts::stream_pacer::{HostSentenceStream, SentenceStreamPacer};
use crate::tts::types::Tts;

/// No default implementation on purpose: segmentation boundaries are audible, and the reference's
/// default is `tokenize.SentenceTokenizer`'s, not a hand-rolled punctuation splitter.
pub trait SentenceSplitter: Send {
    /// Partial text stays buffered.
    fn push(&mut self, text: &str) -> Vec<String>;
    /// Hard boundary: whatever is buffered, complete or not.
    fn flush(&mut self) -> Vec<String>;
}

/// Called once per stream with whether it is expressive: markup is kept whole only while it
/// flows, so a marker is never split across two sentences.
pub type SplitterFactory = Arc<dyn Fn(bool) -> Box<dyn SentenceSplitter> + Send + Sync>;

/// Provider text transformations supplied by the owning speech core.
pub trait SentenceTextProcessor: Send {
    fn normalize(&self, text: &str) -> String;
    fn convert(&self, text: &str) -> String;
}

/// Select a supported provider dialect for one synthesis, or leave text unchanged.
pub type TextProcessorFactory =
    Arc<dyn Fn(&str) -> Option<Box<dyn SentenceTextProcessor>> + Send + Sync>;

/// A host `tokenize.SentenceTokenizer`: each synthesis splits its text by a fresh stream of it,
/// opened on the host loop.
#[crate::host(proxy(host = "livekit.agents.tokenize.SentenceTokenizer"))]
pub trait HostSentenceTokenizer: Send + Sync {
    #[host(await)]
    fn stream(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<Arc<dyn HostSentenceStream>, crate::errors::HostError>>;
}

/// `StreamAdapter(text_pacing=)`: a pacer, or `True` for the default one.
#[crate::host(record(module = "stream_pacer", name = "TextPacing", extract, transparent))]
#[derive(Debug, Clone, Copy)]
pub enum TextPacing {
    Pacer(SentenceStreamPacer),
    On(bool),
}

/// `tts.StreamAdapter`'s engine: a non-streaming TTS as a streaming one, the text split into
/// sentences each synthesized on its own.
#[crate::host(class(name = "TtsStreamAdapter", native = Tts))]
pub trait TtsStreamAdapter: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(class(name = "TtsStream"))]
    type Stream;

    /// `tts` synthesizes each sentence of the text pushed; `sentence_tokenizer` splits it
    /// (`tokenize.SentenceTokenizer`'s by default); `text_pacing` releases sentences as the audio plays out.
    #[host(new)]
    fn new(
        env: &Self::Env,
        #[host(kw)] tts: Arc<dyn Tts>,
        #[host(default = None)] sentence_tokenizer: Option<Arc<dyn HostSentenceTokenizer>>,
        #[host(default = None)] text_pacing: Option<TextPacing>,
    ) -> Self;

    /// Streaming, with word alignment.
    #[host(getter)]
    fn capabilities(&self) -> crate::TtsCapabilities;
    #[host(getter)]
    fn sample_rate(&self) -> u32;
    #[host(getter)]
    fn num_channels(&self) -> u16;

    /// One segment of the text pushed, each sentence retried within `conn_options`. Under
    /// `expressive` the wrapped TTS's markup is lowered before each sentence.
    #[host(await)]
    async fn stream(
        &self,
        cx: crate::telemetry::Cx,
        #[host(kw, default = None)] conn_options: Option<crate::ApiConnectOptions>,
        #[host(default = false)] expressive: bool,
    ) -> Result<Self::Stream, ProviderError>;
}

/// `tts.StreamAdapter`: a non-streaming TTS as a streaming one.
#[crate::host(facade(
    path = "livekit.agents.tts",
    extended,
    name = "StreamAdapter",
    base = "tts.TTS",
    native = TtsStreamAdapter,
    legs(stream),
    base_args(capabilities, sample_rate, num_channels),
    keep(tts)
))]
pub trait TtsStreamAdapterFacade {
    /// A non-streaming TTS as a streaming one: the text is split into sentences, each
    /// synthesized on its own.
    ///
    /// Args:
    ///     tts: The non-streaming TTS.
    ///     sentence_tokenizer: What splits the text; `tokenize.SentenceTokenizer`'s sentences by default.
    ///     text_pacing: Batches sentences as the audio plays out; ``True`` for the default
    ///         ``SentenceStreamPacer``.
    #[host(new, resolve(tts = ".._bindings.core_leg"))]
    fn new(
        #[host(kw)] tts: Arc<dyn Tts>,
        #[host(default = None)] sentence_tokenizer: Option<Arc<dyn HostSentenceTokenizer>>,
        #[host(default = None)] text_pacing: Option<TextPacing>,
    ) -> Self;
}

/// `tts.StreamAdapter.stream()`: the text split into sentences, paced when ``text_pacing`` is
/// on, each sentence synthesized by the wrapped TTS and retried on its own within
/// ``conn_options``. Its time to first audio is measured from the first text pushed.
#[crate::host(facade(
    path = "livekit.agents.tts",
    base = "tts.SynthesizeStream",
    native = TtsStreamAdapter,
    stream(
        op = stream,
        owner = "_tts",
        native_retries,
        method = "stream",
        base_args(conn_options),
        state(expressive),
        default(conn_options = "..types.DEFAULT_API_CONNECT_OPTIONS")
    )
))]
pub trait StreamAdapterWrapper {}
