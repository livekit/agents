use std::sync::Arc;

use crate::tts::stream_pacer::HostSentenceStream;
use crate::tts::types::Tts;

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

/// `tts.StreamAdapter`'s engine: a non-streaming TTS as a streaming one, each chunk of its input
/// synthesized on its own.
#[crate::host(class(name = "TtsStreamAdapter", native = Tts))]
pub trait TtsStreamAdapter: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    /// `tts` synthesizes each chunk of the text pushed.
    #[host(init)]
    fn new(
        env: &Self::Env,
        #[host(label)] label: String,
        events: Arc<dyn crate::types::provider::ProviderEvents>,
        #[host(kw)] tts: Arc<dyn Tts>,
    ) -> Self;

    /// Streaming, with word alignment.
    #[host(getter)]
    fn capabilities(&self) -> crate::TtsCapabilities;
    #[host(getter)]
    fn sample_rate(&self) -> Result<u32, crate::errors::HostError>;
    #[host(getter)]
    fn num_channels(&self) -> u16;
}

/// `tts.StreamAdapter`: a non-streaming TTS as a streaming one.
#[crate::host(facade(
    path = "livekit.agents.tts",
    extended,
    name = "StreamAdapter",
    base = "tts.TTS",
    native = TtsStreamAdapter,
    refuse(_synthesize, _stream),
    node = "session.tts_node",
    base_args(capabilities, sample_rate, num_channels),
    keep(tts)
))]
pub trait TtsStreamAdapterFacade {
    /// A non-streaming TTS as a streaming one: each chunk of the text, as its stage splits it,
    /// is synthesized on its own.
    ///
    /// Args:
    ///     tts: The non-streaming TTS.
    #[host(new)]
    fn new(#[host(kw)] tts: Arc<dyn Tts>) -> Self;
}
