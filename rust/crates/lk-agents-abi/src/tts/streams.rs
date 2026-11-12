use crate::SynthesizedAudio;
use crate::errors::{ProviderError, PullError};

/// One synthesis, `SynthesizeStream`'s shape: text in through `push_text`, `flush` and
/// `end_input`, audio out through `recv`. `aclose` cancels it; it does not end the input.
#[crate::host(class(name = "TtsStream"))]
pub trait TtsStream: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    /// A synthesis by `tts` of `text`, or, without it, of the text pushed. Its text goes through
    /// the TTS's stage (split, paced, lowered to its syntax) and its attempts are retried within
    /// `conn_options`; the first opens when its audio is first read.
    #[host(new)]
    fn new(
        env: &Self::Env,
        #[host(kw)] tts: std::sync::Arc<dyn crate::Tts>,
        #[host(kw, default = None)] text: Option<String>,
        #[host(kw, default = None)] conn_options: Option<crate::ApiConnectOptions>,
    ) -> Self;

    #[host(call)]
    fn push_text(&self, text: String) -> Result<(), ProviderError>;
    #[host(call)]
    fn flush(&self) -> Result<(), ProviderError>;
    #[host(call)]
    fn end_input(&self) -> Result<(), ProviderError>;
    #[host(await)]
    async fn recv(&self) -> Result<Option<SynthesizedAudio>, PullError>;
    /// Every audio it synthesizes into `emitter`, until it ends: its frames, the words they
    /// cover and the usage reported, in one segment when `stream`.
    #[host(await)]
    async fn drain_into(
        &self,
        emitter: std::sync::Arc<dyn crate::tts::stream_pacer::PushedAudio>,
        #[host(kw)] stream: bool,
    ) -> Result<(), DrainError>;
    #[host(await)]
    async fn aclose(&self);
}

/// Why a synthesis stopped draining into its emitter: its own failure, or the emitter's.
#[derive(Debug, thiserror::Error)]
#[crate::host(error(kind = "RuntimeError"))]
pub enum DrainError {
    #[error(transparent)]
    #[host(delegate)]
    Pull(#[from] PullError),
    #[error(transparent)]
    #[host(delegate)]
    Emitter(#[from] crate::media::emitter::EmitterError),
}

/// A streamed synthesis's segments: the flushes asked for (`expected`) and the segments pushed.
#[crate::host(record(module = "tts", dataclass, extract))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentCount {
    pub expected: usize,
    pub actual: usize,
}

/// The rules a host synthesis applies: its pacing until set, and the checks at its end.
#[crate::host(functions(pid = false))]
pub trait SynthesisChecks {
    /// ``TTSOptions.pacing`` until it is set: a ``SentenceStreamPacer`` for a TTS without
    /// streaming, ``None`` for a streaming one.
    #[host(function)]
    fn default_tts_pacing(
        capabilities: crate::TtsCapabilities,
    ) -> Option<crate::tts::stream_pacer::SentenceStreamPacer>;

    /// Why a synthesis of ``text`` that pushed ``duration`` seconds of audio (and, streamed,
    /// ``segments``) failed, or ``None``.
    #[host(function)]
    fn validate_synthesis(
        text: &str,
        duration: f64,
        segments: Option<SegmentCount>,
    ) -> Option<String>;
}
