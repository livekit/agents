//! `stt.FallbackAdapter`'s chain over the host's STTs, implemented by `lk-agents-inference`.

use std::sync::Arc;

use crate::errors::ProviderError;
use crate::llm::fallback::ChainObserver;
use crate::llm::inference::InvalidPayload;
use crate::tokenize::language::LanguageCode;
use crate::utils::stream::{StreamInput, StreamOutput};
use crate::{ApiConnectOptions, AudioFrame, SpeechEvent, Stt, SttCapabilities};

/// `stt.FallbackAdapter`'s engine as the host builds it; the public class stays Python.
#[crate::host(class(name = "FallbackStt", native = Stt))]
pub trait FallbackStt: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(sink(item = AudioFrame, class = "FallbackRecognizeStream"))]
    type Audio: StreamInput<AudioFrame> + 'static;
    #[host(source(item = SpeechEvent, class = "FallbackRecognizeStream"))]
    type Events: StreamOutput<SpeechEvent> + 'static;

    /// `stts` in order, each already streaming (the host wraps a recognize-only entry in its
    /// `StreamAdapter`); the keywords are `FallbackAdapter.__init__`'s.
    #[host(new)]
    fn new(
        env: &Self::Env,
        stts: Vec<Arc<dyn Stt>>,
        #[host(kw, default = 10.0)] attempt_timeout: f64,
        #[host(default = 1)] max_retry_per_stt: u32,
        #[host(default = 5.0)] retry_interval: f64,
        #[host(default = None)] on_availability: Option<Arc<dyn ChainObserver>>,
    ) -> Result<Self, InvalidPayload>;

    /// Which entry most recently served, as a position in the host's list.
    #[host(getter)]
    fn active_index(&self) -> usize;

    /// The merged capabilities, not the primary's: a caller must not wait for interim results a
    /// secondary never sends.
    #[host(getter)]
    fn capabilities(&self) -> SttCapabilities;

    /// Every recovery probe in flight closes its stream, awaited (bounded).
    #[host(await)]
    async fn aclose(&self);

    /// `_recognize_impl`: the batch path, over whichever entry answers.
    #[host(await)]
    async fn recognize(
        &self,
        frames: Vec<AudioFrame>,
        #[host(default = None)] language: Option<LanguageCode>,
        #[host(default = None)] conn_options: Option<ApiConnectOptions>,
    ) -> Result<SpeechEvent, ProviderError>;

    /// Audio in, tee'd to the serving stream and every recovery probe; events out.
    #[host(await)]
    async fn stream(
        &self,
        #[host(default = None)] language: Option<LanguageCode>,
        #[host(default = None)] start_time_offset: Option<f64>,
        #[host(default = None)] conn_options: Option<ApiConnectOptions>,
    ) -> Result<crate::DuplexStream<Self::Audio, Self::Events>, ProviderError>;
}
