use std::sync::Arc;

use crate::errors::Invalid;
use crate::errors::ProviderError;
use crate::llm::fallback::ChainObserver;
use crate::tokenize::language::LanguageCode;
use crate::utils::stream::{StreamInput, StreamOutput};
use crate::vad::types::Vad;
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

    /// `stt` in order; a recognize-only one streams over `vad` (`stt.StreamAdapter`).
    #[host(new)]
    fn new(
        env: &Self::Env,
        stt: Vec<Arc<dyn Stt>>,
        #[host(kw, default = None)] vad: Option<Arc<dyn Vad>>,
        #[host(default = 10.0)] attempt_timeout: f64,
        #[host(default = 1)] max_retry_per_stt: u32,
        #[host(default = 5.0)] retry_interval: f64,
        #[host(default = None)] on_availability: Option<Arc<dyn ChainObserver>>,
    ) -> Result<Self, Invalid>;

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
        #[host(kw, default = None)] language: Option<LanguageCode>,
        #[host(default = None)] start_time_offset: Option<f64>,
        #[host(default = None)] conn_options: Option<ApiConnectOptions>,
    ) -> Result<crate::DuplexStream<Self::Audio, Self::Events>, ProviderError>;
}

/// `stt.FallbackAdapter`: several STT behind one; a failed one is skipped until it recovers.
#[crate::host(facade(
    path = "livekit.agents.stt",
    extended,
    name = "FallbackAdapter",
    base = "stt.STT",
    native = FallbackStt,
    legs(stream, recognize, _recognize_impl),
    base_args(capabilities),
    availability(
        observer = "on_availability",
        members = "stt",
        kind = "STT",
        event = "stt_availability_changed",
        record = ".stt_availability.AvailabilityChangedEvent"
    )
))]
pub trait FallbackSttAdapter {
    /// Several STT behind one: a failed one is skipped until it recovers.
    ///
    /// Args:
    ///     stt: The STT, in order of preference.
    ///     vad: Wraps the non-streaming ones in a ``StreamAdapter``; required when any is.
    ///     attempt_timeout: The timeout of one attempt.
    ///     max_retry_per_stt: Retries of each before moving to the next.
    ///     retry_interval: The wait between retries.
    #[host(new, resolve(stt = ".._bindings.core_legs", vad = ".._bindings.core_leg"))]
    fn new(
        stt: Vec<Arc<dyn Stt>>,
        #[host(kw, default = None)] vad: Option<Arc<dyn Vad>>,
        #[host(default = 10.0)] attempt_timeout: f64,
        #[host(default = 1)] max_retry_per_stt: u32,
        #[host(default = 5.0)] retry_interval: f64,
    ) -> Self;
}

/// `stt.FallbackAdapter.stream()`: the chain's recognition, which reconnects and falls back.
#[crate::host(facade(
    path = "livekit.agents.stt",
    base = "stt.RecognizeStream",
    native = FallbackStt,
    stream(
        op = stream,
        owner = "_stt",
        native_retries,
        method = "stream",
        base_args(conn_options),
        state(start_time_offset),
        default(conn_options = "..types.DEFAULT_API_CONNECT_OPTIONS"),
        annotation(language = "str"),
        not_given
    )
))]
pub trait FallbackRecognizeStream {}
