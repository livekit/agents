use std::sync::Arc;

use crate::errors::{HostError, Invalid, ProviderError};
use crate::utils::stream::{StreamInput, StreamOutput};
use crate::{AudioFrame, Maybe, VadEvent};

/// The host's model factory (`livekit.local_inference.VAD`): called once per stream, as each
/// stream needs its own recurrent state. Never called when the model is native.
#[crate::host(proxy(host = "collections.abc.Callable"))]
pub trait VadModels: Send + Sync {
    #[host(call, name = "__call__")]
    fn create(&self) -> Result<Arc<dyn VadModel>, HostError>;
}

/// One host model, called from its stream's admitted worker.
#[crate::host(proxy(host = "livekit.local_inference.VAD"))]
pub trait VadModel: Send + Sync {
    /// The speech probability of one window of 16 kHz samples.
    #[host(call)]
    fn predict(&self, window: Vec<i16>) -> Result<f64, HostError>;

    /// Clears the recurrent state at a segment boundary.
    #[host(call)]
    fn reset(&self) -> Result<(), HostError>;
}

/// The Silero VAD behind `inference.VAD`.
#[crate::host(class(name = "InferenceVad", native = crate::vad::types::Vad))]
pub trait InferenceVad: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(sink(item = AudioFrame, class = "InferenceVadStream"))]
    type Audio: StreamInput<AudioFrame> + 'static;
    #[host(source(item = VadEvent, class = "InferenceVadStream"))]
    type Events: StreamOutput<VadEvent> + 'static;

    /// `model_factory` builds the host's model when the build links no native one;
    /// `deactivation_threshold` is `max(activation_threshold - 0.15, 0.01)` when `None`. Only the
    /// `"silero"` model exists.
    #[host(init)]
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors the Python constructor's keyword arguments"
    )]
    fn new(
        env: &Self::Env,
        #[host(label)] label: String,
        model_factory: Arc<dyn VadModels>,
        #[host(kw, default = "silero")] model: crate::inference::models::VadModelName,
        #[host(default = 0.05)] min_speech_duration: f64,
        #[host(default = 0.25)] min_silence_duration: f64,
        #[host(default = 0.5)] prefix_padding_duration: f64,
        #[host(default = 60.0)] max_buffered_speech: f64,
        #[host(default = 0.5)] activation_threshold: f64,
        #[host(default = None)] deactivation_threshold: Option<f64>,
    ) -> Result<Self, Invalid>;

    #[host(getter)]
    fn model(&self) -> crate::inference::models::VadModelName;

    #[host(getter)]
    fn capabilities(&self) -> crate::VadCapabilities;

    /// The given options, applied by each running stream before its next frame.
    #[host(call, name = "_core_update_options")]
    fn update_options(
        &self,
        #[host(kw)] min_speech_duration: Maybe<f64>,
        min_silence_duration: Maybe<f64>,
        prefix_padding_duration: Maybe<f64>,
        max_buffered_speech: Maybe<f64>,
        activation_threshold: Maybe<f64>,
        deactivation_threshold: Maybe<f64>,
    );

    #[host(getter)]
    fn min_silence_duration(&self) -> f64;

    /// One detection; each event carries the audio it refers to.
    #[host(call, name = "_core_native_stream")]
    fn stream(&self) -> Result<crate::DuplexStream<Self::Audio, Self::Events>, ProviderError>;
}

/// `inference.VAD`: Silero voice activity detection, the model native to the core when the build
/// links it, else `livekit-local-inference`'s.
#[crate::host(facade(
    path = "livekit.agents.inference",
    extended,
    name = "VAD",
    base = "vad.VAD",
    native = InferenceVad,
    refuse(stream),
    node = "session.stt_node",
    base_args(capabilities),
    host_args(model_factory = ".._bindings.silero_model")
))]
pub trait InferenceVadFacade {
    /// Silero voice activity detection.
    ///
    /// Args:
    ///     model: The VAD model; only ``"silero"``.
    ///     min_speech_duration: Speech needed before a start of speech.
    ///     min_silence_duration: Silence needed before an end of speech.
    ///     prefix_padding_duration: Audio kept before the start of speech.
    ///     max_buffered_speech: The longest speech kept for the end-of-speech event.
    ///     activation_threshold: The probability above which a window is speech.
    ///     deactivation_threshold: The probability below which speech ends;
    ///         ``activation_threshold - 0.15`` by default.
    #[host(new, not_given)]
    fn new(
        #[host(kw, default = "silero")] model: crate::inference::models::VadModelName,
        #[host(default = 0.05)] min_speech_duration: f64,
        #[host(default = 0.25)] min_silence_duration: f64,
        #[host(default = 0.5)] prefix_padding_duration: f64,
        #[host(default = 60.0)] max_buffered_speech: f64,
        #[host(default = 0.5)] activation_threshold: f64,
        #[host(default = None)] deactivation_threshold: Option<f64>,
    ) -> Self;

    #[host(getter)]
    fn model(&self) -> crate::inference::models::VadModelName;

    #[host(getter)]
    fn min_silence_duration(&self) -> f64;

    /// Update the options; running streams apply them before their next frame.
    #[host(native = _core_update_options)]
    fn update_options(
        &self,
        #[host(kw)] min_speech_duration: Maybe<f64>,
        min_silence_duration: Maybe<f64>,
        prefix_padding_duration: Maybe<f64>,
        max_buffered_speech: Maybe<f64>,
        activation_threshold: Maybe<f64>,
        deactivation_threshold: Maybe<f64>,
    );
}

/// `inference.VAD.stream()`: the native detection, frames and flushes in, its events out.
#[crate::host(facade(
    path = "livekit.agents.inference",
    name = "_VADStream",
    base = "vad.VADStream",
    native = InferenceVad,
    stream(op = stream, owner = "_vad", method = "stream")
))]
pub trait InferenceVadStream {}
