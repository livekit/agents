//! `inference.VAD`'s native detector (Silero): the detection machine and stream execution in
//! `lk-agents-inference`, the model native when the build links one, else the host's.

use std::sync::Arc;

use crate::errors::{HostError, ProviderError};
use crate::utils::stream::{StreamSink, StreamSource};
use crate::vad::vad::Vad;
use crate::{AudioFrame, VadEvent};

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
#[crate::host(class(name = "InferenceVad", native = Vad))]
pub trait InferenceVad: Send + Sync + Sized + 'static {
    #[host(sink(item = AudioFrame, class = "InferenceVadStream"))]
    type Audio: StreamSink<AudioFrame> + 'static;
    #[host(source(item = VadEvent, class = "InferenceVadStream"))]
    type Events: StreamSource<VadEvent> + 'static;

    #[host(new)]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model_factory: Arc<dyn VadModels>,
        #[host(kw)] min_speech_duration: f64,
        min_silence_duration: f64,
        prefix_padding_duration: f64,
        max_buffered_speech: f64,
        activation_threshold: f64,
        deactivation_threshold: f64,
    ) -> Self;

    /// Every option, applied by each running stream before its next frame.
    #[host(call)]
    fn update_options(
        &self,
        #[host(kw)] min_speech_duration: f64,
        min_silence_duration: f64,
        prefix_padding_duration: f64,
        max_buffered_speech: f64,
        activation_threshold: f64,
        deactivation_threshold: f64,
    );

    /// One detection; each event carries the audio it refers to.
    #[host(call)]
    fn stream(&self) -> Result<(Self::Audio, Self::Events), ProviderError>;
}
