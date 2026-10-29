//! `inference`'s native turn-taking models: `TurnDetector`'s engine (the threshold table and the
//! streams it opens, cloud with a fallback to the local mini model, native when the build links
//! one, else the host's) and `AdaptiveInterruptionDetector`.

use std::sync::Arc;

use crate::errors::{HostError, ProviderError};
use crate::inference::InvalidOption;
use crate::metrics::base::EotInferenceMetrics;
use crate::tokenize::language::{LanguageCode, ThresholdOverride, TurnDetectorModel};
use crate::{ApiConnectOptions, AudioFrame, EotPrediction};

/// The host's mini model factory (`livekit.local_inference.EOT`), called when a stream mounts the
/// mini model. Never called when the model is native.
#[crate::host(proxy(host = "collections.abc.Callable"))]
pub trait EotModelFactory: Send + Sync {
    #[host(call, name = "__call__")]
    fn create(&self) -> Result<Arc<dyn EotModel>, HostError>;
}

/// One host mini model, called from a blocking thread.
#[crate::host(proxy(host = "livekit.local_inference.EOT"))]
pub trait EotModel: Send + Sync {
    /// The end-of-turn probability of 16 kHz mono samples.
    #[host(call)]
    fn predict(&self, pcm: Vec<i16>) -> Result<f64, HostError>;
}

/// Hears each prediction's metrics (`TurnDetector.emit("metrics_collected", ..)`).
#[crate::host(proxy(host = "collections.abc.Callable"))]
pub trait EotMetricsObserver: Send + Sync {
    #[host(call, name = "__call__")]
    fn collected(&self, metrics: EotInferenceMetrics) -> Result<(), HostError>;
}

/// The detector behind `inference.TurnDetector`: the user's threshold overrides over the model's
/// defaults, shared by every stream it opens, so a stream's cloud-to-local fallback rescales the
/// table the detector answers from.
#[crate::host(class(name = "InferenceEot"))]
pub trait InferenceEot: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(class(name = "EotStream"))]
    type Stream;

    /// `turn-detector-v1` connects to the gateway at `base_url`; `local_fallback` lets it degrade
    /// to the mini model, which `local_model` builds when the build links none.
    #[host(new)]
    #[allow(clippy::too_many_arguments)]
    fn new(
        env: &Self::Env,
        model: TurnDetectorModel,
        overrides: Option<ThresholdOverride>,
        backchannel_overrides: Option<ThresholdOverride>,
        #[host(kw)] local_fallback: bool,
        base_url: String,
        api_key: String,
        api_secret: String,
        sample_rate: u32,
        local_model: Arc<dyn EotModelFactory>,
        #[host(default = None)] on_metrics: Option<Arc<dyn EotMetricsObserver>>,
    ) -> Self;

    /// The active model, which a stream's fallback changes.
    #[host(getter)]
    fn model(&self) -> TurnDetectorModel;

    /// `None` is `NOT_GIVEN`.
    #[host(getter)]
    fn overrides(&self) -> Option<ThresholdOverride>;

    #[host(getter)]
    fn backchannel_overrides(&self) -> Option<ThresholdOverride>;

    /// `language=None` scores as English, not the global default.
    #[host(call)]
    fn lookup(&self, #[host(default = None)] language: Option<LanguageCode>) -> Option<f64>;

    /// `None` disables backchannel detection; a `0.0` threshold also means off.
    #[host(call)]
    fn lookup_backchannel(
        &self,
        #[host(default = None)] language: Option<LanguageCode>,
    ) -> Option<f64>;

    #[host(call)]
    fn supports(&self, #[host(default = None)] language: Option<LanguageCode>) -> bool;

    #[host(call)]
    fn update_overrides(&self, #[host(default = None)] overrides: Option<ThresholdOverride>);

    #[host(call)]
    fn update_backchannel_overrides(
        &self,
        #[host(default = None)] overrides: Option<ThresholdOverride>,
    );

    /// One stream of the user's audio. `headers` (`get_inference_headers()`) identify the job on
    /// each gateway connection.
    #[host(call)]
    fn stream(
        &self,
        headers: Vec<(String, String)>,
        conn_options: ApiConnectOptions,
    ) -> Result<Self::Stream, ProviderError>;
}

/// `TurnDetectorStream`: one stream of the user's audio, answering `predict()` at each pause. The
/// session's own detector when handed to it, which implements its stream operations.
#[crate::host(class(name = "EotStream", native = crate::EotDetector))]
pub trait EotStream: crate::EotDetector + Sized + 'static {
    /// The active model, which changes under the caller after a fallback.
    #[host(getter)]
    fn model(&self) -> TurnDetectorModel;

    #[host(getter)]
    fn is_fallback(&self) -> bool;

    /// True once the cloud transport is gone with no local model to replace it.
    #[host(getter)]
    fn is_degraded(&self) -> bool;

    /// The prediction over the audio so far; supersedes any in flight.
    #[host(await)]
    async fn predict(&self) -> Result<EotPrediction, ProviderError>;

    #[host(call)]
    fn cancel_inference(&self, #[host(kw, default = false)] timed_out: bool) {
        crate::EotDetector::cancel_inference(self, timed_out);
    }

    /// A segment boundary, not a close.
    #[host(call)]
    fn flush(&self, #[host(default = None)] reason: Option<String>) {
        crate::EotDetector::flush(self, reason);
    }

    #[host(call)]
    fn push_audio(&self, frame: AudioFrame) {
        crate::EotDetector::push_audio(self, frame);
    }

    #[host(call)]
    fn end_input(&self);

    #[host(await)]
    async fn aclose(&self);
}

/// `inference.AdaptiveInterruptionDetector`: the gateway model that tells an interruption from a
/// backchannel while the agent speaks. Natively the session's interruption source.
#[crate::host(class(
    name = "InterruptionDetector",
    native = crate::turn::interruption::InterruptionSource,
    subclass
))]
pub trait InterruptionDetector: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    /// `base_url`, `api_key` and `api_secret` are required; the Python class resolves them from
    /// the server's settings.
    #[host(new)]
    #[allow(clippy::too_many_arguments)]
    fn new(
        env: &Self::Env,
        #[host(kw, default = None)] threshold: Option<f64>,
        #[host(default = 0.05)] min_interruption_duration: f64,
        #[host(default = 3.0)] max_audio_duration: f64,
        #[host(default = 1.0)] audio_prefix_duration: f64,
        #[host(default = 0.1)] detection_interval: f64,
        #[host(default = 0.7)] inference_timeout: f64,
        #[host(default = None)] base_url: Option<String>,
        #[host(default = None)] api_key: Option<String>,
        #[host(default = None)] api_secret: Option<String>,
        #[host(default = None)] conn_options: Option<ApiConnectOptions>,
        #[host(default = 600)] token_ttl_secs: u64,
    ) -> Result<Self, InvalidOption>;

    /// Shared with live streams, which reconnect to pick the options up.
    #[host(call)]
    fn update_options(
        &self,
        #[host(kw, default = None)] threshold: Option<f64>,
        #[host(default = None)] min_interruption_duration: Option<f64>,
    );
}
