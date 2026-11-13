pub mod machine;
pub mod plugin;
mod stream;

use crate::abi::{AudioFrame, StreamInput, StreamOutput};
use crate::telemetry;

use lk_agents_abi::errors::{Invalid, ProviderError};
use lk_agents_abi::inference::models::VadModelName;
use lk_agents_abi::types::provider::NativeLabel;
use lk_agents_abi::vad::inference::VadModels;
use lk_agents_abi::{Maybe, ProviderId, Vad, VadCapabilities, VadEvent, VadSink, VadSource};
use machine::{VadOptions, VadOutput};
use std::sync::Arc;

use lk_agents_net::env::ProcessEnv;

/// The Silero model behind `livekit.local_inference`; `predict` also returns how long it took.
pub trait VadModel: Send {
    fn predict(&mut self, window: &[i16]) -> (f32, f64);
    fn reset(&mut self);
}

/// Builds a fresh model per stream: "each stream allocates its own per-instance LSTM/context
/// state".
pub trait VadModelFactory: Send + Sync {
    fn create(&self) -> Box<dyn VadModel>;
}

/// `env` lends the media workers that run the model and the runtime the stream tasks run on.
pub struct LocalVad {
    /// Watched by every running stream: `update_options` reaches them before their next frame.
    opts: lk_agents_net::signal::Producer<VadOptions>,
    models: Arc<dyn VadModelFactory>,
    env: ProcessEnv,
    /// The name logs give it: its host class's when its host built it.
    label: String,
}

impl LocalVad {
    pub fn new(opts: VadOptions, models: Arc<dyn VadModelFactory>, env: &ProcessEnv) -> Self {
        LocalVad {
            opts: lk_agents_net::signal::signal(opts).0,
            models,
            env: env.clone(),
            label: NativeLabel::Vad.as_str().into(),
        }
    }

    /// A detector whose model is the linked native one, else one `models` builds per stream.
    pub fn with_host_models(
        opts: VadOptions,
        models: Arc<dyn VadModels>,
        env: &ProcessEnv,
    ) -> Self {
        let host = Arc::new(HostModels { models, clock: env.clock.clone() });
        LocalVad::new(opts, crate::local_models::vad_factory_or(host), env)
    }

    /// The pair for callers that need the audio and lag alongside the VAD event.
    pub fn stream_native(&self) -> Result<(VadInput, VadOutputSource), ProviderError> {
        self.open(&telemetry::Cx::NONE)
    }

    /// Input drops are measured under `cx`'s job.
    fn open(&self, cx: &telemetry::Cx) -> Result<(VadInput, VadOutputSource), ProviderError> {
        let (sink, source) = stream::open(
            self.opts.cell(),
            self.models.clone(),
            self.env.clock.clone(),
            lk_agents_media::pool::media_pool(),
            cx,
        )?
        .split();
        Ok((VadInput { sink }, VadOutputSource { source }))
    }
}

/// `_core.InferenceVad`: the generated adapter calls this. `model_factory()` builds one host
/// model per stream (each needs its own recurrent state); never called when the model is native.
impl lk_agents_abi::vad::inference::InferenceVad for LocalVad {
    type Audio = VadInput;
    type Events = VadEventSource;
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        label: String,
        model_factory: Arc<dyn VadModels>,
        _model: VadModelName,
        min_speech_duration: f64,
        min_silence_duration: f64,
        prefix_padding_duration: f64,
        max_buffered_speech: f64,
        activation_threshold: f64,
        deactivation_threshold: Option<f64>,
    ) -> Result<Self, Invalid> {
        if deactivation_threshold.is_some_and(|t| t <= 0.0) {
            return Err(Invalid("deactivation_threshold must be greater than 0".into()));
        }
        let opts = VadOptions {
            min_speech_duration,
            min_silence_duration,
            prefix_padding_duration,
            max_buffered_speech,
            activation_threshold,
            // Hysteresis.
            deactivation_threshold: deactivation_threshold
                .unwrap_or((activation_threshold - 0.15).max(0.01)),
        };
        Ok(LocalVad { label, ..LocalVad::with_host_models(opts, model_factory, env) })
    }

    fn model(&self) -> VadModelName {
        VadModelName::Silero
    }

    fn capabilities(&self) -> VadCapabilities {
        Vad::capabilities(self)
    }

    fn update_options(
        &self,
        min_speech_duration: Maybe<f64>,
        min_silence_duration: Maybe<f64>,
        prefix_padding_duration: Maybe<f64>,
        max_buffered_speech: Maybe<f64>,
        activation_threshold: Maybe<f64>,
        deactivation_threshold: Maybe<f64>,
    ) {
        self.opts.modify(|opts| {
            let given = [
                (&mut opts.min_speech_duration, min_speech_duration),
                (&mut opts.min_silence_duration, min_silence_duration),
                (&mut opts.prefix_padding_duration, prefix_padding_duration),
                (&mut opts.max_buffered_speech, max_buffered_speech),
                (&mut opts.activation_threshold, activation_threshold),
                (&mut opts.deactivation_threshold, deactivation_threshold),
            ];
            for (option, value) in given {
                if let Some(value) = value.given() {
                    *option = value;
                }
            }
        });
    }

    fn min_silence_duration(&self) -> f64 {
        self.opts.with(|opts| opts.min_silence_duration)
    }

    fn stream(
        &self,
    ) -> Result<lk_agents_abi::DuplexStream<VadInput, VadEventSource>, ProviderError> {
        let (sink, source) = self.stream_native()?;
        Ok(lk_agents_abi::DuplexStream::new(sink, VadEventSource(source)))
    }
}

/// The host's model factory; `clock` times each model's `predict`.
struct HostModels {
    models: Arc<dyn VadModels>,
    clock: Arc<dyn lk_agents_abi::Clock>,
}

impl VadModelFactory for HostModels {
    /// A factory that raised gives a silent model rather than a panic on a native worker (never
    /// selected on a healthy path).
    fn create(&self) -> Box<dyn VadModel> {
        let clock = self.clock.clone();
        match self.models.create() {
            Ok(model) => Box::new(HostModel { model: Some(model), clock }),
            Err(error) => {
                log::error!(target: lk_agents_abi::LOG_TARGET, "VAD model factory failed: {error}");
                Box::new(HostModel { model: None, clock })
            }
        }
    }
}

/// One host model; its `predict` is timed here, as the reference does. A failed call answers
/// `(0.0, 0.0)`.
struct HostModel {
    model: Option<Arc<dyn lk_agents_abi::vad::inference::VadModel>>,
    clock: Arc<dyn lk_agents_abi::Clock>,
}

impl VadModel for HostModel {
    fn predict(&mut self, window: &[i16]) -> (f32, f64) {
        let Some(model) = &self.model else { return (0.0, 0.0) };
        let started = self.clock.now();
        match model.predict(window.to_vec()) {
            Ok(probability) => {
                let elapsed = self.clock.now().saturating_duration_since(started);
                (probability as f32, elapsed.as_secs_f64())
            }
            Err(error) => {
                log::error!(target: lk_agents_abi::LOG_TARGET, "VAD model failed: {error}");
                (0.0, 0.0)
            }
        }
    }

    fn reset(&mut self) {
        if let Some(model) = &self.model
            && let Err(error) = model.reset()
        {
            log::error!(target: lk_agents_abi::LOG_TARGET, "VAD model reset failed: {error}");
        }
    }
}

impl Vad for LocalVad {
    fn label(&self) -> String {
        self.label.clone()
    }

    fn capabilities(&self) -> VadCapabilities {
        VadCapabilities {
            update_interval: machine::VAD_WINDOW_SAMPLES as f64 / machine::MODEL_SAMPLE_RATE as f64,
        }
    }

    fn id(&self) -> ProviderId {
        ProviderId {
            label: self.label(),
            model: VadModelName::Silero.as_str().into(),
            provider: "livekit-local-inference".into(),
        }
    }

    fn stream(
        &self,
        cx: &telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<lk_agents_abi::DuplexStream<VadSink, VadSource>, ProviderError>,
    > {
        let opened = self.open(cx).map(|(sink, source)| {
            lk_agents_abi::DuplexStream::new(
                Box::new(sink) as VadSink,
                Box::new(VadEventSource(source)) as VadSource,
            )
        });
        Box::pin(async move { opened })
    }
}

/// [`VadOutput`] as a reader takes it: the event, with the audio it refers to; inference slower
/// than realtime is logged.
pub struct VadEventSource(VadOutputSource);

impl StreamOutput<VadEvent> for VadEventSource {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<VadEvent>, ProviderError>> {
        Box::pin(async move {
            let Some(output) = self.0.next().await? else { return Ok(None) };
            if output.slow_inference {
                log::warn!(target: lk_agents_abi::LOG_TARGET, delay = output.extra_inference_time; "inference is slower than realtime");
            }
            let mut event = output.event;
            event.frames = output.frames;
            Ok(Some(event))
        })
    }

    fn close(
        self: Box<Self>,
        deadline: lk_agents_abi::Ts,
    ) -> lk_agents_abi::BoxFuture<'static, lk_agents_abi::Closed> {
        Box::pin(async move { Box::new(self.0).close(deadline).await })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<VadEvent> for VadEventSource {}

/// The stream's audio ingress, a `Now` media queue.
pub struct VadInput {
    sink: Box<dyn StreamInput<AudioFrame>>,
}

impl StreamInput<AudioFrame> for VadInput {
    fn push(&self, item: AudioFrame) -> Result<(), ProviderError> {
        self.sink.push(item)
    }

    fn head(&self, item: AudioFrame) -> Result<(), ProviderError> {
        self.sink.head(item)
    }

    fn flush(&self) -> Result<(), ProviderError> {
        self.sink.flush()
    }

    fn end_input(&self) {
        self.sink.end_input();
    }
}

/// The full output, including the PCM the event refers to.
pub struct VadOutputSource {
    source: Box<dyn StreamOutput<VadOutput>>,
}

impl StreamOutput<VadOutput> for VadOutputSource {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<VadOutput>, ProviderError>> {
        Box::pin(async move { self.source.next().await })
    }

    fn close(
        self: Box<Self>,
        deadline: lk_agents_abi::Ts,
    ) -> lk_agents_abi::BoxFuture<'static, lk_agents_abi::Closed> {
        Box::pin(async move { self.source.close(deadline).await })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<VadOutput> for VadOutputSource {}

#[cfg(test)]
#[path = "vad_tests.rs"]
mod tests;
