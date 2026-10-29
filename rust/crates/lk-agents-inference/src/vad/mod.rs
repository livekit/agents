//! `inference/vad.py`: the Silero VAD, its model from the linked native provider
//! ([`crate::local_models`]) or else the host's `livekit.local_inference`. The decisions are in
//! [`machine`]; [`stream`] runs each detection on an admitted media worker behind a bounded
//! `(Sink, Source)` pair. A native model needs no host call per audio frame.

pub mod machine;
mod stream;

use crate::abi::{AudioFrame, StreamSink, StreamSource};

use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::vad::inference::VadModels;
use lk_agents_abi::{ProviderId, Vad, VadCapabilities, VadEvent, VadSink, VadSource};
use machine::{VadOptions, VadOutput};
use std::marker::PhantomData;
use std::sync::Arc;

use crate::env::{Env, Standalone};

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

/// `E` is the hosting process whose media workers run the model and whose runtime the stream
/// tasks run on.
pub struct NativeVad<E: Env = Standalone> {
    /// Watched by every running stream: `update_options` reaches them before their next frame.
    opts: tokio::sync::watch::Sender<VadOptions>,
    models: Arc<dyn VadModelFactory>,
    clock: Arc<dyn lk_agents_abi::Clock>,
    env: PhantomData<fn() -> E>,
}

impl<E: Env> NativeVad<E> {
    pub fn new(
        opts: VadOptions,
        models: Arc<dyn VadModelFactory>,
        clock: Arc<dyn lk_agents_abi::Clock>,
    ) -> Self {
        NativeVad { opts: tokio::sync::watch::Sender::new(opts), models, clock, env: PhantomData }
    }

    /// The pair for callers that need the audio and lag alongside the VAD event.
    pub fn stream_native(&self) -> Result<(VadSinkHandle, VadOutputSource), ProviderError> {
        let unavailable = |error: String| {
            ProviderError::Local(ApiError::new(format!("VAD execution unavailable: {error}")))
        };
        let workers = E::workers().map_err(unavailable)?;
        let runtime = E::caller_runtime().map_err(unavailable)?;
        let (sink, source) = stream::open(
            self.opts.subscribe(),
            self.models.clone(),
            self.clock.clone(),
            workers,
            runtime,
        )?;
        Ok((VadSinkHandle { sink }, VadOutputSource { source }))
    }
}

/// `_core.InferenceVad`: the generated adapter calls this. `model_factory()` builds one host
/// model per stream (each needs its own recurrent state); never called when the model is native.
impl<E: Env> lk_agents_abi::vad::inference::InferenceVad for NativeVad<E> {
    type Audio = VadSinkHandle;
    type Events = VadEventSource;

    fn new(
        model_factory: Arc<dyn VadModels>,
        min_speech_duration: f64,
        min_silence_duration: f64,
        prefix_padding_duration: f64,
        max_buffered_speech: f64,
        activation_threshold: f64,
        deactivation_threshold: f64,
    ) -> Self {
        let opts = VadOptions {
            min_speech_duration,
            min_silence_duration,
            prefix_padding_duration,
            max_buffered_speech,
            activation_threshold,
            deactivation_threshold,
        };
        let host = Arc::new(HostModels(model_factory));
        NativeVad::new(opts, crate::local_models::vad_factory_or(host), E::clock())
    }

    fn update_options(
        &self,
        min_speech_duration: f64,
        min_silence_duration: f64,
        prefix_padding_duration: f64,
        max_buffered_speech: f64,
        activation_threshold: f64,
        deactivation_threshold: f64,
    ) {
        self.opts.send_replace(VadOptions {
            min_speech_duration,
            min_silence_duration,
            prefix_padding_duration,
            max_buffered_speech,
            activation_threshold,
            deactivation_threshold,
        });
    }

    fn stream(&self) -> Result<(VadSinkHandle, VadEventSource), ProviderError> {
        let (sink, source) = self.stream_native()?;
        Ok((sink, VadEventSource(source)))
    }
}

/// The host's model factory.
struct HostModels(Arc<dyn VadModels>);

impl VadModelFactory for HostModels {
    /// A factory that raised gives a silent model rather than a panic on a native worker (never
    /// selected on a healthy path).
    fn create(&self) -> Box<dyn VadModel> {
        match self.0.create() {
            Ok(model) => Box::new(HostModel(Some(model))),
            Err(error) => {
                log::error!(target: "livekit.agents", "VAD model factory failed: {error}");
                Box::new(HostModel(None))
            }
        }
    }
}

/// One host model; its `predict` is timed here, as the reference does. A failed call answers
/// `(0.0, 0.0)`.
struct HostModel(Option<Arc<dyn lk_agents_abi::vad::inference::VadModel>>);

impl VadModel for HostModel {
    fn predict(&mut self, window: &[i16]) -> (f32, f64) {
        let Some(model) = &self.0 else { return (0.0, 0.0) };
        let started = std::time::Instant::now();
        match model.predict(window.to_vec()) {
            Ok(probability) => (probability as f32, started.elapsed().as_secs_f64()),
            Err(error) => {
                log::error!(target: "livekit.agents", "VAD model failed: {error}");
                (0.0, 0.0)
            }
        }
    }

    fn reset(&mut self) {
        if let Some(model) = &self.0
            && let Err(error) = model.reset()
        {
            log::error!(target: "livekit.agents", "VAD model reset failed: {error}");
        }
    }
}

impl<E: Env> Vad for NativeVad<E> {
    fn label(&self) -> String {
        "livekit.agents.inference.vad.VAD".into()
    }

    fn capabilities(&self) -> VadCapabilities {
        VadCapabilities {
            update_interval: machine::VAD_WINDOW_SAMPLES as f64 / machine::MODEL_SAMPLE_RATE as f64,
        }
    }

    fn id(&self) -> ProviderId {
        ProviderId {
            label: self.label(),
            model: "silero".into(),
            provider: "livekit-local-inference".into(),
        }
    }

    fn stream(&self) -> lk_agents_abi::BoxFuture<'_, Result<(VadSink, VadSource), ProviderError>> {
        let opened = self.stream_native().map(|(sink, source)| {
            (Box::new(sink) as VadSink, Box::new(VadEventSource(source)) as VadSource)
        });
        Box::pin(async move { opened })
    }
}

/// [`VadOutput`] as a reader takes it: the event, with the audio it refers to; inference slower
/// than realtime is logged.
pub struct VadEventSource(VadOutputSource);

impl StreamSource<VadEvent> for VadEventSource {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<VadEvent>, ProviderError>> {
        Box::pin(async move {
            let Some(output) = self.0.next().await? else { return Ok(None) };
            if output.slow_inference {
                log::warn!(target: "livekit.agents", delay = output.extra_inference_time; "inference is slower than realtime");
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

/// The stream's bounded audio ingress.
pub struct VadSinkHandle {
    sink: Box<dyn StreamSink<AudioFrame>>,
}

impl StreamSink<AudioFrame> for VadSinkHandle {
    fn push(&self, item: AudioFrame) -> Result<(), ProviderError> {
        self.sink.push(item)
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
    source: Box<dyn StreamSource<VadOutput>>,
}

impl StreamSource<VadOutput> for VadOutputSource {
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
