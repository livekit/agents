#![cfg_attr(
    feature = "python",
    allow(
        clippy::disallowed_types,
        reason = "S0: generated host siblings need module scope; exact type sites are ratcheted."
    )
)]

//! `inference/vad.py`: the Silero VAD, its model from the linked native provider
//! ([`crate::local_models`]) or else the host's `livekit.local_inference`. The decisions are in
//! [`machine`]; the linked runtime owns stream execution and returns this provider's bounded
//! `(Sink, Source)` pair. A native model needs no host call per audio frame.

pub mod machine;
#[cfg(feature = "python")]
mod python;

use crate::abi::{AudioFrame, StreamSink, StreamSource};
use async_trait::async_trait;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::{
    MediaRef, ProviderId, Vad, VadCapabilities, VadEvent, VadEventType, VadSink, VadSource,
};
use machine::{VadOptions, VadOutput};
use std::sync::Arc;

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

pub struct NativeVad {
    /// Watched by every running stream: `update_options` reaches them before their next frame.
    opts: tokio::sync::watch::Sender<VadOptions>,
    models: Arc<dyn VadModelFactory>,
    clock: Arc<dyn lk_agents_abi::Clock>,
}

impl NativeVad {
    pub fn new(
        opts: VadOptions,
        models: Arc<dyn VadModelFactory>,
        clock: Arc<dyn lk_agents_abi::Clock>,
    ) -> Self {
        NativeVad { opts: tokio::sync::watch::Sender::new(opts), models, clock }
    }

    /// The pair for callers that need the audio and lag alongside the VAD event.
    pub fn stream_native(&self) -> Result<(VadSinkHandle, VadOutputSource), ProviderError> {
        let mut runtimes = inventory::iter::<VadRuntime>.into_iter();
        let runtime = runtimes.next().ok_or_else(|| {
            ProviderError::Local(ApiError::new(
                "VAD execution service unavailable: no runtime linked",
            ))
        })?;
        if runtimes.next().is_some() {
            return Err(ProviderError::Local(ApiError::new(
                "VAD execution service unavailable: multiple runtimes linked",
            )));
        }
        let (sink, source) =
            (runtime.open)(self.opts.subscribe(), self.models.clone(), self.clock.clone())?;
        Ok((VadSinkHandle { sink }, VadOutputSource { source }))
    }
}

/// `inference/vad.py`: Silero VAD, the machine in Rust and the model native when the build links
/// one, else the host's.
///
/// `model_factory()` builds one host model per stream (each needs its own LSTM state); a model has
/// `predict(list[int])` and `reset()`. Never called when the model is native.
#[lk_agents_abi::host(object(name = "InferenceVad", cell = "arc", rust_name = InferenceVad, as_dyn(Vad = as_vad)))]
impl NativeVad {
    #[cfg(feature = "python")]
    #[host(new(kw_only = "min_speech_duration"))]
    #[allow(clippy::too_many_arguments)]
    pub fn for_host(
        model_factory: Arc<dyn VadModelFactory>,
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
        let clock = <Arc<dyn lk_agents_abi::Clock> as lk_agents_abi::python::Ambient>::ambient();
        NativeVad::new(opts, crate::local_models::vad_factory_or(model_factory), clock)
    }

    /// The detector as the provider a session holds directly.
    pub fn as_vad(self: &Arc<Self>) -> Arc<dyn Vad> {
        Arc::clone(self) as Arc<dyn Vad>
    }

    /// `_VADStream.update_options` on a running stream: every option, applied before the next frame.
    #[host(method(kw_only))]
    pub fn update_options(
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

    // One stream per call; each event comes with the audio it refers to.
    #[host(method(name = "stream", stream = InferenceVadStream))]
    pub fn stream_for_host(&self) -> Result<(VadSinkHandle, VadOutputSource), ProviderError> {
        self.stream_native()
    }
}

#[async_trait]
impl Vad for NativeVad {
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

    async fn stream(&self) -> Result<(VadSink, VadSource), ProviderError> {
        let (sink, source) = self.stream_native()?;
        Ok((Box::new(sink), Box::new(VadEventSource(source))))
    }
}

/// [`VadOutput`] as the session reads it: the event, with the speech it ended on.
struct VadEventSource(VadOutputSource);

#[async_trait]
impl StreamSource<VadEvent> for VadEventSource {
    async fn next(&mut self) -> Result<Option<VadEvent>, ProviderError> {
        let Some(output) = self.0.next().await? else { return Ok(None) };
        if output.slow_inference {
            log::warn!(target: "livekit.agents", delay = output.extra_inference_time; "inference is slower than realtime");
        }
        let mut event = output.event;
        // Only an end of speech's audio is read (a non-streaming STT recognizes it).
        if event.kind == VadEventType::EndOfSpeech {
            event.frames = speech_ref(output.frames);
        }
        Ok(Some(event))
    }

    async fn close(self: Box<Self>, deadline: lk_agents_abi::Ts) -> lk_agents_abi::Closed {
        Box::new(self.0).close(deadline).await
    }
}

#[cfg(feature = "python")]
fn speech_ref(frames: Vec<AudioFrame>) -> Option<MediaRef> {
    python::speech_ref(frames)
}

#[cfg(not(feature = "python"))]
fn speech_ref(_frames: Vec<AudioFrame>) -> Option<MediaRef> {
    None
}

/// Runtime-owned stream execution. The inference crate keeps only the synchronous machine and
/// provider interface; the linked runtime owns queues, model threads and close settlement.
pub type VadRuntimeStream = (Box<dyn StreamSink<AudioFrame>>, Box<dyn StreamSource<VadOutput>>);

pub struct VadRuntime {
    pub open: fn(
        tokio::sync::watch::Receiver<VadOptions>,
        Arc<dyn VadModelFactory>,
        Arc<dyn lk_agents_abi::Clock>,
    ) -> Result<VadRuntimeStream, ProviderError>,
}

inventory::collect!(VadRuntime);

/// The native runtime's bounded audio ingress.
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

#[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
/// The full output, including the PCM the event refers to.
#[lk_agents_abi::host(stream(name = "InferenceVadStream", doc = "One VAD stream.", sink = crate::vad::VadSinkHandle,
    input = AudioFrame, input_py = "rtc.AudioFrame", end_input,
    stub_import = "from livekit import rtc",
    output = VadOutput))]
pub struct VadOutputSource {
    source: Box<dyn StreamSource<VadOutput>>,
}

#[async_trait]
impl StreamSource<VadOutput> for VadOutputSource {
    async fn next(&mut self) -> Result<Option<VadOutput>, ProviderError> {
        self.source.next().await
    }

    async fn close(self: Box<Self>, deadline: lk_agents_abi::Ts) -> lk_agents_abi::Closed {
        self.source.close(deadline).await
    }
}
