#![cfg_attr(
    feature = "python",
    allow(
        clippy::disallowed_types,
        reason = "S0: generated host siblings need module scope; exact type sites are ratcheted."
    )
)]

//! `inference/vad.py`: the Silero VAD, its model from the linked native provider
//! ([`crate::local_models`]) or else the host's `livekit.local_inference`. The decisions are in
//! [`machine`]; this gives it the `(Sink, Source)` shape, executes the model windows the machine
//! asks for on a task, and is the session's [`Vad`] as it is, with no host call per frame when the
//! model is native.

pub mod machine;
#[cfg(feature = "python")]
mod python;

use crate::abi::{AudioFrame, StreamSink, StreamSource};
use async_trait::async_trait;
use lk_agents_abi::{MediaRef, ProviderId, Spawner, SpawnerExt, Vad, VadCapabilities, VadEvent, VadEventType, VadSink, VadSource};
use lk_agents_abi::errors::{ApiError, ProviderError};
use machine::{VadMachine, VadOptions, VadOutput, VadPredict, VadResult};
use std::sync::Arc;
use tokio::sync::mpsc;

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
    spawner: Arc<dyn Spawner>,
}

impl NativeVad {
    pub fn new(
        opts: VadOptions,
        models: Arc<dyn VadModelFactory>,
        spawner: Arc<dyn Spawner>,
    ) -> Self {
        NativeVad { opts: tokio::sync::watch::Sender::new(opts), models, spawner }
    }

    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// The unboxed pair, for callers that need [`VadOutput`]'s audio and lag alongside the event.
    pub fn stream_native(&self) -> (VadSinkHandle, VadOutputSource) {
        let mut opts = self.opts.subscribe();
        let mut vad_machine = VadMachine::new(*opts.borrow_and_update());
        let mut model = self.models.create();

        let (input_tx, mut input_rx) = mpsc::unbounded_channel::<VadCmd>();
        let (output_tx, output_rx) = mpsc::unbounded_channel::<VadOutput>();

        self.spawner.spawn_unboxed(async move {
            while let Some(item) = input_rx.recv().await {
                if opts.has_changed().unwrap_or(false) {
                    vad_machine.update_options(*opts.borrow_and_update());
                }
                let mut next = match item {
                    VadCmd::Frame(frame) => match vad_machine.push(frame) {
                        Ok(predict) => predict,
                        // The reference logs and continues.
                        Err(_) => continue,
                    },
                    VadCmd::Flush => {
                        vad_machine.reset();
                        model.reset();
                        None
                    }
                };
                while let Some(VadPredict { window, samples }) = next {
                    let (probability, inference_duration) = model.predict(&samples);
                    let (outputs, following) = vad_machine.predicted(VadResult {
                        window,
                        probability,
                        inference_duration,
                    });
                    for output in outputs {
                        if output_tx.send(output).is_err() {
                            return;
                        }
                    }
                    next = following;
                }
            }
        });

        (VadSinkHandle { input: std::sync::Mutex::new(Some(input_tx)) }, VadOutputSource { outputs: output_rx })
    }
}

/// `inference/vad.py`: Silero VAD, the machine in Rust and the model native when the build links
/// one, else the host's.
///
/// `model_factory()` builds one host model per stream (each needs its own LSTM state); a model has
/// `predict(list[int])` and `reset()`. Never called when the model is native.
#[lk_agents_abi::host(object(name = "InferenceVad", cell = "arc", rust_name = InferenceVad, as_dyn(Vad = as_vad)))]
impl NativeVad {
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
        #[host(ambient)] spawner: Arc<dyn Spawner>,
    ) -> Self {
        let opts = VadOptions {
            min_speech_duration,
            min_silence_duration,
            prefix_padding_duration,
            max_buffered_speech,
            activation_threshold,
            deactivation_threshold,
        };
        NativeVad::new(opts, crate::local_models::vad_factory_or(model_factory), spawner)
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
    pub fn stream_for_host(&self) -> (VadSinkHandle, VadOutputSource) {
        self.stream_native()
    }
}

#[async_trait]
impl Vad for NativeVad {
    fn label(&self) -> String {
        "livekit.agents.inference.vad.VAD".into()
    }

    fn capabilities(&self) -> VadCapabilities {
        VadCapabilities { update_interval: machine::VAD_WINDOW_SAMPLES as f64 / machine::MODEL_SAMPLE_RATE as f64 }
    }

    fn id(&self) -> ProviderId {
        ProviderId { label: self.label(), model: "silero".into(), provider: "livekit-local-inference".into() }
    }

    async fn stream(&self) -> Result<(VadSink, VadSource), ProviderError> {
        let (sink, source) = self.stream_native();
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

/// The `_FlushSentinel` is `flush()` on the trait, so the channel carries both.
#[derive(Debug, Clone, PartialEq)]
enum VadCmd {
    Frame(AudioFrame),
    Flush,
}

/// `end_input` drops the sender: the pump drains what is queued and ends, which ends the output.
pub struct VadSinkHandle {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    input: std::sync::Mutex<Option<mpsc::UnboundedSender<VadCmd>>>,
}

impl StreamSink<AudioFrame> for VadSinkHandle {
    fn push(&self, item: AudioFrame) -> Result<(), ProviderError> {
        self.send(VadCmd::Frame(item))
    }

    fn flush(&self) -> Result<(), ProviderError> {
        self.send(VadCmd::Flush)
    }

    /// `VADStream.end_input()` (`vad.py:167`): the input channel closes, the stream finishes.
    fn end_input(&self) {
        self.input.lock().unwrap_or_else(|p| p.into_inner()).take();
    }
}

impl VadSinkHandle {
    fn send(&self, cmd: VadCmd) -> Result<(), ProviderError> {
        let closed =
            || ProviderError::Local(ApiError::connection(Some("the VAD stream is closed".into())));
        let input = self.input.lock().unwrap_or_else(|p| p.into_inner());
        input.as_ref().ok_or_else(closed)?.send(cmd).map_err(|_| closed())
    }
}

#[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
/// The full output, including the PCM the event refers to.
#[lk_agents_abi::host(stream(name = "InferenceVadStream", doc = "One VAD stream.", sink = crate::vad::VadSinkHandle,
    input = AudioFrame, input_py = "rtc.AudioFrame", end_input,
    stub_import = "from livekit import rtc",
    output = VadOutput))]
pub struct VadOutputSource {
    #[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
    outputs: mpsc::UnboundedReceiver<VadOutput>,
}

#[async_trait]
impl StreamSource<VadOutput> for VadOutputSource {
    async fn next(&mut self) -> Result<Option<VadOutput>, ProviderError> {
        Ok(self.outputs.recv().await)
    }
    /// Nothing here awaits anything, so the deadline is not consulted.
    async fn close(mut self: Box<Self>, _deadline: lk_agents_abi::Ts) -> lk_agents_abi::Closed {
        self.outputs.close();
        lk_agents_abi::Closed::Clean
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use machine::VadOptions;

    struct Silent;

    impl VadModel for Silent {
        fn predict(&mut self, _window: &[i16]) -> (f32, f64) {
            (0.0, 0.0)
        }
        fn reset(&mut self) {}
    }

    struct Factory;

    impl VadModelFactory for Factory {
        fn create(&self) -> Box<dyn VadModel> {
            Box::new(Silent)
        }
    }

    /// Speech wherever the window is not silent.
    struct Energy;

    impl VadModel for Energy {
        fn predict(&mut self, window: &[i16]) -> (f32, f64) {
            (if window.iter().any(|s| *s != 0) { 1.0 } else { 0.0 }, 0.0)
        }
        fn reset(&mut self) {}
    }

    struct EnergyFactory;

    impl VadModelFactory for EnergyFactory {
        fn create(&self) -> Box<dyn VadModel> {
            Box::new(Energy)
        }
    }

    // Catches: the session's `Vad` (the detector held directly, not through the host) losing the
    // machine's start and end of speech.
    #[tokio::test]
    async fn as_the_sessions_vad_it_reports_speech() {
        let spawner: Arc<dyn Spawner> = Arc::new(lk_agents_abi::TokioSpawner::current());
        let vad = Arc::new(NativeVad::new(VadOptions::default(), Arc::new(EnergyFactory), spawner)).as_vad();
        assert_eq!(vad.capabilities().update_interval, 0.032);
        let (sink, mut source) = vad.stream().await.unwrap();
        let frame = |level: i16| AudioFrame {
            data: std::iter::repeat_n(level.to_le_bytes(), 160).flatten().collect::<Vec<u8>>().into(),
            sample_rate: 16000,
            num_channels: 1,
            samples_per_channel: 160,
        };
        for i in 0..200 {
            sink.push(frame(if i < 100 { 1000 } else { 0 })).unwrap();
        }
        sink.end_input();
        let mut kinds = Vec::new();
        while let Ok(Some(ev)) = source.next().await {
            if ev.kind != VadEventType::InferenceDone {
                kinds.push(ev.kind);
            }
        }
        assert_eq!(kinds, [VadEventType::StartOfSpeech, VadEventType::EndOfSpeech]);
    }

    // Catches: `end_input` doing nothing, so a caller reading the stream to its end waits forever
    // (the reference's `end_input` closes the input channel and the stream finishes).
    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    #[tokio::test]
    async fn end_input_ends_the_stream_after_what_was_pushed() {
        let spawner: Arc<dyn Spawner> = Arc::new(lk_agents_abi::TokioSpawner::current());
        let vad = NativeVad::new(VadOptions::default(), Arc::new(Factory), spawner);
        let (sink, mut source) = vad.stream_native();
        let frame = AudioFrame {
            data: vec![0u8; 320 * 2].into(),
            sample_rate: 16000,
            num_channels: 1,
            samples_per_channel: 320,
        };
        sink.push(frame.clone()).unwrap();
        sink.end_input();
        assert!(sink.push(frame).is_err(), "a push after end_input must be refused");
        let drained = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            let mut outputs = 0;
            while let Ok(Some(_)) = source.next().await {
                outputs += 1;
            }
            outputs
        })
        .await;
        assert!(drained.is_ok(), "the stream never ended after end_input");
    }
}
