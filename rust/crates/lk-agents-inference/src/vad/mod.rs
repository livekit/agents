//! `inference/vad.py` -- VAD backed by `livekit.local_inference`. The logic is in [`machine`];
//! this gives it the `(Sink, Source)` shape and runs it on a task.

pub mod machine;
#[cfg(feature = "python")]
mod python;

use crate::abi::{AudioFrame, StreamSink, StreamSource};
use async_trait::async_trait;
use lk_agents_abi::{Spawner, SpawnerExt};
use lk_agents_abi::errors::{ApiError, ProviderError};
use machine::{VadMachine, VadModel, VadOptions, VadOutput};
use std::sync::Arc;
use tokio::sync::mpsc;

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

    /// The unboxed pair, for callers that need [`VadOutput`]'s audio and lag alongside the event.
    pub fn stream_native(&self) -> (VadSinkHandle, VadOutputSource) {
        let mut opts = self.opts.subscribe();
        let mut vad_machine = VadMachine::new(*opts.borrow_and_update(), self.models.create());

        let (input_tx, mut input_rx) = mpsc::unbounded_channel::<VadCmd>();
        let (output_tx, output_rx) = mpsc::unbounded_channel::<VadOutput>();

        self.spawner.spawn_unboxed(async move {
            while let Some(item) = input_rx.recv().await {
                if opts.has_changed().unwrap_or(false) {
                    vad_machine.update_options(*opts.borrow_and_update());
                }
                let outputs = match item {
                    VadCmd::Frame(frame) => match vad_machine.push(frame) {
                        Ok(outputs) => outputs,
                        // The reference logs and continues.
                        Err(_) => continue,
                    },
                    VadCmd::Flush => {
                        vad_machine.reset();
                        Vec::new()
                    }
                };
                for output in outputs {
                    if output_tx.send(output).is_err() {
                        return;
                    }
                }
            }
        });

        (VadSinkHandle { input: std::sync::Mutex::new(Some(input_tx)) }, VadOutputSource { outputs: output_rx })
    }
}

/// `inference/vad.py`: Silero VAD with the ONNX model in the host and the machine in Rust.
///
/// `model_factory()` builds one model per stream (each needs its own LSTM state); a model has
/// `predict(list[int])` and `reset()`.
#[lk_agents_abi::host(object(name = "InferenceVad", cell = "arc", rust_name = InferenceVad))]
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
        NativeVad::new(opts, model_factory, spawner)
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

/// The `_FlushSentinel` is `flush()` on the trait, so the channel carries both.
#[derive(Debug, Clone, PartialEq)]
enum VadCmd {
    Frame(AudioFrame),
    Flush,
}

/// `end_input` drops the sender: the pump drains what is queued and ends, which ends the output.
pub struct VadSinkHandle {
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

/// The full output, including the PCM the event refers to.
#[lk_agents_abi::host(stream(name = "InferenceVadStream", doc = "One VAD stream.", sink = crate::vad::VadSinkHandle,
    input = AudioFrame, input_py = "rtc.AudioFrame", end_input,
    stub_import = "from livekit import rtc",
    output = VadOutput))]
pub struct VadOutputSource {
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

    // Catches: `end_input` doing nothing, so a caller reading the stream to its end waits forever
    // (the reference's `end_input` closes the input channel and the stream finishes).
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
