//! `_LocalTransport` (`transports.py:398`): the in-process mini model behind [`LocalEotModel`], a
//! 1.2 s ring buffer snapshotted at request time, and a failed prediction scored 0.0.

use std::sync::Mutex;
use std::time::Instant;

use async_trait::async_trait;
use lk_agents_abi::errors::ApiError;
use tokio::sync::mpsc;

use lk_agents_abi::AudioFrame;

use crate::eot::{DEFAULT_SAMPLE_RATE, EotRequestId, EotWarning};
use crate::eot::session::TransportPrediction;
use crate::eot::transport::{EotTransport, PredictionSink};

/// `_CLIENT_BUFFER_SECONDS` (`transports.py:70`). How much recent audio one prediction sees.
pub const CLIENT_BUFFER_SECONDS: f64 = 1.2;

/// `_CLIENT_BUFFER_SAMPLES`.
pub const CLIENT_BUFFER_SAMPLES: usize =
    (CLIENT_BUFFER_SECONDS * DEFAULT_SAMPLE_RATE as f64) as usize;

/// The mini model. Blocking; the transport runs it on a blocking thread (`asyncio.to_thread`).
pub trait LocalEotModel: Send + Sync + 'static {
    /// End-of-turn probability for `pcm`, 16 kHz mono. `Err` is the model raising; scored 0.0.
    fn predict(&self, pcm: &[i16]) -> Result<f64, String>;
}

/// `utils.AudioArrayBuffer` (`utils/audio.py:269`), reduced: no resampling or downmix (the host
/// resamples first). An oversized frame keeps its newest `capacity` samples instead of raising.
#[derive(Debug)]
pub struct AudioRingBuffer {
    buf: Vec<i16>,
    capacity: usize,
}

impl AudioRingBuffer {
    pub fn new(capacity: usize) -> Self {
        AudioRingBuffer { buf: Vec::with_capacity(capacity), capacity }
    }

    pub fn push(&mut self, samples: &[i16]) {
        if samples.len() >= self.capacity {
            self.buf.clear();
            self.buf.extend_from_slice(&samples[samples.len() - self.capacity..]);
            return;
        }
        let overflow = (self.buf.len() + samples.len()).saturating_sub(self.capacity);
        self.buf.drain(..overflow);
        self.buf.extend_from_slice(samples);
    }

    /// A copy of everything buffered, taken at request time.
    pub fn read(&self) -> Vec<i16> {
        self.buf.clone()
    }

    /// `flush` (`transports.py:436`): the turn is over, so the next one starts from silence.
    pub fn clear(&mut self) {
        self.buf.clear();
    }
}

/// One queued inference: the id and the audio it was asked about.
type Job = (EotRequestId, Vec<i16>);

pub struct LocalTransport {
    model: std::sync::Arc<dyn LocalEotModel>,
    sink: std::sync::Arc<dyn PredictionSink>,
    buf: Mutex<AudioRingBuffer>,
    /// `None` once the input has ended, which is what lets [`run`](EotTransport::run) return.
    tx: Mutex<Option<mpsc::UnboundedSender<Job>>>,
    rx: Mutex<Option<mpsc::UnboundedReceiver<Job>>>,
}

impl LocalTransport {
    pub fn new(
        model: std::sync::Arc<dyn LocalEotModel>,
        sink: std::sync::Arc<dyn PredictionSink>,
    ) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        LocalTransport {
            model,
            sink,
            buf: Mutex::new(AudioRingBuffer::new(CLIENT_BUFFER_SAMPLES)),
            tx: Mutex::new(Some(tx)),
            rx: Mutex::new(Some(rx)),
        }
    }
}

#[async_trait]
impl EotTransport for LocalTransport {
    fn session_id(&self) -> Option<String> {
        None
    }

    fn run_inference(&self, request_id: &EotRequestId) {
        // Snapshot here, not in `run` (`transports.py:414`).
        let pcm = self.buf.lock().expect("ring buffer mutex").read();
        if let Some(tx) = self.tx.lock().expect("job sender mutex").as_ref() {
            let _ = tx.send((request_id.clone(), pcm));
        }
    }

    fn push_frame(&self, frame: &AudioFrame) {
        let pcm: Vec<i16> = frame.data.as_chunks::<2>().0.iter().copied().map(i16::from_le_bytes).collect();
        self.buf.lock().expect("ring buffer mutex").push(&pcm);
    }

    fn flush(&self) {
        self.buf.lock().expect("ring buffer mutex").clear();
    }

    fn end_input(&self) {
        *self.tx.lock().expect("job sender mutex") = None;
    }

    fn detach(&self) {
        self.end_input();
    }

    async fn run(&self) -> Result<(), ApiError> {
        let Some(mut rx) = self.rx.lock().expect("job receiver mutex").take() else {
            return Ok(());
        };
        while let Some((request_id, pcm)) = rx.recv().await {
            let model = self.model.clone();
            let started = Instant::now();
            // `asyncio.to_thread(self._eot.predict, ...)`: CPU-bound, off the driver's task.
            let result = tokio::task::spawn_blocking(move || model.predict(&pcm)).await;
            let inference_duration = started.elapsed().as_secs_f64();

            // Any failure (panic included) scores 0.0 (`transports.py:421`): "no end of turn".
            let probability = match result {
                Ok(Ok(probability)) => probability,
                Ok(Err(reason)) => {
                    self.sink.warn(EotWarning::LocalPredictionFailed { reason });
                    0.0
                }
                Err(join) => {
                    self.sink.warn(EotWarning::LocalPredictionFailed { reason: join.to_string() });
                    0.0
                }
            };

            self.sink.resolve(
                request_id.as_str(),
                TransportPrediction {
                    probability,
                    detection_delay: None,
                    inference_duration: Some(inference_duration),
                    backchannel_probability: None,
                },
            );
        }
        Ok(())
    }
}
