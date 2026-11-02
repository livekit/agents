//! `_LocalTransport` (`transports.py:398`): the in-process mini model behind [`LocalEotModel`], a
//! 1.2 s ring buffer snapshotted at request time, and a failed prediction scored 0.0.

use parking_lot::Mutex;
use std::time::Instant;

use async_trait::async_trait;
use lk_agents_abi::errors::ApiError;
use lk_agents_media::pool::JobClass;
use lk_agents_media::pool::MediaPool;
use tokio::sync::mpsc;

use lk_agents_abi::AudioFrame;

use crate::eot::session::TransportPrediction;
use crate::eot::transport::{EotTransport, PredictionListener};
use crate::eot::{DEFAULT_SAMPLE_RATE, EotRequestId, EotWarning};

/// `_CLIENT_BUFFER_SECONDS` (`transports.py:70`). How much recent audio one prediction sees.
pub const CLIENT_BUFFER_SECONDS: f64 = 1.2;

/// `_CLIENT_BUFFER_SAMPLES`.
pub const CLIENT_BUFFER_SAMPLES: usize =
    (CLIENT_BUFFER_SECONDS * DEFAULT_SAMPLE_RATE as f64) as usize;

/// The mini model. Blocking; the transport runs it on the process's media worker pool.
pub trait LocalEotModel: Send + Sync + 'static {
    /// End-of-turn probability for `pcm`, 16 kHz mono. `Err` is the model raising; scored 0.0.
    fn predict(&self, pcm: &[i16]) -> Result<f64, String>;
}

/// Recent mono samples, already resampled by the host. An oversized frame keeps its newest
/// `capacity` samples instead of raising.
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
    media_pool: MediaPool,
    sink: std::sync::Arc<dyn PredictionListener>,
    buf: Mutex<AudioRingBuffer>,
    /// `None` once the input has ended, which is what lets [`run`](EotTransport::run) return.
    tx: Mutex<Option<mpsc::UnboundedSender<Job>>>,
    rx: Mutex<Option<mpsc::UnboundedReceiver<Job>>>,
}

impl LocalTransport {
    pub fn new(
        model: std::sync::Arc<dyn LocalEotModel>,
        sink: std::sync::Arc<dyn PredictionListener>,
        media_pool: MediaPool,
    ) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        LocalTransport {
            model,
            media_pool,
            sink,
            buf: Mutex::new(AudioRingBuffer::new(CLIENT_BUFFER_SAMPLES)),
            tx: Mutex::new(Some(tx)),
            rx: Mutex::new(Some(rx)),
        }
    }
}

impl LocalTransport {
    /// A slot on the media pool, waiting for one while the pool is full.
    async fn admit(&self) -> lk_agents_media::pool::Permit<JobClass> {
        loop {
            match self.media_pool.try_admit(JobClass::EotModel) {
                Ok(permit) => return permit,
                Err(_) => self.media_pool.wait_for_capacity(JobClass::EotModel).await,
            }
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
        let pcm = self.buf.lock().read();
        if let Some(tx) = self.tx.lock().as_ref() {
            let _ = tx.send((request_id.clone(), pcm));
        }
    }

    fn push_frame(&self, frame: &AudioFrame) {
        let pcm = lk_agents_media::pcm_samples(&frame.data);
        self.buf.lock().push(&pcm);
    }

    fn flush(&self) {
        self.buf.lock().clear();
    }

    fn end_input(&self) {
        *self.tx.lock() = None;
    }

    fn detach(&self) {
        self.end_input();
    }

    async fn run(&self) -> Result<(), ApiError> {
        let Some(mut rx) = self.rx.lock().take() else {
            return Ok(());
        };
        while let Some((request_id, pcm)) = rx.recv().await {
            let model = self.model.clone();
            let started = Instant::now();
            // CPU-bound, on an admitted media worker; a full pool delays the prediction.
            let permit = self.admit().await;
            let result = self
                .media_pool
                .spawn_admitted(permit, move || model.predict(&pcm))
                .await
                .unwrap_or_else(|panicked| Err(panicked.to_string()));
            let inference_duration = started.elapsed().as_secs_f64();

            // Any failure (panic included) scores 0.0 (`transports.py:421`): "no end of turn".
            let probability = result.unwrap_or_else(|reason| {
                self.sink.warn(EotWarning::LocalPredictionFailed { reason });
                0.0
            });

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
