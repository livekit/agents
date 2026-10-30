//! Background decoding: one admitted native job runs a `lk_agents_media::source` decode and hands
//! its frames to an async consumer through a bounded queue. Pulls yield cooperatively;
//! cancellation retains native ownership through worker settlement. The pool is the process's,
//! lent through [`ProcessEnv`](lk_agents_net::env::ProcessEnv).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use lk_agents_abi::AudioFrame;
use lk_agents_abi::media::codecs::CodecFailure;
use lk_agents_media::decoder::DecodeLimits;
use lk_agents_media::pool::{Job, JobClass, MediaPool, MediaPoolError};
use lk_agents_media::source::{self, Source};
use lk_agents_media::{AudioFormat, CodecError, StreamBuffer};
use lk_agents_net::lock::Mutex;
use tokio::sync::Notify;

use lk_agents_net::env::ProcessEnv;

/// Streamed bytes the native decoder has not read yet; a push beyond it is refused, not queued.
const MAX_PENDING_INPUT: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub struct DecoderLimits {
    pub format: DecodeLimits,
    pub output_samples: usize,
    pub output_chunk_samples: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DecoderError {
    #[error(
        "decoder limits and audio format must be positive, and a PCM chunk must fit output capacity"
    )]
    InvalidLimits,
    #[error(transparent)]
    WorkerAdmission(#[from] MediaPoolError),
    #[error("decoder input is closed")]
    InputClosed,
    #[error("decoder {0} capacity exceeded")]
    Overflow(&'static str),
    #[error(transparent)]
    Codec(#[from] CodecError),
    #[error("decoder worker failed: {0}")]
    WorkerFailed(String),
    #[error("decoder cleanup did not settle before its deadline")]
    IncompleteCleanup,
}

struct State {
    output: VecDeque<AudioFrame>,
    output_samples: usize,
    done: bool,
    error: Option<DecoderError>,
}

enum DecoderPoll {
    Audio(AudioFrame),
    Pending,
    Finished(Result<(), DecoderError>),
    Busy,
}

struct Shared {
    state: Mutex<State>,
    reader: OnceLock<std::thread::Thread>,
    cancelled: AtomicBool,
    ready: Notify,
    limits: DecoderLimits,
    /// A streaming decoder's input; cancellation closes it so a blocked read returns.
    input: Option<Arc<StreamBuffer>>,
}

impl Shared {
    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        if let Some(input) = &self.input {
            input.close();
        }
        self.wake_reader();
        self.ready.notify_one();
    }

    fn fail(state: &mut State, error: DecoderError) {
        if state.error.is_none() {
            state.error = Some(error);
        }
    }

    fn poll_output(&self) -> DecoderPoll {
        let Some(mut state) = self.state.try_lock() else { return DecoderPoll::Busy };
        if let Some(audio) = state.output.pop_front() {
            state.output_samples -= audio.data.len() / 2;
            if state.output.is_empty() {
                self.wake_reader();
            }
            return DecoderPoll::Audio(audio);
        }
        if state.done {
            return DecoderPoll::Finished(state.error.clone().map_or(Ok(()), Err));
        }
        DecoderPoll::Pending
    }

    fn wake_reader(&self) {
        if let Some(reader) = self.reader.get() {
            reader.unpark();
        }
    }

    /// Queues decoded PCM as frames, parking the worker while the queue is full.
    fn output(&self, format: AudioFormat, samples: &[i16]) -> Result<(), DecoderError> {
        if samples.is_empty() {
            return Ok(());
        }
        if format.num_channels == 0 || samples.len() % usize::from(format.num_channels) != 0 {
            return Err(DecoderError::InvalidLimits);
        }
        loop {
            let mut state = self.state.lock();
            if self.cancelled.load(Ordering::Acquire) {
                return Err(DecoderError::InputClosed);
            }
            if let Some(error) = state.error.as_ref() {
                return Err(error.clone());
            }
            if samples.len() <= self.limits.output_samples.saturating_sub(state.output_samples) {
                // Reserve worker-local prepared PCM before releasing the lock.
                state.output_samples += samples.len();
                break;
            }
            if samples.len() <= self.limits.output_samples {
                drop(state);
                std::thread::park();
                continue;
            }
            let error = DecoderError::Overflow("PCM output");
            Self::fail(&mut state, error.clone());
            self.wake_reader();
            self.ready.notify_one();
            return Err(error);
        }
        let prepared = source::frames(format, samples, self.limits.output_chunk_samples);
        let prepared = prepared.collect::<Vec<_>>();
        let mut state = self.state.lock();
        if self.cancelled.load(Ordering::Acquire) {
            return Err(DecoderError::InputClosed);
        }
        if let Some(error) = state.error.clone() {
            state.output_samples -= samples.len();
            return Err(error);
        }
        state.output.extend(prepared);
        drop(state);
        self.ready.notify_one();
        // Give the host and controls scheduling opportunities between native decode chunks.
        std::thread::yield_now();
        Ok(())
    }
}

/// One admitted decoder. EOF follows native completion and delivery of all decoded PCM.
pub struct AudioDecoder {
    shared: Arc<Shared>,
    worker: Option<Job<Result<(), DecoderError>>>,
}

impl AudioDecoder {
    pub fn open_file(
        media_pool: &MediaPool,
        path: String,
        looping: bool,
        sample_rate: u32,
        num_channels: u16,
        limits: DecoderLimits,
    ) -> Result<Self, DecoderError> {
        let source = Source::File { path, looping };
        Self::start(media_pool, source, Some(sample_rate), num_channels, limits)
    }

    /// Bytes pushed as they arrive through the returned input, decoded by one admitted native
    /// job. `sample_rate` `None` keeps the source's rate. EOF follows `end_input` and delivery
    /// of everything it queued.
    pub fn open_stream(
        media_pool: &MediaPool,
        mime: &str,
        sample_rate: Option<u32>,
        num_channels: u16,
        limits: DecoderLimits,
    ) -> Result<(Self, DecoderInput), DecoderError> {
        source::check_stream(mime)?;
        let input = Arc::new(StreamBuffer::new());
        let source = Source::Stream { mime: mime.to_owned(), input: input.clone() };
        let decoder = Self::start(media_pool, source, sample_rate, num_channels, limits)?;
        let shared = decoder.shared.clone();
        let input = DecoderInput { shared, pipe: input, ended: AtomicBool::new(false) };
        Ok((decoder, input))
    }

    fn start(
        media_pool: &MediaPool,
        source: Source,
        sample_rate: Option<u32>,
        num_channels: u16,
        limits: DecoderLimits,
    ) -> Result<Self, DecoderError> {
        if !limits.format.is_valid()
            || limits.output_samples == 0
            || limits.output_chunk_samples == 0
            || limits.output_chunk_samples > limits.output_samples
            || num_channels == 0
            || num_channels > limits.format.max_channels
            || sample_rate.is_some_and(|rate| {
                !(limits.format.min_sample_rate..=limits.format.max_sample_rate).contains(&rate)
            })
            || limits.output_chunk_samples % usize::from(num_channels) != 0
        {
            return Err(DecoderError::InvalidLimits);
        }
        let input = match &source {
            Source::Stream { input, .. } => Some(input.clone()),
            _ => None,
        };
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                output: VecDeque::new(),
                output_samples: 0,
                done: false,
                error: None,
            }),
            reader: OnceLock::new(),
            cancelled: AtomicBool::new(false),
            ready: Notify::new(),
            limits,
            input,
        });
        let resource = shared.clone();
        let worker = media_pool.try_spawn(JobClass::Decoder, move || {
            let _ = resource.reader.set(std::thread::current());
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                source::decode(
                    source,
                    sample_rate,
                    num_channels,
                    resource.limits.format,
                    &|| resource.cancelled.load(Ordering::Acquire),
                    &mut |format, samples| resource.output(format, samples),
                )
            }))
            .unwrap_or_else(|_| Err(DecoderError::WorkerFailed("native decode panicked".into())));
            {
                let mut state = resource.state.lock();
                state.done = true;
                if let Err(error) = &result
                    && !resource.cancelled.load(Ordering::Acquire)
                {
                    Shared::fail(&mut state, error.clone());
                }
            }
            resource.ready.notify_one();
            // Retain the worker's ownership until accepted PCM transfers or cancellation retires
            // it. Large queue deallocation therefore happens here, never in a host-loop drop.
            loop {
                let mut state = resource.state.lock();
                if resource.cancelled.load(Ordering::Acquire) {
                    state.output.clear();
                    state.output_samples = 0;
                    break;
                }
                if state.output.is_empty() {
                    break;
                }
                drop(state);
                std::thread::park();
            }
            result
        })?;
        Ok(Self { shared, worker: Some(worker) })
    }

    pub async fn next(&mut self) -> Result<Option<AudioFrame>, DecoderError> {
        // Each ready pull is its own dispatch; a large decoded backlog must not monopolize controls.
        tokio::task::yield_now().await;
        loop {
            let ready = self.shared.ready.notified();
            match self.shared.poll_output() {
                DecoderPoll::Audio(audio) => return Ok(Some(audio)),
                DecoderPoll::Busy => {
                    drop(ready);
                    tokio::task::yield_now().await;
                }
                DecoderPoll::Finished(outcome) => {
                    drop(ready);
                    let settled = self.settle().await;
                    outcome?;
                    settled?;
                    return Ok(None);
                }
                DecoderPoll::Pending => ready.await,
            }
        }
    }

    async fn settle(&mut self) -> Result<(), DecoderError> {
        if let Some(worker) = self.worker.as_mut() {
            let result = worker.await;
            self.worker = None;
            result.map_err(|e| DecoderError::WorkerFailed(e.to_string()))??;
        }
        Ok(())
    }

    /// Cancel and observe native settlement; the worker's admission is released by then.
    pub async fn close(&mut self) -> Result<(), DecoderError> {
        self.shared.cancel();
        if let Some(worker) = self.worker.as_mut() {
            let settled = worker.await;
            self.worker = None;
            // A cancelled decode's own outcome is not a close failure.
            let _ = settled.map_err(|error| DecoderError::WorkerFailed(error.to_string()))?;
        }
        Ok(())
    }
}

impl Drop for AudioDecoder {
    fn drop(&mut self) {
        // The admitted worker retains both its permit and shared state until native teardown.
        // Drop only requests cancellation; an explicit close observes the cleanup receipt.
        self.shared.cancel();
    }
}

/// A streaming decoder's encoded input. Pushing never waits for the native decoder.
pub struct DecoderInput {
    shared: Arc<Shared>,
    pipe: Arc<StreamBuffer>,
    ended: AtomicBool,
}

impl DecoderInput {
    /// Refused after EOF, cancellation or the end of decoding (whose failure the output side
    /// reports), and past [`MAX_PENDING_INPUT`].
    pub fn push(&self, data: &[u8]) -> Result<(), DecoderError> {
        if self.ended.load(Ordering::Acquire)
            || self.shared.cancelled.load(Ordering::Acquire)
            || self.shared.state.lock().done
        {
            return Err(DecoderError::InputClosed);
        }
        if self.pipe.len().saturating_add(data.len()) > MAX_PENDING_INPUT {
            return Err(DecoderError::Overflow("encoded input"));
        }
        self.pipe.write(data);
        Ok(())
    }

    /// No more input: the decoder drains what was queued, then its output ends.
    pub fn end_input(&self) {
        self.ended.store(true, Ordering::Release);
        self.pipe.end_input();
    }

    /// Stop decoding now; the output side still observes settlement.
    pub fn cancel(&self) {
        self.shared.cancel();
    }
}

/// `_core.AudioDecoder`, the native half of `utils.codecs.AudioStreamDecoder`: input is queued
/// without waiting, frames are awaited, and EOF never blocks the host loop.
pub struct StreamDecoder {
    input: DecoderInput,
    output: tokio::sync::Mutex<AudioDecoder>,
}

fn failure(error: impl std::fmt::Display) -> CodecFailure {
    CodecFailure(error.to_string())
}

impl lk_agents_abi::media::codecs::AudioDecoder for StreamDecoder {
    type Env = ProcessEnv;

    fn new(
        _env: &ProcessEnv,
        mime: String,
        sample_rate: Option<u32>,
        num_channels: u16,
    ) -> Result<Self, CodecFailure> {
        let media_pool = lk_agents_media::pool::media_pool();
        let limits = decoder_limits(sample_rate.unwrap_or(48_000), num_channels);
        let (output, input) =
            AudioDecoder::open_stream(&media_pool, &mime, sample_rate, num_channels, limits)
                .map_err(failure)?;
        Ok(Self { input, output: tokio::sync::Mutex::new(output) })
    }

    fn push(&self, data: &[u8]) -> Result<(), CodecFailure> {
        self.input.push(data).map_err(failure)
    }

    fn end_input(&self) {
        self.input.end_input();
    }

    async fn next(&self) -> Result<Option<AudioFrame>, CodecFailure> {
        self.output.lock().await.next().await.map_err(failure)
    }

    async fn aclose(&self) -> Result<(), CodecFailure> {
        // Cancelling first ends a pending `next`, which releases the output to close it.
        self.input.cancel();
        self.output.lock().await.close().await.map_err(failure)
    }
}

/// Two seconds of queued PCM, delivered in 10 ms chunks.
pub fn decoder_limits(sample_rate: u32, num_channels: u16) -> DecoderLimits {
    let output_samples =
        (sample_rate as usize).saturating_mul(usize::from(num_channels)).saturating_mul(2);
    DecoderLimits {
        format: DecodeLimits {
            min_sample_rate: 8_000,
            max_sample_rate: 192_000,
            max_channels: 8,
            max_streams: 8,
            max_expansion_ratio: 24,
            max_samples_per_frame: output_samples,
        },
        output_samples,
        output_chunk_samples: (sample_rate as usize / 100).max(1) * usize::from(num_channels),
    }
}

#[cfg(test)]
mod tests;
