//! Background decoding: one admitted native job owns reads, codec state and bounded PCM, from a
//! file, a finite buffer or bytes streamed in as they arrive. Pulls yield cooperatively;
//! cancellation retains native ownership through worker settlement.

use std::collections::VecDeque;
use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, TryLockError};
use std::time::Duration;

use lk_agents_abi::AudioFrame;
use lk_agents_abi::media::codecs::CodecFailure;
use lk_agents_abi::media::decoder::ByteDecoderOptions;
use lk_agents_media::decoder::DecodeLimits;
use lk_agents_media::{AudioFormat, CodecError, Decoder};
use tokio::sync::Notify;

use lk_agents_media::workers::{MediaWorker, MediaWorkerClass, MediaWorkerError, MediaWorkers};

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
    WorkerAdmission(#[from] MediaWorkerError),
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
    decoded_samples: usize,
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
    sample_limit: Option<usize>,
    /// A streaming decoder's input; cancellation closes it so a blocked read returns.
    input: Option<Arc<StreamInput>>,
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

    fn try_state(&self) -> Result<MutexGuard<'_, State>, ()> {
        match self.state.try_lock() {
            Ok(state) => Ok(state),
            Err(TryLockError::Poisoned(error)) => Ok(error.into_inner()),
            Err(TryLockError::WouldBlock) => Err(()),
        }
    }

    fn poll_output(&self) -> DecoderPoll {
        let mut state = match self.try_state() {
            Ok(state) => state,
            Err(_) => return DecoderPoll::Busy,
        };
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

    fn output(&self, format: AudioFormat, samples: &[i16]) -> Result<(), DecoderError> {
        if samples.is_empty() {
            return Ok(());
        }
        let channels = usize::from(format.num_channels);
        if channels == 0 {
            return Err(DecoderError::InvalidLimits);
        }
        let chunk_size = self.limits.output_chunk_samples / channels * channels;
        if chunk_size == 0 || samples.len() % channels != 0 {
            return Err(DecoderError::InvalidLimits);
        }
        loop {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if self.cancelled.load(Ordering::Acquire) {
                return Err(DecoderError::InputClosed);
            }
            if let Some(error) = state.error.as_ref() {
                return Err(error.clone());
            }
            if self
                .sample_limit
                .is_some_and(|limit| samples.len() > limit.saturating_sub(state.decoded_samples))
            {
                let error = DecoderError::Overflow("decoded duration");
                Self::fail(&mut state, error.clone());
                return Err(error);
            }
            if samples.len() <= self.limits.output_samples.saturating_sub(state.output_samples) {
                // Reserve worker-local prepared PCM before releasing the lock.
                state.output_samples += samples.len();
                state.decoded_samples += samples.len();
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
        let prepared = samples
            .chunks(chunk_size)
            .map(|chunk| {
                AudioFrame::new(
                    lk_agents_media::pcm_bytes(chunk),
                    format.sample_rate,
                    format.num_channels,
                    u32::try_from(chunk.len() / channels)
                        .expect("validated native frame size fits i32"),
                )
            })
            .collect::<VecDeque<_>>();
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
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
    worker: Option<MediaWorker<Result<(), DecoderError>>>,
}

impl AudioDecoder {
    pub fn open_file(
        workers: &MediaWorkers,
        path: String,
        looping: bool,
        sample_rate: u32,
        num_channels: u16,
        limits: DecoderLimits,
    ) -> Result<Self, DecoderError> {
        let decode = move |shared: &Arc<Shared>| {
            #[cfg(feature = "media-ffmpeg")]
            let mime = "";
            #[cfg(not(feature = "media-ffmpeg"))]
            let mime = if std::path::Path::new(&path)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("wav"))
            {
                "audio/wav"
            } else {
                ""
            };
            loop {
                if shared.cancelled.load(Ordering::Acquire) {
                    return Ok(());
                }
                let mut file = std::fs::File::open(&path)
                    .map_err(|error| DecoderError::WorkerFailed(error.to_string()))?;
                let mut file = Blocks(&mut file);
                let emitted =
                    decode_reader(shared, mime, Some(sample_rate), num_channels, &mut file)?;
                if !looping || !emitted {
                    return Ok(());
                }
            }
        };
        Self::start(workers, Some(sample_rate), num_channels, limits, None, None, decode)
    }

    /// A finite head uses the same admission, bounded output and cleanup receipt as file audio.
    pub fn open_bytes(
        workers: &MediaWorkers,
        options: ByteDecoderOptions,
        limits: DecoderLimits,
    ) -> Result<Self, DecoderError> {
        if options.max_input_bytes == 0 || options.max_duration.is_zero() {
            return Err(DecoderError::InvalidLimits);
        }
        if options.data.len() > options.max_input_bytes {
            return Err(DecoderError::Overflow("encoded bytes"));
        }
        let sample_limit = u128::from(options.sample_rate)
            .checked_mul(u128::from(options.num_channels))
            .and_then(|rate| rate.checked_mul(options.max_duration.as_nanos()))
            .and_then(|samples| usize::try_from(samples / 1_000_000_000).ok())
            .filter(|samples| *samples > 0)
            .ok_or(DecoderError::InvalidLimits)?;
        let sample_rate = options.sample_rate;
        let num_channels = options.num_channels;
        let target = Some(sample_rate);
        Self::start(
            workers,
            target,
            num_channels,
            limits,
            Some(sample_limit),
            None,
            move |shared| {
                if options.mime == "audio/pcm" || options.mime.is_empty() {
                    let (source_rate, source_channels) =
                        options.pcm_format.ok_or(DecoderError::InvalidLimits)?;
                    decode_pcm(
                        shared,
                        &options.data,
                        AudioFormat::new(source_rate, source_channels),
                        sample_rate,
                        num_channels,
                    )
                } else {
                    decode_reader(
                        shared,
                        &options.mime,
                        target,
                        num_channels,
                        &mut Blocks(std::io::Cursor::new(options.data)),
                    )
                    .map(|_| ())
                }
            },
        )
    }

    /// Bytes pushed as they arrive through the returned input, decoded by one admitted native
    /// job. `sample_rate` `None` keeps the source's rate. EOF follows `end_input` and delivery
    /// of everything it queued.
    pub fn open_stream(
        workers: &MediaWorkers,
        mime: &str,
        sample_rate: Option<u32>,
        num_channels: u16,
        limits: DecoderLimits,
    ) -> Result<(Self, DecoderInput), DecoderError> {
        #[cfg(not(feature = "media-ffmpeg"))]
        if lk_agents_media::mime_to_av_format(Some(mime)) != Some("wav") {
            return Err(CodecError::UnsupportedMime {
                mime: mime.into(),
                reason: "container decoding needs the runtime media-ffmpeg feature",
            }
            .into());
        }
        let input = Arc::new(StreamInput::default());
        let pending = input.clone();
        let mime = mime.to_owned();
        let decoder =
            Self::start(workers, sample_rate, num_channels, limits, None, Some(input.clone()), {
                move |shared| {
                    decode_reader(shared, &mime, sample_rate, num_channels, &mut &*pending)
                        .map(drop)
                }
            })?;
        let input = DecoderInput { shared: decoder.shared.clone(), input };
        Ok((decoder, input))
    }

    fn start(
        workers: &MediaWorkers,
        sample_rate: Option<u32>,
        num_channels: u16,
        limits: DecoderLimits,
        sample_limit: Option<usize>,
        input: Option<Arc<StreamInput>>,
        decode: impl FnOnce(&Arc<Shared>) -> Result<(), DecoderError> + Send + 'static,
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
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                output: VecDeque::new(),
                output_samples: 0,
                decoded_samples: 0,
                done: false,
                error: None,
            }),
            reader: OnceLock::new(),
            cancelled: AtomicBool::new(false),
            ready: Notify::new(),
            limits,
            sample_limit,
            input,
        });
        let resource = shared.clone();
        let worker = workers.try_spawn(MediaWorkerClass::Decoder, move || {
            let _ = resource.reader.set(std::thread::current());
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| decode(&resource)))
                    .unwrap_or_else(|_| {
                        Err(DecoderError::WorkerFailed("native decode panicked".into()))
                    });
            {
                let mut state = resource.state.lock().unwrap_or_else(|e| e.into_inner());
                state.done = true;
                if let Err(error) = &result {
                    if !resource.cancelled.load(Ordering::Acquire) {
                        Shared::fail(&mut state, error.clone());
                    }
                }
            }
            resource.ready.notify_one();
            // Retain the worker's ownership until accepted PCM transfers or cancellation retires
            // it. Large queue deallocation therefore happens here, never in a host-loop drop.
            loop {
                let mut state = resource.state.lock().unwrap_or_else(|e| e.into_inner());
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
    input: Arc<StreamInput>,
}

impl DecoderInput {
    /// Refused after EOF, cancellation or the end of decoding (whose failure the output side
    /// reports), and past [`MAX_PENDING_INPUT`].
    pub fn push(&self, data: Vec<u8>) -> Result<(), DecoderError> {
        if self.shared.state.lock().unwrap_or_else(|e| e.into_inner()).done {
            return Err(DecoderError::InputClosed);
        }
        self.input.push(data)
    }

    /// No more input: the decoder drains what was queued, then its output ends.
    pub fn end_input(&self) {
        self.input.state().ended = true;
        self.input.ready.notify_all();
    }

    /// Stop decoding now; the output side still observes settlement.
    pub fn cancel(&self) {
        self.shared.cancel();
    }
}

/// Encoded input as the native decoder reads it. A WAV stream restarts only at a chunk boundary
/// (providers concatenate files, one per push), so the WAV decoder takes chunks whole.
trait Chunks {
    /// The next chunk, `None` at the end of input.
    fn next_chunk(&mut self) -> std::io::Result<Option<Vec<u8>>>;
}

/// A file or a finite buffer, in 4 KiB chunks.
struct Blocks<R>(R);

impl<R: Read> Chunks for Blocks<R> {
    fn next_chunk(&mut self) -> std::io::Result<Option<Vec<u8>>> {
        let mut chunk = vec![0; 4096];
        let n = self.0.read(&mut chunk)?;
        chunk.truncate(n);
        Ok((n > 0).then_some(chunk))
    }
}

/// Pushed chunks waiting for the native decoder.
#[derive(Default)]
struct StreamInput {
    state: Mutex<Pending>,
    ready: Condvar,
}

#[derive(Default)]
struct Pending {
    chunks: VecDeque<Vec<u8>>,
    bytes: usize,
    ended: bool,
    closed: bool,
}

impl StreamInput {
    fn state(&self) -> MutexGuard<'_, Pending> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn push(&self, data: Vec<u8>) -> Result<(), DecoderError> {
        let mut pending = self.state();
        if pending.ended || pending.closed {
            return Err(DecoderError::InputClosed);
        }
        if pending.bytes.saturating_add(data.len()) > MAX_PENDING_INPUT {
            return Err(DecoderError::Overflow("encoded input"));
        }
        pending.bytes += data.len();
        pending.chunks.push_back(data);
        self.ready.notify_all();
        Ok(())
    }

    /// Discards what is queued; a blocked read returns the end of input.
    fn close(&self) {
        let mut pending = self.state();
        pending.closed = true;
        pending.chunks.clear();
        pending.bytes = 0;
        self.ready.notify_all();
    }
}

/// Blocks the native worker until a chunk arrives or input ends.
impl Chunks for &StreamInput {
    fn next_chunk(&mut self) -> std::io::Result<Option<Vec<u8>>> {
        let mut pending = self.state();
        loop {
            if pending.closed {
                return Ok(None);
            }
            if let Some(chunk) = pending.chunks.pop_front() {
                pending.bytes -= chunk.len();
                return Ok(Some(chunk));
            }
            if pending.ended {
                return Ok(None);
            }
            pending = self.ready.wait(pending).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// Chunks as the byte stream a demuxer reads.
#[cfg(feature = "media-ffmpeg")]
struct Joined<'a, C: ?Sized> {
    chunks: &'a mut C,
    chunk: Vec<u8>,
    read: usize,
}

#[cfg(feature = "media-ffmpeg")]
impl<C: Chunks + ?Sized> Read for Joined<'_, C> {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        while self.read == self.chunk.len() {
            match self.chunks.next_chunk()? {
                Some(chunk) => (self.chunk, self.read) = (chunk, 0),
                None => return Ok(0),
            }
        }
        let n = bytes.len().min(self.chunk.len() - self.read);
        bytes[..n].copy_from_slice(&self.chunk[self.read..self.read + n]);
        self.read += n;
        Ok(n)
    }
}

fn decode_reader(
    shared: &Arc<Shared>,
    mime: &str,
    sample_rate: Option<u32>,
    num_channels: u16,
    input: &mut dyn Chunks,
) -> Result<bool, DecoderError> {
    let mut emitted = false;
    if lk_agents_media::mime_to_av_format(Some(mime)) == Some("wav") {
        let mut decoder = Decoder::wav(sample_rate, whole_pushes(shared.limits.format))?;
        while let Some(chunk) =
            input.next_chunk().map_err(|e| DecoderError::WorkerFailed(e.to_string()))?
        {
            let pcm = decoder.push(&chunk)?;
            if let Some(format) = decoder.output_format() {
                let pcm = remix(&pcm, format.num_channels, num_channels)?;
                shared.output(AudioFormat::new(format.sample_rate, num_channels), &pcm)?;
                emitted |= !pcm.is_empty();
            }
        }
        let tail = decoder.finish()?;
        if let Some(format) = decoder.output_format() {
            let tail = remix(&tail, format.num_channels, num_channels)?;
            shared.output(AudioFormat::new(format.sample_rate, num_channels), &tail)?;
            emitted |= !tail.is_empty();
        }
        Ok(emitted)
    } else {
        #[cfg(feature = "media-ffmpeg")]
        {
            lk_agents_media::ffmpeg::decode_stream(
                lk_agents_media::mime_to_av_format(Some(mime)),
                sample_rate.unwrap_or(0),
                num_channels,
                shared.limits.format,
                &mut Joined { chunks: input, chunk: Vec::new(), read: 0 },
                |format, samples| {
                    emitted |= !samples.is_empty();
                    shared.output(format, samples).map_err(|e| CodecError::Ffmpeg {
                        op: "PCM delivery",
                        detail: e.to_string(),
                    })
                },
            )?;
            Ok(emitted)
        }
        #[cfg(not(feature = "media-ffmpeg"))]
        {
            let _ = input;
            Err(CodecError::UnsupportedMime {
                mime: mime.into(),
                reason: "container decoding needs the runtime media-ffmpeg feature",
            }
            .into())
        }
    }
}

fn remix(samples: &[i16], from: u16, to: u16) -> Result<Vec<i16>, DecoderError> {
    if from == to {
        return Ok(samples.to_vec());
    }
    if to == 1 {
        return Ok(lk_agents_media::downmix_mono(samples, from));
    }
    if from == 1 {
        return Ok(samples
            .iter()
            .flat_map(|sample| std::iter::repeat_n(*sample, usize::from(to)))
            .collect());
    }
    Err(DecoderError::InvalidLimits)
}

fn decode_pcm(
    shared: &Arc<Shared>,
    bytes: &[u8],
    source: AudioFormat,
    sample_rate: u32,
    num_channels: u16,
) -> Result<(), DecoderError> {
    shared.limits.format.validate(source, Some(sample_rate), 0)?;
    let width = usize::from(source.num_channels) * 2;
    if !bytes.len().is_multiple_of(width) {
        return Err(DecoderError::Codec(CodecError::DecodeLimit("truncated PCM sample")));
    }
    let mut resampler = if source.sample_rate == sample_rate {
        None
    } else {
        Some(
            lk_agents_media::new_resampler(
                source.sample_rate,
                sample_rate,
                u32::from(num_channels),
                lk_agents_media::Quality::default(),
            )
            .map_err(|error| DecoderError::WorkerFailed(error.to_string()))?,
        )
    };
    let output_format = AudioFormat::new(sample_rate, num_channels);
    // Ten milliseconds of source PCM bounds conversion and native resampling scratch space.
    let chunk_bytes = (source.sample_rate as usize / 100).max(1) * width;
    for chunk in bytes.chunks(chunk_bytes) {
        let samples = chunk
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect::<Vec<_>>();
        let samples = remix(&samples, source.num_channels, num_channels)?;
        if let Some(resampler) = &mut resampler {
            let output = resampler
                .push(&samples)
                .map_err(|error| DecoderError::WorkerFailed(error.to_string()))?;
            shared.output(output_format, output)?;
        } else {
            shared.output(output_format, &samples)?;
        }
    }
    if let Some(resampler) = &mut resampler {
        let tail =
            resampler.flush().map_err(|error| DecoderError::WorkerFailed(error.to_string()))?;
        shared.output(output_format, tail)?;
    }
    Ok(())
}

impl lk_agents_abi::media::decoder::FileDecoder for AudioDecoder {
    fn next(
        &mut self,
    ) -> lk_agents_abi::utils::spawner::BoxFuture<
        '_,
        Result<Option<lk_agents_abi::AudioFrame>, String>,
    > {
        Box::pin(async move { self.next().await.map_err(|error| error.to_string()) })
    }

    fn close(
        &mut self,
        timeout: Duration,
    ) -> lk_agents_abi::utils::spawner::BoxFuture<'_, Result<(), String>> {
        Box::pin(async move {
            match tokio::time::timeout(timeout, self.close()).await {
                Ok(closed) => closed.map_err(|error| error.to_string()),
                Err(_) => Err(DecoderError::IncompleteCleanup.to_string()),
            }
        })
    }
}

lk_agents_abi::inventory::submit! {
    lk_agents_abi::media::decoder::FileDecoderProvider {
        open: |options| {
            let lk_agents_abi::media::decoder::FileDecoderOptions {
                path, looping, sample_rate, num_channels,
            } = options;
            let workers = crate::process::workers().map_err(|error| error.to_string())?;
            let limits = decoder_limits(sample_rate, num_channels);
            AudioDecoder::open_file(&workers, path, looping, sample_rate, num_channels, limits)
                .map(|decoder| Box::new(decoder) as Box<dyn lk_agents_abi::media::decoder::FileDecoder>)
                .map_err(|error| error.to_string())
        },
        open_bytes,
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
    fn new(
        mime: String,
        sample_rate: Option<u32>,
        num_channels: u16,
    ) -> Result<Self, CodecFailure> {
        let workers = crate::process::workers().map_err(failure)?;
        let limits = decoder_limits(sample_rate.unwrap_or(48_000), num_channels);
        let (output, input) =
            AudioDecoder::open_stream(&workers, &mime, sample_rate, num_channels, limits)
                .map_err(failure)?;
        Ok(Self { input, output: tokio::sync::Mutex::new(output) })
    }

    fn push(&self, data: &[u8]) -> Result<(), CodecFailure> {
        self.input.push(data.to_vec()).map_err(failure)
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

pub use lk_agents_media::StreamBuffer;

#[cfg(feature = "python")]
pub use lk_agents_media::blingfire::MediaTokenization;

/// A WAV push decodes whole: the input that carried it bounds its size, and a provider's
/// concatenated files restart only at a push boundary.
pub(crate) fn whole_pushes(limits: DecodeLimits) -> DecodeLimits {
    DecodeLimits { max_samples_per_frame: i32::MAX as usize, ..limits }
}

pub(crate) fn open_bytes(
    options: ByteDecoderOptions,
) -> Result<Box<dyn lk_agents_abi::media::decoder::FileDecoder>, String> {
    let workers = crate::process::workers().map_err(|error| error.to_string())?;
    let limits = decoder_limits(options.sample_rate, options.num_channels);
    AudioDecoder::open_bytes(&workers, options, limits)
        .map(|decoder| Box::new(decoder) as Box<dyn lk_agents_abi::media::decoder::FileDecoder>)
        .map_err(|error| error.to_string())
}

/// Two seconds of queued PCM, delivered in 10 ms chunks.
pub(crate) fn decoder_limits(sample_rate: u32, num_channels: u16) -> DecoderLimits {

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
mod tests {
    use super::*;
    use lk_agents_media::workers::MediaWorkerLimits;

    fn workers() -> MediaWorkers {
        MediaWorkers::new(MediaWorkerLimits {
            total: 3,
            decoders: 1,
            recordings: 1,
            images: 1,
            formatting: 1,
            eot_models: 1,
            vad_models: 1,
        })
        .unwrap()
    }

    fn limits() -> DecoderLimits {
        DecoderLimits {
            format: DecodeLimits {
                min_sample_rate: 8000,
                max_sample_rate: 192000,
                max_channels: 8,
                max_streams: 8,
                max_expansion_ratio: 24,
                max_samples_per_frame: 65536,
            },
            output_samples: 16 * 1024,
            output_chunk_samples: 128,
        }
    }

    fn wav(channels: u16, samples: &[i16]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + samples.len() as u32 * 2).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&24000u32.to_le_bytes());
        bytes.extend_from_slice(&(48000 * u32::from(channels)).to_le_bytes());
        bytes.extend_from_slice(&(2 * channels).to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(samples.len() as u32 * 2).to_le_bytes());
        bytes.extend(lk_agents_media::pcm_bytes(samples));
        bytes
    }

    struct AudioFile(std::path::PathBuf);

    impl AudioFile {
        fn wav(name: &str, pcm: &[i16]) -> Self {
            let path =
                std::env::temp_dir().join(format!("lk-decoder-{name}-{}.wav", std::process::id()));
            std::fs::write(&path, wav(1, pcm)).unwrap();
            Self(path)
        }

        fn path(&self) -> String {
            self.0.to_str().unwrap().to_owned()
        }
    }

    impl Drop for AudioFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[tokio::test]
    async fn decoded_stereo_is_normalized_before_delivery_to_a_mono_input() {
        let file = AudioFile::wav("stereo-normalization", &[]);
        std::fs::write(&file.0, wav(2, &[100, 200].repeat(480))).unwrap();
        let mut decoder =
            AudioDecoder::open_file(&workers(), file.path(), false, 24_000, 1, limits()).unwrap();
        let mut samples = Vec::new();
        while let Some(frame) = decoder.next().await.unwrap() {
            assert_eq!((frame.sample_rate, frame.num_channels), (24_000, 1));
            samples.extend(frame.data.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])));
        }
        assert_eq!(samples, vec![150; 480]);
        decoder.close().await.unwrap();
    }

    fn byte_options(data: Vec<u8>) -> ByteDecoderOptions {
        ByteDecoderOptions {
            mime: "audio/pcm".into(),
            data,
            pcm_format: Some((24_000, 1)),
            sample_rate: 24_000,
            num_channels: 1,
            max_input_bytes: 2 * 1024 * 1024,
            max_duration: Duration::from_secs(10),
        }
    }

    #[tokio::test]
    async fn buffered_pcm_is_normalized_on_the_admitted_decoder_worker() {
        for (source_rate, target_rate) in
            [(24_000, 24_000), (48_000, 24_000), (24_000, 96_000), (24_000, 192_000)]
        {
            let options = ByteDecoderOptions {
                data: lk_agents_media::pcm_bytes(&[100, 200].repeat(source_rate / 10)),
                pcm_format: Some((source_rate as u32, 2)),
                sample_rate: target_rate,
                ..byte_options(Vec::new())
            };
            let mut decoder = AudioDecoder::open_bytes(&workers(), options, limits()).unwrap();
            let mut samples = Vec::new();
            while let Some(frame) = decoder.next().await.unwrap() {
                assert_eq!((frame.sample_rate, frame.num_channels), (target_rate, 1));
                samples
                    .extend(frame.data.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])));
            }
            assert_eq!(
                samples.len(),
                target_rate as usize / 10,
                "normalization must preserve the head's duration"
            );
            if source_rate as u32 == target_rate {
                assert_eq!(samples, vec![150; 2400]);
            }
            decoder.close().await.unwrap();
        }
    }

    #[tokio::test]
    async fn buffered_decoder_refuses_encoded_and_decoded_excess_without_delivering_it() {
        let options = ByteDecoderOptions { max_input_bytes: 4, ..byte_options(vec![0; 6]) };
        let workers = workers();
        assert!(matches!(
            AudioDecoder::open_bytes(&workers, options, limits()),
            Err(DecoderError::Overflow("encoded bytes"))
        ));
        let options = ByteDecoderOptions {
            max_duration: Duration::from_millis(10),
            ..byte_options(lk_agents_media::pcm_bytes(&vec![100; 480]))
        };
        let mut decoder = AudioDecoder::open_bytes(&workers, options, limits()).unwrap();
        let mut samples = 0;
        loop {
            match decoder.next().await {
                Ok(Some(frame)) => samples += frame.samples_per_channel,
                Err(DecoderError::Overflow("decoded duration")) => break,
                result => panic!("expected an explicit duration overflow, got {result:?}"),
            }
        }
        assert!(samples <= 240, "excess decoded audio must never enter the output queue");
        decoder.close().await.unwrap();
        let mut valid =
            AudioDecoder::open_bytes(&workers, byte_options(vec![0; 480]), limits()).unwrap();
        assert!(
            valid.next().await.unwrap().is_some(),
            "refused heads must release decoder admission"
        );
        valid.close().await.unwrap();
    }

    #[cfg(feature = "media-ffmpeg")]
    #[tokio::test]
    async fn compressed_heads_stop_before_retaining_audio_beyond_the_duration_limit() {
        let mut encoder = lk_agents_media::ffmpeg::OggOpusEncoder::new(48_000, 1, 24_000).unwrap();
        let mut data = encoder.push(&vec![100; 4800]).unwrap();
        data.extend(encoder.finish().unwrap());
        let options = ByteDecoderOptions {
            mime: "audio/ogg;codecs=opus".into(),
            data,
            pcm_format: None,
            max_duration: Duration::from_millis(20),
            ..byte_options(Vec::new())
        };
        let mut decoder = AudioDecoder::open_bytes(&workers(), options, limits()).unwrap();
        let mut samples = 0;
        loop {
            match decoder.next().await {
                Ok(Some(frame)) => samples += frame.samples_per_channel,
                Err(DecoderError::Overflow("decoded duration")) => break,
                result => panic!("compressed duration excess must be explicit: {result:?}"),
            }
        }
        assert!(samples <= 480);
        decoder.close().await.unwrap();
    }

    #[tokio::test]
    async fn background_file_service_preserves_audio_and_reports_file_and_codec_failures() {
        let pcm = (0..48_000).map(|sample| (sample % 16_000) as i16).collect::<Vec<_>>();
        let file = AudioFile::wav("service", &pcm);
        let mut decoder = lk_agents_abi::media::decoder::open(
            lk_agents_abi::media::decoder::FileDecoderOptions {
                path: file.path(),
                looping: false,
                sample_rate: 24_000,
                num_channels: 1,
            },
        )
        .unwrap();
        let mut actual = Vec::new();
        while let Some(frame) = decoder.next().await.unwrap() {
            assert_eq!((frame.sample_rate, frame.num_channels), (24_000, 1));
            actual.extend(
                frame.data.chunks_exact(2).map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]])),
            );
        }
        assert_eq!(actual, pcm);
        decoder.close(Duration::from_secs(1)).await.unwrap();
        let mut missing = lk_agents_abi::media::decoder::open(
            lk_agents_abi::media::decoder::FileDecoderOptions {
                path: format!("{}.missing", file.path()),
                looping: false,
                sample_rate: 24_000,
                num_channels: 1,
            },
        )
        .unwrap();
        assert!(missing.next().await.unwrap_err().contains("No such file"));
        missing.close(Duration::from_secs(1)).await.unwrap();

        let garbage = AudioFile::wav("garbage", &[]);
        std::fs::write(&garbage.0, b"not audio at all").unwrap();
        let mut unreadable = lk_agents_abi::media::decoder::open(
            lk_agents_abi::media::decoder::FileDecoderOptions {
                path: garbage.path(),
                looping: false,
                sample_rate: 24_000,
                num_channels: 1,
            },
        )
        .unwrap();
        let error = tokio::time::timeout(Duration::from_secs(1), unreadable.next())
            .await
            .expect("garbage input must settle its pull")
            .unwrap_err();
        #[cfg(feature = "media-ffmpeg")]
        assert!(error.starts_with("ffmpeg:"), "expected codec failure, got {error}");
        #[cfg(not(feature = "media-ffmpeg"))]
        assert!(error.starts_with("invalid WAV file:"), "expected codec failure, got {error}");
        unreadable.close(Duration::from_secs(1)).await.unwrap();
    }

    #[tokio::test]
    async fn a_stream_keeps_its_slot_until_cancellation_settles_then_refuses_input() {
        let workers = workers();
        let open = || AudioDecoder::open_stream(&workers, "audio/wav", Some(24_000), 1, limits());
        let (mut first, input) = open().unwrap();
        // A header alone leaves the native reader waiting for more input.
        input.push(wav(1, &[1; 480])[..44].to_vec()).unwrap();
        assert!(matches!(
            open(),
            Err(DecoderError::WorkerAdmission(MediaWorkerError::Exhausted(
                MediaWorkerClass::Decoder
            )))
        ));
        first.close().await.unwrap();
        assert!(matches!(input.push(vec![0; 4]), Err(DecoderError::InputClosed)));

        let (mut second, input) = open().unwrap();
        let encoded = wav(1, &[7; 480]);
        for chunk in encoded.chunks(37) {
            input.push(chunk.to_vec()).unwrap();
        }
        input.end_input();
        let mut samples = Vec::new();
        while let Some(frame) = second.next().await.unwrap() {
            samples.extend(frame.data.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])));
        }
        assert_eq!(samples, vec![7; 480]);
        assert!(matches!(input.push(vec![0; 4]), Err(DecoderError::InputClosed)));
        second.close().await.unwrap();
    }

    #[tokio::test]
    async fn looping_file_backpressure_keeps_its_slot_until_cancelled_decode_settles() {
        let pcm = (0..48_000).map(|sample| (sample % 16_000) as i16).collect::<Vec<_>>();
        let file = AudioFile::wav("looping", &pcm);
        let workers = workers();
        let limits = DecoderLimits { output_samples: 4096, ..limits() };
        let mut decoder =
            AudioDecoder::open_file(&workers, file.path(), true, 24_000, 1, limits).unwrap();
        let mut actual = Vec::new();
        while actual.len() < pcm.len() + 4096 {
            let frame = decoder.next().await.unwrap().unwrap();
            actual.extend(
                frame.data.chunks_exact(2).map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]])),
            );
        }
        assert_eq!(&actual[..pcm.len()], pcm);
        assert_eq!(&actual[pcm.len()..], &pcm[..actual.len() - pcm.len()]);
        assert!(matches!(
            AudioDecoder::open_file(&workers, file.path(), false, 24_000, 1, limits),
            Err(DecoderError::WorkerAdmission(MediaWorkerError::Exhausted(
                MediaWorkerClass::Decoder
            )))
        ));
        decoder.close().await.unwrap();
        let mut replacement =
            AudioDecoder::open_file(&workers, file.path(), false, 24_000, 1, limits).unwrap();
        replacement.close().await.unwrap();
    }

    #[cfg(unix)]
    fn thread_cpu() -> Duration {
        let mut time = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        // SAFETY: clock_gettime writes to a valid timespec for this thread's CPU clock.
        let status = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut time) };
        assert_eq!(status, 0, "thread CPU clock unavailable");
        Duration::new(time.tv_sec as u64, time.tv_nsec as u32)
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn ready_file_pcm_yields_to_control_without_dropping_audio() {
        let file = AudioFile::wav("fairness", &[1; 16_000]);
        let limits = DecoderLimits { output_chunk_samples: 1, ..limits() };
        let mut decoder =
            AudioDecoder::open_file(&workers(), file.path(), false, 24_000, 1, limits).unwrap();
        loop {
            let ready = decoder.shared.ready.notified();
            if decoder.shared.try_state().map(|state| state.done).unwrap_or(false) {
                break;
            }
            ready.await;
        }
        let (ready, mut control) = tokio::sync::oneshot::channel();
        let started = thread_cpu();
        let control_task = tokio::spawn(async move {
            let _ = ready.send(thread_cpu() - started);
        });
        let mut delivered = 0;
        let control_cpu = loop {
            assert!(decoder.next().await.unwrap().is_some(), "control must run before EOF");
            delivered += 1;
            if let Ok(elapsed) = control.try_recv() {
                break elapsed;
            }
        };
        assert!(
            control_cpu <= Duration::from_millis(2),
            "ready control exceeded the 2ms synchronous CPU budget: {control_cpu:?}",
        );
        while decoder.next().await.unwrap().is_some() {
            delivered += 1;
        }
        assert_eq!(delivered, 16_000);
        decoder.close().await.unwrap();
        control_task.await.unwrap();
    }
}
