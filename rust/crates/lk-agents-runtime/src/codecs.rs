//! Streaming decode resources. The runtime owns admission, AVIO waits, bounded input/PCM queues
//! and native settlement; media receives only supplied chunks or synchronous codec callbacks.

use std::collections::VecDeque;
use std::io::{self, Read};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, TryLockError};
use std::time::Duration;

use lk_agents_media::decoder::DecodeLimits;
use lk_agents_media::{AudioFormat, CodecError, Decoder};
use tokio::sync::Notify;

use crate::media_workers::{MediaWorker, MediaWorkerClass, MediaWorkerError, MediaWorkers};

#[derive(Debug, Clone, Copy)]
pub struct DecoderLimits {
    pub format: DecodeLimits,
    pub input_bytes: usize,
    pub input_chunks: usize,
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
    #[error("decoder ingress is busy; input was not accepted")]
    Busy,
    #[error("decoder {0} capacity exceeded")]
    Overflow(&'static str),
    #[error(transparent)]
    Codec(#[from] CodecError),
    #[error("decoder worker failed: {0}")]
    WorkerFailed(String),
    #[error("decoder cleanup did not settle before its deadline")]
    IncompleteCleanup,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedAudio {
    pub data: Vec<u8>,
    pub sample_rate: u32,
    pub num_channels: u16,
}

struct State {
    input: VecDeque<Vec<u8>>,
    input_offset: usize,
    input_bytes: usize,
    eof: bool,
    output: VecDeque<DecodedAudio>,
    output_samples: usize,
    done: bool,
    error: Option<DecoderError>,
}

enum DecoderPoll {
    Audio(DecodedAudio),
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
}

impl Shared {
    fn fail(state: &mut State, error: DecoderError) {
        if state.error.is_none() {
            state.error = Some(error);
        }
        state.eof = true;
    }

    fn try_state(&self) -> Result<MutexGuard<'_, State>, DecoderError> {
        match self.state.try_lock() {
            Ok(state) => Ok(state),
            Err(TryLockError::Poisoned(error)) => Ok(error.into_inner()),
            Err(TryLockError::WouldBlock) => Err(DecoderError::Busy),
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
        {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if self.cancelled.load(Ordering::Acquire) {
                return Err(DecoderError::InputClosed);
            }
            if let Some(error) = state.error.as_ref() {
                return Err(error.clone());
            }
            if samples.len() > self.limits.output_samples.saturating_sub(state.output_samples) {
                let error = DecoderError::Overflow("PCM output");
                Self::fail(&mut state, error.clone());
                self.wake_reader();
                self.ready.notify_one();
                return Err(error);
            }
            // Count worker-local prepared PCM as retained output before releasing the lock.
            state.output_samples += samples.len();
        }
        let prepared = samples
            .chunks(chunk_size)
            .map(|chunk| DecodedAudio {
                data: lk_agents_media::pcm_bytes(chunk),
                sample_rate: format.sample_rate,
                num_channels: format.num_channels,
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
        Ok(())
    }
}

struct Input(Arc<Shared>);

impl Read for Input {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if output.is_empty() {
            return Ok(0);
        }
        loop {
            let mut state = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
            if self.0.cancelled.load(Ordering::Acquire) || state.error.is_some() {
                return Err(io::Error::new(io::ErrorKind::Interrupted, "decoder input cancelled"));
            }
            if let Some(bytes) = state.input.front() {
                let n = output.len().min(4096).min(bytes.len() - state.input_offset);
                output[..n].copy_from_slice(&bytes[state.input_offset..state.input_offset + n]);
                state.input_offset += n;
                if state.input_offset == state.input.front().expect("present").len() {
                    state.input_bytes -= state.input.front().expect("present").capacity();
                    state.input.pop_front();
                    state.input_offset = 0;
                }
                return Ok(n);
            }
            if state.eof {
                return Ok(0);
            }
            drop(state);
            std::thread::park();
        }
    }
}

/// One admitted decode stream. Ingress never waits for native work; EOF is reported only after
/// successful native completion and delivery of every accepted PCM chunk.
pub struct AudioDecoder {
    shared: Arc<Shared>,
    worker: Option<MediaWorker<Result<(), DecoderError>>>,
}

impl AudioDecoder {
    pub fn open(
        workers: &MediaWorkers,
        mime: &str,
        sample_rate: Option<u32>,
        num_channels: u16,
        limits: DecoderLimits,
    ) -> Result<Self, DecoderError> {
        if !limits.format.is_valid()
            || limits.input_chunks == 0
            || limits.input_bytes == 0
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
                input: VecDeque::new(),
                input_offset: 0,
                input_bytes: 0,
                eof: false,
                output: VecDeque::new(),
                output_samples: 0,
                done: false,
                error: None,
            }),
            reader: OnceLock::new(),
            cancelled: AtomicBool::new(false),
            ready: Notify::new(),
            limits,
        });
        let mime = mime.to_owned();
        let resource = shared.clone();
        let worker = workers.try_spawn(MediaWorkerClass::Decoder, move || {
            let _ = resource.reader.set(std::thread::current());
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                decode(&resource, &mime, sample_rate, num_channels)
            }))
            .unwrap_or_else(|_| Err(DecoderError::WorkerFailed("native decode panicked".into())));
            {
                let mut state = resource.state.lock().unwrap_or_else(|e| e.into_inner());
                state.done = true;
                state.input.clear();
                state.input_bytes = 0;
                state.input_offset = 0;
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

    /// Move a chunk only after admission. Busy leaves the caller's buffer unchanged for retry.
    pub fn try_push(&self, data: &mut Vec<u8>) -> Result<(), DecoderError> {
        let mut state = self.shared.try_state()?;
        if let Some(error) = &state.error {
            return Err(error.clone());
        }
        if state.eof || self.shared.cancelled.load(Ordering::Acquire) || state.done {
            return Err(DecoderError::InputClosed);
        }
        if data.capacity() > self.shared.limits.input_bytes.saturating_sub(state.input_bytes)
            || (!data.is_empty() && state.input.len() >= self.shared.limits.input_chunks)
        {
            let error = DecoderError::Overflow("compressed input");
            Shared::fail(&mut state, error.clone());
            drop(state);
            self.shared.wake_reader();
            self.shared.ready.notify_one();
            return Err(error);
        }
        if !data.is_empty() {
            state.input_bytes += data.capacity();
            state.input.push_back(std::mem::take(data));
        }
        drop(state);
        self.shared.wake_reader();
        Ok(())
    }

    pub fn end_input(&self) -> Result<(), DecoderError> {
        let mut state = self.shared.try_state()?;
        if let Some(error) = &state.error {
            return Err(error.clone());
        }
        if self.shared.cancelled.load(Ordering::Acquire) {
            return Err(DecoderError::InputClosed);
        }
        state.eof = true;
        drop(state);
        self.shared.wake_reader();
        Ok(())
    }

    pub async fn next(&mut self) -> Result<Option<DecodedAudio>, DecoderError> {
        tokio::task::consume_budget().await;
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

    pub async fn close(&mut self, timeout: Duration) -> Result<(), DecoderError> {
        self.cancel();
        if let Some(worker) = self.worker.as_mut() {
            match tokio::time::timeout(timeout, worker).await {
                Err(_) => return Err(DecoderError::IncompleteCleanup),
                Ok(Err(error)) => {
                    self.worker = None;
                    return Err(DecoderError::WorkerFailed(error.to_string()));
                }
                Ok(Ok(_)) => self.worker = None,
            }
        }
        Ok(())
    }

    fn cancel(&self) {
        self.shared.cancelled.store(true, Ordering::Release);
        self.shared.wake_reader();
        self.shared.ready.notify_one();
    }
}

impl Drop for AudioDecoder {
    fn drop(&mut self) {
        // The admitted worker retains both its permit and shared state until native teardown.
        // Drop only requests cancellation; an explicit close observes the cleanup receipt.
        self.cancel();
    }
}

fn decode(
    shared: &Arc<Shared>,
    mime: &str,
    sample_rate: Option<u32>,
    num_channels: u16,
) -> Result<(), DecoderError> {
    let mut input = Input(shared.clone());
    if lk_agents_media::mime_to_av_format(Some(mime)) == Some("wav") {
        let mut decoder =
            Decoder::new_bounded(mime, sample_rate, num_channels, shared.limits.format)?;
        let mut bytes = vec![0; 4096.min(shared.limits.input_bytes)];
        loop {
            let n =
                input.read(&mut bytes).map_err(|e| DecoderError::WorkerFailed(e.to_string()))?;
            if n == 0 {
                break;
            }
            let pcm = decoder.push(&bytes[..n])?;
            if let Some(format) = decoder.output_format() {
                shared.output(format, &pcm)?;
            }
        }
        let tail = decoder.finish()?;
        if let Some(format) = decoder.output_format() {
            shared.output(format, &tail)?;
        }
        Ok(())
    } else {
        #[cfg(feature = "media-ffmpeg")]
        {
            lk_agents_media::ffmpeg::decode_stream(
                lk_agents_media::mime_to_av_format(Some(mime)),
                sample_rate.unwrap_or(0),
                num_channels,
                shared.limits.format,
                &mut input,
                |format, samples| {
                    shared.output(format, samples).map_err(|e| CodecError::Ffmpeg {
                        op: "PCM delivery",
                        detail: e.to_string(),
                    })
                },
            )?;
            Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media_workers::MediaWorkerLimits;

    async fn push(decoder: &AudioDecoder, mut data: Vec<u8>) -> Result<(), DecoderError> {
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                match decoder.try_push(&mut data) {
                    Err(DecoderError::Busy) => tokio::task::yield_now().await,
                    result => return result,
                }
            }
        })
        .await
        .map_err(|_| DecoderError::Busy)?
    }

    async fn end_input(decoder: &AudioDecoder) -> Result<(), DecoderError> {
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                match decoder.end_input() {
                    Err(DecoderError::Busy) => tokio::task::yield_now().await,
                    result => return result,
                }
            }
        })
        .await
        .map_err(|_| DecoderError::Busy)?
    }

    fn workers() -> MediaWorkers {
        MediaWorkers::new(
            tokio::runtime::Handle::current(),
            MediaWorkerLimits { total: 3, decoders: 1, recordings: 1, images: 1 },
        )
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
            input_bytes: 16 * 1024,
            input_chunks: 1024,
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

    #[tokio::test]
    async fn input_end_drains_accepted_pcm_before_eof_and_rejects_late_input() {
        let mut decoder =
            AudioDecoder::open(&workers(), "audio/wav", Some(24000), 1, limits()).unwrap();
        let samples = (0..2000).map(|n| n as i16).collect::<Vec<_>>();
        for chunk in wav(1, &samples).chunks(37) {
            push(&decoder, chunk.to_vec()).await.unwrap();
        }
        end_input(&decoder).await.unwrap();
        assert_eq!(push(&decoder, vec![0; 2]).await, Err(DecoderError::InputClosed));
        let mut pcm = Vec::new();
        while let Some(frame) = decoder.next().await.unwrap() {
            assert_eq!((frame.sample_rate, frame.num_channels), (24000, 1));
            assert!(frame.data.len() <= 256);
            pcm.extend(frame.data);
        }
        assert_eq!(pcm, lk_agents_media::pcm_bytes(&samples));
        decoder.close(Duration::from_secs(1)).await.unwrap();
    }

    #[tokio::test]
    async fn ragged_stereo_input_never_splits_a_channel_frame() {
        let mut decoder =
            AudioDecoder::open(&workers(), "audio/wav", Some(24000), 2, limits()).unwrap();
        let samples = [1, -1, 2, -2, 3, -3, 4, -4];
        for chunk in wav(2, &samples).chunks(7) {
            push(&decoder, chunk.to_vec()).await.unwrap();
        }
        end_input(&decoder).await.unwrap();
        let mut pcm = Vec::new();
        while let Some(frame) = decoder.next().await.unwrap() {
            assert_eq!(frame.num_channels, 2);
            assert_eq!(frame.data.len() % 4, 0);
            pcm.extend(frame.data);
        }
        assert_eq!(pcm, lk_agents_media::pcm_bytes(&samples));
        decoder.close(Duration::from_secs(1)).await.unwrap();
    }

    #[tokio::test]
    async fn malformed_audio_is_a_decode_error_instead_of_normal_eof() {
        let mut decoder =
            AudioDecoder::open(&workers(), "audio/wav", Some(24000), 1, limits()).unwrap();
        push(&decoder, b"this is not a wave file".to_vec()).await.unwrap();
        end_input(&decoder).await.unwrap();
        assert!(matches!(
            decoder.next().await,
            Err(DecoderError::Codec(CodecError::MissingRiffWave(_)))
        ));
        decoder.close(Duration::from_secs(1)).await.unwrap();
    }

    #[tokio::test]
    async fn truncated_wave_is_an_error_instead_of_normal_eof() {
        for bytes in [b"RIF".to_vec(), wav(1, &[1])[..45].to_vec()] {
            let mut decoder =
                AudioDecoder::open(&workers(), "audio/wav", Some(24000), 1, limits()).unwrap();
            push(&decoder, bytes).await.unwrap();
            end_input(&decoder).await.unwrap();
            assert!(
                decoder.next().await.is_err(),
                "truncation cannot become a successful empty stream"
            );
            decoder.close(Duration::from_secs(1)).await.unwrap();
        }
    }

    #[tokio::test]
    async fn an_extreme_source_rate_is_rejected_before_native_resampling() {
        let mut bytes = wav(1, &[1; 1024]);
        bytes[24..28].copy_from_slice(&1u32.to_le_bytes());
        let mut decoder =
            AudioDecoder::open(&workers(), "audio/wav", Some(48000), 1, limits()).unwrap();
        push(&decoder, bytes).await.unwrap();
        end_input(&decoder).await.unwrap();
        assert_eq!(
            decoder.next().await,
            Err(DecoderError::Codec(CodecError::DecodeLimit("sample rate")))
        );
        decoder.close(Duration::from_secs(1)).await.unwrap();
    }

    #[tokio::test]
    async fn compressed_input_overflow_fails_the_stream_and_releases_queued_data() {
        let mut decoder =
            AudioDecoder::open(&workers(), "audio/wav", Some(24000), 1, limits()).unwrap();
        let error = DecoderError::Overflow("compressed input");
        assert_eq!(push(&decoder, vec![0; limits().input_bytes + 1]).await, Err(error.clone()));
        assert_eq!(decoder.next().await, Err(error));
        decoder.close(Duration::from_secs(1)).await.unwrap();
        assert!(decoder.shared.state.lock().unwrap().input.is_empty());
    }

    #[tokio::test]
    async fn stalled_pcm_consumer_fails_with_bounded_retained_output() {
        let limits = DecoderLimits { output_samples: 128, ..limits() };
        let mut decoder =
            AudioDecoder::open(&workers(), "audio/wav", Some(24000), 1, limits).unwrap();
        push(&decoder, wav(1, &[1; 4096])).await.unwrap();
        end_input(&decoder).await.unwrap();
        let result = decoder.worker.as_mut().unwrap().await.unwrap();
        decoder.worker = None;
        assert!(matches!(result, Err(DecoderError::Overflow("PCM output"))));
        assert_eq!(decoder.next().await, Err(DecoderError::Overflow("PCM output")));
        assert!(decoder.shared.state.lock().unwrap().output_samples <= limits.output_samples);
        decoder.close(Duration::from_secs(1)).await.unwrap();
    }

    #[tokio::test]
    async fn busy_admission_preserves_the_callers_chunk_and_does_not_publish_eof() {
        let mut decoder =
            AudioDecoder::open(&workers(), "audio/wav", Some(24000), 1, limits()).unwrap();
        let mut bytes = wav(1, &[1, 2, 3, 4]);
        let expected = bytes.clone();
        {
            let _worker_state = decoder.shared.state.lock().unwrap();
            assert_eq!(decoder.try_push(&mut bytes), Err(DecoderError::Busy));
            assert_eq!(bytes, expected);
            assert_eq!(decoder.end_input(), Err(DecoderError::Busy));
        }
        push(&decoder, bytes).await.unwrap();
        end_input(&decoder).await.unwrap();
        let mut pcm = Vec::new();
        while let Some(frame) = decoder.next().await.unwrap() {
            pcm.extend(frame.data);
        }
        assert_eq!(pcm, lk_agents_media::pcm_bytes(&[1, 2, 3, 4]));
        decoder.close(Duration::from_secs(1)).await.unwrap();
    }

    #[tokio::test]
    async fn a_ready_pcm_backlog_does_not_starve_queued_control() {
        let limits = DecoderLimits { input_bytes: 65536, output_chunk_samples: 1, ..limits() };
        let mut decoder =
            AudioDecoder::open(&workers(), "audio/wav", Some(24000), 1, limits).unwrap();
        push(&decoder, wav(1, &[1; 16000])).await.unwrap();
        end_input(&decoder).await.unwrap();
        loop {
            let ready = decoder.shared.ready.notified();
            let done = decoder.shared.try_state().map(|state| state.done).unwrap_or(false);
            if done {
                break;
            }
            ready.await;
        }
        let (ready, mut control) = tokio::sync::oneshot::channel();
        let control_task = tokio::spawn(async move {
            let _ = ready.send(());
        });
        let mut delivered = 0;
        while decoder.next().await.unwrap().is_some() {
            delivered += 1;
            if control.try_recv().is_ok() {
                break;
            }
        }
        assert!(delivered < 16000, "ready PCM must yield before draining the entire backlog");
        decoder.close(Duration::from_secs(1)).await.unwrap();
        control_task.await.unwrap();
    }

    #[tokio::test]
    async fn cancelling_a_blocked_reader_settles_before_admitting_a_replacement() {
        let workers = workers();
        let mut first =
            AudioDecoder::open(&workers, "audio/wav", Some(24000), 1, limits()).unwrap();
        assert!(matches!(
            AudioDecoder::open(&workers, "audio/wav", Some(24000), 1, limits()),
            Err(DecoderError::WorkerAdmission(MediaWorkerError::Exhausted(
                MediaWorkerClass::Decoder
            )))
        ));
        first.close(Duration::from_secs(1)).await.unwrap();
        let mut next = AudioDecoder::open(&workers, "audio/wav", Some(24000), 1, limits()).unwrap();
        next.close(Duration::from_secs(1)).await.unwrap();
    }

    #[cfg(feature = "media-ffmpeg")]
    #[tokio::test]
    async fn ragged_compressed_input_delivers_resampled_audio_and_the_encoder_tail() {
        let samples = (0..4800)
            .map(|n| (12000. * (n as f64 * 440. * std::f64::consts::TAU / 48000.).sin()) as i16)
            .collect::<Vec<_>>();
        let mut encoder = lk_agents_media::OggOpusEncoder::new(48000, 1, 64000).unwrap();
        let mut encoded = encoder.push(&samples).unwrap();
        encoded.extend(encoder.finish().unwrap());
        let mut decoder =
            AudioDecoder::open(&workers(), "audio/ogg", Some(24000), 1, limits()).unwrap();
        for chunk in encoded.chunks(7) {
            push(&decoder, chunk.to_vec()).await.unwrap();
        }
        end_input(&decoder).await.unwrap();
        let mut pcm = Vec::new();
        while let Some(frame) = decoder.next().await.unwrap() {
            assert_eq!((frame.sample_rate, frame.num_channels), (24000, 1));
            pcm.extend(frame.data.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])));
        }
        assert!((2300..=2800).contains(&pcm.len()), "all 100ms and the drained tail must arrive");
        assert!(
            pcm.iter().any(|sample| *sample > 1000) && pcm.iter().any(|sample| *sample < -1000),
            "decoded audio must contain the supplied signal"
        );
        assert_eq!(decoder.next().await.unwrap(), None);
        decoder.close(Duration::from_secs(1)).await.unwrap();
    }

    #[cfg(feature = "media-ffmpeg")]
    #[tokio::test]
    async fn cancelling_blocked_ffmpeg_avio_observes_native_cleanup() {
        let workers = workers();
        let mut decoder =
            AudioDecoder::open(&workers, "audio/mpeg", Some(24000), 1, limits()).unwrap();
        push(&decoder, b"ID3".to_vec()).await.unwrap();
        decoder.close(Duration::from_secs(1)).await.unwrap();
        let mut replacement =
            AudioDecoder::open(&workers, "audio/mpeg", Some(24000), 1, limits()).unwrap();
        replacement.close(Duration::from_secs(1)).await.unwrap();
    }
}
