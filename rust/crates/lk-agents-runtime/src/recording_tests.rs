use parking_lot::Mutex;
use std::io;
use std::sync::mpsc as sync_mpsc;

use lk_agents_media::recorder::{EncodeError, PCM_ENCODER_TRAILER};

use super::*;
use lk_agents_media::pool::{MediaPool, MediaPoolLimits};

fn media_pool(total: usize) -> MediaPool {
    MediaPool::new(MediaPoolLimits {
        total,
        decoders: 1,
        recordings: 1,
        images: 1,
        formatting: 1,
        eot_models: 1,
        vad_models: 1,
    })
    .unwrap()
}

fn limits() -> RecordingLimits {
    RecordingLimits {
        queued_commands: 32,
        queued_bytes: 4096,
        frame_bytes: 1024,
        retained_bytes: 4096,
        segment_bytes: 1024,
        block_duration: Duration::from_millis(100),
    }
}

fn frame(samples: &[i16]) -> AudioFrame {
    AudioFrame::new(lk_agents_media::pcm_bytes(samples), 10, 1, samples.len() as u32)
}

struct Encoder(Arc<Mutex<Vec<&'static str>>>);

impl OggOpusEncoder for Encoder {
    fn push(&mut self, pcm: &[i16]) -> Result<Vec<u8>, EncodeError> {
        self.0.lock().push("encode");
        Ok(lk_agents_media::pcm_bytes(pcm))
    }
    fn finish(self) -> Result<Vec<u8>, EncodeError> {
        self.0.lock().push("finalize");
        Ok(PCM_ENCODER_TRAILER.to_vec())
    }
}

struct Writer {
    bytes: Arc<Mutex<Vec<u8>>>,
    trace: Arc<Mutex<Vec<&'static str>>>,
    blocked: Option<(oneshot::Sender<()>, sync_mpsc::Receiver<()>)>,
    fail: bool,
}

impl Write for Writer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if let Some((started, release)) = self.blocked.take() {
            let _ = started.send(());
            release.recv().unwrap();
        }
        if self.fail {
            return Err(io::Error::new(io::ErrorKind::StorageFull, "full disk"));
        }
        self.trace.lock().push("write");
        self.bytes.lock().extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct Sink {
    bytes: Arc<Mutex<Vec<u8>>>,
    trace: Arc<Mutex<Vec<&'static str>>>,
    writer: Writer,
}

impl Sink {
    fn new(blocked: bool, fail: bool) -> (Self, oneshot::Receiver<()>, sync_mpsc::SyncSender<()>) {
        let bytes = Arc::new(Mutex::new(Vec::new()));
        let trace = Arc::new(Mutex::new(Vec::new()));
        let (started, ready) = oneshot::channel();
        let (release, wait) = sync_mpsc::sync_channel(1);
        let writer = Writer {
            bytes: bytes.clone(),
            trace: trace.clone(),
            blocked: blocked.then_some((started, wait)),
            fail,
        };
        (Self { bytes, trace, writer }, ready, release)
    }
}

#[tokio::test]
async fn recording_orders_flush_and_finish_after_accepted_audio() {
    let media_pool = media_pool(1);
    let (sink, started, release) = Sink::new(true, false);
    let (bytes, trace) = (sink.bytes, sink.trace);
    let encoder = Encoder(trace.clone());
    let recorder = SessionRecorder::open_with(&media_pool, 10, Some(0.0), limits(), move || {
        Ok((encoder, sink.writer))
    })
    .await
    .unwrap();
    recorder.input(0.1, &frame(&[1234]), false).unwrap();
    let flushing = {
        let recorder = recorder.clone();
        tokio::spawn(async move { recorder.flush(0.1).await })
    };
    started.await.unwrap();
    recorder.input(0.2, &frame(&[2345]), false).unwrap();
    let finishing = {
        let recorder = recorder.clone();
        tokio::spawn(async move { recorder.finish(0.2).await })
    };
    tokio::task::yield_now().await;
    assert!(!flushing.is_finished(), "flush must await its write outcome");
    assert!(!finishing.is_finished(), "finish must await the older write");
    assert!(!trace.lock().contains(&"finalize"));
    release.send(()).unwrap();
    flushing.await.unwrap().unwrap();
    let summary = finishing.await.unwrap().unwrap();
    assert_eq!(recorder.finish(0.3).await.unwrap(), summary);
    assert_eq!(recorder.input(0.4, &frame(&[9999]), false), Err(RecordingFailure::Closed));
    let mut want = lk_agents_media::pcm_bytes(&[1234, 0, 2345, 0]);
    want.extend_from_slice(PCM_ENCODER_TRAILER);
    assert_eq!(*bytes.lock(), want);
    assert_eq!(*trace.lock(), ["encode", "write", "encode", "write", "finalize", "write"]);
    assert_eq!(recorder.retained_bytes(), 0);
}

#[tokio::test]
async fn recording_overflow_releases_pcm_and_preserves_other_work() {
    // Keep two payload slots available alongside the reserved worker-health slot.
    let media_pool = media_pool(3);
    let (sink, started, release) = Sink::new(true, false);
    let encoder = Encoder(sink.trace.clone());
    let mut limits = limits();
    limits.queued_commands = 2;
    let recorder = SessionRecorder::open_with(&media_pool, 10, Some(0.0), limits, move || {
        Ok((encoder, sink.writer))
    })
    .await
    .unwrap();
    recorder.input(0.1, &frame(&[1]), false).unwrap();
    let flushing = {
        let recorder = recorder.clone();
        tokio::spawn(async move { recorder.flush(0.1).await })
    };
    started.await.unwrap();
    recorder.record(RecordingCommand::Capture(0.2, frame(&[2]))).unwrap();
    recorder.record(RecordingCommand::Capture(0.3, frame(&[3]))).unwrap();
    assert_eq!(
        recorder.record(RecordingCommand::Capture(0.4, frame(&[4]))),
        Err(RecordingFailure::Overflow("staging commands"))
    );
    assert!(matches!(
        recorder.status(),
        RecordingStatus::Failing(RecordingFailure::Overflow("staging commands"))
    ));
    assert_eq!(
        recorder.close(Some(0.4), Duration::from_millis(10)).await,
        Err(RecordingFailure::IncompleteCleanup)
    );
    assert!(recorder.retained_bytes() <= limits.queued_bytes + limits.retained_bytes);
    media_pool.try_spawn(JobClass::Image, || "next turn continues").unwrap().await.unwrap();
    release.send(()).unwrap();
    let _ = flushing.await.unwrap();
    assert_eq!(recorder.finish(0.4).await, Err(RecordingFailure::Overflow("staging commands")));
    assert_eq!(recorder.retained_bytes(), 0);
    assert!(matches!(recorder.status(), RecordingStatus::Failed(RecordingFailure::Overflow(_))));
}

#[tokio::test]
async fn recording_close_deadline_keeps_stalled_worker_owned_until_settlement() {
    let media_pool = media_pool(1);
    let (sink, started, release) = Sink::new(true, false);
    let encoder = Encoder(sink.trace.clone());
    let recorder = SessionRecorder::open_with(&media_pool, 10, Some(0.0), limits(), move || {
        Ok((encoder, sink.writer))
    })
    .await
    .unwrap();
    recorder.input(0.1, &frame(&[1]), false).unwrap();
    let flushing = {
        let recorder = recorder.clone();
        tokio::spawn(async move { recorder.flush(0.1).await })
    };
    started.await.unwrap();
    assert_eq!(
        recorder.close(Some(0.1), Duration::from_millis(10)).await,
        Err(RecordingFailure::IncompleteCleanup)
    );
    assert!(
        media_pool.try_spawn(JobClass::Image, || ()).is_err(),
        "deadline cannot release a live worker permit"
    );
    release.send(()).unwrap();
    flushing.await.unwrap().unwrap();
    recorder.finish(0.1).await.unwrap();
    media_pool.try_spawn(JobClass::Image, || ()).unwrap().await.unwrap();
}

#[tokio::test]
async fn recording_write_failure_reaches_flush_and_terminal_outcome_once() {
    let media_pool = media_pool(1);
    let (sink, _, _) = Sink::new(false, true);
    let encoder = Encoder(sink.trace.clone());
    let recorder = SessionRecorder::open_with(&media_pool, 10, Some(0.0), limits(), move || {
        Ok((encoder, sink.writer))
    })
    .await
    .unwrap();
    recorder.input(0.1, &frame(&[1]), false).unwrap();
    let error = recorder.flush(0.1).await.unwrap_err();
    assert!(matches!(&error, RecordingFailure::Io(detail) if detail == "full disk"));
    assert_eq!(recorder.finish(0.1).await, Err(error));
    assert_eq!(recorder.retained_bytes(), 0);
}

#[tokio::test]
async fn sequential_playback_segments_release_previous_pcm() {
    let media_pool = media_pool(1);
    let (sink, _, _) = Sink::new(false, false);
    let bytes = sink.bytes;
    let encoder = Encoder(sink.trace.clone());
    let recorder = SessionRecorder::open_with(&media_pool, 10, Some(0.0), limits(), move || {
        Ok((encoder, sink.writer))
    })
    .await
    .unwrap();
    recorder.record(RecordingCommand::Capture(0.0, frame(&[100, 200]))).unwrap();
    recorder.record(RecordingCommand::PlaybackStarted(0.0)).unwrap();
    recorder.record(RecordingCommand::Progress(0.0, 0.0, 0.2)).unwrap();
    recorder.record(RecordingCommand::PlaybackFinished(0.2, 0.2)).unwrap();
    recorder.record(RecordingCommand::Capture(0.2, frame(&[300]))).unwrap();
    recorder.record(RecordingCommand::PlaybackStarted(0.2)).unwrap();
    recorder.record(RecordingCommand::PlaybackFinished(0.3, 0.1)).unwrap();
    recorder.finish(0.3).await.unwrap();
    let mut want = lk_agents_media::pcm_bytes(&[0, 100, 0, 200, 0, 300]);
    want.extend_from_slice(PCM_ENCODER_TRAILER);
    assert_eq!(*bytes.lock(), want);
    assert_eq!(recorder.retained_bytes(), 0);
}

#[tokio::test]
async fn cancelled_finish_waiter_does_not_abandon_accepted_audio() {
    let media_pool = media_pool(1);
    let (sink, started, release) = Sink::new(true, false);
    let bytes = sink.bytes;
    let encoder = Encoder(sink.trace.clone());
    let recorder = SessionRecorder::open_with(&media_pool, 10, None, limits(), move || {
        Ok((encoder, sink.writer))
    })
    .await
    .unwrap();
    recorder.input(100.1, &frame(&[4444]), false).unwrap();
    let finishing = {
        let recorder = recorder.clone();
        tokio::spawn(async move { recorder.finish(100.1).await })
    };
    started.await.unwrap();
    finishing.abort();
    assert!(finishing.await.unwrap_err().is_cancelled());
    release.send(()).unwrap();
    let summary = recorder.finish(101.0).await.unwrap();
    assert_eq!(summary.started_at, Some(100.0));
    let mut want = lk_agents_media::pcm_bytes(&[4444, 0]);
    want.extend_from_slice(PCM_ENCODER_TRAILER);
    assert_eq!(*bytes.lock(), want);
}

#[tokio::test]
async fn segment_pcm_limits_fail_recording_and_release_the_buffer() {
    for field in ["segment PCM", "retained PCM"] {
        let media_pool = media_pool(1);
        let (sink, _, _) = Sink::new(false, false);
        let encoder = Encoder(sink.trace.clone());
        let mut limits = limits();
        match field {
            "segment PCM" => limits.segment_bytes = 2,
            "retained PCM" => limits.retained_bytes = 4,
            _ => unreachable!(),
        }
        let recorder = SessionRecorder::open_with(&media_pool, 10, Some(0.0), limits, move || {
            Ok((encoder, sink.writer))
        })
        .await
        .unwrap();
        recorder.record(RecordingCommand::Capture(0.0, frame(&[1, 2, 3]))).unwrap();
        assert_eq!(recorder.finish(0.4).await, Err(RecordingFailure::Overflow(field)));
        assert_eq!(recorder.retained_bytes(), 0);
    }
}

#[tokio::test]
async fn finalization_preserves_the_input_resamplers_tail() {
    let mut resampler =
        lk_agents_media::Resampler::new(16_000, 48_000, 1, lk_agents_media::Quality::Medium)
            .unwrap();
    let mut expected = resampler.push(&[16000; 160]).unwrap().to_vec();
    expected.extend_from_slice(resampler.flush().unwrap());
    assert_eq!(expected.len(), 480);
    for flush_before_finish in [false, true] {
        let media_pool = media_pool(1);
        let (sink, _, _) = Sink::new(false, false);
        let bytes = sink.bytes;
        let encoder = Encoder(sink.trace.clone());
        let mut limits = limits();
        limits.retained_bytes = 64 * 1024;
        let recorder =
            SessionRecorder::open_with(&media_pool, 48_000, Some(0.0), limits, move || {
                Ok((encoder, sink.writer))
            })
            .await
            .unwrap();
        let input = AudioFrame::new(lk_agents_media::pcm_bytes(&[16000; 160]), 16_000, 1, 160);
        recorder.input(0.01, &input, false).unwrap();
        if flush_before_finish {
            recorder.flush(0.01).await.unwrap();
        }
        recorder.finish(0.01).await.unwrap();
        let bytes = bytes.lock();
        let left = bytes[..bytes.len() - PCM_ENCODER_TRAILER.len()]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|sample| i16::from_le_bytes([sample[0], sample[1]]))
            .collect::<Vec<_>>();
        assert_eq!(left.len(), expected.len());
        // The recorder must retain supplied codec output, including its EOF filter response.
        // Two int16 quantization steps cover independent native-resampler dither.
        for (index, (actual, expected)) in left.iter().zip(&expected).enumerate() {
            assert!(
                (i32::from(*actual) - i32::from(*expected)).abs() <= 2,
                "sample {index}, periodic={flush_before_finish}: recorded {actual}, codec produced {expected}"
            );
        }
        assert!(
            bytes[..bytes.len() - PCM_ENCODER_TRAILER.len()]
                .as_chunks::<4>()
                .0
                .iter()
                .all(|sample| sample[2..] == [0, 0]),
            "input must remain on the left channel"
        );
    }
}

#[tokio::test]
async fn host_finish_preserves_the_session_close_time_after_a_settled_flush() {
    let media_pool = media_pool(1);
    let (sink, _, _) = Sink::new(false, false);
    let bytes = sink.bytes;
    let encoder = Encoder(sink.trace.clone());
    let recorder = SessionRecorder::open_with(&media_pool, 10, Some(0.0), limits(), move || {
        Ok((encoder, sink.writer))
    })
    .await
    .unwrap();
    recorder.input(0.1, &frame(&[1234]), false).unwrap();
    // A recent microphone is only settled through its most recent frame; the session closes
    // later. Finalization must still extend the timeline through the requested close time.
    assert_eq!(recorder.flush(0.5).await.unwrap().last_flushed, Some(0.1));
    lk_agents_abi::recorder::SessionRecorder::finish(&*recorder, None).await.unwrap();
    let mut expected = lk_agents_media::pcm_bytes(&[1234, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    expected.extend_from_slice(PCM_ENCODER_TRAILER);
    assert_eq!(*bytes.lock(), expected);
}

#[tokio::test]
async fn recording_burst_preserves_input_timing_and_discarded_output_offsets() {
    let media_pool = media_pool(1);
    let (sink, _, _) = Sink::new(false, false);
    let bytes = sink.bytes;
    let encoder = Encoder(sink.trace);
    let recorder = SessionRecorder::open_with(&media_pool, 10, Some(0.), limits(), move || {
        Ok((encoder, sink.writer))
    })
    .await
    .unwrap();
    recorder.input(1., &frame(&(1..=10).collect::<Vec<_>>()), false).unwrap();
    recorder.record(RecordingCommand::Capture(0., frame(&(100..110).collect::<Vec<_>>()))).unwrap();
    recorder.record(RecordingCommand::Progress(1., 0.2, 0.6)).unwrap();
    recorder.finish(2.).await.unwrap();

    let mut expected = Vec::new();
    for input in 1..=10 {
        expected.extend([input, 0]);
    }
    for output in 102..108 {
        expected.extend([0, output]);
    }
    expected.extend([0; 8]);
    let mut encoded = lk_agents_media::pcm_bytes(&expected);
    encoded.extend_from_slice(PCM_ENCODER_TRAILER);
    assert_eq!(*bytes.lock(), encoded);
}

#[tokio::test]
async fn recording_fallback_drains_played_audio_and_retires_each_segment_once() {
    let media_pool = media_pool(1);
    let (sink, _, _) = Sink::new(false, false);
    let bytes = sink.bytes;
    let encoder = Encoder(sink.trace);
    let recorder = SessionRecorder::open_with(&media_pool, 10, Some(0.), limits(), move || {
        Ok((encoder, sink.writer))
    })
    .await
    .unwrap();

    // Without a start/progress report, the played endpoint determines the anchor and truncation.
    recorder.record(RecordingCommand::Capture(0., frame(&(100..110).collect::<Vec<_>>()))).unwrap();
    recorder.record(RecordingCommand::PlaybackFinished(2., 0.6)).unwrap();
    // The next segment has its own offset and honors an explicit start over the fallback anchor.
    recorder.record(RecordingCommand::Capture(2., frame(&(200..204).collect::<Vec<_>>()))).unwrap();
    recorder.record(RecordingCommand::PlaybackStarted(2.3)).unwrap();
    recorder.record(RecordingCommand::PlaybackFinished(3., 0.4)).unwrap();
    // A progress report suppresses endpoint fallback for the remaining unplayed samples.
    recorder.record(RecordingCommand::Capture(3., frame(&(300..304).collect::<Vec<_>>()))).unwrap();
    recorder.record(RecordingCommand::Progress(3., 0.1, 0.1)).unwrap();
    recorder.record(RecordingCommand::PlaybackFinished(3.4, 0.4)).unwrap();
    recorder.finish(4.).await.unwrap();

    let mut expected = vec![0; 80];
    for (index, sample) in (100..106).enumerate() {
        expected[(14 + index) * 2 + 1] = sample;
    }
    for (index, sample) in (200..204).enumerate() {
        expected[(23 + index) * 2 + 1] = sample;
    }
    expected[30 * 2 + 1] = 301;
    let mut encoded = lk_agents_media::pcm_bytes(&expected);
    encoded.extend_from_slice(PCM_ENCODER_TRAILER);
    assert_eq!(*bytes.lock(), encoded);
}

/// The left (input) channel of what the stub encoder wrote, without its trailer.
fn input_channel(bytes: &[u8]) -> Vec<i16> {
    let pcm = &bytes[..bytes.len() - PCM_ENCODER_TRAILER.len()];
    pcm.as_chunks::<4>().0.iter().map(|sample| i16::from_le_bytes([sample[0], sample[1]])).collect()
}

// Catches: a burst behind a stalled worker failing the recording, or the audio its bound drops
// collapsing the timeline instead of becoming silence of the same length.
#[tokio::test]
async fn a_stalled_recording_keeps_call_time_with_the_dropped_audio_as_silence() {
    let media_pool = media_pool(1);
    let (sink, started, release) = Sink::new(true, false);
    let bytes = sink.bytes;
    let encoder = Encoder(sink.trace.clone());
    let recorder = SessionRecorder::open_with(&media_pool, 10, Some(0.0), limits(), move || {
        Ok((encoder, sink.writer))
    })
    .await
    .unwrap();
    let value = |index: usize| index as i16 + 1;
    recorder.input(0.1, &frame(&[value(0)]), false).unwrap();
    let flushing = {
        let recorder = recorder.clone();
        tokio::spawn(async move { recorder.flush(0.1).await })
    };
    started.await.unwrap();
    // 16s of microphone while the worker is stuck in a write: 6s more than the bound keeps.
    for index in 1..=160 {
        recorder.input((index + 1) as f64 / 10., &frame(&[value(index)]), false).unwrap();
    }
    release.send(()).unwrap();
    flushing.await.unwrap().unwrap();
    let summary = recorder.finish(16.1).await.unwrap();
    assert_eq!(summary.dropped_input_samples, 0);

    // The first frame, the 6s the bound dropped as silence, then the 10s it kept, on call time.
    let expected: Vec<i16> =
        (0..=160).map(|index| if (1..=60).contains(&index) { 0 } else { value(index) }).collect();
    assert_eq!(input_channel(&bytes.lock()), expected);
}

// Catches: pre-connect audio replayed in a burst being stacked on itself or over the live audio
// instead of ending where the live audio begins.
#[tokio::test]
async fn pre_connect_audio_ends_where_the_live_audio_begins() {
    let media_pool = media_pool(1);
    let (sink, _, _) = Sink::new(false, false);
    let bytes = sink.bytes;
    let encoder = Encoder(sink.trace.clone());
    let recorder = SessionRecorder::open_with(&media_pool, 10, None, limits(), move || {
        Ok((encoder, sink.writer))
    })
    .await
    .unwrap();
    for sample in 1..=5 {
        recorder.input(1.0, &frame(&[sample]), true).unwrap();
    }
    recorder.input(1.1, &frame(&[6]), false).unwrap();
    recorder.input(1.2, &frame(&[7]), false).unwrap();
    let summary = recorder.finish(1.2).await.unwrap();
    assert_eq!(summary.started_at, Some(0.5));
    assert_eq!(input_channel(&bytes.lock()), [1, 2, 3, 4, 5, 6, 7]);
}

// Catches: a periodic flush landing between the pre-connect burst and the first live frame,
// writing past where the held pre-connect audio will be placed so it is dropped as late input.
#[tokio::test]
async fn pre_connect_audio_survives_a_flush_before_the_live_audio() {
    let media_pool = media_pool(1);
    let (sink, _, _) = Sink::new(false, false);
    let bytes = sink.bytes;
    let encoder = Encoder(sink.trace.clone());
    let recorder = SessionRecorder::open_with(&media_pool, 10, None, limits(), move || {
        Ok((encoder, sink.writer))
    })
    .await
    .unwrap();
    for sample in 1..=5 {
        recorder.input(1.0, &frame(&[sample]), true).unwrap();
    }
    recorder.flush(1.0).await.unwrap();
    recorder.input(1.1, &frame(&[6]), false).unwrap();
    recorder.flush(1.1).await.unwrap();
    recorder.input(1.2, &frame(&[7]), false).unwrap();
    let summary = recorder.finish(1.2).await.unwrap();
    assert_eq!(summary.dropped_input_samples, 0);
    assert_eq!(summary.started_at, Some(0.5));
    assert_eq!(input_channel(&bytes.lock()), [1, 2, 3, 4, 5, 6, 7]);
}
