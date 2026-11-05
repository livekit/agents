//! VAD constructors and recurrent model operations must not occupy the calling runtime thread.

use std::sync::Arc;
use std::thread::ThreadId;
use std::time::Duration;

use lk_agents_abi::{AudioFrame, StreamInput, StreamOutput, VadEventType};
use lk_agents_inference::vad::machine::VadOptions;
use lk_agents_inference::vad::{LocalVad, VadModel, VadModelFactory};
use parking_lot::Mutex;

struct Model {
    calls: Arc<Mutex<Vec<(&'static str, ThreadId)>>>,
}

impl VadModel for Model {
    fn predict(&mut self, _: &[i16]) -> (f32, f64) {
        self.calls.lock().push(("predict", std::thread::current().id()));
        std::thread::sleep(Duration::from_millis(20));
        (0.0, 0.02)
    }
    fn reset(&mut self) {
        self.calls.lock().push(("reset", std::thread::current().id()));
        std::thread::sleep(Duration::from_millis(20));
    }
}

struct SlowFactory(Arc<Mutex<Vec<(&'static str, ThreadId)>>>);

impl VadModelFactory for SlowFactory {
    fn create(&self) -> Box<dyn VadModel> {
        self.0.lock().push(("create", std::thread::current().id()));
        std::thread::sleep(Duration::from_millis(20));
        Box::new(Model { calls: self.0.clone() })
    }
}

#[tokio::test(flavor = "current_thread")]
async fn model_construction_prediction_and_reset_leave_the_runtime_thread_free() {
    let caller = std::thread::current().id();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let vad = LocalVad::new(
        VadOptions::default(),
        Arc::new(SlowFactory(calls.clone())),
        &lk_agents_inference::testkit::env(),
    );
    let (sink, mut source) = vad.stream_native().unwrap();
    sink.push(AudioFrame::new(vec![0; 1024], 16_000, 1, 512)).unwrap();
    sink.flush().unwrap();
    sink.end_input();
    tokio::time::timeout(Duration::from_secs(2), async {
        while source.next().await.unwrap().is_some() {}
    })
    .await
    .unwrap();
    let calls = calls.lock();
    assert_eq!(
        calls.iter().map(|(kind, _)| *kind).collect::<Vec<_>>(),
        ["create", "predict", "reset"]
    );
    for (kind, thread) in calls.iter() {
        assert_ne!(*thread, caller, "VAD {kind} occupied the runtime thread");
    }
}

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
    let vad: Arc<dyn lk_agents_abi::Vad> = Arc::new(LocalVad::new(
        VadOptions::default(),
        Arc::new(EnergyFactory),
        &lk_agents_inference::testkit::env(),
    ));
    assert_eq!(vad.capabilities().update_interval, 0.032);
    let (sink, mut source) = vad.stream(&lk_agents_abi::telemetry::Cx::NONE).await.unwrap().split();
    let frame = |level: i16| AudioFrame {
        data: std::iter::repeat_n(level.to_le_bytes(), 512).flatten().collect::<Vec<u8>>().into(),
        sample_rate: 16000,
        num_channels: 1,
        samples_per_channel: 512,
    };
    // One window per push, each pushed once the last was scored: the input never lags.
    let mut kinds = Vec::new();
    for i in 0..60 {
        sink.push(frame(if i < 30 { 1000 } else { 0 })).unwrap();
        loop {
            let ev = source.next().await.unwrap().unwrap();
            if ev.kind == VadEventType::InferenceDone {
                break;
            }
            kinds.push(ev.kind);
        }
    }
    sink.end_input();
    while let Ok(Some(ev)) = source.next().await {
        if ev.kind != VadEventType::InferenceDone {
            kinds.push(ev.kind);
        }
    }
    assert_eq!(kinds, [VadEventType::StartOfSpeech, VadEventType::EndOfSpeech]);
}

// Catches: `end_input` doing nothing, so a caller reading the stream to its end waits forever
// (the reference's `end_input` closes the input channel and the stream finishes).
#[tokio::test]
async fn end_input_ends_the_stream_after_what_was_pushed() {
    let vad = LocalVad::new(
        VadOptions::default(),
        Arc::new(Factory),
        &lk_agents_inference::testkit::env(),
    );
    let (sink, mut source) = vad.stream_native().unwrap();
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

struct GatedModel {
    started: Option<tokio::sync::oneshot::Sender<()>>,
    release: std::sync::mpsc::Receiver<()>,
    settled: Option<tokio::sync::oneshot::Sender<()>>,
}

impl VadModel for GatedModel {
    fn predict(&mut self, _: &[i16]) -> (f32, f64) {
        self.started.take().unwrap().send(()).unwrap();
        self.release.recv_timeout(Duration::from_secs(2)).unwrap();
        (0.0, 0.0)
    }
    fn reset(&mut self) {}
}

impl Drop for GatedModel {
    fn drop(&mut self) {
        if let Some(settled) = self.settled.take() {
            let _ = settled.send(());
        }
    }
}

struct GatedFactory(Mutex<Option<GatedModel>>);

impl VadModelFactory for GatedFactory {
    fn create(&self) -> Box<dyn VadModel> {
        Box::new(self.0.lock().take().unwrap())
    }
}

#[tokio::test(flavor = "current_thread")]
async fn closing_a_pending_model_reports_incomplete_cleanup_and_retires_queued_audio() {
    use lk_agents_abi::{Closed, Ts};

    let (started, ready) = tokio::sync::oneshot::channel();
    let (release, blocked) = std::sync::mpsc::channel();
    let (settled, done) = tokio::sync::oneshot::channel();
    let factory = GatedFactory(Mutex::new(Some(GatedModel {
        started: Some(started),
        release: blocked,
        settled: Some(settled),
    })));
    let vad = LocalVad::new(
        VadOptions::default(),
        Arc::new(factory),
        &lk_agents_inference::testkit::env(),
    );
    let (sink, source) = vad.stream_native().unwrap();
    let frame = AudioFrame::new(vec![0; 1024], 16_000, 1, 512);
    sink.push(frame.clone()).unwrap();
    ready.await.unwrap();
    sink.push(frame.clone()).unwrap();
    assert_eq!(Box::new(source).close(Ts::ZERO).await, Closed::TimedOut);
    assert!(
        sink.push(frame).is_err(),
        "close stops admission while the native model remains pending"
    );
    release.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(2), done).await.unwrap().unwrap();
}

#[tokio::test]
async fn a_mismatched_frame_does_not_end_the_stream_or_discard_following_valid_audio() {
    let vad = LocalVad::new(
        VadOptions::default(),
        Arc::new(Factory),
        &lk_agents_inference::testkit::env(),
    );
    let (sink, mut source) = vad.stream_native().unwrap();
    sink.push(AudioFrame::new(vec![0; 1024], 16_000, 1, 512)).unwrap();
    sink.push(AudioFrame::new(vec![0; 1024], 8_000, 1, 512)).unwrap();
    sink.push(AudioFrame::new(vec![0; 1024], 16_000, 1, 512)).unwrap();
    sink.end_input();
    let mut windows = 0;
    while let Some(output) = source.next().await.unwrap() {
        windows += usize::from(output.event.kind == VadEventType::InferenceDone);
    }
    assert_eq!(windows, 2, "only the frame with the mismatched rate is dropped");
}

#[tokio::test]
async fn one_second_of_48khz_stereo_is_accepted_without_doubling_the_vad_timeline() {
    let vad = LocalVad::new(
        VadOptions::default(),
        Arc::new(Factory),
        &lk_agents_inference::testkit::env(),
    );
    let (sink, mut source) = vad.stream_native().unwrap();
    sink.push(AudioFrame::new(vec![0; 192_000], 48_000, 2, 48_000)).unwrap();
    sink.end_input();
    let mut timestamp = 0.0;
    while let Some(output) = source.next().await.unwrap() {
        timestamp = output.event.timestamp;
        assert!(output.frames.iter().all(|frame| frame.num_channels == 1));
    }
    assert!(
        (0.8..=1.0).contains(&timestamp),
        "one second of stereo produced {timestamp}s of VAD audio"
    );
}

/// Blocks its first prediction until released, then records the first sample of every window.
struct StalledModel {
    started: Option<tokio::sync::oneshot::Sender<()>>,
    release: std::sync::mpsc::Receiver<()>,
    windows: Arc<Mutex<Vec<i16>>>,
}

impl VadModel for StalledModel {
    fn predict(&mut self, window: &[i16]) -> (f32, f64) {
        if let Some(started) = self.started.take() {
            started.send(()).unwrap();
            self.release.recv_timeout(Duration::from_secs(2)).unwrap();
        }
        self.windows.lock().push(window[0]);
        (0.0, 0.0)
    }
    fn reset(&mut self) {}
}

struct StalledFactory(Mutex<Option<StalledModel>>);

impl VadModelFactory for StalledFactory {
    fn create(&self) -> Box<dyn VadModel> {
        Box::new(self.0.lock().take().unwrap())
    }
}

// Catches: a stalled VAD failing its input, or replaying the backlog once it resumes instead of
// deciding on the newest audio.
#[tokio::test(flavor = "current_thread")]
async fn a_stalled_vad_resumes_on_current_audio_within_its_lag() {
    let (started, ready) = tokio::sync::oneshot::channel();
    let (release, blocked) = std::sync::mpsc::channel();
    let windows = Arc::new(Mutex::new(Vec::new()));
    let factory = StalledFactory(Mutex::new(Some(StalledModel {
        started: Some(started),
        release: blocked,
        windows: windows.clone(),
    })));
    let vad = LocalVad::new(
        VadOptions::default(),
        Arc::new(factory),
        &lk_agents_inference::testkit::env(),
    );
    let (sink, mut source) = vad.stream_native().unwrap();
    // One 32 ms window per frame, each frame marked by its index.
    let frame = |n: i16| AudioFrame::new(n.to_le_bytes().repeat(512), 16_000, 1, 512);
    sink.push(frame(0)).unwrap();
    ready.await.unwrap();
    for n in 1..=100 {
        sink.push(frame(n)).expect("a stalled VAD drops old audio, it never refuses it");
    }
    release.send(()).unwrap();
    sink.end_input();
    while source.next().await.expect("a stall is not a failure").is_some() {}

    let windows = windows.lock();
    let resumed = &windows[1..];
    // 500 ms is under 16 windows of 32 ms.
    assert!(resumed.len() <= 16, "the backlog was replayed: {resumed:?}");
    assert!(resumed.iter().all(|n| *n > 100 - 16), "old audio after the stall: {resumed:?}");
    assert_eq!(resumed.last(), Some(&100), "the newest audio is scored");
}

#[tokio::test]
async fn vad_admission_leaves_capacity_for_image_and_recording_work() {
    use lk_agents_media::pool::{JobClass, MediaPool, MediaPoolError, MediaPoolLimits};

    let media_pool = MediaPool::new(MediaPoolLimits::default()).unwrap();
    let models =
        (0..32).map(|_| media_pool.try_admit(JobClass::VadModel).unwrap()).collect::<Vec<_>>();
    assert!(matches!(
        media_pool.try_admit(JobClass::VadModel),
        Err(MediaPoolError::Exhausted(JobClass::VadModel))
    ));
    let image = media_pool.try_admit(JobClass::Image).unwrap();
    let recording = media_pool.try_admit(JobClass::Recording).unwrap();
    drop((models, image, recording));
}

// Catches: a flush or gap during speech resetting the detector without ending the speech, so the
// session stays speaking.
#[tokio::test]
async fn a_flush_during_speech_ends_the_speech() {
    let vad = LocalVad::new(
        VadOptions::default(),
        Arc::new(EnergyFactory),
        &lk_agents_inference::testkit::env(),
    );
    let (sink, mut source) = vad.stream_native().unwrap();
    for _ in 0..10 {
        sink.push(AudioFrame::new(1000_i16.to_le_bytes().repeat(512), 16_000, 1, 512)).unwrap();
    }
    sink.flush().unwrap();
    sink.end_input();
    let mut kinds = Vec::new();
    while let Some(output) = source.next().await.unwrap() {
        if output.event.kind != VadEventType::InferenceDone {
            kinds.push(output.event.kind);
        }
    }
    assert_eq!(kinds, [VadEventType::StartOfSpeech, VadEventType::EndOfSpeech]);
}

// Catches: a pre-connect head cut to its last 500 ms by the `Now` policy, a head scored at real
// time, or live audio heard during the catch-up queued behind the head instead of cut to the
// newest.
#[tokio::test(flavor = "current_thread")]
async fn a_ten_second_head_is_scored_in_full_far_faster_than_real_time() {
    let (started, ready) = tokio::sync::oneshot::channel();
    let (release, blocked) = std::sync::mpsc::channel();
    let windows = Arc::new(Mutex::new(Vec::new()));
    let factory = StalledFactory(Mutex::new(Some(StalledModel {
        started: Some(started),
        release: blocked,
        windows: windows.clone(),
    })));
    let vad: Arc<dyn lk_agents_abi::Vad> = Arc::new(LocalVad::new(
        VadOptions::default(),
        Arc::new(factory),
        &lk_agents_inference::testkit::env(),
    ));
    let measurements = lk_agents_inference::testkit::Measurements::default();
    let (sink, mut source) = vad.stream(&measurements.cx()).await.unwrap().split();
    // One 32 ms window per frame, the head's marked negative, the live ones positive.
    let frame = |n: i16| AudioFrame::new(n.to_le_bytes().repeat(512), 16_000, 1, 512);
    const HEAD: i16 = 313;
    for n in 1..=HEAD {
        sink.head(frame(-n)).unwrap();
    }
    ready.await.unwrap();
    for n in 1..=100 {
        sink.push(frame(n)).unwrap();
    }
    let began = std::time::Instant::now();
    release.send(()).unwrap();
    sink.end_input();
    while source.next().await.unwrap().is_some() {}
    let took = began.elapsed();

    let windows = windows.lock();
    let (head, live) = windows.split_at(HEAD as usize);
    assert_eq!(head, (1..=HEAD).map(|n| -n).collect::<Vec<_>>(), "the head in full, in order");
    assert!(live.len() <= 16, "the live backlog was replayed: {live:?}");
    assert_eq!(live.last(), Some(&100), "the newest live audio is scored");
    eprintln!("VAD: 10.0 s head scored in {took:?}");
    assert!(took < Duration::from_secs(1), "10 s of head took {took:?}");
    let measured = measurements.of("lk.agents.audio.catch_up_time");
    assert!(matches!(measured[..], [seconds] if seconds < 1.0), "{measured:?}");
}
