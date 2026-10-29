//! Run the lock inversion in a child: a blocked native mutex cannot be cancelled by Tokio.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use lk_agents_abi::errors::ApiError;
use lk_agents_abi::tokenize::language::{ThresholdOptions, TurnDetectorModel};

use lk_agents_inference::env::Env as _;
use lk_agents_inference::eot::*;
use lk_agents_runtime::process::{self, Process};

struct CloudFailure(Arc<tokio::sync::Notify>);

#[async_trait]
impl EotTransport for CloudFailure {
    fn session_id(&self) -> Option<String> {
        None
    }
    fn run_inference(&self, _: &EotRequestId) {}
    fn push_frame(&self, _: &lk_agents_abi::AudioFrame) {}
    fn flush(&self) {}
    fn end_input(&self) {}
    fn detach(&self) {}
    async fn run(&self) -> Result<(), ApiError> {
        self.0.notified().await;
        Err(ApiError::new("cloud disconnected"))
    }
}

struct Model(Arc<Mutex<Vec<Vec<i16>>>>);
impl LocalEotModel for Model {
    fn predict(&self, pcm: &[i16]) -> Result<f64, String> {
        self.0.lock().unwrap().push(pcm.to_vec());
        Ok(0.7)
    }
}

struct HostLogger(std::sync::mpsc::SyncSender<()>);
impl log::Log for HostLogger {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        if record.target() == "livekit.agents" && record.level() == log::Level::Warn {
            self.0.send(()).unwrap();
            with_host_lock(|| ());
        }
    }
    fn flush(&self) {}
}

/// The host's execution lock (the GIL in Python), held while a host log line is written.
fn with_host_lock<T>(f: impl FnOnce() -> T) -> T {
    static HOST: Mutex<()> = Mutex::new(());
    let _host = HOST.lock().unwrap();
    f()
}

#[test]
fn cloud_fallback_does_not_deadlock_a_host_audio_push() {
    const CHILD: &str = "LK_EOT_LOCK_INVERSION_CHILD";
    if let Ok(mode) = std::env::var(CHILD) {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(if mode == "logger" { 2 } else { 1 })
            .enable_all()
            .build()
            .unwrap();
        let _runtime = runtime.enter();
        let _workers = process::workers().unwrap();
        let failure = Arc::new(tokio::sync::Notify::new());
        let thresholds =
            Arc::new(Mutex::new(ThresholdOptions::new(TurnDetectorModel::V1, None, None)));
        let session = EotSession::new(TurnDetectorModel::V1, thresholds.clone(), true);
        let (entered, waiting) = std::sync::mpsc::sync_channel(1);
        if mode == "logger" {
            log::set_logger(Box::leak(Box::new(HostLogger(entered.clone())))).unwrap();
            log::set_max_level(log::LevelFilter::Warn);
        }
        let trigger = failure.clone();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let local_model = Arc::new(Model(seen.clone()));
        let log_mode = mode == "logger";
        let make: TransportFactory = Arc::new(move |model| {
            if model == TurnDetectorModel::V1Mini && !log_mode {
                entered.send(()).unwrap();
                with_host_lock(|| ());
            }
            TransportShell::mount(model.as_str(), thresholds.clone(), |sink| {
                if model == TurnDetectorModel::V1 {
                    Arc::new(CloudFailure(trigger.clone())) as Arc<dyn EotTransport>
                } else {
                    Arc::new(LocalTransport::new(local_model.clone(), sink))
                        as Arc<dyn EotTransport>
                }
            })
        });
        let stream = EotStream::open(
            session,
            16_000,
            make,
            Process::workers,
            None,
            &lk_agents_abi::TokioSpawner::current(),
        );
        let abandoned = stream.predict();
        let mut queued = None;
        with_host_lock(|| {
            failure.notify_one();
            waiting
                .recv_timeout(Duration::from_secs(2))
                .expect("fallback reached host construction");
            // The factory now waits for the host lock this caller owns. This must still return.
            stream.push_audio(lk_agents_abi::AudioFrame::new(vec![0; 320], 16_000, 1, 160));
            let cancelled = stream.predict();
            stream.cancel_inference(false);
            assert_eq!(cancelled.blocking_recv().unwrap().end_of_turn_probability, 0.0);
            stream.flush();
            stream.push_audio(lk_agents_abi::AudioFrame::new(vec![1; 320], 16_000, 1, 160));
            queued = Some(stream.predict());
            // Later audio belongs to the next snapshot, even if construction has not finished.
            stream.push_audio(lk_agents_abi::AudioFrame::new(vec![2; 320], 16_000, 1, 160));
            if mode == "close" {
                stream.aclose();
            } else if !log_mode {
                let (tx, rx) = std::sync::mpsc::sync_channel(1);
                runtime.spawn(async move {
                    tx.send(()).unwrap();
                });
                rx.recv_timeout(Duration::from_secs(2))
                    .expect("Tokio progresses during host construction");
            }
        });
        assert_eq!(abandoned.blocking_recv().unwrap().end_of_turn_probability, 1.0);
        runtime.block_on(async {
            let answer = tokio::time::timeout(Duration::from_secs(2), queued.unwrap())
                .await
                .unwrap()
                .unwrap();
            if mode == "close" {
                assert_eq!(answer.end_of_turn_probability, 0.0);
                assert!(seen.lock().unwrap().is_empty(), "close cannot revive a queued prediction");
            } else {
                assert_eq!(answer.end_of_turn_probability, 0.7);
                assert_eq!(
                    *seen.lock().unwrap(),
                    vec![vec![257; 160]],
                    "fallback keeps the request's audio snapshot"
                );
            }
        });
        stream.aclose();
        runtime.block_on(async {
            tokio::time::timeout(Duration::from_secs(2), stream.wait_closed()).await.unwrap();
        });
        if mode == "close" {
            assert!(seen.lock().unwrap().is_empty(), "settled replacement never ran after close");
        }
        return;
    }

    for mode in ["factory", "logger", "close"] {
        run_child("cloud_fallback_does_not_deadlock_a_host_audio_push", CHILD, mode);
    }
}

fn run_child(test: &str, flag: &str, mode: &str) {
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture"])
        .env(flag, mode)
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "{test} child ({mode}) failed: {status}");
            return;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("{test} child ({mode}) hung (5s watchdog)");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// A gateway connection that dies as soon as it runs.
struct DyingTransport;

#[async_trait]
impl EotTransport for DyingTransport {
    fn session_id(&self) -> Option<String> {
        None
    }
    fn run_inference(&self, _request_id: &EotRequestId) {}
    fn push_frame(&self, _frame: &lk_agents_abi::AudioFrame) {}
    fn flush(&self) {}
    fn end_input(&self) {}
    fn detach(&self) {}
    async fn run(&self) -> Result<(), ApiError> {
        Err(ApiError::new("gateway gone"))
    }
}

// Catches: the stream driver not remounting after a cloud failure (the swap's new transport never
// run), the waiter left hanging, or a request the dead transport held never answered.
#[tokio::test]
async fn a_cloud_failure_swaps_the_stream_to_the_local_model_which_answers_the_next_request() {
    let _workers = process::workers().unwrap();
    let shared = Arc::new(Mutex::new(ThresholdOptions::new(TurnDetectorModel::V1, None, None)));
    let session = EotSession::new(TurnDetectorModel::V1, shared.clone(), true);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(Model(Arc::clone(&seen)));
    let make: TransportFactory = Arc::new(move |m: TurnDetectorModel| {
        let model = Arc::clone(&model) as Arc<dyn LocalEotModel>;
        TransportShell::mount(m.as_str(), Arc::clone(&shared), move |sink| match m {
            TurnDetectorModel::V1 => Arc::new(DyingTransport) as Arc<dyn EotTransport>,
            _ => Arc::new(LocalTransport::new(model, sink)) as Arc<dyn EotTransport>,
        })
    });
    let spawner = lk_agents_abi::TokioSpawner::current();
    let stream = EotStream::open(session, 16_000, make, Process::workers, None, &spawner);
    tokio::time::timeout(Duration::from_secs(5), async {
        while !stream.is_fallback() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the stream fell back to local inference");
    assert_eq!(stream.model(), TurnDetectorModel::V1Mini);

    stream.push_audio(lk_agents_abi::AudioFrame::new(vec![0; 3200], 16_000, 1, 1600));
    let answer =
        tokio::time::timeout(Duration::from_secs(5), stream.predict()).await.unwrap().unwrap();
    assert!((answer.end_of_turn_probability - 0.7).abs() < 1e-9, "{answer:?}");
    assert!(!seen.lock().unwrap().is_empty(), "the local model heard the audio");

    stream.aclose();
    let after =
        tokio::time::timeout(Duration::from_secs(5), stream.predict()).await.unwrap().unwrap();
    assert_eq!(after.end_of_turn_probability, 1.0, "a closed stream answers 1.0 at once");
    stream.wait_closed().await;
}

#[test]
fn saturated_model_construction_defaults_then_retries_on_the_next_prediction() {
    const CHILD: &str = "LK_EOT_SATURATION_CHILD";
    if std::env::var_os(CHILD).is_none() {
        run_child(
            "saturated_model_construction_defaults_then_retries_on_the_next_prediction",
            CHILD,
            "1",
        );
        return;
    }

    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    runtime.block_on(async {
        use lk_agents_runtime::media::workers::MediaWorkerClass;

        let workers = process::workers().unwrap();
        let (release, waiting) = std::sync::mpsc::channel();
        let blocker = workers
            .try_spawn(MediaWorkerClass::EotModel, move || waiting.recv().unwrap())
            .expect("reserve the one model-construction slot");
        let shared =
            Arc::new(Mutex::new(ThresholdOptions::new(TurnDetectorModel::V1Mini, None, None)));
        let session = EotSession::new(TurnDetectorModel::V1Mini, shared.clone(), true);
        let constructions = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let constructed = constructions.clone();
        let make: TransportFactory = Arc::new(move |model| {
            constructed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            TransportShell::mount(model.as_str(), shared.clone(), |sink| {
                Arc::new(LocalTransport::new(Arc::new(Model(Arc::default())), sink))
            })
        });
        let stream = EotStream::open(
            session,
            16_000,
            make,
            Process::workers,
            None,
            &lk_agents_abi::TokioSpawner::current(),
        );
        let first = tokio::time::timeout(Duration::from_secs(1), stream.predict())
            .await
            .expect("refusal settles the pending prediction")
            .unwrap();
        assert_eq!(first.end_of_turn_probability, 1.0);
        assert!(!stream.is_degraded(), "temporary capacity refusal must preserve local detection");
        assert_eq!(constructions.load(std::sync::atomic::Ordering::SeqCst), 0);
        let refused_again = tokio::time::timeout(Duration::from_secs(1), stream.predict())
            .await
            .expect("another prediction while saturated is explicitly refused")
            .unwrap();
        assert_eq!(refused_again.end_of_turn_probability, 1.0);
        assert!(!stream.is_degraded());
        release.send(()).unwrap();
        blocker.await.unwrap();
        tokio::task::yield_now().await;
        assert_eq!(
            constructions.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "capacity returning must not start a queued constructor"
        );
        stream.push_audio(lk_agents_abi::AudioFrame::new(vec![0; 3200], 16_000, 1, 1600));
        let next = tokio::time::timeout(Duration::from_secs(1), stream.predict())
            .await
            .expect("the next prediction remounts local inference")
            .unwrap();
        assert_eq!(next.end_of_turn_probability, 0.7);
        assert_eq!(constructions.load(std::sync::atomic::Ordering::SeqCst), 1);
        stream.aclose();
        stream.wait_closed().await;
    });
}
