//! Run the lock inversion in a child: a blocked native mutex cannot be cancelled by Tokio.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use lk_agents_abi::errors::ApiError;
use lk_agents_abi::tokenize::language::{ThresholdOptions, TurnDetectorModel};

use super::*;

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

#[cfg(feature = "python")]
fn with_host_lock<T>(f: impl FnOnce() -> T) -> T {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_| f())
}

// The feature-free suite exercises the identical ordering with a host execution lock.
#[cfg(not(feature = "python"))]
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
        let stream =
            EotStream::open(session, 16_000, make, None, &lk_agents_abi::TokioSpawner::current());
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
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "eot::deadlock_tests::cloud_fallback_does_not_deadlock_a_host_audio_push",
                "--nocapture",
            ])
            .env(CHILD, mode)
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(status.success(), "fallback child failed: {status}");
                break;
            }
            if Instant::now() >= deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("host audio push deadlocked with cloud fallback (5s watchdog)");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}
