//! The event-loop monitor's watchdog thread and its clock: it samples the host loop and feeds the
//! core's stall policy (`lk_agents_core::telemetry::loop_monitor`).

/// The runtime's monotonic and wall-clock sampling boundary for the host monitor adapter.
pub struct LoopMonitorClock(std::time::Instant);

impl Default for LoopMonitorClock {
    fn default() -> Self {
        Self(std::time::Instant::now())
    }
}

impl LoopMonitorClock {
    pub fn now(&self) -> f64 {
        self.0.elapsed().as_secs_f64()
    }

    pub fn epoch_seconds(&self) -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
    }
}

/// A watchdog owns one sampling thread. Stopping only signals it; the runtime retains its join
/// until settlement, including after a host sample misses the cleanup deadline.
pub struct LoopWatchdog {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    thread: std::thread::Thread,
}

impl LoopWatchdog {
    pub fn start(
        runtime: &tokio::runtime::Handle,
        name: String,
        tick: f64,
        mut sample: impl FnMut() + Send + 'static,
        incomplete_cleanup: impl FnOnce() + Send + 'static,
    ) -> std::io::Result<Self> {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::time::{Duration, Instant};

        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        let join = std::thread::Builder::new().name(name).spawn(move || {
            while !stopped.load(Ordering::Acquire) {
                std::thread::park_timeout(Duration::from_secs_f64(tick));
                if stopped.load(Ordering::Acquire) {
                    break;
                }
                sample();
            }
        })?;
        let thread = join.thread().clone();
        let stopped = stop.clone();
        runtime.spawn(async move {
            let mut stopping_at = None;
            let mut incomplete_cleanup = Some(incomplete_cleanup);
            while !join.is_finished() {
                if stopped.load(Ordering::Acquire) {
                    let since = stopping_at.get_or_insert_with(Instant::now);
                    if since.elapsed() >= Duration::from_secs(1)
                        && let Some(observe) = incomplete_cleanup.take()
                    {
                        observe();
                    }
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            // `is_finished` guarantees there is no thread wait on the Tokio worker.
            let _ = join.join();
        });
        Ok(Self { stop, thread })
    }

    pub fn stop(&self) {
        self.stop.store(true, std::sync::atomic::Ordering::Release);
        self.thread.unpark();
    }
}

impl Drop for LoopWatchdog {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod watchdog_tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn stopping_does_not_wait_for_a_blocked_sample_and_runtime_keeps_its_owner() {
        let (started, ready) = tokio::sync::oneshot::channel();
        let (release, wait) = std::sync::mpsc::channel();
        let (late, incomplete) = tokio::sync::oneshot::channel();
        let (dropped, mut settled) = tokio::sync::oneshot::channel();
        struct SampleOwner(Option<tokio::sync::oneshot::Sender<()>>);
        impl Drop for SampleOwner {
            fn drop(&mut self) {
                let _ = self.0.take().unwrap().send(());
            }
        }
        let owner = SampleOwner(Some(dropped));
        let mut started = Some(started);
        let watchdog = LoopWatchdog::start(
            &tokio::runtime::Handle::current(),
            "watchdog-cleanup-test".to_owned(),
            0.001,
            move || {
                let _owner = &owner;
                if let Some(started) = started.take() {
                    let _ = started.send(());
                    let _ = wait.recv();
                }
            },
            move || {
                let _ = late.send(());
            },
        )
        .unwrap();
        ready.await.unwrap();
        watchdog.stop();
        drop(watchdog);
        // A deadline observation does not release the blocked sample or its admission.
        tokio::time::timeout(Duration::from_secs(2), incomplete).await.unwrap().unwrap();
        assert!(matches!(
            settled.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ));
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), settled).await.unwrap().unwrap();
    }
}
