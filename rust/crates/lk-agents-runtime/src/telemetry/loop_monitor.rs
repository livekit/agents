//! Watchdog thresholds, stall classification, sampling and report admission. Every transition
//! consumes supplied time; host adapters only take clock/stack samples and emit the result.

use std::collections::VecDeque;

use lk_agents_abi::telemetry::loop_monitor::{LoopBlock, LoopBlockSeverity, LoopSampleRequest};

pub const DEFAULT_WARN_THRESHOLD: f64 = 0.1;
pub const DEFAULT_ERROR_THRESHOLD: f64 = 0.5;
pub const ENV_WARN_THRESHOLD_MS: &str = "LIVEKIT_AGENTS_LOOP_BLOCK_WARN_MS";
pub const ENV_ERROR_THRESHOLD_MS: &str = "LIVEKIT_AGENTS_LOOP_BLOCK_ERROR_MS";
const MIN_TICK_INTERVAL: f64 = 0.02;
const MAX_TICK_INTERVAL: f64 = 0.05;
const MAX_SPANS_PER_MINUTE: usize = 30;
const MAX_LOGS_PER_MINUTE: usize = 5;

#[derive(Debug, Clone, Copy)]
pub struct LoopMonitorOptions {
    pub warn: f64,
    pub error: f64,
    pub tick: f64,
    pub emit_spans: bool,
}

impl LoopMonitorOptions {
    pub fn new(
        mut warn: f64,
        mut error: f64,
        tick: Option<f64>,
        emit_spans: bool,
    ) -> Result<Self, &'static str> {
        if !warn.is_finite() || !error.is_finite() || tick.is_some_and(|v| !v.is_finite()) {
            return Err("loop monitor timing values must be finite");
        }
        if warn <= 0.0 {
            return Err("warn_threshold must be > 0");
        }
        if error < warn {
            return Err("error_threshold must be >= warn_threshold");
        }
        let tick = match tick {
            Some(tick) => tick,
            None => {
                if warn < MIN_TICK_INTERVAL {
                    log::warn!(target: "livekit.agents", "loop monitor warning threshold raised to the 20ms sampling floor");
                    warn = MIN_TICK_INTERVAL;
                    error = error.max(warn);
                }
                (warn / 5.0).clamp(MIN_TICK_INTERVAL, MAX_TICK_INTERVAL)
            }
        };
        if tick <= 0.0 || tick > warn {
            return Err("tick_interval must be > 0 and <= warn_threshold");
        }
        Ok(Self { warn, error, tick, emit_spans })
    }
}

fn env_seconds(name: &str, default: f64) -> f64 {
    let Ok(raw) = std::env::var(name) else { return default };
    if raw.trim().is_empty() {
        return default;
    }
    match raw.trim().parse::<f64>() {
        Ok(ms) if ms.is_finite() && ms >= 0.0 => ms / 1000.0,
        _ => {
            log::warn!(target: "livekit.agents", "invalid {name}={raw:?}; expected finite milliseconds >= 0; using {:.0}ms", default * 1000.0);
            default
        }
    }
}

pub fn thresholds_from_env() -> Option<(f64, f64)> {
    let warn = env_seconds(ENV_WARN_THRESHOLD_MS, DEFAULT_WARN_THRESHOLD);
    if warn == 0.0 {
        return None;
    }
    let error = env_seconds(ENV_ERROR_THRESHOLD_MS, DEFAULT_ERROR_THRESHOLD);
    if error < warn {
        log::warn!(target: "livekit.agents", "{ENV_ERROR_THRESHOLD_MS} is below {ENV_WARN_THRESHOLD_MS}; using the warning threshold for both");
    }
    Some((warn, error.max(warn)))
}

struct RateLimiter {
    limit: usize,
    events: VecDeque<f64>,
    suppressed: u64,
}

impl RateLimiter {
    fn new(limit: usize) -> Self {
        Self { limit, events: VecDeque::new(), suppressed: 0 }
    }

    fn allow(&mut self, now: f64) -> bool {
        while self.events.front().is_some_and(|at| *at < now - 60.0) {
            self.events.pop_front();
        }
        if self.events.len() == self.limit {
            self.suppressed = self.suppressed.saturating_add(1);
            return false;
        }
        self.events.push_back(now);
        true
    }
}

pub struct LoopMonitorPolicy {
    options: LoopMonitorOptions,
    last_tick_at: f64,
    tick_sequence: u64,
    last_thread_cpu: f64,
    gc_started_at: Option<f64>,
    gc_time: f64,
    watchdog_late: Option<(f64, f64)>,
    watchdog_last_wake: f64,
    samples: u8,
    late_sampled: bool,
    spans: RateLimiter,
    logs: RateLimiter,
}

impl LoopMonitorPolicy {
    pub fn new(options: LoopMonitorOptions) -> Self {
        Self {
            options,
            last_tick_at: 0.0,
            tick_sequence: 0,
            last_thread_cpu: 0.0,
            gc_started_at: None,
            gc_time: 0.0,
            watchdog_late: None,
            watchdog_last_wake: 0.0,
            samples: 0,
            late_sampled: false,
            spans: RateLimiter::new(MAX_SPANS_PER_MINUTE),
            logs: RateLimiter::new(MAX_LOGS_PER_MINUTE),
        }
    }

    pub fn arm(&mut self, now: f64, thread_cpu: f64) {
        self.last_tick_at = now;
        self.watchdog_last_wake = now;
        self.last_thread_cpu = thread_cpu;
    }

    pub fn gc_started(&mut self, now: f64) {
        self.gc_started_at = Some(now);
    }

    pub fn gc_ended(&mut self, now: f64) {
        if let Some(start) = self.gc_started_at.take() {
            self.gc_time += (now - start).max(0.0);
        }
    }

    pub fn watchdog_woke(&mut self, before: f64, now: f64) {
        let gap = (now - before - self.options.tick).max(0.0);
        if self.watchdog_late.is_none_or(|(_, late)| gap > late) {
            self.watchdog_late = Some((now, gap));
        }
        self.watchdog_last_wake = now;
    }

    /// Reserves at most two samples per incident before invoking the host outside the lock.
    pub fn sample_due(&mut self, now: f64) -> Option<LoopSampleRequest> {
        let lag = now - self.last_tick_at - self.options.tick;
        let first_lag = (self.options.warn - self.options.tick).max(self.options.tick);
        if lag < first_lag
            || (self.samples > 0 && (self.late_sampled || lag < self.options.warn * 10.0))
        {
            return None;
        }
        self.samples += 1;
        self.late_sampled = lag >= self.options.warn * 10.0;
        Some(LoopSampleRequest { tick_sequence: self.tick_sequence, lag })
    }

    pub fn tick(&mut self, now: f64, epoch: f64, thread_cpu: f64) -> Option<LoopBlock> {
        let tick_sequence = self.tick_sequence;
        let window_start = self.last_tick_at;
        let lag = now - self.last_tick_at - self.options.tick;
        let cpu_time = (thread_cpu - self.last_thread_cpu).max(0.0);
        let gc_time = std::mem::take(&mut self.gc_time).min(lag.max(0.0));
        self.tick_sequence += 1;
        self.last_tick_at = now;
        self.last_thread_cpu = thread_cpu;
        self.samples = 0;
        self.late_sampled = false;
        let mut watchdog_gap = self
            .watchdog_late
            .take()
            .filter(|(at, _)| *at >= window_start)
            .map_or(0.0, |(_, gap)| gap);
        if self.watchdog_last_wake < window_start {
            watchdog_gap = watchdog_gap.max(now - self.watchdog_last_wake - self.options.tick);
        }
        if lag < self.options.warn {
            return None;
        }
        let process_descheduled = watchdog_gap >= lag * 0.5 && cpu_time < lag * 0.5;
        let severity = if lag >= self.options.error && !process_descheduled {
            LoopBlockSeverity::Error
        } else {
            LoopBlockSeverity::Warning
        };
        let span_allowed = self.spans.allow(now);
        let emit_span = self.options.emit_spans && span_allowed;
        let emit_log = !process_descheduled && self.logs.allow(now);
        let suppressed = if emit_span { std::mem::take(&mut self.spans.suppressed) } else { 0 };
        Some(LoopBlock {
            tick_sequence,
            duration: lag,
            started_at: epoch - lag,
            warn_threshold: self.options.warn,
            severity,
            gc_time,
            cpu_time,
            watchdog_gap,
            process_descheduled,
            emit_span,
            emit_log,
            reported: span_allowed || emit_log,
            suppressed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> LoopMonitorPolicy {
        LoopMonitorPolicy::new(LoopMonitorOptions::new(0.1, 0.5, Some(0.02), true).unwrap())
    }

    #[test]
    fn nonfinite_timing_is_rejected_before_a_watchdog_is_started() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(LoopMonitorOptions::new(value, 0.5, Some(0.02), true).is_err());
            assert!(LoopMonitorOptions::new(0.1, value, Some(0.02), true).is_err());
            assert!(LoopMonitorOptions::new(0.1, 0.5, Some(value), true).is_err());
        }
    }

    #[test]
    fn thresholds_classify_supplied_time_and_sampling_is_bounded() {
        let mut p = policy();
        p.arm(0.0, 0.0);
        assert!(p.sample_due(0.09).is_none());
        let first = p.sample_due(0.11).unwrap();
        assert_eq!(first.tick_sequence, 0);
        assert!(p.sample_due(0.8).is_none());
        assert_eq!(p.sample_due(1.1).unwrap().tick_sequence, 0);
        assert!(p.sample_due(10.0).is_none());
        assert!(p.tick(0.1, 100.0, 0.1).is_none());
        let warning = p.tick(0.23, 101.0, 0.2).unwrap();
        assert_eq!(warning.severity, LoopBlockSeverity::Warning);
        assert!((warning.started_at - 100.89).abs() < 1e-10);
        let error = p.tick(0.9, 102.0, 0.85).unwrap();
        assert_eq!(error.severity, LoopBlockSeverity::Error);
    }

    #[test]
    fn descheduling_does_not_spend_log_quota_and_limits_recover() {
        let mut p = policy();
        p.arm(0.0, 0.0);
        p.watchdog_woke(0.0, 1.0);
        let descheduled = p.tick(1.0, 1.0, 0.0).unwrap();
        assert!(descheduled.process_descheduled);
        assert!(!descheduled.emit_log);
        assert_eq!(descheduled.severity, LoopBlockSeverity::Warning);
        for n in 1..=31 {
            let at = 1.0 + n as f64;
            p.watchdog_woke(at - 0.02, at);
            let block = p.tick(at, at, n as f64).unwrap();
            assert_eq!(block.emit_log, n <= 5);
            assert_eq!(block.emit_span, n < 30);
        }
        p.arm(100.0, 31.0);
        p.watchdog_woke(100.98, 101.0);
        let recovered = p.tick(101.0, 101.0, 32.0).unwrap();
        assert!(recovered.emit_span && recovered.emit_log);
        assert_eq!(recovered.suppressed, 2);
    }

    #[test]
    fn watchdog_wake_order_and_stale_gaps_do_not_cross_attribute_stalls() {
        let mut p = policy();
        p.arm(0.0, 0.0);
        assert!(p.tick(0.02, 100.0, 0.02).is_none());
        let loop_first = p.tick(0.54, 101.0, 0.021).unwrap();
        assert!(loop_first.process_descheduled);

        p.arm(2.0, 1.0);
        p.watchdog_woke(2.0, 2.5);
        let watchdog_first = p.tick(2.54, 102.0, 1.001).unwrap();
        assert!(watchdog_first.process_descheduled);

        p.arm(4.0, 3.0);
        p.watchdog_woke(3.0, 3.8);
        p.watchdog_woke(4.49, 4.51);
        let blocking_wait = p.tick(4.54, 103.0, 3.001).unwrap();
        assert_eq!(blocking_wait.watchdog_gap, 0.0);
        assert!(!blocking_wait.process_descheduled);
        assert_eq!(blocking_wait.severity, LoopBlockSeverity::Error);

        p.arm(6.0, 4.0);
        p.watchdog_woke(6.0, 6.5);
        let gil_held = p.tick(6.54, 104.0, 4.52).unwrap();
        assert!(!gil_held.process_descheduled);
        assert_eq!(gil_held.severity, LoopBlockSeverity::Error);
    }

    #[test]
    fn disabling_spans_preserves_report_observers_after_the_log_quota() {
        let options = LoopMonitorOptions::new(0.1, 0.5, Some(0.02), false).unwrap();
        let mut p = LoopMonitorPolicy::new(options);
        p.arm(0.0, 0.0);
        for n in 1..=31 {
            let at = n as f64;
            p.watchdog_woke(at - 0.02, at);
            let report = p.tick(at, at, at).unwrap();
            assert!(!report.emit_span);
            assert_eq!(report.emit_log, n <= 5);
            assert_eq!(report.reported, n <= 30);
        }
    }
}

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

        let permit = super::work::try_admit_monitor().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "loop monitor worker capacity exhausted",
            )
        })?;
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
            let _permit = permit;
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

impl lk_agents_abi::telemetry::loop_monitor::LoopMonitorThresholds for LoopMonitorOptions {
    fn _core_loop_monitor_thresholds_from_env() -> Option<(f64, f64)> {
        thresholds_from_env()
    }
}
