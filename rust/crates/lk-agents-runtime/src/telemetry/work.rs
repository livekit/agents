//! A fixed SDK worker pool. Admission covers queued and executing callbacks until settlement,
//! even when flush times out. No SDK or user callback executes under the admission lock.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use lk_agents_abi::HostRef;
use lk_agents_abi::telemetry::work::{
    TelemetryWorkObservation, TelemetryWorkOutcome, TelemetryWorker,
};

const WORKERS: usize = 4;
const JOBS: usize = 64;
const BYTES: usize = 8 * 1024 * 1024;
const MONITORS: usize = 32;
include!(concat!(env!("OUT_DIR"), "/telemetry_event_limits.rs"));

static PROCESS_WORK: OnceLock<Arc<Work>> = OnceLock::new();

type Callback = Box<dyn FnOnce() -> bool + Send>;

struct Pending {
    bytes: usize,
    admitted_at: Instant,
    monitor: bool,
    timeout_reported: bool,
}

struct State {
    queue: VecDeque<(Callback, WorkPermit)>,
    sdk_available: bool,
    pending: BTreeMap<String, Pending>,
    bytes: usize,
    monitors: usize,
    reserved: BTreeMap<String, (usize, Instant)>,
    observations: VecDeque<TelemetryWorkObservation>,
    observation_drops: u64,
    reserved_bytes: usize,
    conflicts: u64,
    next_upload: u64,
    dropped: u64,
    failed: u64,
    completed: u64,
    incomplete_cleanup: u64,
}

pub(super) struct Work {
    pid: u32,
    started_at: Instant,
    sdk_started: OnceLock<()>,
    events: EventQuota,
    state: Mutex<State>,
    metrics: Mutex<super::metrics::MetricSeries>,
    changed: Condvar,
    settled: tokio::sync::Notify,
}

/// Capacity belongs to the work, including across caller cancellation.
pub struct WorkPermit {
    work: Arc<Work>,
    key: String,
    shared_class: Option<crate::media_workers::MediaWorkerPermit>,
}

impl WorkPermit {
    pub(super) fn try_add_bytes(&self, bytes: usize) -> bool {
        let mut state = self.work.state.lock().unwrap_or_else(|p| p.into_inner());
        let Some(total) = state.bytes.checked_add(bytes).filter(|total| *total <= BYTES) else {
            state.dropped += 1;
            return false;
        };
        state.pending.get_mut(&self.key).expect("a live permit owns its pending work").bytes +=
            bytes;
        state.bytes = total;
        true
    }
}

impl Drop for WorkPermit {
    fn drop(&mut self) {
        drop(self.shared_class.take());
        let mut state = self.work.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(pending) = state.pending.remove(&self.key) {
            state.bytes -= pending.bytes;
            state.monitors -= usize::from(pending.monitor);
            state.completed += 1;
            observe(&mut state, &self.key, TelemetryWorkOutcome::Settled, &self.work);
        }
        drop(state);
        self.work.changed.notify_all();
        self.work.settled.notify_waiters();
    }
}

impl Work {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            pid: std::process::id(),
            started_at: Instant::now(),
            sdk_started: OnceLock::new(),
            events: EventQuota::default(),
            state: Mutex::new(State {
                queue: VecDeque::new(),
                sdk_available: false,
                pending: BTreeMap::new(),
                bytes: 0,
                monitors: 0,
                reserved: BTreeMap::new(),
                observations: VecDeque::new(),
                observation_drops: 0,
                reserved_bytes: 0,
                conflicts: 0,
                next_upload: 0,
                dropped: 0,
                failed: 0,
                completed: 0,
                incomplete_cleanup: 0,
            }),
            metrics: Mutex::new(super::metrics::MetricSeries::default()),
            changed: Condvar::new(),
            settled: tokio::sync::Notify::new(),
        })
    }

    #[cfg(test)]
    pub(super) fn start(workers: usize) -> Arc<Self> {
        let work = Self::new();
        work.sdk_started.get_or_init(|| work.start_workers(workers));
        work
    }

    fn start_workers(self: &Arc<Self>, workers: usize) {
        if workers == 0 {
            self.state.lock().unwrap_or_else(|p| p.into_inner()).sdk_available = true;
            return;
        }
        // Starting host SDK workers starts native process work, so the existing runtime fork
        // guard must observe this before a child could inherit dead worker handles.
        if crate::process::handle().is_err() {
            self.state.lock().unwrap_or_else(|p| p.into_inner()).failed += 1;
            return;
        }
        for worker in 0..workers {
            let owned = self.clone();
            if std::thread::Builder::new()
                .name(format!("lk-telemetry-{worker}"))
                .spawn(move || owned.run())
                .is_err()
            {
                self.state.lock().unwrap_or_else(|p| p.into_inner()).failed += 1;
            } else {
                self.state.lock().unwrap_or_else(|p| p.into_inner()).sdk_available = true;
            }
        }
    }

    fn admit(self: &Arc<Self>, key: String, bytes: usize, monitor: bool) -> Option<WorkPermit> {
        if self.pid != std::process::id() {
            return None;
        }
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let retained = bytes.checked_add(key.len());
        if key.len() > 128
            || state.pending.contains_key(&key)
            || state.pending.len() >= JOBS
            || retained.is_none_or(|bytes| bytes > BYTES.saturating_sub(state.bytes))
            || (monitor && state.monitors >= MONITORS)
        {
            state.dropped += 1;
            return None;
        }
        let bytes = retained.expect("validated above");
        state.bytes += bytes;
        state.monitors += usize::from(monitor);
        state.pending.insert(
            key.clone(),
            Pending { bytes, admitted_at: Instant::now(), monitor, timeout_reported: false },
        );
        Some(WorkPermit { work: self.clone(), key, shared_class: None })
    }

    fn submit(self: &Arc<Self>, key: String, bytes: usize, callback: Callback) -> bool {
        if self.pid != std::process::id() {
            return false;
        }
        self.sdk_started.get_or_init(|| self.start_workers(WORKERS));
        {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if !state.sdk_available {
                state.dropped += 1;
                return false;
            }
        }
        let Some(permit) = self.admit(key, bytes, false) else {
            return false;
        };
        self.state.lock().unwrap_or_else(|p| p.into_inner()).queue.push_back((callback, permit));
        self.changed.notify_one();
        true
    }

    fn run(&self) {
        loop {
            let (callback, permit) = {
                let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
                loop {
                    if let Some(queued) = state.queue.pop_front() {
                        break queued;
                    }
                    state = self.changed.wait(state).unwrap_or_else(|p| p.into_inner());
                }
            };
            if !std::panic::catch_unwind(std::panic::AssertUnwindSafe(callback)).unwrap_or(false) {
                let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
                state.failed += 1;
                observe(&mut state, &permit.key, TelemetryWorkOutcome::Failed, self);
            }
            drop(permit);
        }
    }

    fn timeout(&self, key: &str) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(pending) = state.pending.get_mut(key) {
            if !pending.timeout_reported {
                pending.timeout_reported = true;
                state.incomplete_cleanup += 1;
                observe(&mut state, key, TelemetryWorkOutcome::IncompleteCleanup, self);
            }
        }
    }

    fn wait(&self, key: &str, timeout: f64) -> bool {
        if self.pid != std::process::id() {
            return false;
        }
        let Some(timeout) = duration(timeout) else {
            return false;
        };
        let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let (state, _) = self
            .changed
            .wait_timeout_while(state, timeout, |s| s.pending.contains_key(key))
            .unwrap_or_else(|p| p.into_inner());
        let done = !state.pending.contains_key(key);
        drop(state);
        if !done {
            self.timeout(key);
        }
        done
    }

    async fn wait_async(&self, key: &str, timeout: f64) -> bool {
        if self.pid != std::process::id() {
            return false;
        }
        let Some(timeout) = duration(timeout) else {
            return false;
        };
        let wait = async {
            loop {
                let notified = self.settled.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                if !self.state.lock().unwrap_or_else(|p| p.into_inner()).pending.contains_key(key) {
                    return;
                }
                notified.await;
            }
        };
        if tokio::time::timeout(timeout, wait).await.is_ok() {
            true
        } else {
            self.timeout(key);
            false
        }
    }
}

fn observe(state: &mut State, key: &str, outcome: TelemetryWorkOutcome, work: &Work) {
    if state.observations.len() == JOBS {
        state.observations.pop_front();
        state.observation_drops = state.observation_drops.saturating_add(1);
    }
    state.observations.push_back(TelemetryWorkObservation {
        key: key.to_owned(),
        monotonic_ns: work.started_at.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        outcome,
    });
}

fn duration(seconds: f64) -> Option<Duration> {
    if seconds.is_finite() && seconds >= 0.0 && seconds <= 3600.0 {
        Some(Duration::from_secs_f64(seconds))
    } else {
        None
    }
}

fn process_work() -> &'static Arc<Work> {
    PROCESS_WORK.get_or_init(Work::new)
}

/// Recording preparation and transport share the process native upload class. SDK byte accounting
/// remains attached to the same work permit until actual settlement.
pub fn try_admit_upload(retained_bytes: usize) -> Option<WorkPermit> {
    let work = process_work();
    if work.pid != std::process::id() {
        return None;
    }
    let shared_class = crate::process::workers()
        .ok()
        .and_then(|workers| workers.try_admit(crate::media_workers::MediaWorkerClass::Upload).ok());
    let Some(shared_class) = shared_class else {
        work.state.lock().unwrap_or_else(|p| p.into_inner()).dropped += 1;
        return None;
    };
    let key = {
        let mut state = work.state.lock().unwrap_or_else(|p| p.into_inner());
        state.next_upload = state.next_upload.wrapping_add(1);
        format!("upload/{}", state.next_upload)
    };
    let mut permit = work.admit(key, retained_bytes, false)?;
    permit.shared_class = Some(shared_class);
    Some(permit)
}

/// Watchdogs use the same admission owner and retain capacity until their thread settles.
pub fn try_admit_monitor() -> Option<WorkPermit> {
    let work = process_work();
    if work.pid != std::process::id() {
        return None;
    }
    let key = {
        let mut state = work.state.lock().unwrap_or_else(|p| p.into_inner());
        state.next_upload = state.next_upload.wrapping_add(1);
        format!("monitor/{}", state.next_upload)
    };
    work.admit(key, 0, true)
}

pub fn record_incomplete_cleanup() {
    process_work().state.lock().unwrap_or_else(|p| p.into_inner()).incomplete_cleanup += 1;
}

/// Charges follow the retained snapshots, independently of SDK callbacks and their settlement.
/// All session stores share the process telemetry admission owner.
pub(super) struct EventBudget {
    work: Option<Arc<Work>>,
}

#[derive(Default)]
struct EventQuota {
    items: AtomicUsize,
    bytes: AtomicUsize,
    dropped: AtomicU64,
}

impl EventQuota {
    // Quotas publish no payload; in-flight reservations may conservatively refuse optional data.
    fn admit(&self, bytes: usize) -> bool {
        if self
            .items
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |items| {
                (items < PROCESS_EVENT_ITEMS).then_some(items + 1)
            })
            .is_err()
        {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        if self
            .bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |retained| {
                retained.checked_add(bytes).filter(|total| *total <= PROCESS_EVENT_BYTES)
            })
            .is_err()
        {
            self.items.fetch_sub(1, Ordering::Relaxed);
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        true
    }

    fn release(&self, items: usize, bytes: usize) {
        self.bytes.fetch_sub(bytes, Ordering::Relaxed);
        self.items.fetch_sub(items, Ordering::Relaxed);
    }
}

impl EventBudget {
    #[cfg(test)]
    pub(super) fn new(work: Arc<Work>) -> Self {
        Self { work: Some(work) }
    }

    pub(super) fn admit(&mut self, bytes: usize) -> bool {
        let work = self.work.get_or_insert_with(|| process_work().clone());
        if work.pid != std::process::id() {
            return false;
        }
        work.events.admit(bytes)
    }

    pub(super) fn release(&mut self, items: usize, bytes: usize) {
        if items == 0 {
            return;
        }
        self.work.as_ref().expect("retained events own their budget").events.release(items, bytes);
    }
}

impl Default for EventBudget {
    fn default() -> Self {
        // Configuration may be created during forkserver preload, before a process owns events.
        Self { work: None }
    }
}

pub struct SdkWorker;

impl TelemetryWorker for SdkWorker {
    fn new() -> Self {
        Self
    }

    fn submit(&self, key: String, callback: HostRef, retained_bytes: usize) -> bool {
        #[cfg(feature = "python")]
        let work: Callback = Box::new(move || {
            use pyo3::prelude::*;
            Python::attach(|py| -> PyResult<bool> {
                let callback = lk_agents_abi::python::resolve_key(py, callback.id())?;
                let result = callback.bind(py).call0()?;
                // SDK force_flush reports failure with False; shutdown/report callbacks return None.
                if result.is_instance_of::<pyo3::types::PyBool>() {
                    result.extract()
                } else {
                    Ok(true)
                }
            })
            .unwrap_or(false)
        });
        #[cfg(not(feature = "python"))]
        let work: Callback = Box::new(move || {
            drop(callback);
            false
        });
        process_work().submit(key, retained_bytes, work)
    }

    fn reserve(&self, key: String, retained_bytes: usize) -> bool {
        if process_work().pid != std::process::id() {
            return false;
        }
        let mut state = process_work().state.lock().unwrap_or_else(|p| p.into_inner());
        let bytes = retained_bytes.checked_add(key.len());
        if key.len() > 128
            || state.reserved.contains_key(&key)
            || state.reserved.len() >= JOBS
            || bytes.is_none_or(|bytes| bytes > BYTES.saturating_sub(state.reserved_bytes))
        {
            state.dropped += 1;
            return false;
        }
        let bytes = bytes.expect("validated above");
        state.reserved_bytes += bytes;
        state.reserved.insert(key, (bytes, Instant::now()));
        true
    }

    fn release(&self, key: String) {
        if process_work().pid != std::process::id() {
            return;
        }
        let mut state = process_work().state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((bytes, _)) = state.reserved.remove(&key) {
            state.reserved_bytes -= bytes;
        }
    }

    fn record_drop(&self, kind: String, count: u64) {
        if process_work().pid != std::process::id() {
            return;
        }
        let mut state = process_work().state.lock().unwrap_or_else(|p| p.into_inner());
        let counter = match kind.as_str() {
            "dropped" | "oversize" | "capacity" => &mut state.dropped,
            "failed" | "export_failed" => &mut state.failed,
            "conflict" => &mut state.conflicts,
            _ => return,
        };
        *counter = counter.saturating_add(count);
    }

    fn admit_metric(&self, name: String, attributes: Vec<(String, String)>) -> bool {
        let work = process_work();
        if work.pid != std::process::id() {
            return false;
        }
        if attributes.len() > 8 || name.len() > 128 {
            work.state.lock().unwrap_or_else(|p| p.into_inner()).dropped += 1;
            return false;
        }
        let admitted = work
            .metrics
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .admit(&name, attributes.into_iter().collect());
        if !admitted {
            work.state.lock().unwrap_or_else(|p| p.into_inner()).dropped += 1;
        }
        admitted
    }

    fn wait(&self, key: String, timeout: f64) -> bool {
        #[cfg(feature = "python")]
        {
            pyo3::Python::attach(|py| py.detach(move || process_work().wait(&key, timeout)))
        }
        #[cfg(not(feature = "python"))]
        {
            process_work().wait(&key, timeout)
        }
    }

    async fn wait_async(&self, key: String, timeout: f64) -> bool {
        process_work().wait_async(&key, timeout).await
    }

    fn drain_observations(&self) -> Vec<TelemetryWorkObservation> {
        let work = process_work();
        if work.pid != std::process::id() {
            return Vec::new();
        }
        work.state.lock().unwrap_or_else(|p| p.into_inner()).observations.drain(..).collect()
    }

    fn diagnostics(&self) -> Vec<(String, u64)> {
        if process_work().pid != std::process::id() {
            return vec![("foreign_process".into(), 1)];
        }
        let state = process_work().state.lock().unwrap_or_else(|p| p.into_inner());
        [
            ("dropped", state.dropped),
            ("session_event_items", process_work().events.items.load(Ordering::Relaxed) as u64),
            ("session_event_bytes", process_work().events.bytes.load(Ordering::Relaxed) as u64),
            ("session_event_drops", process_work().events.dropped.load(Ordering::Relaxed)),
            ("observation_drops", state.observation_drops),
            ("failed", state.failed),
            ("completed", state.completed),
            ("incomplete_cleanup", state.incomplete_cleanup),
            ("conflicts", state.conflicts),
            ("reserved_records", state.reserved.len() as u64),
            ("reserved_bytes", state.reserved_bytes as u64),
            ("pending", state.pending.len() as u64),
            ("retained_bytes", state.bytes as u64),
            (
                "oldest_work_age_ms",
                state
                    .pending
                    .values()
                    .map(|p| p.admitted_at.elapsed().as_millis() as u64)
                    .max()
                    .unwrap_or(0),
            ),
            (
                "oldest_record_age_ms",
                state
                    .reserved
                    .values()
                    .map(|(_, at)| at.elapsed().as_millis() as u64)
                    .max()
                    .unwrap_or(0),
            ),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value))
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timed_out_work_stays_admitted_and_reports_cleanup_once() {
        let work = Work::start(1);
        let (send, recv) = std::sync::mpsc::channel();
        assert!(work.submit("blocked".into(), BYTES - 7, Box::new(move || recv.recv().is_ok())));
        let start = Instant::now();
        for _ in 0..10 {
            assert!(!work.wait("blocked", 0.001));
        }
        assert!(start.elapsed() < Duration::from_secs(1));
        assert!(!work.submit("extra".into(), 1, Box::new(|| true)));
        assert!(!work.submit("blocked".into(), 0, Box::new(|| true)));
        let state = work.state.lock().unwrap();
        assert_eq!(state.incomplete_cleanup, 1);
        assert_eq!(state.observations.len(), 1);
        assert_eq!(state.observations[0].key, "blocked");
        assert_eq!(state.observations[0].outcome, TelemetryWorkOutcome::IncompleteCleanup);
        drop(state);
        send.send(()).unwrap();
        assert!(work.wait("blocked", 1.0));
        assert!(work.submit("extra".into(), 1, Box::new(|| true)));
        assert!(work.wait("extra", 1.0));
    }

    #[tokio::test]
    async fn slow_sdk_work_does_not_stop_async_progress_or_other_workers() {
        let work = Work::start(2);
        let (send, recv) = std::sync::mpsc::channel();
        assert!(work.submit("blocked".into(), 0, Box::new(move || recv.recv().is_ok())));
        assert!(work.submit("fast".into(), 0, Box::new(|| false)));
        assert!(work.wait_async("fast", 1.0).await);
        assert_eq!(work.state.lock().unwrap().failed, 1);
        assert!(!work.wait_async("blocked", 0.001).await);
        send.send(()).unwrap();
        assert!(work.wait_async("blocked", 1.0).await);
    }

    #[test]
    fn no_worker_can_turn_queue_saturation_into_unbounded_retention() {
        let work = Work::start(0);
        for index in 0..JOBS {
            assert!(work.submit(index.to_string(), 1, Box::new(|| true)));
        }
        assert!(!work.submit("over".into(), 1, Box::new(|| true)));
        assert_eq!(work.state.lock().unwrap().pending.len(), JOBS);
    }
}
