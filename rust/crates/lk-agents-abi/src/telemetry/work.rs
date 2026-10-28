//! The host SDK's bounded work port. The runtime owns admission and settlement; SDK I/O stays host-side.

use crate::HostRef;

#[crate::host(record(module = "telemetry", emit))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelemetryWorkOutcome {
    IncompleteCleanup,
    Failed,
    Settled,
}

/// Runtime work identity and supplied monotonic time, without an SDK object or exception text.
#[crate::host(record(module = "telemetry", emit))]
#[derive(Debug, Clone)]
pub struct TelemetryWorkObservation {
    pub key: String,
    pub monotonic_ns: u64,
    pub outcome: TelemetryWorkOutcome,
}

/// Process-owned SDK work. Constructing another handle shares the same queue and worker ceiling.
#[crate::host(class(name = "_core_TelemetryWorker"))]
#[async_trait::async_trait]
pub trait TelemetryWorker: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new() -> Self;

    /// Refuses overload or a duplicate live key before retaining the callback. `retained_bytes`
    /// includes the immutable SDK batch captured by the callback; no producer waits for capacity.
    #[host(call)]
    fn submit(&self, key: String, callback: HostRef, retained_bytes: usize) -> bool;

    /// Retention admission for an SDK batch queue, independent of cleanup callback slots.
    #[host(call)]
    fn reserve(&self, key: String, retained_bytes: usize) -> bool;

    #[host(call)]
    fn release(&self, key: String);

    /// `kind` is one of `dropped`, `failed`, `conflict`; unknown categories are refused.
    #[host(call)]
    fn record_drop(&self, kind: String, count: u64);

    /// Refuse unbounded labels/series before host SDK instrument creation.
    #[host(call)]
    fn admit_metric(&self, name: String, attributes: Vec<(String, String)>) -> bool;

    /// Off-loop cleanup only. A deadline never detaches work or releases its capacity early.
    #[host(call)]
    fn wait(&self, key: String, timeout: f64) -> bool;

    #[host(await)]
    async fn wait_async(&self, key: String, timeout: f64) -> bool;

    /// Drain bounded lifecycle observations; timed-out work keeps its captured key and owner.
    #[host(call)]
    fn drain_observations(&self) -> Vec<TelemetryWorkObservation>;

    /// Nonrecursive counters, read without exporting/logging them through this same pipeline.
    #[host(getter)]
    fn diagnostics(&self) -> Vec<(String, u64)>;
}
