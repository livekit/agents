use std::sync::Arc;

use lk_agents_abi::telemetry::decl::{MetricDecl, Number, ProductStats};
use lk_agents_abi::telemetry::record::{KeyValues, Measurement};
use parking_lot::RwLock;

// The process's one product statistics recorder: every measurement point records here, never a
// plugin. Nothing records until the process's exporter is installed, which an opted-out process
// never does; a measurement belongs to no job.

/// Where the process's product statistics go.
pub trait Sink: Send + Sync {
    fn record(&self, measurement: Measurement);
}

static SINK: RwLock<Option<Arc<dyn Sink>>> = RwLock::new(None);

/// Installs the process's sink, unless one is installed: false when one was.
pub fn install(sink: Arc<dyn Sink>) -> bool {
    let mut slot = SINK.write();
    if slot.is_some() {
        return false;
    }
    *slot = Some(sink);
    true
}

/// Removes the installed sink: records are dropped until the next install.
pub fn uninstall() -> Option<Arc<dyn Sink>> {
    SINK.write().take()
}

/// A forked child abandons its parent's sink without dropping it: its task belongs to the
/// parent's runtime.
pub fn forget() {
    std::mem::forget(SINK.write().take());
}

/// Whether a sink is installed.
pub fn enabled() -> bool {
    SINK.read().is_some()
}

/// Records one measurement of `M` under `labels`. Only a product statistic records here:
///
/// ```
/// use lk_agents_abi::productstats::testkit::TestDuration;
/// lk_agents_telemetry::productstats::measure(&TestDuration, 1.0);
/// ```
///
/// A customer instrument does not compile:
///
/// ```compile_fail,E0271
/// use lk_agents_abi::telemetry::instruments::WorkerProcessReadyDuration;
/// lk_agents_telemetry::productstats::measure(&WorkerProcessReadyDuration, 1.0);
/// ```
pub fn measure<M: MetricDecl<Audience = ProductStats>>(labels: &M, value: impl Into<Number>) {
    let Some(sink) = SINK.read().clone() else { return };
    let mut out = KeyValues::new();
    labels.write(&mut out);
    sink.record(Measurement {
        instrument: M::NAME,
        kind: M::KIND,
        unit: M::UNIT,
        bounds: M::BOUNDS,
        value: value.into(),
        labels: out,
        job: None,
    });
}
