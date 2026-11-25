use std::sync::Arc;

use lk_agents_abi::productstats::{ProviderLabels, STALL_SIGNATURE_FRAMES};
use lk_agents_abi::telemetry::decl::{Label, MetricDecl, Number, ProductStats};
use lk_agents_abi::telemetry::record::{KeyValues, Measurement, StackFrame};
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
/// use lk_agents_abi::productstats::JobStarts;
/// lk_agents_telemetry::productstats::record(&JobStarts, 1u64);
/// ```
///
/// A customer instrument does not compile:
///
/// ```compile_fail,E0271
/// use lk_agents_abi::telemetry::instruments::WorkerProcessReadyDuration;
/// lk_agents_telemetry::productstats::record(&WorkerProcessReadyDuration, 1.0);
/// ```
pub fn record<M: MetricDecl<Audience = ProductStats>>(labels: &M, value: impl Into<Number>) {
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

/// Adds one to counter `M`.
pub fn count<M: MetricDecl<Audience = ProductStats>>(labels: &M) {
    record(labels, 1u64);
}

/// The labels of a provider statistic; an empty provider or model is `unknown`.
pub fn provider(
    operation: lk_agents_abi::productstats::Operation,
    provider: &str,
    model: &str,
) -> ProviderLabels {
    let known = |v: &str| Label(if v.is_empty() { "unknown".to_owned() } else { v.to_owned() });
    ProviderLabels { operation, provider: known(provider), model: known(model) }
}

/// A stall's signature: the function names of its innermost frames, innermost first.
pub fn stall_signature(stack: Option<&[StackFrame]>) -> Label {
    let names: Vec<&str> = stack
        .unwrap_or_default()
        .iter()
        .take(STALL_SIGNATURE_FRAMES)
        .map(|frame| frame.function.as_str())
        .collect();
    Label(if names.is_empty() { "unknown".to_owned() } else { names.join(" < ") })
}
