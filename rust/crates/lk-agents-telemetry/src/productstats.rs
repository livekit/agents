use std::sync::Arc;

use lk_agents_abi::productstats::ProductStat;
use lk_agents_abi::telemetry::decl::{MetricDecl, Number, ProductStats};
use lk_agents_abi::telemetry::record::Measurement;
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
    if recording() {
        record(ProductStat::of(labels, value));
    }
}

/// Records a measurement made elsewhere: a sans-I/O machine's, or several of one model's usage.
pub fn record(stat: ProductStat) {
    let measurement = stat.into_measurement();
    #[cfg(any(test, feature = "testkit"))]
    let Some(measurement) = testkit::capture(measurement) else { return };
    if let Some(sink) = SINK.read().clone() {
        sink.record(measurement);
    }
}

fn recording() -> bool {
    #[cfg(any(test, feature = "testkit"))]
    if testkit::capturing() {
        return true;
    }
    enabled()
}

/// A test's own product statistics: what its thread records, instead of the process's sink.
#[cfg(any(test, feature = "testkit"))]
pub mod testkit {
    use std::cell::RefCell;

    use lk_agents_abi::telemetry::record::{Measurement, Number, Value};

    thread_local! {
        static CAPTURED: RefCell<Option<Vec<Measurement>>> = const { RefCell::new(None) };
    }

    /// One recorded statistic: its metric, its labels in declaration order, and its value.
    pub type Stat = (&'static str, Vec<(String, String)>, f64);

    /// Captures what this thread records until dropped: a test on a current-thread runtime sees
    /// its own statistics and no other test's.
    pub struct Capture(());

    impl Capture {
        pub fn start() -> Self {
            CAPTURED.with(|c| *c.borrow_mut() = Some(Vec::new()));
            Capture(())
        }

        /// What was recorded since the last call.
        pub fn take(&self) -> Vec<Stat> {
            let taken = CAPTURED.with(|c| c.borrow_mut().as_mut().map(std::mem::take));
            taken.unwrap_or_default().into_iter().map(stat).collect()
        }
    }

    impl Drop for Capture {
        fn drop(&mut self) {
            CAPTURED.with(|c| *c.borrow_mut() = None);
        }
    }

    /// A measurement as a test asserts it.
    pub fn stat(m: Measurement) -> Stat {
        let labels = m.labels.into_iter().map(|(key, value)| match value {
            Value::String(s) => (key.into_owned(), s),
            other => (key.into_owned(), format!("{other:?}")),
        });
        let value = match m.value {
            Number::Double(v) => v,
            Number::Int(v) => v as f64,
        };
        (m.instrument, labels.collect(), value)
    }

    pub(super) fn capturing() -> bool {
        CAPTURED.with(|c| c.borrow().is_some())
    }

    /// `None` when the thread's capture took the measurement.
    pub(super) fn capture(measurement: Measurement) -> Option<Measurement> {
        CAPTURED.with(|c| match c.borrow_mut().as_mut() {
            Some(captured) => {
                captured.push(measurement);
                None
            }
            None => Some(measurement),
        })
    }
}
