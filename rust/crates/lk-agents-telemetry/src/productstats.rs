use lk_agents_abi::productstats::MeasureProduct as _;
use lk_agents_abi::telemetry::decl::{MetricDecl, Number, ProductStats};

use crate::Cx;

// LiveKit's product statistics are recorded like observability: `measure` under a context in
// async code, `Trace::measure_product` in a sans-I/O machine. Both make a `Record::ProductStat`
// for the context's recorder; no state lives here.

/// Records one measurement of `M` under `labels`, through `cx`'s recorder; under [`Cx::NONE`]
/// nothing is recorded. Only a product statistic records here:
///
/// ```
/// use lk_agents_abi::productstats::testkit::TestDuration;
/// use lk_agents_telemetry::{Cx, productstats};
/// productstats::measure(&Cx::NONE, &TestDuration, 1.0);
/// ```
///
/// A customer instrument does not compile:
///
/// ```compile_fail,E0271
/// use lk_agents_abi::telemetry::instruments::WorkerProcessReadyDuration;
/// use lk_agents_telemetry::{Cx, productstats};
/// productstats::measure(&Cx::NONE, &WorkerProcessReadyDuration, 1.0);
/// ```
pub fn measure<M: MetricDecl<Audience = ProductStats>>(
    cx: &Cx,
    labels: &M,
    value: impl Into<Number>,
) {
    let mut cx = cx;
    cx.measure_product(labels, value.into());
}

/// What a test's own recorder was handed.
#[cfg(any(test, feature = "testkit"))]
pub mod testkit {
    use std::sync::Arc;

    use lk_agents_abi::VirtualClock;
    use lk_agents_abi::productstats::testkit::{Stat, stat};
    use lk_agents_abi::telemetry::record::{Record, RecordSink};
    use parking_lot::Mutex;

    #[derive(Default)]
    struct Records(Mutex<Vec<Record>>);

    impl RecordSink for Records {
        fn record(&self, record: Record) {
            self.0.lock().push(record);
        }
    }

    /// A tracer of the test's own: the product statistics recorded under its context.
    pub struct Recorded {
        records: Arc<Records>,
        tracer: crate::Tracer,
    }

    impl Default for Recorded {
        fn default() -> Self {
            let records = Arc::new(Records::default());
            let tracer = crate::Tracer::new(records.clone(), Arc::new(VirtualClock::new()), 1);
            Recorded { records, tracer }
        }
    }

    impl Recorded {
        /// The context to record under.
        pub fn cx(&self) -> crate::Cx {
            self.tracer.root()
        }

        /// The product statistics among `records`, in order.
        pub fn stats(records: impl IntoIterator<Item = Record>) -> Vec<Stat> {
            let stats = records.into_iter().filter_map(|record| match record {
                Record::ProductStat(measured) => Some(stat(measured)),
                _ => None,
            });
            stats.collect()
        }

        /// The product statistics recorded since the last call.
        pub fn take(&self) -> Vec<Stat> {
            Recorded::stats(std::mem::take(&mut *self.records.0.lock()))
        }
    }
}
