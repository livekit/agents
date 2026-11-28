// What tests of the crates that record read back: a recorder of their own.

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

/// A tracer of the test's own, and what was recorded under its context.
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
    /// The tracer, for a test that builds what records under it.
    pub fn tracer(&self) -> crate::Tracer {
        self.tracer.clone()
    }

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

    /// Every record made since the last call.
    pub fn records(&self) -> Vec<Record> {
        std::mem::take(&mut *self.records.0.lock())
    }

    /// The product statistics recorded since the last call.
    pub fn take(&self) -> Vec<Stat> {
        Recorded::stats(self.records())
    }
}
