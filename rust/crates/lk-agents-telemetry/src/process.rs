use lk_agents_abi::telemetry::decl::{Instrument, MetricDecl};
use lk_agents_abi::telemetry::record::InstrumentKind;
use parking_lot::Mutex;

// Unlike `crate::measure`, these values belong to no job and never reach a sink: a pull exporter
// reads them whole.

/// An instrument's current value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Gauge(f64),
    /// One count per bound (not cumulative), then the overflow count; and the observations' sum.
    Histogram {
        buckets: Vec<u64>,
        sum: f64,
    },
}

/// The process's instruments: [`INSTRUMENTS`], or a test's own.
#[derive(Debug)]
pub struct Instruments(Mutex<Vec<(Instrument, Value)>>);

/// The instruments the worker records and its `/metrics` serves.
pub static INSTRUMENTS: Instruments = Instruments::new();

impl Instruments {
    pub const fn new() -> Self {
        Self(parking_lot::const_mutex(Vec::new()))
    }

    /// Sets gauge `M` to `value`.
    pub fn set<M: MetricDecl>(&self, value: f64) {
        const { assert!(matches!(M::KIND, InstrumentKind::Gauge), "set takes a gauge") };
        self.with::<M>(|current| *current = Value::Gauge(value));
    }

    /// Records one observation of histogram `M`: in the first bucket whose bound is at least
    /// `value`, else the overflow bucket.
    pub fn observe<M: MetricDecl>(&self, value: f64) {
        const { assert!(matches!(M::KIND, InstrumentKind::Histogram), "observe takes a histogram") };
        self.with::<M>(|current| {
            if let Value::Histogram { buckets, sum } = current {
                buckets[M::BOUNDS.partition_point(|bound| *bound < value)] += 1;
                *sum += value;
            }
        });
    }

    /// Every instrument recorded so far, in first-recorded order.
    pub fn snapshot(&self) -> Vec<(Instrument, Value)> {
        self.0.lock().clone()
    }

    fn with<M: MetricDecl>(&self, update: impl FnOnce(&mut Value)) {
        let mut instruments = self.0.lock();
        let index = match instruments.iter().position(|(i, _)| i.name == M::NAME) {
            Some(index) => index,
            None => {
                let value = match M::KIND {
                    InstrumentKind::Histogram => {
                        Value::Histogram { buckets: vec![0; M::BOUNDS.len() + 1], sum: 0.0 }
                    }
                    InstrumentKind::Counter | InstrumentKind::Gauge => Value::Gauge(0.0),
                };
                instruments.push((Instrument::of::<M>(), value));
                instruments.len() - 1
            }
        };
        update(&mut instruments[index].1);
    }
}

impl Default for Instruments {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "process_tests.rs"]
mod tests;
