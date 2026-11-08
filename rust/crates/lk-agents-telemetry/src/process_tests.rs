use super::*;
use lk_agents_abi::telemetry::decl::{Attrs, metric};

const BOUNDS: &[f64] = &[1.0, 5.0];

#[metric(name = "test.gauge", kind = Gauge, unit = "1")]
#[derive(Attrs)]
struct TestGauge;

#[metric(name = "test.other", kind = Gauge, unit = "{item}")]
#[derive(Attrs)]
struct OtherGauge;

#[metric(name = "test.time", kind = Histogram, unit = "s", bounds = BOUNDS)]
#[derive(Attrs)]
struct TestTime;

fn value(instruments: &Instruments, name: &str) -> Option<Value> {
    instruments.snapshot().into_iter().find(|(i, _)| i.name == name).map(|(_, value)| value)
}

// Catches: a gauge that accumulates instead of holding its last value, or two gauges sharing one.
#[test]
fn a_gauge_holds_the_last_value_set() {
    let instruments = Instruments::new();
    instruments.set::<TestGauge>(0.25);
    instruments.set::<OtherGauge>(3.0);
    instruments.set::<TestGauge>(0.75);
    assert_eq!(value(&instruments, "test.gauge"), Some(Value::Gauge(0.75)));
    assert_eq!(value(&instruments, "test.other"), Some(Value::Gauge(3.0)));
}

// Catches: an observation on a bound counted in the next bucket (`le` is inclusive), one past
// the last bound dropped instead of overflowing, or a sum that misses an observation.
#[test]
fn a_histogram_counts_each_observation_in_its_bucket() {
    let instruments = Instruments::new();
    for seconds in [0.25, 1.0, 1.5, 5.0, 7.0, 30.0] {
        instruments.observe::<TestTime>(seconds);
    }
    assert_eq!(
        value(&instruments, "test.time"),
        Some(Value::Histogram { buckets: vec![2, 2, 2], sum: 44.75 })
    );
}

// Catches: a snapshot losing the declaration the exporter names and buckets by, or an order that
// changes between scrapes.
#[test]
fn a_snapshot_carries_each_declaration_in_first_recorded_order() {
    let instruments = Instruments::new();
    assert!(instruments.snapshot().is_empty(), "nothing is served before it is recorded");
    instruments.observe::<TestTime>(0.5);
    instruments.set::<TestGauge>(1.0);
    instruments.observe::<TestTime>(0.5);
    let snapshot = instruments.snapshot();
    let declarations: Vec<_> = snapshot.iter().map(|(i, _)| *i).collect();
    assert_eq!(declarations, [Instrument::of::<TestTime>(), Instrument::of::<TestGauge>()]);
    assert_eq!(declarations[0].bounds, BOUNDS);
}
