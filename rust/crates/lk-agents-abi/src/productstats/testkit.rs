use crate::telemetry::decl::{Attrs, Label, metric};
use crate::telemetry::instruments::DURATION_BOUNDS;

// What the pipeline's tests record: no shipped build declares or sends these.

/// A counter only tests record.
#[metric(
    name = "lk.product.agents.test.counts",
    kind = Counter,
    unit = "{count}",
    audience = ProductStats
)]
#[derive(Attrs)]
pub struct TestCounts {
    /// What a test counted.
    #[attr("lk.product.agents.test.label")]
    pub label: Label,
}

/// A histogram only tests record.
#[metric(
    name = "lk.product.agents.test.duration",
    kind = Histogram,
    unit = "s",
    bounds = DURATION_BOUNDS,
    audience = ProductStats
)]
#[derive(Attrs)]
pub struct TestDuration;

/// One recorded statistic as a test asserts it: its metric, its labels in declaration order, and
/// its value.
pub type Stat = (&'static str, Vec<(String, String)>, f64);

pub fn stat(stat: super::ProductStat) -> Stat {
    use crate::telemetry::record::{Number, Value};
    let m = stat.into_measurement();
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
