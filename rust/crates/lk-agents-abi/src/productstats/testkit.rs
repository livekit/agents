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
