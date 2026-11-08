use std::fmt::Write as _;

use lk_agents_abi::telemetry::decl::Instrument;
use lk_agents_telemetry::process::{Instruments, Value};

// An instrument's name is its OpenTelemetry name with `.` as `_`, plus a `_seconds` or `_ratio`
// suffix for those units.

pub const CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

/// The exposition of every instrument recorded in `instruments`.
pub fn exposition(instruments: &Instruments) -> String {
    let mut out = String::new();
    for (instrument, value) in instruments.snapshot() {
        let name = name(&instrument);
        // Writing to a `String` cannot fail.
        let _ = match value {
            Value::Gauge(value) => {
                write!(out, "# TYPE {name} gauge\n{name} {}\n", Sample(value))
            }
            Value::Histogram { buckets, sum } => {
                histogram(&mut out, &name, instrument.bounds, &buckets, sum)
            }
        };
    }
    out
}

fn histogram(
    out: &mut String,
    name: &str,
    bounds: &[f64],
    buckets: &[u64],
    sum: f64,
) -> std::fmt::Result {
    writeln!(out, "# TYPE {name} histogram")?;
    let mut cumulative = 0;
    for (bound, count) in bounds.iter().zip(buckets) {
        cumulative += count;
        writeln!(out, "{name}_bucket{{le=\"{}\"}} {cumulative}", Sample(*bound))?;
    }
    let count: u64 = buckets.iter().sum();
    writeln!(out, "{name}_bucket{{le=\"+Inf\"}} {count}")?;
    write!(out, "{name}_sum {}\n{name}_count {count}\n", Sample(sum))
}

fn name(instrument: &Instrument) -> String {
    let suffix = match instrument.unit {
        "s" => "_seconds",
        "1" => "_ratio",
        _ => "",
    };
    format!("{}{suffix}", instrument.name.replace('.', "_"))
}

/// A sample value as the format spells it: Rust's `inf` is `+Inf`.
struct Sample(f64);

impl std::fmt::Display for Sample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            v if v == f64::INFINITY => f.write_str("+Inf"),
            v if v == f64::NEG_INFINITY => f.write_str("-Inf"),
            v => write!(f, "{v}"),
        }
    }
}

#[cfg(test)]
#[path = "prometheus_tests.rs"]
mod tests;
