//! What the host's logger and meters record for a metrics event.

use lk_agents_abi::MetricsReport;
use lk_agents_abi::metrics::{AgentMetrics, LogValue, OtelWire, log_fields, turn_records};
use serde_json::{Map, Value};

pub struct MetricsRecords;

impl lk_agents_abi::metrics::MetricsRecords for MetricsRecords {
    fn metrics_log_fields(metrics: AgentMetrics) -> Option<(String, Map<String, Value>)> {
        let out = log_fields(&metrics)?;
        let fields = out
            .fields
            .into_iter()
            .map(|(key, value)| {
                let value = match value {
                    LogValue::Str(s) => s.into(),
                    LogValue::Float(f) => f.into(),
                    LogValue::Int(i) => i.into(),
                };
                (key.to_owned(), value)
            })
            .collect();
        Some((out.message.to_owned(), fields))
    }

    fn otel_turn_records(report: Option<MetricsReport>) -> Vec<OtelWire> {
        turn_records(&report.unwrap_or_default()).into_iter().map(|r| r.wire()).collect()
    }

    fn otel_usage_records(metrics: AgentMetrics) -> Vec<OtelWire> {
        lk_agents_abi::metrics::usage_records(&metrics).into_iter().map(|r| r.wire()).collect()
    }
}
