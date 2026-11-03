//! What the host's logger records for a metrics event.

pub mod legacy_json;
mod metrics_log;

use lk_agents_abi::metrics::AgentMetrics;
use serde_json::{Map, Value};

use metrics_log::{LogValue, log_fields};

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
}
