//! Typed session-report observations shared by host SDK adapters.

use crate::telemetry::{LogRecord, epoch_nanos};
use serde_json::Value;

/// A [`LogRecord`] as the dict `_upload_session_report` hands `logger.emit`: one integer of epoch
/// nanos, where the log handler's records carry an `HrTime` pair.
#[crate::host(record(module = "telemetry", emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct LogRecordWire {
    pub logger: String,
    pub body: String,
    pub timestamp_nanos: i64,
    pub severity_number: u8,
    pub severity_text: String,
    pub attributes: serde_json::Map<String, Value>,
}

impl From<LogRecord> for LogRecordWire {
    fn from(r: LogRecord) -> Self {
        LogRecordWire {
            logger: r.logger,
            body: r.body,
            timestamp_nanos: epoch_nanos(r.timestamp).min(i64::MAX as u128) as i64,
            severity_number: r.severity_number,
            severity_text: r.severity_text,
            attributes: match r.attributes {
                Value::Object(attributes) => attributes,
                _ => unreachable!("session report attributes are always objects"),
            },
        }
    }
}
