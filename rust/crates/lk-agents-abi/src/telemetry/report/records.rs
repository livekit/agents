//! Typed session-report observations shared by host SDK adapters.

use serde_json::Value;

/// A session report record as the dict `_upload_session_report` hands `logger.emit`: one integer
/// of epoch nanos, where the log handler's records carry an `HrTime` pair.
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
