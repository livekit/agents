//! The report projection's records and class; `lk-agents-net`'s session report implements it.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::RecordingOptions;

/// The serialized SDK snapshot. Native projection keeps its schema stable when content is denied.
#[crate::host(record(module = "telemetry", extract, emit))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionReportSnapshot {
    pub job_id: String,
    pub room_id: String,
    pub room: String,
    pub events: Vec<Value>,
    #[host(default = 0)]
    pub omitted_events: usize,
    pub audio_recording_path: Option<String>,
    pub audio_recording_started_at: Option<f64>,
    pub options: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_history: Option<Value>,
    pub timestamp: f64,
    pub usage: Option<crate::usage::Usage>,
    pub sdk_version: String,
}

/// Captured once at the destination boundary; emission does not decode or project it again.
#[crate::host(record(module = "telemetry", extract, emit))]
#[derive(Debug, Clone)]
pub struct ProjectedReportEvent {
    pub value: Value,
    pub content: bool,
    pub retained_bytes: usize,
}

#[crate::host(class(name = "_core_SessionReportProjection"))]
pub trait SessionReportProjection: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(capture_content: bool, options: &RecordingOptions, project_redaction: bool) -> Self;

    #[host(getter)]
    fn content(&self) -> bool;

    #[host(getter)]
    fn transcript(&self) -> bool;

    #[host(getter)]
    fn audio(&self) -> bool;

    /// Optional observations beyond this prefix are omitted without failing the transcript.
    #[host(call)]
    fn event_limit(&self, items: usize) -> usize;

    /// Admission precedes host SDK serialization and native JSON extraction.
    #[host(call)]
    fn accepts_event(&self, retained_bytes: usize, event_bytes: usize) -> bool;

    #[host(call)]
    fn project(&self, report: SessionReportSnapshot) -> SessionReportSnapshot;

    #[host(call)]
    fn project_event(&self, event: Value, retained_bytes: usize) -> Option<ProjectedReportEvent>;

    #[host(call)]
    fn event_record(
        &self,
        event: ProjectedReportEvent,
        identity: (String, String, String),
        timestamp: f64,
        disabled: bool,
    ) -> Option<super::LogRecordWire>;
}
