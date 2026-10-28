//! One native privacy projection for retained report events, console snapshots and Cloud records.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use super::RecordingOptions;
use crate::metrics::usage::ModelUsage;
use crate::voice::events::{
    AgentStateChanged, SessionUsageUpdated, UserStateChanged, UserTranscriptionTimeout,
};

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
    pub usage: Option<Vec<ModelUsage>>,
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

#[derive(Deserialize)]
struct EventHeader {
    #[serde(rename = "type")]
    kind: String,
}

// Reuse the actual event/usage schemas. Typed decoding drops extra fields before safe events leave
// the destination boundary; retaining a safe kind never authorizes arbitrary attached content.
#[derive(Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SafeEvent {
    UserStateChanged(UserStateChanged),
    AgentStateChanged(AgentStateChanged),
    UserTranscriptionTimeout(UserTranscriptionTimeout),
    SessionUsageUpdated(SessionUsageUpdated),
}

#[derive(Clone, Copy)]
enum EventPrivacy {
    Safe,
    Content,
    Refused,
}

impl EventPrivacy {
    fn for_kind(kind: &str) -> Self {
        match kind {
            "user_state_changed"
            | "agent_state_changed"
            | "user_transcription_timeout"
            | "session_usage_updated" => Self::Safe,
            "user_input_transcribed"
            | "conversation_item_added"
            | "function_tools_executed"
            | "tool_execution_updated"
            | "agent_false_interruption"
            | "error"
            | "close" => Self::Content,
            // Live handles, media arrays and unknown schemas are never report snapshots.
            _ => Self::Refused,
        }
    }
}

pub fn event_allowed(kind: &str, capture_content: bool, redaction: bool) -> bool {
    match EventPrivacy::for_kind(kind) {
        EventPrivacy::Safe => true,
        EventPrivacy::Content => super::super::policy::content_allowed(capture_content, redaction),
        EventPrivacy::Refused => false,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReportProjection {
    content: bool,
    transcript: bool,
    audio: bool,
    recording: bool,
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

impl SessionReportProjection for ReportProjection {
    fn new(capture_content: bool, options: &RecordingOptions, project_redaction: bool) -> Self {
        let content = super::super::policy::content_allowed(
            capture_content,
            options.redaction || project_redaction,
        );
        Self {
            content,
            transcript: content && options.transcript,
            audio: options.audio && !options.redaction && !project_redaction,
            recording: options.recording_enabled(),
        }
    }

    fn content(&self) -> bool {
        self.content
    }
    fn transcript(&self) -> bool {
        self.transcript
    }
    fn audio(&self) -> bool {
        self.audio
    }

    fn event_limit(&self, items: usize) -> usize {
        items.min(super::super::policy::REPORT_EVENT_ITEMS)
    }

    fn accepts_event(&self, retained_bytes: usize, event_bytes: usize) -> bool {
        use super::super::policy;
        event_bytes <= policy::REPORT_EVENT_RECORD_BYTES
            && event_bytes <= policy::REPORT_EVENT_BYTES.saturating_sub(retained_bytes)
    }

    fn project(&self, mut report: SessionReportSnapshot) -> SessionReportSnapshot {
        if !self.content {
            report.options = None;
        }
        if !self.transcript {
            report.chat_history = Some(json!({ "items": [] }));
        }
        if !self.audio {
            report.audio_recording_path = None;
            report.audio_recording_started_at = None;
        }
        report
    }

    fn project_event(&self, event: Value, retained_bytes: usize) -> Option<ProjectedReportEvent> {
        if !super::report_value_fits(&event) {
            return None;
        }
        let header = EventHeader::deserialize(&event).ok()?;
        let (value, content) = match EventPrivacy::for_kind(&header.kind) {
            EventPrivacy::Safe => {
                let safe = serde_json::from_value::<SafeEvent>(event).ok()?;
                (serde_json::to_value(safe).ok()?, false)
            }
            EventPrivacy::Content if self.content => (event, true),
            _ => return None,
        };
        // Include a separator, and never charge less than either the SDK's retained graph or
        // the native wire representation. Both destinations consequently share the same bound.
        let bytes = super::report_value_bytes(&value)?.saturating_add(1).max(retained_bytes);
        Some(ProjectedReportEvent { value, content, retained_bytes: bytes })
    }

    fn event_record(
        &self,
        event: ProjectedReportEvent,
        identity: (String, String, String),
        timestamp: f64,
        disabled: bool,
    ) -> Option<super::LogRecordWire> {
        if disabled || !self.recording || (event.content && !self.content) {
            return None;
        }
        let key = if event.content { "lk.pii.session_event" } else { "session.event" };
        let value = serde_json::to_string(&event.value).ok()?;
        let attributes = Value::Object(Map::from_iter([(key.to_owned(), Value::String(value))]));
        let mut record = super::record(
            super::LoggerName::ChatHistory,
            "session event",
            super::hr_seconds(timestamp),
            attributes,
            false,
        );
        super::attribute_report_record(
            &mut record,
            &identity.0,
            &identity.1,
            &identity.2,
            self.content,
        );
        super::report_value_fits(&record.attributes).then(|| record.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_safe_event_schemas_survive_without_content_capture() {
        let projection = ReportProjection::new(false, &RecordingOptions::default(), true);
        let samples = [
            json!({"type":"user_state_changed","old_state":"listening","new_state":"speaking","created_at":1.0}),
            json!({"type":"agent_state_changed","old_state":"listening","new_state":"thinking","created_at":2.0}),
            json!({"type":"user_transcription_timeout","speech_duration":0.5,"vad_speech_started_at":2.0,"created_at":3.0}),
            json!({"type":"session_usage_updated","usage":{"model_usage":[{"type":"llm_usage","provider":"provider","model":"model","input_tokens":7}]},"created_at":4.0}),
        ];
        for sample in samples {
            let mut with_extra = sample.clone();
            with_extra.as_object_mut().unwrap().insert("private".into(), json!("not a safe field"));
            let projected = projection.project_event(with_extra, 1024).unwrap();
            assert!(!projected.content);
            assert!(projected.value.get("private").is_none());
            assert_eq!(projected.value["type"], sample["type"]);
            assert_eq!(projected.value["created_at"], sample["created_at"]);
        }
        assert!(
            projection
                .project_event(json!({"type":"user_input_transcribed","transcript":"private"}), 128)
                .is_none()
        );
    }
}
