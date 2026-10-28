//! One native privacy projection for retained report events, console snapshots and Cloud records.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::RecordingOptions;
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
    pub usage: Option<Value>,
    pub sdk_version: String,
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

    /// Admission precedes host SDK serialization and native JSON extraction.
    #[host(call)]
    fn check_events(
        &self,
        items: usize,
        retained_bytes: usize,
        event_bytes: usize,
    ) -> Result<(), super::super::upload::UploadError>;

    #[host(call)]
    fn project(&self, report: SessionReportSnapshot) -> SessionReportSnapshot;

    #[host(call)]
    fn project_event(&self, event: Value) -> Option<Value>;

    #[host(call)]
    fn event_record(
        &self,
        event: Value,
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

    fn check_events(
        &self,
        items: usize,
        retained_bytes: usize,
        event_bytes: usize,
    ) -> Result<(), super::super::upload::UploadError> {
        use super::super::policy;
        if items > policy::REPORT_EVENT_ITEMS
            || retained_bytes > policy::REPORT_EVENT_BYTES
            || event_bytes > policy::REPORT_EVENT_RECORD_BYTES
        {
            return Err(super::super::upload::UploadError(
                "session report events exceed the native input budget".into(),
            ));
        }
        Ok(())
    }

    fn project(&self, mut report: SessionReportSnapshot) -> SessionReportSnapshot {
        let before = report.events.len();
        report.events = self.events(report.events);
        report.omitted_events = report.omitted_events.saturating_add(before - report.events.len());
        report.usage = report.usage.and_then(|usage| {
            serde_json::from_value::<Vec<crate::metrics::usage::ModelUsage>>(usage)
                .ok()
                .map(|usage| Value::Array(usage.iter().map(super::usage_attributes).collect()))
        });
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
    fn project_event(&self, event: Value) -> Option<Value> {
        if !super::report_value_fits(&event) {
            return None;
        }
        let header: EventHeader = serde_json::from_value(event.clone()).ok()?;
        let event = match EventPrivacy::for_kind(&header.kind) {
            EventPrivacy::Safe => serde_json::from_value::<SafeEvent>(event)
                .ok()
                .and_then(|event| serde_json::to_value(event).ok()),
            EventPrivacy::Content if self.content => Some(event),
            _ => None,
        }?;
        super::report_value_fits(&event).then_some(event)
    }

    fn event_record(
        &self,
        event: Value,
        identity: (String, String, String),
        timestamp: f64,
        disabled: bool,
    ) -> Option<super::LogRecordWire> {
        if disabled || !self.recording {
            return None;
        }
        let mut record = super::session_event_record(self, event, timestamp)?;
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

impl ReportProjection {
    pub fn events(&self, events: Vec<Value>) -> Vec<Value> {
        let mut bytes = 0usize;
        events
            .into_iter()
            .take(super::super::policy::REPORT_EVENT_ITEMS)
            .filter_map(|event| {
                let event = self.project_event(event)?;
                let charge = super::report_value_bytes(&event)?;
                if charge > super::super::policy::REPORT_EVENT_BYTES.saturating_sub(bytes) {
                    return None;
                }
                bytes += charge;
                Some(event)
            })
            .collect()
    }

    pub fn event_attribute(event: &Value) -> Option<&'static str> {
        let header: EventHeader = serde_json::from_value(event.clone()).ok()?;
        match EventPrivacy::for_kind(&header.kind) {
            EventPrivacy::Safe => Some("session.event"),
            EventPrivacy::Content => Some("lk.pii.session_event"),
            EventPrivacy::Refused => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn destination_projection_preserves_safe_events_and_withholds_only_denied_formats() {
        for (capture, redaction) in [(false, false), (true, false), (true, true)] {
            let options = RecordingOptions { transcript: true, audio: true, ..Default::default() };
            let projection = ReportProjection::new(capture, &options, redaction);
            let report = projection.project(SessionReportSnapshot {
                job_id: "job".into(),
                room_id: "room".into(),
                room: "call".into(),
                timestamp: 12.0,
                sdk_version: "test".into(),
                options: Some(json!({ "instructions": "private prompt" })),
                chat_history: Some(json!({ "items": [{ "content": "private history" }] })),
                audio_recording_path: Some("audio.ogg".into()),
                audio_recording_started_at: Some(10.0),
                usage: None,
                omitted_events: 0,
                events: vec![
                    json!({
                        "type": "agent_state_changed", "old_state": "listening",
                        "new_state": "thinking", "created_at": 11.0, "private": "extra content"
                    }),
                    json!({
                        "type": "session_usage_updated", "created_at": 11.5,
                        "usage": { "model_usage": [{
                            "type": "llm_usage", "provider": "provider", "model": "model",
                            "input_tokens": 7, "private": "extra usage content"
                        }] }
                    }),
                    json!({ "type": "user_input_transcribed", "transcript": "private event" }),
                    json!({ "type": "unknown", "payload": "private unknown content" }),
                ],
            });
            assert_eq!(report.events[0]["new_state"], "thinking");
            assert!(report.events[0].get("private").is_none());
            assert_eq!(report.events[1]["usage"]["model_usage"][0]["input_tokens"], 7);
            assert!(report.events[1]["usage"]["model_usage"][0].get("private").is_none());
            assert_eq!(report.events.len(), if capture && !redaction { 3 } else { 2 });
            assert_eq!(report.omitted_events, if capture && !redaction { 1 } else { 2 });
            assert_eq!(report.options.is_some(), capture && !redaction);
            if !capture || redaction {
                assert_eq!(report.chat_history, Some(json!({ "items": [] })));
            }
            assert_eq!(report.audio_recording_path.is_some(), !redaction);
            assert_eq!(report.audio_recording_started_at.is_some(), !redaction);
        }
    }
}
