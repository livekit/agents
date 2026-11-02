//! The session report's privacy projection and record shape: which events and fields a report
//! may retain, and the bounded log records the host SDK emits ([`records`]). `lk-agents-net`
//! uploads the report.

use lk_agents_abi::telemetry::policy::{
    REPORT_EVENT_BYTES, REPORT_EVENT_ITEMS, REPORT_EVENT_RECORD_BYTES, SDK_RECORD_BYTES,
};
use lk_agents_abi::telemetry::report::{LogRecordWire, RecordingOptions};
use lk_agents_abi::telemetry::report::{
    ProjectedReportEvent, SessionReportProjection, SessionReportSnapshot,
};
use lk_agents_abi::voice::events::{AgentStateChanged, UserStateChanged, UserTranscriptionTimeout};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

pub mod records;

/// OTel JS's `HrTime`: epoch nanos do not fit a JS `number`.
pub type HrTime = (i64, u32);

pub fn hr_time(epoch_nanos: u128) -> HrTime {
    ((epoch_nanos / 1_000_000_000) as i64, (epoch_nanos % 1_000_000_000) as u32)
}

pub fn epoch_nanos(t: HrTime) -> u128 {
    t.0.max(0) as u128 * 1_000_000_000 + t.1 as u128
}

/// OTel's `SeverityNumber.ERROR`, identical in both SDKs.
const SEVERITY_ERROR: u8 = 17;

#[derive(Debug, Clone, PartialEq)]
pub struct LogRecord {
    pub logger: String,
    pub body: String,
    pub timestamp: HrTime,
    pub severity_number: u8,
    pub severity_text: String,
    /// Recursive, unlike a span's.
    pub attributes: Value,
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

// What `_upload_session_report` (`telemetry/traces.py:480`) emits as OTel log records; Rust builds
// them, the host's SDK emits them. Tag-with-metadata and outcome records are not behind
// `recording_enabled` (hence `job.py:320`'s `recording_enabled(..) or has_evals`). Buffered early
// logs (`job.py:265`) are the log handler's.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoggerName {
    ChatHistory,
    /// Also carries the tag and outcome records.
    Evaluations,
}

/// A report record: `SeverityNumber.ERROR` / `"error"` when `error`, else `UNSPECIFIED` /
/// `"unspecified"` -- the report's own lowercase spelling, not the log handler's.
pub fn record(
    logger: LoggerName,
    body: &str,
    timestamp: HrTime,
    attributes: Value,
    error: bool,
) -> LogRecord {
    let (severity_number, severity_text) =
        if error { (SEVERITY_ERROR, "error") } else { (0, "unspecified") };
    LogRecord {
        logger: match logger {
            LoggerName::ChatHistory => "chat_history",
            LoggerName::Evaluations => "evaluations",
        }
        .to_owned(),
        body: body.to_owned(),
        timestamp,
        severity_number,
        severity_text: severity_text.to_owned(),
        attributes,
    }
}

/// `int(seconds * 1e9)`.
pub fn hr_seconds(seconds: f64) -> HrTime {
    hr_time((seconds * 1e9) as u128)
}

/// Count serialized bytes without allocating another copy of arbitrary application metadata.
pub fn report_value_fits(value: &(impl serde::Serialize + ?Sized)) -> bool {
    report_value_bytes(value).is_some()
}

pub fn report_value_bytes(value: &(impl serde::Serialize + ?Sized)) -> Option<usize> {
    struct Counter(usize);
    impl std::io::Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if bytes.len() > SDK_RECORD_BYTES.saturating_sub(self.0 + 1024) {
                return Err(std::io::ErrorKind::FileTooLarge.into());
            }
            self.0 += bytes.len();
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut counter = Counter(0);
    serde_json::to_writer(&mut counter, value).ok()?;
    Some(counter.0)
}

pub fn attribute_report_record(
    record: &mut LogRecord,
    job_id: &str,
    room_id: &str,
    room: &str,
    capture_content: bool,
) {
    let attributes = record.attributes.as_object_mut().expect("report attributes are objects");
    attributes.insert("job_id".into(), job_id.into());
    attributes.insert("room_id".into(), room_id.into());
    attributes.insert("room".into(), room.into());
    attributes.insert("lk.telemetry.capture_content".into(), capture_content.into());
    // Bodies are protocol discriminators; free-form content lives in gated attributes.
    attributes.insert("lk.telemetry.safe_body".into(), true.into());
}

#[derive(Deserialize)]
struct EventHeader {
    #[serde(rename = "type")]
    kind: String,
}

// Reuse the actual event schemas. Typed decoding drops extra fields before safe events leave
// the destination boundary; retaining a safe kind never authorizes arbitrary attached content.
#[derive(Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SafeEvent {
    UserStateChanged(UserStateChanged),
    AgentStateChanged(AgentStateChanged),
    UserTranscriptionTimeout(UserTranscriptionTimeout),
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
            "user_state_changed" | "agent_state_changed" | "user_transcription_timeout" => {
                Self::Safe
            }
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
        EventPrivacy::Content => super::policy::content_allowed(capture_content, redaction),
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

impl SessionReportProjection for ReportProjection {
    fn new(capture_content: bool, options: &RecordingOptions, project_redaction: bool) -> Self {
        let content =
            super::policy::content_allowed(capture_content, options.redaction || project_redaction);
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
        items.min(REPORT_EVENT_ITEMS)
    }

    fn accepts_event(&self, retained_bytes: usize, event_bytes: usize) -> bool {
        event_bytes <= REPORT_EVENT_RECORD_BYTES
            && event_bytes <= REPORT_EVENT_BYTES.saturating_sub(retained_bytes)
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
        if !report_value_fits(&event) {
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
        let bytes = report_value_bytes(&value)?.saturating_add(1).max(retained_bytes);
        Some(ProjectedReportEvent { value, content, retained_bytes: bytes })
    }

    fn event_record(
        &self,
        event: ProjectedReportEvent,
        identity: (String, String, String),
        timestamp: f64,
        disabled: bool,
    ) -> Option<LogRecordWire> {
        if disabled || !self.recording || (event.content && !self.content) {
            return None;
        }
        let key = if event.content { "lk.pii.session_event" } else { "session.event" };
        let value = serde_json::to_string(&event.value).ok()?;
        let attributes = Value::Object(Map::from_iter([(key.to_owned(), Value::String(value))]));
        let mut record = record(
            LoggerName::ChatHistory,
            "session event",
            hr_seconds(timestamp),
            attributes,
            false,
        );
        attribute_report_record(&mut record, &identity.0, &identity.1, &identity.2, self.content);
        report_value_fits(&record.attributes).then(|| record.into())
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
