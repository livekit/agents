//! The session report's recording upload, decided before any I/O: the header and its session
//! summary, the `chat_history.json` bytes within their bound, and the audio file to send.

use lk_agents_abi::livekit_protocol::agent as proto;
use lk_agents_abi::telemetry::report::{
    ErrorCounts, RecordingHeader, RecordingOptions, SESSION_SUMMARY_BYTES, SessionReport,
    SessionSummary,
};
use lk_agents_abi::voice::events::CloseReason;
use lk_agents_abi::{ChatItem, Role, ToDictHost, ToDictOptions};
use serde::Deserialize;
use serde_json::{Map, Value};

/// The uploaded transcript's bounds: whole items beyond them are dropped from the oldest end.
pub const TRANSCRIPT_BYTES: usize = 2 * 1024 * 1024;
pub const TRANSCRIPT_ITEMS: usize = 10_000;

/// One recording upload: what the multipart request carries.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordingUpload {
    pub header: RecordingHeader,
    /// `chat_history.json`: `{"items": [...]}`, the `ChatContext` dictionary form.
    pub chat_history: Option<String>,
    /// The session's OGG recording.
    pub audio_path: Option<String>,
}

/// The upload the report and its options make; `None` when there is nothing to upload (neither a
/// transcript nor audio). A redacted session always uploads its transcript, so Cloud can rebuild
/// it after redaction.
pub fn upload(
    report: &SessionReport,
    options: &RecordingOptions,
    simulated: bool,
) -> Option<RecordingUpload> {
    let audio_path = report
        .audio_recording_path
        .clone()
        .filter(|_| options.audio && report.audio_recording_started_at.is_some());
    let transcript = options.transcript || options.redaction;
    if !transcript && audio_path.is_none() {
        return None;
    }
    let (chat_history, omitted) = if transcript {
        let (json, omitted) = chat_history(&report.chat_history);
        (Some(json), omitted)
    } else {
        (None, 0)
    };
    let header = RecordingHeader {
        room_id: report.room_id.clone(),
        job_id: report.job_id.clone(),
        start_time_millis: (report.audio_recording_started_at.unwrap_or(0.0) * 1000.0) as i64,
        duration_millis: (report.duration.unwrap_or(0.0).max(0.0) * 1000.0) as u64,
        simulated,
        redaction_enabled: options.redaction,
        session_summary: session_summary(report, omitted),
    };
    Some(RecordingUpload { header, chat_history, audio_path })
}

/// The newest items that fit the transcript's bounds, and how many older ones were left out.
fn chat_history(items: &[ChatItem]) -> (String, u64) {
    let options = ToDictOptions { exclude_timestamp: false, ..Default::default() };
    let host = ToDictHost::default();
    let mut kept: Vec<String> = Vec::new();
    // `{"items":[` and `]}`.
    let mut bytes = 12;
    for item in items.iter().rev() {
        let Ok(Some(value)) = crate::llm::chat_dict::item_to_dict(item, &options, &host) else {
            continue;
        };
        let Ok(json) = serde_json::to_string(&value) else { continue };
        let separator = usize::from(!kept.is_empty());
        if kept.len() == TRANSCRIPT_ITEMS || bytes + json.len() + separator > TRANSCRIPT_BYTES {
            break;
        }
        bytes += json.len() + separator;
        kept.push(json);
    }
    let omitted = (items.len() - kept.len()) as u64;
    kept.reverse();
    (format!("{{\"items\":[{}]}}", kept.join(",")), omitted)
}

/// The summary agent evaluations read, within [`SESSION_SUMMARY_BYTES`]: past it, the options,
/// then the tags, then the usage are left out.
fn session_summary(report: &SessionReport, omitted_chat_items: u64) -> String {
    let mut tags: Vec<String> = report.tags.iter().map(|t| t.name.clone()).collect();
    tags.sort();
    let mut summary = SessionSummary {
        job_id: report.job_id.clone(),
        room_id: report.room_id.clone(),
        room: report.room.clone(),
        audio_recording_path: report.audio_recording_path.clone(),
        audio_recording_started_at: report.audio_recording_started_at,
        options: Some(summary_options(&report.options)),
        timestamp: report.timestamp,
        usage: (!report.usage.models.is_empty()).then(|| usage(&report.usage.models)),
        sdk_version: report.sdk_version.clone(),
        agent_name: report.agent_name.clone(),
        tags,
        duration_ms: report.duration.map(|d| (d * 1000.0) as i64),
        user_turns: report
            .chat_history
            .iter()
            .filter(|i| matches!(i, ChatItem::Message(m) if m.role == Role::User))
            .count() as u64,
        errors: error_counts(&report.events),
        close_reason: close_reason(&report.events),
        omitted_chat_items,
    };
    let encode = |s: &SessionSummary| serde_json::to_string(s).unwrap_or_else(|_| "{}".into());
    let mut json = encode(&summary);
    let shrink: [fn(&mut SessionSummary); 3] =
        [|s| s.options = None, |s| s.tags.clear(), |s| s.usage = None];
    for step in shrink {
        if json.len() <= SESSION_SUMMARY_BYTES {
            break;
        }
        step(&mut summary);
        json = encode(&summary);
    }
    json
}

/// Each model's entry without its absent fields.
fn usage(models: &[lk_agents_abi::usage::ModelUsage]) -> Vec<Value> {
    fn strip(v: &mut Value) {
        if let Value::Object(o) = v {
            o.retain(|_, v| !v.is_null());
            o.values_mut().for_each(strip);
        }
    }
    models
        .iter()
        .filter_map(|m| serde_json::to_value(m).ok())
        .map(|mut v| {
            strip(&mut v);
            v
        })
        .collect()
}

/// 1.8.10's summary options: the interruption, endpointing, tool-step and recording settings.
fn summary_options(options: &Value) -> Value {
    let at = |path: &[&str]| {
        path.iter().try_fold(options, |v, k| v.get(k)).cloned().unwrap_or(Value::Null)
    };
    let fields = [
        ("allow_interruptions", at(&["interruption", "enabled"])),
        (
            "discard_audio_if_uninterruptible",
            at(&["interruption", "discard_audio_if_uninterruptible"]),
        ),
        ("min_interruption_duration", at(&["interruption", "min_duration"])),
        ("min_interruption_words", at(&["interruption", "min_words"])),
        ("min_endpointing_delay", at(&["endpointing", "min_delay"])),
        ("max_endpointing_delay", at(&["endpointing", "max_delay"])),
        ("max_tool_steps", at(&["max_tool_steps"])),
        ("user_away_timeout", at(&["user_away_timeout"])),
        ("min_consecutive_speech_delay", at(&["min_consecutive_speech_delay"])),
        ("preemptive_generation", at(&["preemptive_generation"])),
        ("recording_options", at(&["recording_options"])),
    ];
    Value::Object(fields.into_iter().map(|(k, v)| (k.to_owned(), v)).collect::<Map<_, _>>())
}

/// What the summary reads of a report event.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ReportEvent {
    Error {
        error: ReportedError,
    },
    Close {
        reason: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct ReportedError {
    #[serde(rename = "type")]
    source: ErrorSource,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum ErrorSource {
    LlmError,
    SttError,
    TtsError,
    RealtimeModelError,
    InterruptionDetectionError,
    #[serde(other)]
    Other,
}

fn report_events(events: &[Value]) -> impl DoubleEndedIterator<Item = ReportEvent> + '_ {
    events.iter().filter_map(|e| ReportEvent::deserialize(e).ok())
}

fn error_counts(events: &[Value]) -> ErrorCounts {
    let mut errors = ErrorCounts::default();
    for event in report_events(events) {
        let ReportEvent::Error { error } = event else { continue };
        let count = match error.source {
            ErrorSource::LlmError => &mut errors.llm,
            ErrorSource::SttError => &mut errors.stt,
            ErrorSource::TtsError => &mut errors.tts,
            ErrorSource::RealtimeModelError => &mut errors.realtime,
            ErrorSource::InterruptionDetectionError => &mut errors.interruption,
            ErrorSource::Other => continue,
        };
        *count += 1;
    }
    errors
}

/// The last close event's reason.
fn close_reason(events: &[Value]) -> Option<proto::CloseReason> {
    let reason = report_events(events).rev().find_map(|e| match e {
        ReportEvent::Close { reason } => Some(reason),
        _ => None,
    })?;
    Some(CloseReason::from_wire(&reason).map(Into::into).unwrap_or_default())
}
