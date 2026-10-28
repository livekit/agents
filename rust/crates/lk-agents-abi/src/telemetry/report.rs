//! `telemetry/traces.py`'s `_upload_session_report` records, its `_UploadGate`, and
//! `observability.py`'s `Tagger`. The records are [`crate::telemetry`]'s `LogRecord`.

use crate::ChatItem;
use crate::Clock;
use crate::metrics::ModelUsage;
use lk_agents_host_derive::Host;
use serde_json::{Map, Value, json};
use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::SystemTime;

use super::{HrTime, LogRecord, Severity, epoch_nanos, hr_time, hr_wall};

// -----------------------------------------------------------------------------------------
// the session report's log records
// -----------------------------------------------------------------------------------------
// What `_upload_session_report` (`telemetry/traces.py:480`) emits as OTel log records; Rust builds
// them, the host's SDK emits them. Tag-with-metadata and outcome records are not behind
// `recording_enabled` (hence `job.py:320`'s `recording_enabled(..) or has_evals`). Buffered early
// logs (`job.py:265`) are the log handler's.

#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoggerName {
    ChatHistory,
    /// Also carries the tag and outcome records.
    Evaluations,
}

/// A report record: `SeverityNumber.ERROR` / `"error"` when `error`, else `UNSPECIFIED` /
/// `"unspecified"` -- the report's own lowercase spelling, not the log handler's.
fn record(
    logger: LoggerName,
    body: &str,
    timestamp: HrTime,
    attributes: Value,
    error: bool,
) -> LogRecord {
    let (severity_number, severity_text) =
        if error { (Severity::Error as u8, "error") } else { (0, "unspecified") };
    LogRecord {
        logger: logger.as_str().to_owned(),
        body: body.to_owned(),
        timestamp,
        severity_number,
        severity_text: severity_text.to_owned(),
        attributes,
    }
}

/// `int(seconds * 1e9)`.
fn hr_seconds(seconds: f64) -> HrTime {
    hr_time((seconds * 1e9) as u128)
}

/// `RecordingOptions`, the four keys `recording_enabled` reads plus `redaction`.
///
/// `redaction = False` never disables project-level redaction; it only declines to ask for it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Host)]
#[binding(emit)]
#[host(host_type = "livekit.agents._core.RecordingOptions", handwritten(extract))]
pub struct RecordingOptions {
    pub audio: bool,
    pub traces: bool,
    pub logs: bool,
    pub transcript: bool,
    /// Not part of `_RECORDING_OPTION_KEYS` -- `recording_enabled` ignores it.
    pub redaction: bool,
}

/// `RecordingOptions`, as five booleans.
#[crate::host(object(
    cell = "input",
    pid = false,
    pyclass(from_py_object),
    derive(Clone, Copy, Default)
))]
impl RecordingOptions {
    #[host(new(
        kw_only,
        defaults(
            audio = false,
            traces = false,
            logs = false,
            transcript = false,
            redaction = false
        )
    ))]
    pub fn new(audio: bool, traces: bool, logs: bool, transcript: bool, redaction: bool) -> Self {
        RecordingOptions { audio, traces, logs, transcript, redaction }
    }

    #[host(getter)]
    pub fn audio(&self) -> bool {
        self.audio
    }

    #[host(getter)]
    pub fn traces(&self) -> bool {
        self.traces
    }

    #[host(getter)]
    pub fn logs(&self) -> bool {
        self.logs
    }

    #[host(getter)]
    pub fn transcript(&self) -> bool {
        self.transcript
    }

    #[host(getter)]
    pub fn redaction(&self) -> bool {
        self.redaction
    }

    /// `types.py:64`: any of audio, traces, logs, transcript. `redaction` alone is not recording.
    #[host(method)]
    pub fn recording_enabled(&self) -> bool {
        self.audio || self.traces || self.logs || self.transcript
    }
}

/// Whether recording is enabled for the job.
#[crate::host(export(pid = false))]
pub fn recording_enabled(options: &RecordingOptions) -> bool {
    options.recording_enabled()
}

/// `{k: v for k, v in u.model_dump().items() if v != 0 and v != 0.0}`, in declaration order. Only
/// numeric zeros are stripped: `provider=""` survives, since `"" != 0` in Python.
pub fn usage_attributes(u: &ModelUsage) -> Value {
    let (provider, model) = match u {
        ModelUsage::Llm(u) => (&u.provider, &u.model),
        ModelUsage::Tts(u) => (&u.provider, &u.model),
        ModelUsage::Stt(u) => (&u.provider, &u.model),
        ModelUsage::Interruption(u) => (&u.provider, &u.model),
        ModelUsage::Eot(u) => (&u.provider, &u.model),
    };
    if !report_value_fits(&(provider, model)) {
        return Value::Null;
    }
    let mut m = Map::new();
    m.insert("type".into(), Value::String(u.type_literal().into()));
    let int_field = |m: &mut Map<String, Value>, k: &str, v: u64| {
        if v != 0 {
            m.insert(k.into(), json!(v));
        }
    };
    match u {
        ModelUsage::Llm(x) => {
            m.insert("provider".into(), json!(x.provider));
            m.insert("model".into(), json!(x.model));
            int_field(&mut m, "input_tokens", x.input_tokens);
            int_field(&mut m, "input_cached_tokens", x.input_cached_tokens);
            int_field(&mut m, "input_cache_creation_tokens", x.input_cache_creation_tokens);
            int_field(&mut m, "input_audio_tokens", x.input_audio_tokens);
            int_field(&mut m, "input_cached_audio_tokens", x.input_cached_audio_tokens);
            int_field(&mut m, "input_text_tokens", x.input_text_tokens);
            int_field(&mut m, "input_cached_text_tokens", x.input_cached_text_tokens);
            int_field(&mut m, "input_image_tokens", x.input_image_tokens);
            int_field(&mut m, "input_cached_image_tokens", x.input_cached_image_tokens);
            int_field(&mut m, "output_tokens", x.output_tokens);
            int_field(&mut m, "output_audio_tokens", x.output_audio_tokens);
            int_field(&mut m, "output_text_tokens", x.output_text_tokens);
            if x.session_duration != 0.0 {
                m.insert("session_duration".into(), json!(x.session_duration));
            }
        }
        ModelUsage::Tts(x) => {
            m.insert("provider".into(), json!(x.provider));
            m.insert("model".into(), json!(x.model));
            int_field(&mut m, "input_tokens", x.input_tokens);
            int_field(&mut m, "output_tokens", x.output_tokens);
            int_field(&mut m, "characters_count", x.characters_count);
            if x.audio_duration != 0.0 {
                m.insert("audio_duration".into(), json!(x.audio_duration));
            }
        }
        ModelUsage::Stt(x) => {
            m.insert("provider".into(), json!(x.provider));
            m.insert("model".into(), json!(x.model));
            int_field(&mut m, "input_tokens", x.input_tokens);
            int_field(&mut m, "output_tokens", x.output_tokens);
            if x.audio_duration != 0.0 {
                m.insert("audio_duration".into(), json!(x.audio_duration));
            }
        }
        ModelUsage::Interruption(x) => {
            m.insert("provider".into(), json!(x.provider));
            m.insert("model".into(), json!(x.model));
            int_field(&mut m, "total_requests", x.total_requests);
        }
        ModelUsage::Eot(x) => {
            m.insert("provider".into(), json!(x.provider));
            m.insert("model".into(), json!(x.model));
            int_field(&mut m, "total_requests", x.total_requests);
        }
    }
    Value::Object(m)
}

fn chat_item_is_error(item: &ChatItem) -> bool {
    matches!(item, ChatItem::FunctionCallOutput(o) if o.is_error)
}

pub mod projection;
use projection::SessionReportProjection as _;
mod records;
pub use records::LogRecordWire;

/// One item at a time lets a host SDK emit a large report without constructing another
/// whole-history record vector under its interpreter lock.
#[crate::host(export(name = "_core_session_chat_record", pid = false))]
pub fn session_chat_record(
    gate: &UploadGate,
    options: &RecordingOptions,
    item: ChatItem,
    projected: Value,
    capture_content: bool,
    identity: (String, String, String),
) -> Option<LogRecordWire> {
    if gate.disabled()
        || !options.transcript
        || !super::policy::content_allowed(capture_content, options.redaction)
    {
        return None;
    }
    let mut record = chat_item_record(&item, projected);
    attribute_report_record(&mut record, &identity.0, &identity.1, &identity.2, true);
    report_value_fits(&record.attributes).then(|| record.into())
}

fn session_event_record(
    projection: &projection::ReportProjection,
    event: Value,
    timestamp: f64,
) -> Option<LogRecord> {
    let event = projection.project_event(event)?;
    let key = projection::ReportProjection::event_attribute(&event)?;
    let event = serde_json::to_string(&event).ok()?;
    let attributes = Value::Object(Map::from_iter([(key.to_owned(), Value::String(event))]));
    Some(record(LoggerName::ChatHistory, "session event", hr_seconds(timestamp), attributes, false))
}

fn chat_item_record(item: &ChatItem, projected: Value) -> LogRecord {
    record(
        LoggerName::ChatHistory,
        "chat item",
        hr_wall(item.created_at()),
        json!({ "chat.item": projected }),
        chat_item_is_error(item),
    )
}

/// In the reference's order; empty when the [`UploadGate`] is closed.
pub fn session_records(
    gate: &UploadGate,
    report: &crate::worker::worker::SessionReportInput,
    tagger: &Tagger,
    project_item: &dyn Fn(&ChatItem) -> Value,
    capture_content: bool,
) -> Vec<LogRecord> {
    if gate.disabled()
        || !report_value_fits(&(
            &report.agent_name,
            &report.job_id,
            &report.room_id,
            &report.room,
            &report.sdk_version,
        ))
    {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut omitted = report.omitted_events;
    let opts = report.recording_options;
    let projection = projection::ReportProjection::new(capture_content, &opts, false);
    let capture_content = projection.content();

    if opts.recording_enabled() {
        // An `or` chain: a `started_at` of exactly 0.0 falls through to `timestamp`.
        let seconds = match report.started_at {
            Some(s) if s != 0.0 => s,
            _ => report.timestamp,
        };
        let tags = if capture_content {
            let mut tags: Vec<_> = tagger
                .entries()
                .take(super::policy::SDK_PENDING_RECORDS)
                .map(|(name, _)| name)
                .filter(|name| name.len() <= 256)
                .map(str::to_owned)
                .collect();
            tags.sort();
            tags
        } else {
            Vec::new()
        };
        let mut usage = Vec::new();
        let mut usage_bytes = 2usize;
        for item in report.model_usage.iter().take(super::policy::SDK_PENDING_RECORDS) {
            let value = usage_attributes(item);
            let Some(bytes) = report_value_bytes(&value) else {
                omitted += 1;
                continue;
            };
            if value.is_null()
                || usage_bytes.saturating_add(bytes + 1) > super::policy::SDK_RECORD_BYTES / 2
            {
                omitted += 1;
                continue;
            }
            usage_bytes += bytes + 1;
            usage.push(value);
        }
        let summary = json!({
            "session.options": if capture_content && report_value_fits(&report.options) { report.options.clone() } else { Value::Null },
            "session.report_timestamp": report.timestamp,
            // An empty tag set is `None`, not `[]`.
            "session.tags": if tags.is_empty() { Value::Null } else { json!(tags) },
            "agent_name": report.agent_name,
            "sdk_version": report.sdk_version,
            // No usage at all is `None`, not `[]`.
            "usage": if usage.is_empty() { Value::Null } else { Value::Array(usage) },
        });
        out.push(record(
            LoggerName::ChatHistory,
            "session report",
            hr_seconds(seconds),
            summary,
            false,
        ));
    }

    let event_limit = if opts.recording_enabled() {
        super::policy::SDK_PENDING_RECORDS.saturating_sub(out.len() + 1)
    } else {
        0
    };
    omitted += report.events.len().saturating_sub(event_limit);
    for event in report.events.iter().take(event_limit) {
        match session_event_record(&projection, event.clone(), report.timestamp) {
            Some(event) => out.push(event),
            None => omitted += 1,
        }
    }

    if projection.transcript() {
        for item in report
            .chat_history
            .iter()
            .take(super::policy::SDK_PENDING_RECORDS.saturating_sub(out.len() + 1))
        {
            out.push(chat_item_record(item, project_item(item)));
        }
    }

    for e in tagger
        .evaluations()
        .iter()
        .take(super::policy::SDK_PENDING_RECORDS.saturating_sub(out.len() + 1))
    {
        let fields: Vec<_> = e
            .attributes()
            .into_iter()
            .filter(|(k, _)| capture_content || !matches!(*k, "reasoning" | "instructions"))
            .collect();
        if !report_value_fits(&fields) {
            omitted += 1;
            continue;
        }
        let mut m = Map::new();
        for (k, v) in fields {
            if capture_content || !matches!(k, "reasoning" | "instructions") {
                m.insert(k.into(), Value::String(v.into()));
            }
        }
        let attributes = json!({ "evaluation": m });
        let at = hr_seconds(report.timestamp);
        out.push(record(
            LoggerName::Evaluations,
            "evaluation",
            at,
            attributes,
            e.verdict == "fail",
        ));
    }

    for (name, entry) in
        tagger.entries().take(super::policy::SDK_PENDING_RECORDS.saturating_sub(out.len() + 1))
    {
        if !capture_content {
            continue;
        }
        let Some(metadata) = entry.metadata.as_ref() else { continue };
        if !report_value_fits(&(name, metadata)) {
            omitted += 1;
            continue;
        }
        // `if entry.metadata:` -- an *empty* dict is falsy in Python and emits nothing.
        if metadata.as_object().is_some_and(|o| o.is_empty()) {
            continue;
        }
        let attributes = json!({ "tag": { "name": name, "metadata": metadata } });
        out.push(record(
            LoggerName::Evaluations,
            "tag",
            hr_wall(entry.timestamp),
            attributes,
            false,
        ));
    }

    if let Some(outcome) = tagger.outcome() {
        let mut m = Map::new();
        m.insert("outcome".into(), Value::String(outcome.as_str().into()));
        // `if tagger.outcome_reason:` -- an empty reason is falsy and omits the key entirely.
        if let Some(r) = tagger
            .outcome_reason()
            .filter(|r| capture_content && !r.is_empty() && report_value_fits(r))
        {
            m.insert("reason".into(), Value::String(r.into()));
        }
        let attributes = json!({ "outcome": m });
        let at = hr_seconds(report.timestamp);
        out.push(record(
            LoggerName::Evaluations,
            "outcome",
            at,
            attributes,
            outcome == Outcome::Fail,
        ));
    }

    // Attribution belongs to the report snapshot, independent of the host exporter thread.
    for record in &mut out {
        attribute_report_record(
            record,
            &report.job_id,
            &report.room_id,
            &report.room,
            capture_content,
        );
    }
    let before = out.len();
    out.retain(|record| report_value_fits(&record.attributes));
    omitted += report
        .chat_history
        .len()
        .saturating_add(tagger.evaluations().len())
        .saturating_add(tagger.tags.len())
        .saturating_sub(super::policy::SDK_PENDING_RECORDS)
        .saturating_add(before - out.len());
    if omitted > 0 {
        if let Some(record) = out.first_mut() {
            record
                .attributes
                .as_object_mut()
                .expect("report attributes are objects")
                .insert("lk.telemetry.omitted_report_entries".into(), omitted.into());
        }
    }
    out
}

/// Count serialized bytes without allocating another copy of arbitrary application metadata.
fn report_value_fits(value: &(impl serde::Serialize + ?Sized)) -> bool {
    report_value_bytes(value).is_some()
}

fn report_value_bytes(value: &(impl serde::Serialize + ?Sized)) -> Option<usize> {
    struct Counter(usize);
    impl std::io::Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if bytes.len() > super::policy::SDK_RECORD_BYTES.saturating_sub(self.0 + 1024) {
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

fn attribute_report_record(
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

// -----------------------------------------------------------------------------------------
// the upload gate
// -----------------------------------------------------------------------------------------
// `_UploadGate` (`telemetry/traces.py:127`), made per-session: the reference's process-wide
// singleton would let one project's "recording disabled" 401 suppress another session's uploads.
// It matches the message text, not the status: other 401s must not disable uploads.

/// The 401 that means "the project owner turned recording off", matched case-insensitively.
pub const DISABLED_MARKERS: [&str; 2] = ["data recording is disabled", "disabled by owner"];

/// Per-session upload gate.
#[crate::class(mutable)]
#[derive(Debug, Default)]
pub struct UploadGate {
    disabled: bool,
    /// Set once, so a caller can log the warning exactly once per session the way `disable()` does.
    warned: bool,
}

/// The "data recording is disabled" latch (`upload_gate.py`): one transition, one warning.
#[crate::host(object)]
impl UploadGate {
    #[host(new)]
    pub fn new() -> Self {
        UploadGate::default()
    }

    /// `reset()`. Called when a session re-initialises recording.
    #[host(method)]
    pub fn reset(&mut self) {
        self.disabled = false;
        self.warned = false;
    }

    #[host(getter)]
    pub fn disabled(&self) -> bool {
        self.disabled
    }

    /// `disable()`. Returns `true` on the transition only, where the reference logs its warning.
    #[host(method)]
    pub fn disable(&mut self) -> bool {
        if self.disabled {
            return false;
        }
        self.disabled = true;
        let first = !self.warned;
        self.warned = true;
        first
    }

    /// `is_disabled_response(status_code, body)`; static because the exporter has no gate in hand.
    #[host(method)]
    pub fn is_disabled_response(status_code: u16, body: &[u8]) -> bool {
        if status_code != 401 {
            return false;
        }
        // `body.decode("utf-8", "ignore").lower()`: invalid UTF-8 is dropped, not an error.
        let text = String::from_utf8_lossy(body).to_lowercase();
        DISABLED_MARKERS.iter().any(|m| text.contains(m))
    }
}

/// Build upload records for a completed session. `project_item` serializes chat items, which may
/// reference host media.
#[cfg(feature = "python")]
#[pyo3::pyfunction]
#[pyo3(name = "session_records", signature = (gate, report, tagger, project_item, capture_content=false))]
pub fn host_session_records<'py>(
    py: pyo3::Python<'py>,
    gate: &UploadGate,
    report: pyo3::Bound<'py, pyo3::types::PyDict>,
    tagger: &Tagger,
    project_item: pyo3::Bound<'py, pyo3::PyAny>,
    capture_content: bool,
) -> pyo3::PyResult<Vec<pyo3::Bound<'py, pyo3::PyAny>>> {
    use pyo3::prelude::*;
    crate::python::check_pid()?;
    let input: crate::worker::worker::SessionReportInput = report.as_any().extract()?;

    // A raise inside `project_item` cannot unwind through the core's walk, so it is parked and
    // re-raised afterwards. The walk is single-threaded under the GIL, hence `RefCell`.
    let failure = std::cell::RefCell::new(None::<PyErr>);
    let project = |item: &ChatItem| -> Value {
        if failure.borrow().is_some() {
            return Value::Null;
        }
        match project_item
            .call1((item.clone(),))
            .and_then(|v| crate::python::value_from_py(&v, "project_item(...)"))
        {
            Ok(v) => v,
            Err(e) => {
                *failure.borrow_mut() = Some(e);
                Value::Null
            }
        }
    };

    let out = session_records(gate, &input, tagger, &project, capture_content);
    if let Some(e) = failure.borrow_mut().take() {
        return Err(e);
    }
    out.into_iter().map(|r| LogRecordWire::from(r).into_pyobject(py)).collect()
}

#[cfg(feature = "python")]
crate::register_host!(fn host_session_records);

#[cfg(test)]
mod upload_gate_tests {
    use super::*;

    #[test]
    fn incremental_chat_records_keep_supplied_identity_time_and_enforce_recording_consent() {
        let item = ChatItem::Message(crate::ChatMessage::new(
            "message-a",
            crate::Role::User,
            vec![crate::ChatContent::Text("private".into())],
            std::time::UNIX_EPOCH + std::time::Duration::from_nanos(123_456),
        ));
        let identity: (String, String, String) =
            ("job-a".into(), "room-a".into(), "room-name".into());
        let projected = json!({"id": "message-a", "content": ["private"]});
        let mut gate = UploadGate::new();
        let mut options = RecordingOptions { transcript: true, ..Default::default() };
        let record = session_chat_record(
            &gate,
            &options,
            item.clone(),
            projected.clone(),
            true,
            identity.clone(),
        )
        .unwrap();
        assert_eq!(record.timestamp_nanos, 123_456);
        assert_eq!(record.attributes["chat.item"], projected);
        assert_eq!(record.attributes["job_id"], "job-a");
        assert_eq!(record.attributes["lk.telemetry.capture_content"], true);
        assert!(
            session_chat_record(
                &gate,
                &options,
                item.clone(),
                projected.clone(),
                false,
                identity.clone()
            )
            .is_none()
        );
        options.redaction = true;
        assert!(
            session_chat_record(
                &gate,
                &options,
                item.clone(),
                projected.clone(),
                true,
                identity.clone()
            )
            .is_none()
        );
        options.redaction = false;
        options.transcript = false;
        assert!(
            session_chat_record(
                &gate,
                &options,
                item.clone(),
                projected.clone(),
                true,
                identity.clone()
            )
            .is_none()
        );
        options.transcript = true;
        gate.disable();
        assert!(
            session_chat_record(&gate, &options, item, projected, true, identity.clone()).is_none()
        );
    }

    #[test]
    fn large_tag_state_cannot_create_an_unbounded_sdk_report_or_remove_the_business_outcome() {
        let clock = Arc::new(crate::utils::clock::VirtualClock::new());
        let mut tagger = Tagger::new(clock);
        for index in 0..1_000 {
            tagger.add(&format!("tag-{index}"), Some(json!({"detail": "value"})));
        }
        tagger.success(Some("completed"));
        let report = crate::worker::worker::SessionReportInput {
            recording_options: RecordingOptions { logs: true, ..Default::default() },
            ..Default::default()
        };
        let records = session_records(&UploadGate::new(), &report, &tagger, &|_| Value::Null, true);
        assert!(records.len() <= super::super::policy::SDK_PENDING_RECORDS);
        assert!(records.iter().any(|record| record.body == "outcome"
            && record.attributes["outcome"]["outcome"] == "success"));
        assert!(records[0].attributes["lk.telemetry.omitted_report_entries"].as_u64().unwrap() > 0);
        assert_eq!(
            tagger.sorted_tags().len(),
            1_001,
            "business tag state remains application-owned"
        );
        assert!(records.iter().all(|record| report_value_fits(&record.attributes)));
    }

    // Catches: gating on the status code alone (a missing-grant 401 would disable uploads).
    #[test]
    fn only_the_recording_disabled_401_closes_the_gate() {
        assert!(UploadGate::is_disabled_response(401, b"data recording is disabled for project"));
        assert!(UploadGate::is_disabled_response(401, b"Disabled By Owner"));
        assert!(!UploadGate::is_disabled_response(401, b"missing project id"));
        assert!(!UploadGate::is_disabled_response(
            401,
            b"operation requires observability write grant"
        ));
        // Right text, wrong code -- the reference checks the code first.
        assert!(!UploadGate::is_disabled_response(403, b"data recording is disabled"));
        assert!(!UploadGate::is_disabled_response(500, b"data recording is disabled"));
    }

    // Catches: panicking or missing the marker on a non-UTF-8 body.
    #[test]
    fn invalid_utf8_body_still_matches() {
        let mut body = b"data recording is disabled ".to_vec();
        body.extend_from_slice(&[0xff, 0xfe]);
        assert!(UploadGate::is_disabled_response(401, &body));
    }

    // Catches: the gate becoming a global again.
    #[test]
    fn two_sessions_do_not_share_the_disabled_flag() {
        let mut a = UploadGate::new();
        let b = UploadGate::new();
        a.disable();
        assert!(a.disabled());
        assert!(!b.disabled(), "session B must still be allowed to upload");
    }

    // Catches: warning on every rejected upload instead of once, and `reset()` not re-arming.
    #[test]
    fn disable_reports_only_the_transition_and_reset_rearms() {
        let mut g = UploadGate::new();
        assert!(g.disable());
        assert!(!g.disable());
        g.reset();
        assert!(!g.disabled());
        assert!(g.disable());
    }
}

// -----------------------------------------------------------------------------------------
// the tagger
// -----------------------------------------------------------------------------------------
// `observability.py`'s `Tagger`. Timestamps come from a [`Clock`]. A `Vec`, not a map: insertion
// order is observable in the tag-metadata records, and re-adding a tag keeps its position.

/// `observability.py:12`.
#[derive(Debug, Clone, PartialEq)]
pub struct TagEntry {
    /// Only tags with metadata produce a "tag" log record.
    pub metadata: Option<Value>,
    /// Taken from the session clock.
    pub timestamp: SystemTime,
}

/// One entry of `Tagger.evaluations` (`observability.py:120`).
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(emit)]
pub struct EvaluationRecord {
    pub name: String,
    pub tag: String,
    pub verdict: String,
    pub reasoning: String,
    pub instructions: String,
}

impl EvaluationRecord {
    /// In the reference's insertion order.
    pub fn attributes(&self) -> Vec<(&'static str, &str)> {
        vec![
            ("name", &self.name),
            ("tag", &self.tag),
            ("verdict", &self.verdict),
            ("reasoning", &self.reasoning),
            ("instructions", &self.instructions),
        ]
    }
}

/// The session outcome. `"fail"` bumps the outcome record's severity.
#[crate::host(wire_enum)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Success,
    Fail,
}

/// Public because the outcome is derived from tag *presence*.
pub const TAG_SUCCESS: &str = "lk.success";
pub const TAG_FAIL: &str = "lk.fail";

/// `observability.py:17`.
#[crate::class(mutable)]
pub struct Tagger {
    clock: Arc<dyn Clock>,
    tags: Vec<(String, TagEntry)>,
    evaluation_results: Vec<EvaluationRecord>,
    outcome_reason: Option<String>,
}

impl std::fmt::Debug for Tagger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tagger")
            .field("tags", &self.tags)
            .field("evaluation_results", &self.evaluation_results)
            .field("outcome_reason", &self.outcome_reason)
            .finish()
    }
}

/// Tag sessions with metadata for observability.
///
/// The Tagger allows adding string tags (key:value format) with optional
/// structured metadata to sessions. Tags and metadata are uploaded to
/// LiveKit Cloud at session end.
///
/// Example:
/// ```python
/// # Mark session as successful
/// ctx.tagger.success(reason="Task completed successfully")
///
/// # Mark session as failed
/// ctx.tagger.fail(reason="User hung up before completing booking")
///
/// # Add custom tags
/// ctx.tagger.add("voicemail:true")
/// ctx.tagger.add("language:es")
///
/// # Add tags with structured metadata
/// ctx.tagger.add(
///     "appointment:booked",
///     metadata={"slot_id": "abc123", "calendar": "cal.com"},
/// )
///
/// # Remove a tag
/// ctx.tagger.remove("voicemail:true")
/// ```
#[crate::host(object)]
impl Tagger {
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        Tagger { clock, tags: Vec::new(), evaluation_results: Vec::new(), outcome_reason: None }
    }

    /// Timestamps are strictly increasing, so same-instant tags keep insertion order.
    #[host(new)]
    pub fn with_real_clock() -> Self {
        Tagger::new(Arc::new(crate::RealClock::new()))
    }

    /// Python `dict` assignment: replace in place (keeping position), else append.
    fn put(&mut self, tag: &str, metadata: Option<Value>) {
        let entry = TagEntry { metadata, timestamp: self.clock.wall() };
        match self.tags.iter_mut().find(|(k, _)| k == tag) {
            Some(slot) => slot.1 = entry,
            None => self.tags.push((tag.to_string(), entry)),
        }
    }

    fn pop(&mut self, tag: &str) {
        self.tags.retain(|(k, _)| k != tag);
    }

    /// Mark the session as successful; ``reason`` is kept apart from the tag.
    // Clears any existing `lk.fail` first.
    #[host(method(defaults(reason = None)))]
    pub fn success(&mut self, reason: Option<impl Into<String>>) {
        self.pop(TAG_FAIL);
        self.put(TAG_SUCCESS, None);
        self.outcome_reason = reason.map(Into::into);
    }

    /// Mark the session as failed; ``reason`` is kept apart from the tag.
    #[host(method(defaults(reason = None)))]
    pub fn fail(&mut self, reason: Option<impl Into<String>>) {
        self.pop(TAG_SUCCESS);
        self.put(TAG_FAIL, None);
        self.outcome_reason = reason.map(Into::into);
    }

    /// Add a ``key:value`` tag, with optional structured metadata.
    #[host(method(kw_only = "metadata", defaults(metadata = None)))]
    pub fn add(&mut self, tag: &str, metadata: Option<Value>) {
        self.put(tag, metadata);
    }

    /// Remove a tag; removing an absent one does nothing.
    #[host(method)]
    pub fn remove(&mut self, tag: &str) {
        self.pop(tag);
    }

    /// `tags`, in insertion order; see [`Self::sorted_tags`] for the reference's `sorted(...)`.
    pub fn tags(&self) -> impl Iterator<Item = &str> {
        self.tags.iter().map(|(k, _)| k.as_str())
    }

    /// The session's tags.
    #[host(getter(name = "tags"))]
    pub fn tag_set(&self) -> BTreeSet<String> {
        self.tags().map(str::to_owned).collect()
    }

    /// `sorted(tagger.tags)` -- what the session summary record carries.
    #[host(method)]
    pub fn sorted_tags(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.tags().collect();
        v.sort_unstable();
        v
    }

    /// `tagger._tags.items()`, in insertion order. This is what the tag-metadata records walk.
    pub fn entries(&self) -> impl Iterator<Item = (&str, &TagEntry)> {
        self.tags.iter().map(|(k, e)| (k.as_str(), e))
    }

    /// `(tag, metadata, epoch seconds)`, in insertion order.
    #[host(method(name = "entries"))]
    pub fn entries_wire(&self) -> Vec<(String, Option<Value>, SystemTime)> {
        self.tags.iter().map(|(k, e)| (k.clone(), e.metadata.clone(), e.timestamp)).collect()
    }

    #[host(method)]
    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }

    /// The judgments recorded by judge groups, one dict each.
    // The reference returns a `.copy()`; a shared slice is the Rust equivalent.
    #[host(getter)]
    pub fn evaluations(&self) -> &[EvaluationRecord] {
        &self.evaluation_results
    }

    /// ``"success"``, ``"fail"`` or ``None``.
    // From tag presence; `lk.success` wins if both are set.
    #[host(getter)]
    pub fn outcome(&self) -> Option<Outcome> {
        if self.tags.iter().any(|(k, _)| k == TAG_SUCCESS) {
            Some(Outcome::Success)
        } else if self.tags.iter().any(|(k, _)| k == TAG_FAIL) {
            Some(Outcome::Fail)
        } else {
            None
        }
    }

    /// The reason given with the outcome.
    // Only another `success`/`fail` overwrites it.
    #[host(getter)]
    pub fn outcome_reason(&self) -> Option<&str> {
        self.outcome_reason.as_deref()
    }

    /// One judgment of a judge group's result: its `lk.judge.<name>:<verdict>` tag and record.
    #[host(method(name = "_core_record_judgment"))]
    pub fn record_judgment(
        &mut self,
        name: &str,
        verdict: &str,
        reasoning: &str,
        instructions: &str,
    ) {
        let tag = format!("lk.judge.{name}:{verdict}");
        self.put(&tag, None);
        self.evaluation_results.push(EvaluationRecord {
            name: name.to_owned(),
            tag,
            verdict: verdict.to_owned(),
            reasoning: reasoning.to_owned(),
            instructions: instructions.to_owned(),
        });
    }

    /// `has_evals` at `job.py:320`: `bool(self._tagger.evaluations or self._tagger.outcome)`.
    #[host(method)]
    pub fn has_evals(&self) -> bool {
        !self.evaluation_results.is_empty() || self.outcome().is_some()
    }
}

// -----------------------------------------------------------------------------------------
// the interruption detector's span update
// -----------------------------------------------------------------------------------------
// `{"type": "span_update", ..}`: five attributes the host writes onto a span it owns.

/// `{"type": "span_update", "span": .., "attrs": {..}}`.
#[derive(Debug, Clone, Host)]
#[binding(emit, tag = "type", tag_value = "span_update")]
pub struct SpanUpdateWire {
    /// The span, as a `HostRef` id the host resolves.
    pub span: u64,
    pub attrs: SpanUpdateAttrsWire,
}

/// The nested `attrs` dict: the five values `_apply_span_update` writes onto the span. A wire twin
/// of `lk_agents_inference::interruption::SpanAttributes`, whose `&'static str` field the
/// generator cannot map.
#[derive(Debug, Clone, Host)]
#[binding(emit)]
pub struct SpanUpdateAttrsWire {
    pub is_interruption: String,
    pub probability: f64,
    pub total_duration: f64,
    pub prediction_duration: f64,
    pub detection_delay: f64,
}

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "python"))]
mod python_tests {
    use pyo3::prelude::*;

    use super::Tagger;

    // Catches: tag order lost, or non-JSON metadata coerced instead of refused.
    #[test]
    fn the_tagger_keeps_insertion_order_and_refuses_non_json_metadata() {
        Python::initialize();
        Python::attach(|py| {
            crate::python::declare().unwrap();
            let t = Bound::new(py, Tagger::with_real_clock()).unwrap();
            t.call_method1("add", ("b",)).unwrap();
            t.call_method1("add", ("a",)).unwrap();
            let order: Vec<String> = t
                .call_method0("entries")
                .unwrap()
                .extract::<Vec<(String, Option<Py<PyAny>>, f64)>>()
                .unwrap()
                .into_iter()
                .map(|(tag, _, _)| tag)
                .collect();
            assert_eq!(order, vec!["b", "a"]);
            let bad = py.eval(c"object()", None, None).unwrap();
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("metadata", bad).unwrap();
            assert!(
                t.call_method("add", ("c",), Some(&kwargs)).is_err(),
                "metadata must be JSON, not coerced"
            );
        });
    }
}
