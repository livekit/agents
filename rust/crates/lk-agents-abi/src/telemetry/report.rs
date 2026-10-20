//! `telemetry/traces.py`'s `_upload_session_report` records, its `_UploadGate`, and
//! `observability.py`'s `Tagger`. The records are [`crate::telemetry`]'s `LogRecord`.

use crate::ChatItem;
use crate::Clock;
use crate::evals::EvaluationResult;
use crate::metrics::ModelUsage;
use lk_agents_host_derive::Host;
use serde_json::{Map, Value, json};
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
fn record(logger: LoggerName, body: &str, timestamp: HrTime, attributes: Value, error: bool) -> LogRecord {
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
    #[host(new(kw_only, defaults(audio = false, traces = false, logs = false, transcript = false, redaction = false)))]
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

/// A [`LogRecord`] as the dict `_upload_session_report` hands `logger.emit`: one integer of epoch
/// nanos, where the log handler's records carry an `HrTime` pair.
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(emit)]
pub struct LogRecordWire {
    pub logger: String,
    pub body: String,
    pub timestamp_nanos: i64,
    pub severity_number: u8,
    pub severity_text: String,
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
            attributes: r.attributes,
        }
    }
}

/// In the reference's order; empty when the [`UploadGate`] is closed.
pub fn session_records(
    gate: &UploadGate,
    report: &crate::worker::worker::SessionReportInput,
    tagger: &Tagger,
    project_item: &dyn Fn(&ChatItem) -> Value,
) -> Vec<LogRecord> {
    if gate.disabled() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let opts = report.recording_options;

    if opts.recording_enabled() {
        // An `or` chain: a `started_at` of exactly 0.0 falls through to `timestamp`.
        let seconds = match report.started_at {
            Some(s) if s != 0.0 => s,
            _ => report.timestamp,
        };
        let tags = tagger.sorted_tags();
        let summary = json!({
            "session.options": report.options,
            "session.report_timestamp": report.timestamp,
            // An empty tag set is `None`, not `[]`.
            "session.tags": if tags.is_empty() { Value::Null } else { json!(tags) },
            "agent_name": report.agent_name,
            "sdk_version": report.sdk_version,
            // No usage at all is `None`, not `[]`.
            "usage": if report.model_usage.is_empty() {
                Value::Null
            } else {
                Value::Array(report.model_usage.iter().map(usage_attributes).collect())
            },
        });
        out.push(record(LoggerName::ChatHistory, "session report", hr_seconds(seconds), summary, false));
    }

    if opts.transcript {
        for item in &report.chat_history {
            let attributes = json!({ "chat.item": project_item(item) });
            let at = hr_wall(item.created_at());
            out.push(record(LoggerName::ChatHistory, "chat item", at, attributes, chat_item_is_error(item)));
        }
    }

    for e in tagger.evaluations() {
        let mut m = Map::new();
        for (k, v) in e.attributes() {
            m.insert(k.into(), Value::String(v.into()));
        }
        let attributes = json!({ "evaluation": m });
        let at = hr_seconds(report.timestamp);
        out.push(record(LoggerName::Evaluations, "evaluation", at, attributes, e.verdict == "fail"));
    }

    for (name, entry) in tagger.entries() {
        let Some(metadata) = entry.metadata.as_ref() else { continue };
        // `if entry.metadata:` -- an *empty* dict is falsy in Python and emits nothing.
        if metadata.as_object().is_some_and(|o| o.is_empty()) {
            continue;
        }
        let attributes = json!({ "tag": { "name": name, "metadata": metadata } });
        out.push(record(LoggerName::Evaluations, "tag", hr_wall(entry.timestamp), attributes, false));
    }

    if let Some(outcome) = tagger.outcome() {
        let mut m = Map::new();
        m.insert("outcome".into(), Value::String(outcome.as_str().into()));
        // `if tagger.outcome_reason:` -- an empty reason is falsy and omits the key entirely.
        if let Some(r) = tagger.outcome_reason().filter(|r| !r.is_empty()) {
            m.insert("reason".into(), Value::String(r.into()));
        }
        let attributes = json!({ "outcome": m });
        let at = hr_seconds(report.timestamp);
        out.push(record(LoggerName::Evaluations, "outcome", at, attributes, outcome == Outcome::Fail));
    }

    out
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
#[pyo3(name = "session_records", signature = (gate, report, tagger, project_item))]
pub fn host_session_records<'py>(
    py: pyo3::Python<'py>,
    gate: &UploadGate,
    report: pyo3::Bound<'py, pyo3::types::PyDict>,
    tagger: &Tagger,
    project_item: pyo3::Bound<'py, pyo3::PyAny>,
) -> pyo3::PyResult<Vec<pyo3::Bound<'py, pyo3::types::PyDict>>> {
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
        match project_item.call1((item.clone(),)).and_then(|v| crate::python::value_from_py(&v, "project_item(...)")) {
            Ok(v) => v,
            Err(e) => {
                *failure.borrow_mut() = Some(e);
                Value::Null
            }
        }
    };

    let out = session_records(gate, &input, tagger, &project);
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
pub const TAG_SUCCESS: &str = super::semconv::LK_SUCCESS;
pub const TAG_FAIL: &str = super::semconv::LK_FAIL;

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

/// Session tags and outcome used in job reports.
///
/// `entries` is the binding's own method (`worker.rs`).
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

    /// `success(reason=None)`. Clears any existing `lk.fail` first.
    #[host(method)]
    pub fn success(&mut self, reason: Option<impl Into<String>>) {
        self.pop(TAG_FAIL);
        self.put(TAG_SUCCESS, None);
        self.outcome_reason = reason.map(Into::into);
    }

    /// `fail(reason=None)`.
    #[host(method)]
    pub fn fail(&mut self, reason: Option<impl Into<String>>) {
        self.pop(TAG_SUCCESS);
        self.put(TAG_FAIL, None);
        self.outcome_reason = reason.map(Into::into);
    }

    /// `add(tag, metadata=None)`. Public API, unchanged.
    #[host(method)]
    pub fn add(&mut self, tag: &str, metadata: Option<Value>) {
        self.put(tag, metadata);
    }

    /// `remove(tag)`. Idempotent -- the reference is `dict.pop(tag, None)`.
    #[host(method)]
    pub fn remove(&mut self, tag: &str) {
        self.pop(tag);
    }

    /// `tags`, in insertion order; see [`Self::sorted_tags`] for the reference's `sorted(...)`.
    #[host(method)]
    pub fn tags(&self) -> impl Iterator<Item = &str> {
        self.tags.iter().map(|(k, _)| k.as_str())
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

    /// `evaluations`. The reference returns a `.copy()`; a shared slice is the Rust equivalent.
    #[host(method)]
    pub fn evaluations(&self) -> &[EvaluationRecord] {
        &self.evaluation_results
    }

    /// `outcome`, from tag presence; `lk.success` wins if both are set.
    #[host(method)]
    pub fn outcome(&self) -> Option<Outcome> {
        if self.tags.iter().any(|(k, _)| k == TAG_SUCCESS) {
            Some(Outcome::Success)
        } else if self.tags.iter().any(|(k, _)| k == TAG_FAIL) {
            Some(Outcome::Fail)
        } else {
            None
        }
    }

    /// `outcome_reason`. Only another `success`/`fail` overwrites it.
    #[host(method)]
    pub fn outcome_reason(&self) -> Option<&str> {
        self.outcome_reason.as_deref()
    }

    /// `_evaluation(result)` (`observability.py:115`): one tag and one record per judgment.
    #[host(method)]
    pub fn record_evaluation(&mut self, result: &EvaluationResult) {
        let rows: Vec<EvaluationRecord> = result
            .judgments()
            .map(|(name, j)| EvaluationRecord {
                name: name.to_string(),
                tag: format!("lk.judge.{name}:{}", j.verdict.as_str()),
                verdict: j.verdict.as_str().to_string(),
                reasoning: j.reasoning.clone(),
                instructions: j.instructions.clone(),
            })
            .collect();
        for row in rows {
            self.put(&row.tag, None);
            self.evaluation_results.push(row);
        }
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
mod tests {
    
    use std::time::{Duration, SystemTime};

    use lk_agents_abi::evals::{EvaluationResult, JudgmentResult, Verdict};
    use lk_agents_abi::metrics::{LlmModelUsage, ModelUsage, TtsModelUsage};
    use lk_agents_abi::{ChatItem, ChatMessage, Role, VirtualClock};
    use serde_json::{Value, json};

    use super::*;
    use crate::worker::worker::SessionReportInput;

    // ============================================================================================
    // records
    // ============================================================================================

    fn view(usage: &[ModelUsage], items: &[ChatItem], opts: RecordingOptions) -> SessionReportInput {
        SessionReportInput {
            job_id: "J1".into(),
            room_id: "RM_1".into(),
            room: "room-1".into(),
            agent_name: "my-agent".into(),
            sdk_version: "1.0.0".into(),
            started_at: Some(100.0),
            timestamp: 200.0,
            options: json!({"turn_detection": "vad"}),
            recording_options: opts,
            model_usage: usage.to_vec(),
            chat_history: items.to_vec(),
        }
    }

    fn all_on() -> RecordingOptions {
        RecordingOptions { audio: true, traces: true, logs: true, transcript: true, redaction: false }
    }

    fn tagger() -> Tagger {
        Tagger::new(std::sync::Arc::new(VirtualClock::new()))
    }

    fn find<'a>(recs: &'a [LogRecord], body: &str) -> Vec<&'a LogRecord> {
        recs.iter().filter(|r| r.body == body).collect()
    }

    fn attr<'a>(r: &'a LogRecord, k: &str) -> &'a Value {
        r.attributes.get(k).expect("missing attribute")
    }

    // Catches: serializing zero-valued usage fields.
    #[test]
    fn zero_valued_usage_fields_are_stripped() {
        let u = ModelUsage::Llm(LlmModelUsage {
            provider: "openai".into(),
            model: "gpt-4o".into(),
            input_tokens: 100,
            output_tokens: 50,
            ..Default::default()
        });
        let v = usage_attributes(&u);
        let o = v.as_object().unwrap();
        assert_eq!(o.get("input_tokens"), Some(&json!(100)));
        assert_eq!(o.get("output_tokens"), Some(&json!(50)));
        assert!(o.get("input_cached_tokens").is_none(), "{v}");
        assert!(o.get("session_duration").is_none(), "float zeros go too: {v}");
        assert_eq!(o.get("type"), Some(&json!("llm_usage")));
        assert_eq!(o.len(), 5, "type, provider, model, input_tokens, output_tokens: {v}");
    }

    // Catches: stripping empty strings along with the zeros.
    #[test]
    fn empty_provider_and_model_strings_survive_the_zero_strip() {
        let u = ModelUsage::Tts(TtsModelUsage::default());
        let o = usage_attributes(&u);
        let o = o.as_object().unwrap();
        assert_eq!(o.get("provider"), Some(&json!("")));
        assert_eq!(o.get("model"), Some(&json!("")));
        assert_eq!(o.len(), 3, "only type, provider, model: {o:?}");
    }


    // Catches: a failing judge at the same severity as a passing one.
    #[test]
    fn a_failing_verdict_bumps_the_evaluation_severity() {
        let mut t = tagger();
        let mut r = EvaluationResult::new();
        r.insert("politeness", JudgmentResult::new(Verdict::Pass, "fine"));
        r.insert("booked", JudgmentResult::new(Verdict::Fail, "never booked"));
        r.insert("unsure", JudgmentResult::new(Verdict::Maybe, "hmm"));
        t.record_evaluation(&r);

        let recs = session_records(&UploadGate::new(), &view(&[], &[], all_on()), &t, &|_| json!({}));
        let evals = find(&recs, "evaluation");
        assert_eq!(evals.len(), 3);
        assert_eq!(
            evals.iter().map(|r| (r.severity_number, r.severity_text.as_str())).collect::<Vec<_>>(),
            vec![(0, "unspecified"), (17, "error"), (0, "unspecified")],
            "only `fail` bumps -- `maybe` does not"
        );
        assert!(evals.iter().all(|r| r.logger == "evaluations"));
        assert_eq!(attr(evals[1], "evaluation")["tag"], json!("lk.judge.booked:fail"));
    }

    // Catches: the outcome record losing its severity bump, or a `reason` key with no reason.
    #[test]
    fn a_failed_outcome_bumps_severity_and_an_empty_reason_is_omitted() {
        let mut t = tagger();
        t.fail(Some("user hung up"));
        let recs = session_records(&UploadGate::new(), &view(&[], &[], all_on()), &t, &|_| json!({}));
        let o = find(&recs, "outcome");
        assert_eq!(o.len(), 1);
        assert_eq!(o[0].severity_number, 17);
        assert_eq!(attr(o[0], "outcome"), &json!({"outcome": "fail", "reason": "user hung up"}));

        let mut t = tagger();
        t.success(None::<String>);
        let recs = session_records(&UploadGate::new(), &view(&[], &[], all_on()), &t, &|_| json!({}));
        let o = find(&recs, "outcome");
        assert_eq!(o[0].severity_number, 0);
        assert_eq!(attr(o[0], "outcome"), &json!({"outcome": "success"}));
    }

    // Catches: emitting `[]` for an empty tag set or usage list instead of `None`.
    #[test]
    fn empty_tags_and_usage_are_null_not_empty_lists() {
        let recs =
            session_records(&UploadGate::new(), &view(&[], &[], all_on()), &tagger(), &|_| json!({}));
        let s = find(&recs, "session report")[0];
        assert_eq!(attr(s, "session.tags"), &Value::Null);
        assert_eq!(attr(s, "usage"), &Value::Null);
    }

    // Catches: shipping the tags unsorted.
    #[test]
    fn session_tags_are_sorted() {
        let mut t = tagger();
        t.add("zebra:1", None);
        t.add("apple:2", None);
        t.add("mango:3", None);
        let recs = session_records(&UploadGate::new(), &view(&[], &[], all_on()), &t, &|_| json!({}));
        let s = find(&recs, "session report")[0];
        assert_eq!(attr(s, "session.tags"), &json!(["apple:2", "mango:3", "zebra:1"]));
    }

    // Catches: collapsing the two gates (summary on `recording_enabled`, items on `transcript`).
    #[test]
    fn the_summary_and_transcript_gates_are_independent() {
        let t = SystemTime::UNIX_EPOCH + Duration::from_secs(1);
        let items = vec![ChatItem::Message(ChatMessage::text("i", Role::User, "hi", t))];
        let logs_only = RecordingOptions { logs: true, ..Default::default() };
        assert!(logs_only.recording_enabled());
        let recs =
            session_records(&UploadGate::new(), &view(&[], &items, logs_only), &tagger(), &|_| {
                json!({})
            });
        assert_eq!(find(&recs, "session report").len(), 1);
        assert_eq!(find(&recs, "chat item").len(), 0);

        // Nothing enabled: no summary, no transcript -- but see the next test.
        let recs = session_records(
            &UploadGate::new(),
            &view(&[], &items, RecordingOptions::default()),
            &tagger(),
            &|_| json!({}),
        );
        assert!(recs.is_empty(), "{recs:?}");
    }

    // Catches: putting the tag/outcome records behind `recording_enabled`.
    #[test]
    fn tag_and_outcome_records_are_emitted_with_recording_off() {
        let mut t = tagger();
        t.add("voicemail:true", Some(json!({"slot_id": "abc"})));
        t.fail(Some("no answer"));
        let recs = session_records(
            &UploadGate::new(),
            &view(&[], &[], RecordingOptions::default()),
            &t,
            &|_| json!({}),
        );
        assert_eq!(find(&recs, "session report").len(), 0);
        assert_eq!(find(&recs, "tag").len(), 1);
        assert_eq!(find(&recs, "outcome").len(), 1);
        assert!(t.has_evals(), "a bare fail() must trip the upload gate");
    }

    // Catches: a tag record for every tag instead of those carrying metadata.
    #[test]
    fn only_tags_with_non_empty_metadata_produce_a_record() {
        let mut t = tagger();
        t.add("plain:1", None);
        t.add("empty:1", Some(json!({})));
        t.add("rich:1", Some(json!({"k": "v"})));
        let recs = session_records(&UploadGate::new(), &view(&[], &[], all_on()), &t, &|_| json!({}));
        let tags = find(&recs, "tag");
        assert_eq!(tags.len(), 1, "{tags:?}");
        assert_eq!(attr(tags[0], "tag"), &json!({"name": "rich:1", "metadata": {"k": "v"}}));
    }

    // Catches: the `started_at or timestamp` chain handled wrongly, including 0.0.
    #[test]
    fn the_summary_timestamp_falls_through_a_zero_started_at() {
        let mut v = view(&[], &[], all_on());
        let recs = session_records(&UploadGate::new(), &v, &tagger(), &|_| json!({}));
        assert_eq!(find(&recs, "session report")[0].timestamp, (100, 0));

        v.started_at = Some(0.0);
        let recs = session_records(&UploadGate::new(), &v, &tagger(), &|_| json!({}));
        assert_eq!(find(&recs, "session report")[0].timestamp, (200, 0));
    }


    // ============================================================================================
    // tagger
    // ============================================================================================

    // Catches: `success()` and `fail()` accumulating instead of replacing.
    #[test]
    fn the_outcome_tags_are_mutually_exclusive() {
        let mut t = tagger();
        t.success(Some("done"));
        t.fail(Some("actually no"));
        assert_eq!(t.sorted_tags(), vec![TAG_FAIL]);
        assert_eq!(t.outcome(), Some(Outcome::Fail));
        assert_eq!(t.outcome_reason(), Some("actually no"));
    }

    // Catches: re-adding a tag moving it to the end of the insertion order.
    #[test]
    fn re_adding_a_tag_keeps_its_position() {
        let mut t = tagger();
        t.add("a:1", Some(json!({"v": 1})));
        t.add("b:1", Some(json!({"v": 2})));
        t.add("a:1", Some(json!({"v": 3})));
        let names: Vec<&str> = t.entries().map(|(n, _)| n).collect();
        assert_eq!(names, vec!["a:1", "b:1"]);
        assert_eq!(t.entries().next().unwrap().1.metadata, Some(json!({"v": 3})));
    }


    // Catches: judge tags not landing in the tag set.
    #[test]
    fn recording_an_evaluation_writes_both_a_tag_and_a_record() {
        let mut t = tagger();
        let mut r = EvaluationResult::new();
        r.insert("booked", JudgmentResult::new(Verdict::Fail, "no"));
        t.record_evaluation(&r);
        assert_eq!(t.sorted_tags(), vec!["lk.judge.booked:fail"]);
        assert_eq!(t.evaluations().len(), 1);
        assert_eq!(t.evaluations()[0].name, "booked");
        assert!(t.has_evals());
    }

    // Catches: tag timestamps from the wall clock rather than the session clock.
    #[test]
    fn tag_timestamps_come_from_the_session_clock() {
        let clock = std::sync::Arc::new(VirtualClock::new());
        let mut t = Tagger::new(clock.clone());
        t.add("a:1", None);
        clock.advance(Duration::from_secs(10));
        t.add("b:1", None);
        let ts: Vec<SystemTime> = t.entries().map(|(_, e)| e.timestamp).collect();
        // Not exactly 10s: `Clock::wall` nudges by 1 µs per read.
        let delta = ts[1].duration_since(ts[0]).unwrap();
        assert!(delta > Duration::from_millis(9_990) && delta < Duration::from_secs(11), "{ts:?}");
    }
}

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
            assert_eq!(t.call_method0("tags").unwrap().extract::<Vec<String>>().unwrap(), vec!["b", "a"]);
            let bad = py.eval(c"object()", None, None).unwrap();
            assert!(t.call_method1("add", ("c", bad)).is_err(), "metadata must be JSON, not coerced");
        });
    }
}
