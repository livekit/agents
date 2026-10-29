//! `telemetry/traces.py`'s `_upload_session_report` records, its `_UploadGate`, and
//! `observability.py`'s `Tagger`. The records are ABI's telemetry `LogRecord`s, built here and
//! emitted by the host's SDK.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::SystemTime;

use lk_agents_abi::errors::HostError;
use lk_agents_abi::metrics::ModelUsage;
use lk_agents_abi::telemetry::policy;
use lk_agents_abi::telemetry::report::projection::SessionReportProjection;
use lk_agents_abi::telemetry::report::{
    ChatItemProjection, EvaluationRecord, LogRecordWire, Outcome, RecordingOptions,
    SessionReportInput, TAG_FAIL, TAG_SUCCESS,
};
use lk_agents_abi::telemetry::{HrTime, LogRecord, Severity, hr_time, hr_wall};
use lk_agents_abi::{ChatItem, Clock};
use serde_json::{Map, Value, json};

// -----------------------------------------------------------------------------------------
// the session report's log records
// -----------------------------------------------------------------------------------------
// What `_upload_session_report` (`telemetry/traces.py:480`) emits as OTel log records; Rust builds
// them, the host's SDK emits them. Tag-with-metadata and outcome records are not behind
// `recording_enabled` (hence `job.py:320`'s `recording_enabled(..) or has_evals`). Buffered early
// logs (`job.py:265`) are the log handler's.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LoggerName {
    ChatHistory,
    /// Also carries the tag and outcome records.
    Evaluations,
}

/// A report record: `SeverityNumber.ERROR` / `"error"` when `error`, else `UNSPECIFIED` /
/// `"unspecified"` -- the report's own lowercase spelling, not the log handler's.
pub(super) fn record(
    logger: LoggerName,
    body: &str,
    timestamp: HrTime,
    attributes: Value,
    error: bool,
) -> LogRecord {
    let (severity_number, severity_text) =
        if error { (Severity::Error as u8, "error") } else { (0, "unspecified") };
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
pub(super) fn hr_seconds(seconds: f64) -> HrTime {
    hr_time((seconds * 1e9) as u128)
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

/// One item at a time lets a host SDK emit a large report without constructing another
/// whole-history record vector under its interpreter lock.
pub fn session_chat_record(
    disabled: bool,
    options: &RecordingOptions,
    item: ChatItem,
    projected: Value,
    capture_content: bool,
    identity: (String, String, String),
) -> Option<LogRecordWire> {
    if disabled
        || !options.transcript
        || !lk_agents_core::telemetry::policy::content_allowed(capture_content, options.redaction)
    {
        return None;
    }
    let mut record = chat_item_record(&item, projected);
    attribute_report_record(&mut record, &identity.0, &identity.1, &identity.2, true);
    report_value_fits(&record.attributes).then(|| record.into())
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

fn outcome_str(outcome: Outcome) -> &'static str {
    match outcome {
        Outcome::Success => "success",
        Outcome::Fail => "fail",
    }
}

/// In the reference's order; empty once uploads are `disabled`.
pub fn session_records(
    disabled: bool,
    report: &SessionReportInput,
    tagger: &Tagger,
    project_item: &dyn Fn(&ChatItem) -> Value,
    capture_content: bool,
) -> Vec<LogRecord> {
    if disabled
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
    let projection = <super::projection::ReportProjection as SessionReportProjection>::new(
        capture_content,
        &opts,
        false,
    );
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
                .take(policy::SDK_PENDING_RECORDS)
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
        for item in report.model_usage.iter().take(policy::SDK_PENDING_RECORDS) {
            let value = usage_attributes(item);
            let Some(bytes) = report_value_bytes(&value) else {
                omitted += 1;
                continue;
            };
            if value.is_null()
                || usage_bytes.saturating_add(bytes + 1) > policy::SDK_RECORD_BYTES / 2
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

    if projection.transcript() {
        for item in report
            .chat_history
            .iter()
            .take(policy::SDK_PENDING_RECORDS.saturating_sub(out.len() + 1))
        {
            out.push(chat_item_record(item, project_item(item)));
        }
    }

    for e in
        tagger.evaluations().iter().take(policy::SDK_PENDING_RECORDS.saturating_sub(out.len() + 1))
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
        tagger.entries().take(policy::SDK_PENDING_RECORDS.saturating_sub(out.len() + 1))
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
        m.insert("outcome".into(), Value::String(outcome_str(outcome).into()));
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
        .saturating_sub(policy::SDK_PENDING_RECORDS)
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
pub(super) fn report_value_fits(value: &(impl serde::Serialize + ?Sized)) -> bool {
    report_value_bytes(value).is_some()
}

pub(super) fn report_value_bytes(value: &(impl serde::Serialize + ?Sized)) -> Option<usize> {
    struct Counter(usize);
    impl std::io::Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if bytes.len() > policy::SDK_RECORD_BYTES.saturating_sub(self.0 + 1024) {
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

pub(super) fn attribute_report_record(
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
#[derive(Debug, Default)]
pub struct UploadGate {
    disabled: bool,
    /// Set once, so a caller can log the warning exactly once per session the way `disable()` does.
    warned: bool,
}

/// The "data recording is disabled" latch (`upload_gate.py`): one transition, one warning.
impl UploadGate {
    pub fn new() -> Self {
        UploadGate::default()
    }

    /// `reset()`. Called when a session re-initialises recording.
    pub fn reset(&mut self) {
        self.disabled = false;
        self.warned = false;
    }

    pub fn disabled(&self) -> bool {
        self.disabled
    }

    /// `disable()`. Returns `true` on the transition only, where the reference logs its warning.
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
    pub fn is_disabled_response(status_code: u16, body: &[u8]) -> bool {
        if status_code != 401 {
            return false;
        }
        // `body.decode("utf-8", "ignore").lower()`: invalid UTF-8 is dropped, not an error.
        let text = String::from_utf8_lossy(body).to_lowercase();
        DISABLED_MARKERS.iter().any(|m| text.contains(m))
    }
}

/// `observability.py:12`.
#[derive(Debug, Clone, PartialEq)]
pub struct TagEntry {
    /// Only tags with metadata produce a "tag" log record.
    pub metadata: Option<Value>,
    /// Taken from the session clock.
    pub timestamp: SystemTime,
}

/// `observability.py:17`.
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

impl Tagger {
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        Tagger { clock, tags: Vec::new(), evaluation_results: Vec::new(), outcome_reason: None }
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
    pub fn success(&mut self, reason: Option<impl Into<String>>) {
        self.pop(TAG_FAIL);
        self.put(TAG_SUCCESS, None);
        self.outcome_reason = reason.map(Into::into);
    }

    /// Mark the session as failed; ``reason`` is kept apart from the tag.
    pub fn fail(&mut self, reason: Option<impl Into<String>>) {
        self.pop(TAG_SUCCESS);
        self.put(TAG_FAIL, None);
        self.outcome_reason = reason.map(Into::into);
    }

    /// Add a ``key:value`` tag, with optional structured metadata.
    pub fn add(&mut self, tag: &str, metadata: Option<Value>) {
        self.put(tag, metadata);
    }

    /// Remove a tag; removing an absent one does nothing.
    pub fn remove(&mut self, tag: &str) {
        self.pop(tag);
    }

    /// `tags`, in insertion order; see [`Self::sorted_tags`] for the reference's `sorted(...)`.
    pub fn tags(&self) -> impl Iterator<Item = &str> {
        self.tags.iter().map(|(k, _)| k.as_str())
    }

    /// `sorted(tagger.tags)` -- what the session summary record carries.
    pub fn sorted_tags(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.tags().collect();
        v.sort_unstable();
        v
    }

    /// `tagger._tags.items()`, in insertion order. This is what the tag-metadata records walk.
    pub fn entries(&self) -> impl Iterator<Item = (&str, &TagEntry)> {
        self.tags.iter().map(|(k, e)| (k.as_str(), e))
    }

    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }

    /// The judgments recorded by judge groups.
    // The reference returns a `.copy()`; a shared slice is the Rust equivalent.
    pub fn evaluations(&self) -> &[EvaluationRecord] {
        &self.evaluation_results
    }

    /// ``"success"``, ``"fail"`` or ``None``.
    // From tag presence; `lk.success` wins if both are set.
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
    pub fn outcome_reason(&self) -> Option<&str> {
        self.outcome_reason.as_deref()
    }

    /// One judgment of a judge group's result: its `lk.judge.<name>:<verdict>` tag and record.
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
    pub fn has_evals(&self) -> bool {
        !self.evaluation_results.is_empty() || self.outcome().is_some()
    }
}

impl lk_agents_abi::telemetry::report::Tagger for Tagger {
    /// Timestamps are strictly increasing, so same-instant tags keep insertion order.
    fn new() -> Self {
        Tagger::new(Arc::new(lk_agents_abi::RealClock::new()))
    }

    fn success(&mut self, reason: Option<String>) {
        Tagger::success(self, reason);
    }

    fn fail(&mut self, reason: Option<String>) {
        Tagger::fail(self, reason);
    }

    fn add(&mut self, tag: String, metadata: Option<Value>) {
        Tagger::add(self, &tag, metadata);
    }

    fn remove(&mut self, tag: String) {
        Tagger::remove(self, &tag);
    }

    fn tags(&self) -> BTreeSet<String> {
        Tagger::tags(self).map(str::to_owned).collect()
    }

    fn sorted_tags(&self) -> Vec<String> {
        Tagger::sorted_tags(self).into_iter().map(str::to_owned).collect()
    }

    fn entries(&self) -> Vec<(String, Option<Value>, f64)> {
        let seconds = |at: SystemTime| {
            at.duration_since(SystemTime::UNIX_EPOCH).map_or(0.0, |since| since.as_secs_f64())
        };
        self.tags
            .iter()
            .map(|(k, e)| (k.clone(), e.metadata.clone(), seconds(e.timestamp)))
            .collect()
    }

    fn is_empty(&self) -> bool {
        Tagger::is_empty(self)
    }

    fn evaluations(&self) -> Vec<EvaluationRecord> {
        Tagger::evaluations(self).to_vec()
    }

    fn outcome(&self) -> Option<Outcome> {
        Tagger::outcome(self)
    }

    fn outcome_reason(&self) -> Option<String> {
        Tagger::outcome_reason(self).map(str::to_owned)
    }

    fn _core_record_judgment(
        &mut self,
        name: String,
        verdict: String,
        reasoning: String,
        instructions: String,
    ) {
        self.record_judgment(&name, &verdict, &reasoning, &instructions);
    }

    fn has_evals(&self) -> bool {
        Tagger::has_evals(self)
    }

    fn _core_session_records(
        &self,
        report: SessionReportInput,
        disabled: bool,
        project_item: Arc<dyn ChatItemProjection>,
        capture_content: bool,
    ) -> Result<Vec<LogRecordWire>, HostError> {
        // A failing projection cannot unwind through the walk: the first is kept and returned.
        let failure = std::cell::RefCell::new(None::<HostError>);
        let project = |item: &ChatItem| -> Value {
            if failure.borrow().is_some() {
                return Value::Null;
            }
            project_item.project(item.clone()).unwrap_or_else(|error| {
                *failure.borrow_mut() = Some(error);
                Value::Null
            })
        };
        let records = session_records(disabled, &report, self, &project, capture_content);
        match failure.into_inner() {
            Some(error) => Err(error),
            None => Ok(records.into_iter().map(LogRecordWire::from).collect()),
        }
    }
}

impl lk_agents_abi::telemetry::report::UploadGate for UploadGate {
    fn new() -> Self {
        UploadGate::new()
    }

    fn reset(&mut self) {
        UploadGate::reset(self);
    }

    fn disabled(&self) -> bool {
        UploadGate::disabled(self)
    }

    fn disable(&mut self) -> bool {
        UploadGate::disable(self)
    }

    fn is_disabled_response(status_code: u16, body: Vec<u8>) -> bool {
        UploadGate::is_disabled_response(status_code, &body)
    }
}

/// The report records a host builds one at a time.
pub struct ReportRecords;

impl lk_agents_abi::telemetry::report::ReportRecords for ReportRecords {
    fn recording_enabled(options: RecordingOptions) -> bool {
        options.recording_enabled()
    }

    fn _core_session_chat_record(
        disabled: bool,
        options: RecordingOptions,
        item: ChatItem,
        projected: Value,
        capture_content: bool,
        identity: (String, String, String),
    ) -> Option<LogRecordWire> {
        session_chat_record(disabled, &options, item, projected, capture_content, identity)
    }
}

#[cfg(test)]
mod tests;
