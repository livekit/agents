//! The session report's records: `telemetry/traces.py`'s `_upload_session_report` records, its
//! `_UploadGate`, and `observability.py`'s `Tagger` state. The records are built here as
//! [`LogRecord`]s and emitted by the host's SDK as ABI's `LogRecordWire`.

use std::collections::BTreeSet;
use std::time::SystemTime;

use lk_agents_abi::ChatItem;
use lk_agents_abi::telemetry::policy;
use lk_agents_abi::telemetry::report::projection::SessionReportProjection;
use lk_agents_abi::telemetry::report::{
    EvaluationRecord, LogRecordWire, Outcome, RecordingOptions, SessionReportInput, TAG_FAIL,
    TAG_SUCCESS,
};
use lk_agents_abi::usage::ModelUsage;
use serde_json::{Map, Value, json};

use super::{
    HrTime, LogRecord, LoggerName, ReportProjection, attribute_report_record, hr_seconds, hr_time,
    record, report_value_bytes, report_value_fits,
};

// -----------------------------------------------------------------------------------------
// log records and their timestamps
// -----------------------------------------------------------------------------------------

/// A wall-clock stamp; pre-epoch reads as the epoch.
pub fn hr_wall(t: SystemTime) -> HrTime {
    hr_time(t.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0))
}

/// An evaluation's fields, in the reference's insertion order.
fn evaluation_attributes(e: &EvaluationRecord) -> [(&'static str, &str); 5] {
    [
        ("name", &e.name),
        ("tag", &e.tag),
        ("verdict", &e.verdict),
        ("reasoning", &e.reasoning),
        ("instructions", &e.instructions),
    ]
}

/// One model's usage as the report carries it: the normalized record without its absent
/// fields (a reported zero stays). `None` when it does not fit a record.
pub fn usage_attributes(u: &ModelUsage) -> Value {
    fn absent_dropped(value: &mut Value) {
        if let Value::Object(map) = value {
            map.retain(|_, v| !v.is_null());
            map.values_mut().for_each(absent_dropped);
        }
    }
    let mut value = serde_json::to_value(u).unwrap_or(Value::Null);
    absent_dropped(&mut value);
    if report_value_fits(&value) { value } else { Value::Null }
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
        || !crate::telemetry::policy::content_allowed(capture_content, options.redaction)
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
    let projection =
        <ReportProjection as SessionReportProjection>::new(capture_content, &opts, false);
    let capture_content = projection.content();

    if opts.recording_enabled() {
        // An `or` chain: a `started_at` of exactly 0.0 falls through to `timestamp`.
        let seconds = match report.started_at {
            Some(s) if s != 0.0 => s,
            _ => report.timestamp,
        };
        let tags = if capture_content {
            let mut tags: Vec<_> = tagger
                .tags
                .iter()
                .take(policy::SDK_PENDING_RECORDS)
                .map(|(name, _)| name)
                .filter(|name| name.len() <= 256)
                .cloned()
                .collect();
            tags.sort();
            tags
        } else {
            Vec::new()
        };
        let mut usage = Vec::new();
        let mut usage_bytes = 2usize;
        for item in report.usage.models.iter().take(policy::SDK_PENDING_RECORDS) {
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

    for e in tagger
        .evaluation_results
        .iter()
        .take(policy::SDK_PENDING_RECORDS.saturating_sub(out.len() + 1))
    {
        let fields: Vec<_> = evaluation_attributes(e)
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
        tagger.tags.iter().take(policy::SDK_PENDING_RECORDS.saturating_sub(out.len() + 1))
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
            .outcome_reason
            .as_deref()
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
        .saturating_add(tagger.evaluation_results.len())
        .saturating_add(tagger.tags.len())
        .saturating_sub(policy::SDK_PENDING_RECORDS)
        .saturating_add(before - out.len());
    if omitted > 0
        && let Some(record) = out.first_mut()
    {
        record
            .attributes
            .as_object_mut()
            .expect("report attributes are objects")
            .insert("lk.telemetry.omitted_report_entries".into(), omitted.into());
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
#[derive(Debug, Default)]
pub struct UploadGate {
    disabled: bool,
    /// Set once, so a caller can log the warning exactly once per session the way `disable()` does.
    warned: bool,
}

/// The "data recording is disabled" latch (`upload_gate.py`): one transition, one warning.
impl lk_agents_abi::telemetry::report::UploadGate for UploadGate {
    fn new() -> Self {
        UploadGate::default()
    }

    /// `reset()`. Called when a session re-initialises recording.
    fn reset(&mut self) {
        self.disabled = false;
        self.warned = false;
    }

    fn disabled(&self) -> bool {
        self.disabled
    }

    /// `disable()`. Returns `true` on the transition only, where the reference logs its warning.
    fn disable(&mut self) -> bool {
        if self.disabled {
            return false;
        }
        self.disabled = true;
        let first = !self.warned;
        self.warned = true;
        first
    }

    /// `is_disabled_response(status_code, body)`; static because the exporter has no gate in hand.
    fn is_disabled_response(status_code: u16, body: Vec<u8>) -> bool {
        if status_code != 401 {
            return false;
        }
        // `body.decode("utf-8", "ignore").lower()`: invalid UTF-8 is dropped, not an error.
        let text = String::from_utf8_lossy(&body).to_lowercase();
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

/// `observability.py:17`: the session's tags, judgments and outcome. Times are supplied by the
/// caller from the session clock.
#[derive(Debug, Default)]
pub struct Tagger {
    tags: Vec<(String, TagEntry)>,
    evaluation_results: Vec<EvaluationRecord>,
    outcome_reason: Option<String>,
}

impl Tagger {
    /// Clears any existing `lk.fail` first.
    pub fn success(&mut self, reason: Option<String>, at: SystemTime) {
        self.remove(TAG_FAIL);
        self.add(TAG_SUCCESS.to_owned(), None, at);
        self.outcome_reason = reason;
    }

    pub fn fail(&mut self, reason: Option<String>, at: SystemTime) {
        self.remove(TAG_SUCCESS);
        self.add(TAG_FAIL.to_owned(), None, at);
        self.outcome_reason = reason;
    }

    /// Python `dict` assignment: replace in place (keeping position), else append.
    pub fn add(&mut self, tag: String, metadata: Option<Value>, at: SystemTime) {
        let entry = TagEntry { metadata, timestamp: at };
        match self.tags.iter_mut().find(|(k, _)| *k == tag) {
            Some(slot) => slot.1 = entry,
            None => self.tags.push((tag, entry)),
        }
    }

    pub fn remove(&mut self, tag: &str) {
        self.tags.retain(|(k, _)| k != tag);
    }

    pub fn tags(&self) -> BTreeSet<String> {
        self.tags.iter().map(|(k, _)| k.clone()).collect()
    }

    pub fn sorted_tags(&self) -> Vec<String> {
        self.tags().into_iter().collect()
    }

    pub fn entries(&self) -> Vec<(String, Option<Value>, f64)> {
        self.tags
            .iter()
            .map(|(k, e)| (k.clone(), e.metadata.clone(), lk_agents_abi::unix_secs(e.timestamp)))
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }

    pub fn evaluations(&self) -> &[EvaluationRecord] {
        &self.evaluation_results
    }

    /// From tag presence; `lk.success` wins if both are set.
    pub fn outcome(&self) -> Option<Outcome> {
        if self.tags.iter().any(|(k, _)| k == TAG_SUCCESS) {
            Some(Outcome::Success)
        } else if self.tags.iter().any(|(k, _)| k == TAG_FAIL) {
            Some(Outcome::Fail)
        } else {
            None
        }
    }

    /// Only another `success`/`fail` overwrites it.
    pub fn outcome_reason(&self) -> Option<&str> {
        self.outcome_reason.as_deref()
    }

    pub fn record_judgment(
        &mut self,
        name: String,
        verdict: String,
        reasoning: String,
        instructions: String,
        at: SystemTime,
    ) {
        let tag = format!("lk.judge.{name}:{verdict}");
        self.add(tag.clone(), None, at);
        self.evaluation_results.push(EvaluationRecord {
            name,
            tag,
            verdict,
            reasoning,
            instructions,
        });
    }

    /// `has_evals` at `job.py:320`: `bool(self._tagger.evaluations or self._tagger.outcome)`.
    pub fn has_evals(&self) -> bool {
        !self.evaluation_results.is_empty() || self.outcome().is_some()
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
#[path = "records_tests.rs"]
mod tests;
