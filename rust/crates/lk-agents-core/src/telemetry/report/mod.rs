//! The session report's pure projections (docs/internals/telemetry.md "Session report"): one
//! `SessionReport` and its recording options become the Cloud records ([`records()`]) and the
//! recording upload ([`upload()`]), bounds decided here, before any I/O. `lk-agents-net` exports
//! and uploads.
//!
//! The Cloud records (OTLP logs) are as 1.8.10's `_upload_session_report` emitted them: the
//! `chat_history` scope's session summary and chat items, the `evaluations` scope's evaluations,
//! tags and outcome.

mod chat_item;
mod upload;

use lk_agents_abi::ChatItem;
use lk_agents_abi::telemetry::decl::{JsonObject, KvList, RecordDecl, Value};
use lk_agents_abi::telemetry::record::{JobKey, Key, LogRecord, SpanId, SpanRef, TraceId};
use lk_agents_abi::telemetry::report::records::{
    ChatItemLog, EvaluationLog, OutcomeLog, SessionReportLog, TagLog,
};
use lk_agents_abi::telemetry::report::{RecordingOptions, SessionReport};
use serde_json::{Map, Value as Json};

pub use upload::{RecordingUpload, TRANSCRIPT_BYTES, TRANSCRIPT_ITEMS, upload};

/// Whether the session's event retention may keep an event of `kind`: state changes, errors and
/// the close always, content-bearing events with content consent, nothing else.
pub fn event_allowed(kind: &str, capture_content: bool, redaction: bool) -> bool {
    match kind {
        // The summary's error counts and close reason read these; they carry no content.
        "user_state_changed"
        | "agent_state_changed"
        | "user_transcription_timeout"
        | "error"
        | "close" => true,
        "user_input_transcribed"
        | "conversation_item_added"
        | "function_tools_executed"
        | "tool_execution_updated"
        | "agent_false_interruption" => super::policy::content_allowed(capture_content, redaction),
        // Live handles, media arrays and unknown schemas are never report snapshots.
        _ => false,
    }
}

/// OTel's `SeverityNumber.ERROR`.
const SEVERITY_ERROR: i32 = 17;

/// The report's records, in 1.8.10's order. The recording options decide which exist: the
/// summary while anything is recorded, the chat items with the transcript; evaluations, tags with
/// metadata and the outcome always.
pub fn records(report: &SessionReport, options: &RecordingOptions) -> Vec<LogRecord> {
    let job = JobKey(report.job_id.clone());
    let mut out = Vec::new();
    let mut push = |decl: &dyn Decl, at: u64, error: bool| out.push(decl.log(at, error, &job));

    if options.recording_enabled() {
        let mut tags: Vec<String> = report.tags.iter().map(|t| t.name.clone()).collect();
        tags.sort();
        let usage = &report.usage.models;
        let summary = SessionReportLog {
            options: KvList(session_options(&report.options)),
            report_timestamp: report.timestamp,
            tags: (!tags.is_empty()).then_some(tags),
            agent_name: report.agent_name.clone(),
            sdk_version: report.sdk_version.clone(),
            usage: (!usage.is_empty()).then(|| usage.iter().cloned().map(KvList).collect()),
        };
        // A `started_at` of exactly zero is unknown.
        let at = report.started_at.filter(|s| *s != 0.0).unwrap_or(report.timestamp);
        push(&summary, nanos(at), false);
    }

    if options.transcript {
        for item in &report.chat_history {
            let turn = report.turns.get(item.id()).map(|&(trace, span, sampled)| SpanRef {
                trace_id: TraceId(trace),
                span_id: SpanId(span),
                sampled,
            });
            let record = ChatItemLog { item: KvList(chat_item::chat_item(item)), turn };
            let error = matches!(item, ChatItem::FunctionCallOutput(o) if o.is_error);
            let at = item.created_at().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
            push(&record, at.as_nanos() as u64, error);
        }
    }

    for e in &report.evaluations {
        let evaluation = object([
            ("name", e.name.as_str()),
            ("tag", &e.tag),
            ("verdict", &e.verdict),
            ("reasoning", &e.reasoning),
            ("instructions", &e.instructions),
        ]);
        let fail = e.verdict == "fail";
        push(&EvaluationLog { evaluation: KvList(evaluation) }, nanos(report.timestamp), fail);
    }

    for tag in &report.tags {
        let Some(metadata) = tag.metadata.as_ref().filter(|m| truthy(m)) else { continue };
        let mut m = Map::new();
        m.insert("name".into(), tag.name.clone().into());
        m.insert("metadata".into(), metadata.clone());
        push(&TagLog { tag: KvList(m) }, nanos(tag.timestamp), false);
    }

    if let Some(outcome) = &report.outcome {
        let mut m = object([("outcome", outcome.outcome.as_str())]);
        if let Some(reason) = outcome.reason.as_ref().filter(|r| !r.is_empty()) {
            m.insert("reason".into(), reason.clone().into());
        }
        let fail = outcome.outcome == lk_agents_abi::Outcome::Fail;
        push(&OutcomeLog { outcome: KvList(m) }, nanos(report.timestamp), fail);
    }
    out
}

/// A record declaration, written as a log of its job.
trait Decl {
    fn log(&self, at: u64, error: bool, job: &JobKey) -> LogRecord;
}

impl<R: RecordDecl> Decl for R {
    fn log(&self, at: u64, error: bool, job: &JobKey) -> LogRecord {
        let mut attributes = Vec::new();
        self.write(&mut attributes);
        let (severity_number, severity_text) =
            if error { (SEVERITY_ERROR, "error") } else { (0, "unspecified") };
        LogRecord {
            scope: Key::Borrowed(R::SCOPE),
            body: Value::String(R::BODY.to_owned()),
            time_unix_nanos: at,
            severity_number,
            severity_text: Key::Borrowed(severity_text),
            attributes,
            span: self.span(),
            job: Some(job.clone()),
        }
    }
}

/// Unix seconds as nanoseconds.
fn nanos(secs: f64) -> u64 {
    (secs.max(0.0) * 1e9) as u64
}

fn object<'a>(fields: impl IntoIterator<Item = (&'static str, &'a str)>) -> JsonObject {
    fields.into_iter().map(|(k, v)| (k.to_owned(), Json::String(v.to_owned()))).collect()
}

/// Python truthiness: an empty or zero value is no metadata.
fn truthy(v: &Json) -> bool {
    match v {
        Json::Null => false,
        Json::Bool(b) => *b,
        Json::Number(n) => n.as_f64() != Some(0.0),
        Json::String(s) => !s.is_empty(),
        Json::Array(a) => !a.is_empty(),
        Json::Object(o) => !o.is_empty(),
    }
}

/// The resolved options as the summary record carries them: `instructions` (customer prompt text)
/// omitted and `keyterms` keyed `lk.pii.keyterms`, at any depth.
pub(super) fn session_options(options: &Json) -> JsonObject {
    fn walk(v: &Json) -> Json {
        match v {
            Json::Object(o) => Json::Object(
                o.iter()
                    .filter(|(k, _)| k.as_str() != "instructions")
                    .map(|(k, v)| {
                        let key = if k == "keyterms" { "lk.pii.keyterms" } else { k.as_str() };
                        (key.to_owned(), walk(v))
                    })
                    .collect(),
            ),
            Json::Array(a) => Json::Array(a.iter().map(walk).collect()),
            v => v.clone(),
        }
    }
    match walk(options) {
        Json::Object(o) => o,
        _ => JsonObject::new(),
    }
}

#[cfg(test)]
#[path = "report_tests.rs"]
mod tests;
