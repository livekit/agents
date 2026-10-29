//! The session report's upload records (`telemetry/traces.py`'s `_upload_session_report`), its
//! upload gate and `observability.py`'s `Tagger`: records and host contracts. The records are
//! built by `lk-agents-net`'s session report.

use std::collections::BTreeSet;
use std::sync::Arc;

use serde_json::Value;

use crate::ChatItem;
use crate::errors::HostError;
use crate::metrics::ModelUsage;

/// `RecordingOptions`, the four keys `recording_enabled` reads plus `redaction`.
///
/// `redaction = False` never disables project-level redaction; it only declines to ask for it.
#[crate::host(record(module = "telemetry", extract, emit))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RecordingOptions {
    #[host(default = false)]
    pub audio: bool,
    #[host(default = false)]
    pub traces: bool,
    #[host(default = false)]
    pub logs: bool,
    #[host(default = false)]
    pub transcript: bool,
    /// Not part of `_RECORDING_OPTION_KEYS` -- `recording_enabled` ignores it.
    #[host(default = false)]
    pub redaction: bool,
}

impl RecordingOptions {
    /// Redaction alone does not enable recording.
    pub fn recording_enabled(&self) -> bool {
        self.audio || self.traces || self.logs || self.transcript
    }
}

pub mod projection;

// -----------------------------------------------------------------------------------------
// the tagger
// -----------------------------------------------------------------------------------------

/// One entry of `Tagger.evaluations` (`observability.py:120`).
#[crate::host(record(module = "telemetry", emit))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluationRecord {
    pub name: String,
    pub tag: String,
    pub verdict: String,
    pub reasoning: String,
    pub instructions: String,
}

/// The session outcome. `"fail"` bumps the outcome record's severity.
#[crate::host(record(module = "telemetry", emit))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Success,
    Fail,
}

/// Public because the outcome is derived from tag *presence*.
pub const TAG_SUCCESS: &str = "lk.success";
pub const TAG_FAIL: &str = "lk.fail";

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
#[crate::host(class(name = "Tagger"))]
pub trait Tagger: Send + Sized + 'static {
    /// What the hosting process lends the class: its clock stamps the tags.
    #[host(env)]
    type Env;

    #[host(new)]
    fn new(env: &Self::Env) -> Self;

    /// Mark the session as successful; ``reason`` is kept apart from the tag.
    #[host(call)]
    fn success(&mut self, #[host(default = None)] reason: Option<String>);

    /// Mark the session as failed; ``reason`` is kept apart from the tag.
    #[host(call)]
    fn fail(&mut self, #[host(default = None)] reason: Option<String>);

    /// Add a ``key:value`` tag, with optional structured metadata.
    #[host(call)]
    fn add(&mut self, tag: String, #[host(kw, default = None)] metadata: Option<Value>);

    /// Remove a tag; removing an absent one does nothing.
    #[host(call)]
    fn remove(&mut self, tag: String);

    /// The session's tags.
    #[host(getter)]
    fn tags(&self) -> BTreeSet<String>;

    /// ``sorted(tagger.tags)``.
    #[host(call)]
    fn sorted_tags(&self) -> Vec<String>;

    /// ``(tag, metadata, epoch seconds)``, in insertion order.
    #[host(call)]
    fn entries(&self) -> Vec<(String, Option<Value>, f64)>;

    #[host(call)]
    fn is_empty(&self) -> bool;

    /// The judgments recorded by judge groups.
    #[host(getter)]
    fn evaluations(&self) -> Vec<EvaluationRecord>;

    /// ``"success"``, ``"fail"`` or ``None``.
    #[host(getter)]
    fn outcome(&self) -> Option<Outcome>;

    /// The reason given with the outcome.
    #[host(getter)]
    fn outcome_reason(&self) -> Option<String>;

    /// One judgment of a judge group's result: its ``lk.judge.<name>:<verdict>`` tag and record.
    #[host(call)]
    fn _core_record_judgment(
        &mut self,
        name: String,
        verdict: String,
        reasoning: String,
        instructions: String,
    );

    /// Whether the session has judgments or an outcome to upload.
    #[host(call)]
    fn has_evals(&self) -> bool;

    /// The session report's upload records, in the reference's order; none once uploads are
    /// ``disabled``. A raise in ``project_item`` (a chat item's span attribute) is re-raised once
    /// the records are built.
    #[host(call)]
    fn _core_session_records(
        &self,
        report: SessionReportInput,
        disabled: bool,
        project_item: Arc<dyn ChatItemProjection>,
        #[host(default = false)] capture_content: bool,
    ) -> Result<Vec<LogRecordWire>, HostError>;
}

/// Per-session upload gate: the "data recording is disabled" latch, one transition, one warning.
#[crate::host(class(name = "UploadGate"))]
pub trait UploadGate: Send + Sized + 'static {
    #[host(new)]
    fn new() -> Self;

    /// Called when a session re-initialises recording.
    #[host(call)]
    fn reset(&mut self);

    #[host(getter)]
    fn disabled(&self) -> bool;

    /// Returns ``True`` on the transition only, where the reference logs its warning.
    #[host(call)]
    fn disable(&mut self) -> bool;

    /// Whether an upload response is the project's "recording disabled" 401.
    #[host(call)]
    fn is_disabled_response(status_code: u16, body: Vec<u8>) -> bool;
}

/// What the session report's chat item records carry for one item (its span attribute).
#[crate::host(proxy(host = "collections.abc.Callable"))]
pub trait ChatItemProjection: Send + Sync {
    #[host(call, name = "__call__")]
    fn project(&self, item: ChatItem) -> Result<Value, HostError>;
}

/// `SessionReport._core_view()`: what the report's upload records are built from.
#[crate::host(record(module = "telemetry", extract))]
#[derive(Debug, Clone, Default)]
pub struct SessionReportInput {
    pub job_id: String,
    pub room_id: String,
    pub room: String,
    pub agent_name: String,
    pub sdk_version: String,
    pub started_at: Option<f64>,
    pub timestamp: f64,
    pub options: Value,
    pub recording_options: RecordingOptions,
    pub model_usage: Vec<ModelUsage>,
    pub chat_history: Vec<ChatItem>,
    #[host(default = 0)]
    pub omitted_events: usize,
}

/// The report records a host builds one at a time.
#[crate::host(functions(pid = false))]
pub trait ReportRecords {
    /// Whether recording is enabled for the job.
    #[host(function)]
    fn recording_enabled(options: RecordingOptions) -> bool;

    /// One chat item's upload record; one at a time lets a host SDK emit a large report without
    /// another whole-history vector under its interpreter lock. ``None`` once uploads are
    /// ``disabled`` or the transcript is not recorded.
    #[host(function)]
    fn _core_session_chat_record(
        disabled: bool,
        options: RecordingOptions,
        item: ChatItem,
        projected: Value,
        capture_content: bool,
        identity: (String, String, String),
    ) -> Option<LogRecordWire>;
}

// -----------------------------------------------------------------------------------------
// the interruption detector's span update
// -----------------------------------------------------------------------------------------
// `{"type": "span_update", ..}`: five attributes the host writes onto a span it owns.

/// `{"type": "span_update", "span": .., "attrs": {..}}`.
#[crate::host(record(module = "telemetry", emit, constant(type = "span_update")))]
#[derive(Debug, Clone)]
pub struct SpanUpdateWire {
    /// The span, as a `HostRef` id the host resolves.
    pub span: u64,
    pub attrs: SpanUpdateAttrsWire,
}

/// The nested `attrs` dict: the five values `_apply_span_update` writes onto the span. A wire twin
/// of `lk_agents_inference::interruption::SpanAttributes`, whose `&'static str` field the
/// generator cannot map.
#[crate::host(record(module = "telemetry", emit))]
#[derive(Debug, Clone)]
pub struct SpanUpdateAttrsWire {
    pub is_interruption: String,
    pub probability: f64,
    pub total_duration: f64,
    pub prediction_duration: f64,
    pub detection_delay: f64,
}

/// A session report record as the dict `_upload_session_report` hands `logger.emit`: one integer
/// of epoch nanos, where the log handler's records carry an `HrTime` pair.
#[crate::host(record(module = "telemetry", emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct LogRecordWire {
    pub logger: String,
    pub body: String,
    pub timestamp_nanos: i64,
    pub severity_number: u8,
    pub severity_text: String,
    pub attributes: serde_json::Map<String, Value>,
}
