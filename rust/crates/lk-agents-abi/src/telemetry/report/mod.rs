//! The session report's upload records (`telemetry/traces.py`'s `_upload_session_report`), its
//! upload gate and `observability.py`'s `Tagger`: records and host contracts. The records are
//! built by `lk-agents-net`'s session report.

use std::collections::BTreeSet;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ChatItem;
use crate::errors::HostError;

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
    pub usage: crate::usage::Usage,
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

// Public data contracts. Python behavior remains in its adapters.

/// Which recording features are active; keys default to ``True`` when not specified.
#[crate::host(record(
    module = "recording_options",
    path = "livekit.agents.voice._session_types",
    typed_dict,
    name = "RecordingOptions"
))]
pub struct RecordingOptionsPatch {
    #[host(annotation = "bool")]
    pub audio: crate::Maybe<bool>,
    #[host(annotation = "bool")]
    pub traces: crate::Maybe<bool>,
    #[host(annotation = "bool")]
    pub logs: crate::Maybe<bool>,
    #[host(annotation = "bool")]
    pub transcript: crate::Maybe<bool>,
    #[host(annotation = "bool")]
    pub redaction: crate::Maybe<bool>,
}

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
    pub usage: Option<crate::usage::Usage>,
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
    ) -> Option<LogRecordWire>;
}

/// Bounded retention of a session report's host event snapshots. The native policy retains only
/// byte charges and supplied monotonic timestamps; the adapter owns independent model copies.
#[crate::host(class(name = "_core_EventRetention"))]
pub trait EventRetention: Send + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(new)]
    fn new(env: &Self::Env, capture_content: bool) -> Result<Self, super::upload::UploadError>;

    /// Bind report access to the first job that starts the session. Ownership cannot transfer.
    #[host(call)]
    fn claim(&mut self, job_id: String);

    /// Sessions started outside a job retain their existing application-owned access contract.
    #[host(call)]
    fn allows_read(&self, job_id: String) -> bool;

    /// Immutable consent captured when this session's report retention was created.
    #[host(getter)]
    fn capture_content(&self) -> bool;

    /// Reject content without creation-time consent, unknown schemas and live-object events.
    #[host(call)]
    fn accepts(&self, kind: String, redaction: bool) -> bool;

    /// Expire old entries, then admit only within the item and byte budgets. The adapter removes
    /// exactly the returned prefix before appending the newly admitted snapshot.
    #[host(call)]
    fn admit(&mut self, retained_bytes: usize, now_ns: u64) -> (usize, bool);

    #[host(call)]
    fn expire(&mut self, now_ns: u64) -> usize;

    #[host(getter)]
    fn diagnostics(&self) -> Vec<(String, u64)>;
}
