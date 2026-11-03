//! The session report (docs/internals/telemetry.md "Session report"): the one value a session's
//! end produces, the recording options that decide its outputs, the tagger the application tags
//! the session with, and the upload's header and summary. `lk_agents_core::telemetry::report`
//! projects the report; `lk-agents-net` exports and uploads the projections.

use std::collections::BTreeSet;

use serde::Serialize;
use serde_json::Value;

use crate::ChatItem;
use crate::usage::Usage;

/// What the session records to LiveKit Cloud. Every part is on unless the session turns it off.
///
/// `redaction = False` never disables project-level redaction; it only declines to ask for it.
#[crate::host(record(module = "telemetry", extract, emit))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordingOptions {
    #[host(default = true)]
    pub audio: bool,
    #[host(default = true)]
    pub traces: bool,
    #[host(default = true)]
    pub logs: bool,
    #[host(default = true)]
    pub transcript: bool,
    /// Not a recorded part: `recording_enabled` ignores it.
    #[host(default = false)]
    pub redaction: bool,
}

impl Default for RecordingOptions {
    fn default() -> Self {
        RecordingOptions {
            audio: true,
            traces: true,
            logs: true,
            transcript: true,
            redaction: false,
        }
    }
}

impl RecordingOptions {
    /// Redaction alone does not enable recording.
    pub fn recording_enabled(&self) -> bool {
        self.audio || self.traces || self.logs || self.transcript
    }
}

/// One judge's judgment of the session.
#[crate::host(record(module = "telemetry", extract, emit))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluationRecord {
    pub name: String,
    pub tag: String,
    pub verdict: String,
    pub reasoning: String,
    pub instructions: String,
}

/// The session outcome. `"fail"` bumps the outcome record's severity.
#[crate::host(record(module = "telemetry", extract, emit))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Success,
    Fail,
}

impl Outcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Outcome::Success => "success",
            Outcome::Fail => "fail",
        }
    }
}

/// Public because the outcome is derived from tag *presence*.
pub const TAG_SUCCESS: &str = "lk.success";
pub const TAG_FAIL: &str = "lk.fail";

/// One tag of the session, with its optional structured metadata and when it was added.
#[crate::host(record(module = "telemetry", extract, emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct ReportTag {
    pub name: String,
    pub metadata: Option<Value>,
    /// Unix seconds.
    pub timestamp: f64,
}

/// The session's outcome and the reason given with it.
#[crate::host(record(module = "telemetry", extract, emit))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReportOutcome {
    pub outcome: Outcome,
    pub reason: Option<String>,
}

/// The summary of one voice session: its transcript, usage, options, events, tags and
/// evaluations, and its audio recording. One snapshot every reader shares: `on_session_end`
/// (through `JobContext.make_session_report()`), the Cloud records and the recording upload.
#[crate::host(record(module = "telemetry", extract, emit))]
#[derive(Debug, Clone, Default)]
pub struct SessionReport {
    pub job_id: String,
    pub room_id: String,
    pub room: String,
    /// The name the agent was dispatched with.
    #[host(default = "")]
    pub agent_name: String,
    /// The resolved session options, its recording options under `recording_options`.
    pub options: Value,
    /// The session's events, as dictionaries.
    #[host(default_factory = "list")]
    pub events: Vec<Value>,
    pub chat_history: Vec<ChatItem>,
    /// The span of the turn each chat item belongs to, by item id: (trace id, span id, sampled).
    #[host(default_factory = "dict")]
    pub turns: std::collections::BTreeMap<String, (u128, u64, bool)>,
    pub usage: Usage,
    #[host(default_factory = "list")]
    pub tags: Vec<ReportTag>,
    #[host(default_factory = "list")]
    pub evaluations: Vec<EvaluationRecord>,
    #[host(default = None)]
    pub outcome: Option<ReportOutcome>,
    /// When the session started, in unix seconds.
    #[host(default = None)]
    pub started_at: Option<f64>,
    /// When the report was made, in unix seconds.
    pub timestamp: f64,
    /// The audio recording's length, in seconds.
    #[host(default = None)]
    pub duration: Option<f64>,
    #[host(default = None)]
    pub audio_recording_path: Option<String>,
    /// When the audio recording started, in unix seconds.
    #[host(default = None)]
    pub audio_recording_started_at: Option<f64>,
    pub sdk_version: String,
}

/// The recording upload's `MetricsRecordingHeader`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordingHeader {
    pub room_id: String,
    pub job_id: String,
    /// The audio recording's start, in unix milliseconds.
    pub start_time_millis: i64,
    pub duration_millis: u64,
    pub simulated: bool,
    pub redaction_enabled: bool,
    /// [`SessionSummary`] as JSON, at most [`SESSION_SUMMARY_BYTES`].
    pub session_summary: String,
}

/// The header's `session_summary` bound.
pub const SESSION_SUMMARY_BYTES: usize = 16 * 1024;

/// The session facts agent evaluations read (`evalschema/session-summary.schema.json`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SessionSummary {
    pub job_id: String,
    pub room_id: String,
    pub room: String,
    pub audio_recording_path: Option<String>,
    pub audio_recording_started_at: Option<f64>,
    /// The interruption, endpointing, tool-step and recording settings.
    pub options: Option<Value>,
    pub timestamp: f64,
    /// The `Usage` model's entries, absent fields omitted, as the summary record carries them.
    pub usage: Option<Vec<Value>>,
    pub sdk_version: String,
    pub agent_name: String,
    pub tags: Vec<String>,
    pub duration_ms: Option<i64>,
    pub user_turns: u64,
    pub errors: ErrorCounts,
    /// `CR_*`, from the last close event.
    pub close_reason: Option<String>,
    /// Chat items left out of the uploaded transcript to keep it within its bounds.
    #[serde(skip_serializing_if = "is_zero")]
    pub omitted_chat_items: u64,
}

fn is_zero(n: &u64) -> bool {
    *n == 0
}

/// Error events by source; every key is always present.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ErrorCounts {
    pub llm: u64,
    pub stt: u64,
    pub tts: u64,
    pub realtime: u64,
    pub interruption: u64,
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

    /// The tags in insertion order, with their metadata and when they were added.
    #[host(call)]
    fn entries(&self) -> Vec<ReportTag>;

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

/// Bounded retention of a session report's host event snapshots. The native policy retains only
/// byte charges and supplied monotonic timestamps; the adapter owns independent model copies.
#[crate::host(class(name = "_core_EventRetention"))]
pub trait EventRetention: Send + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    /// Takes the process's GenAI message capture setting as it is now.
    #[host(new)]
    fn new(env: &Self::Env) -> Self;

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

    /// Expire old entries, then admit `event` only within the item and byte budgets. The adapter
    /// removes exactly the returned prefix before appending its snapshot of an admitted event.
    #[host(call)]
    fn admit(&mut self, event: super::policy::TelemetryValue, now_ns: u64) -> (usize, bool);

    #[host(call)]
    fn expire(&mut self, now_ns: u64) -> usize;

    #[host(getter)]
    fn diagnostics(&self) -> Vec<(String, u64)>;
}
