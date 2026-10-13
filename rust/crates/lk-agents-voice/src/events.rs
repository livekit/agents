use lk_agents_abi::voice::events::{AgentState, UserState};
// The host records that are these events' payloads as they are.
pub use lk_agents_abi::voice::events::{
    AgentBackchannelOpportunityEvent, EndOfSpeechEvent, OverlapInterruptionEvent,
    PreemptiveGenerationEvent, StartOfSpeechEvent, UserTurnExceededEvent,
};
use lk_agents_abi::{SpeechEvent, Ts};
use std::sync::Arc;
use std::time::Duration;

/// `ErrorEvent` (`events.py`): which leg failed, whether the attempt loop goes on (`recoverable`),
/// and the error itself (a `ProviderError::Host` is moved, never stringified).
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorEvent {
    pub source: ErrorSource,
    pub recoverable: bool,
    pub error: lk_agents_abi::errors::ProviderError,
    /// The speech whose request failed; `None` for the STT leg and the realtime socket.
    pub speech: Option<crate::agent_activity::scheduling::SpeechId>,
}

/// The four legs the reference's `_on_error` names. An interruption detector failure degrades to
/// VAD silently and the EOT model's `error` is never subscribed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, lk_agents_abi::Vocabulary)]
pub enum ErrorSource {
    Stt,
    Llm,
    Tts,
    Realtime,
}

impl SessionEvent {
    /// The event as the journal keeps it: a `ProviderError::Host` replaced by a `Local` copy, so
    /// the journal never pins a host exception.
    pub fn journal_form(&self) -> SessionEvent {
        match self {
            SessionEvent::Error(e) => SessionEvent::Error(Box::new(ErrorEvent {
                source: e.source,
                recoverable: e.recoverable,
                error: journal_error(&e.error),
                speech: e.speech,
            })),
            SessionEvent::Close { reason, error } => {
                SessionEvent::Close { reason: *reason, error: error.as_ref().map(journal_error) }
            }
            other => other.clone(),
        }
    }
}

fn journal_error(
    error: &lk_agents_abi::errors::ProviderError,
) -> lk_agents_abi::errors::ProviderError {
    use lk_agents_abi::errors::{ApiError, ApiErrorKind, ProviderError};
    match error {
        ProviderError::Host { error, api_kind, retryable, status_code, .. } => {
            ProviderError::Local(ApiError {
                kind: api_kind.unwrap_or(ApiErrorKind::Api),
                message: error.to_string(),
                retryable: *retryable,
                status_code: *status_code,
                request_id: None,
            })
        }
        ProviderError::Local(api) => ProviderError::Local(api.clone()),
    }
}

impl ErrorSource {
    /// `LLMError.type` and its siblings: the wire strings, which are public API.
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorSource::Stt => "stt_error",
            ErrorSource::Llm => "llm_error",
            ErrorSource::Tts => "tts_error",
            ErrorSource::Realtime => "realtime_model_error",
        }
    }
}

/// `EotPredictionEvent`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EotPredictionEvent {
    pub probability: f64,
    pub threshold: f64,
    /// Server-side inference time in seconds; `0.0` when the detector reports none.
    pub inference_duration: f64,
    pub delay: std::time::Duration,
}

/// One transcript reaching the session (becomes `user_input_transcribed`). Boxed in
/// [`SessionEvent`]: [`SpeechEvent`] is large.
#[derive(Debug, Clone, PartialEq)]
pub struct UserTranscriptEvent {
    pub ev: SpeechEvent,
    /// `is_final` on the wire. Preflight transcripts are not final; a manual commit re-emits a held
    /// interim as final.
    pub is_final: bool,
    /// The reference's `speaking=`: `None` when the session has no trusted VAD. Three-valued.
    pub speaking: Option<bool>,
    /// `UserInputTranscribedEvent.item_id`, shared by every segment; `None` from the turn machine,
    /// stamped by the driver.
    pub item_id: Option<String>,
}

/// `on_end_of_turn`'s `EndOfTurnInfo`, as much of it as the host needs.
#[derive(Debug, Clone, PartialEq)]
pub struct UserTurnCompletedEvent {
    /// The turn that ended. A `TurnCommand::HookDone` answers for this turn and no other.
    pub turn: crate::audio_recognition::TurnId,
    pub new_transcript: String,
    pub transcript_confidence: f64,
}

/// VAD heard speech and no transcript arrived (`UserTranscriptionTimeoutEvent`, #6182).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TranscriptionTimeoutEvent {
    /// Total VAD speech this turn that produced no transcript, summed across segments.
    pub speech_duration: Duration,
    pub turn_start: Ts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticLevel {
    Info,
    Warning,
    Error,
}

impl DiagnosticLevel {
    /// The name the host's `logging` module takes.
    pub fn as_str(self) -> &'static str {
        match self {
            DiagnosticLevel::Info => "info",
            DiagnosticLevel::Warning => "warning",
            DiagnosticLevel::Error => "error",
        }
    }
}

/// A sentence the core decided and only the host can say: a core crate has no logger. `message` is
/// the reference's own sentence (users grep for it); `fields` is its `extra={...}`, `lk.pii.*` included.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticEvent {
    pub level: DiagnosticLevel,
    pub message: &'static str,
    pub fields: Vec<(&'static str, String)>,
}

impl DiagnosticEvent {
    pub fn wire(&self) -> lk_agents_abi::voice::events::DiagnosticWire {
        lk_agents_abi::voice::events::DiagnosticWire {
            level: self.level.as_str().to_owned(),
            message: self.message.to_owned(),
            fields: self.fields.iter().map(|(k, v)| ((*k).to_owned(), v.clone())).collect(),
        }
    }

    pub fn warning(message: &'static str) -> Self {
        DiagnosticEvent { level: DiagnosticLevel::Warning, message, fields: Vec::new() }
    }

    pub fn info(message: &'static str) -> Self {
        DiagnosticEvent { level: DiagnosticLevel::Info, message, fields: Vec::new() }
    }

    pub fn error(message: &'static str) -> Self {
        DiagnosticEvent { level: DiagnosticLevel::Error, message, fields: Vec::new() }
    }

    pub fn with(mut self, key: &'static str, value: impl Into<String>) -> Self {
        self.fields.push((key, value.into()));
        self
    }
}

/// The sentences the core says, verbatim from the reference (`tools/check-diagnostics.py` checks them).
pub mod diagnostics {
    /// `AgentActivity._user_turn_completed_task`, the reply-path variant of `.on_end_of_turn`'s.
    pub const SCHEDULING_PAUSED: &str = "skipping reply to user input, speech scheduling is paused";

    /// `generation._execute_tools_task` (error level): a call answering `tool_choice="none"` is
    /// dropped, so the budget's forced `"none"` ends the turn.
    pub const TOOL_CHOICE_NONE: &str =
        "received a tool call with tool_choice set to 'none', ignoring";

    /// `agent_activity.AgentActivity._pipeline_reply_task_impl`.
    pub const MAX_TOOL_STEPS: &str = "maximum number of function calls steps reached, generating final response with tool_choice='none'";
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, lk_agents_abi::Vocabulary)]
pub enum SpeechOutcome {
    Completed,
    /// Cut short: a barge-in, `interrupt()`, or the interruption watchdog.
    Interrupted,
    /// A provider failed it; the handle's `exception()` carries the error.
    Failed,
}

impl SpeechOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            SpeechOutcome::Completed => "completed",
            SpeechOutcome::Interrupted => "interrupted",
            SpeechOutcome::Failed => "failed",
        }
    }
}

/// One `session.run()` / `start(capture_run=True)`, numbered by the driver; tags what the run owns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RunId(pub u64);

/// Why a `Command::Run` opened nothing (`SessionEvent::RunRefused`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, lk_agents_abi::Vocabulary)]
pub enum RunRefusal {
    /// A run is open: `RuntimeError("nested runs are not supported")`.
    Nested,
    Closing,
}

impl RunRefusal {
    /// The `run_refused` event's `cause`; the host raises the reference's error for it.
    pub fn as_str(self) -> &'static str {
        match self {
            RunRefusal::Nested => "nested",
            RunRefusal::Closing => "closing",
        }
    }
}

/// How a run ended (`RunResult._mark_done`), read off the last owned speech.
#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum RunOutcome {
    /// Nothing the run owned is live. `final_output` is what `AgentTask.complete()` got, if anything.
    Completed { final_output: Option<lk_agents_abi::HostRef> },
    /// The run raises: the last speech failed, or a tool completed its task with an exception.
    Failed(RunFailure),
    /// The session closed under the run: `RuntimeError("session closed")`
    /// (`AgentSession._aclose_impl`).
    Closed,
}

impl RunOutcome {
    pub fn as_str(&self) -> &'static str {
        match self {
            RunOutcome::Completed { .. } => "completed",
            RunOutcome::Failed(_) => "failed",
            RunOutcome::Closed => "closed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum RunFailure {
    /// The last owned speech failed; its `exception()` is the error.
    Speech,
    /// `AgentTask.complete(exc)` from inside the last speech.
    Output(lk_agents_abi::HostRef),
}

/// One entry of a tool call's lifecycle (`ToolExecutionUpdatedEvent.update`).
#[derive(Debug, Clone, PartialEq)]
pub enum ToolLifecycle {
    Started { call: lk_agents_abi::FunctionCall },
    /// `ctx.update(message)`: `id` is the call's own first, then `{call_id}_update_N`.
    Updated { id: String, call_id: String, message: String },
    /// The call's single terminal entry (`{call_id}_final` once detached); `message` is `None`
    /// when there is nothing to voice.
    Ended { id: String, call_id: String, message: Option<String>, status: ToolEndStatus },
    /// The deferred reply covering `update_ids`: `Scheduled`, then one of the three ends.
    Reply {
        update_ids: Vec<String>,
        status: ToolReplyStatus,
        speech: crate::agent_activity::scheduling::SpeechId,
    },
}

impl ToolLifecycle {
    pub fn wire(&self) -> lk_agents_abi::voice::events::ToolUpdate {
        use lk_agents_abi::voice::events as w;
        match self {
            ToolLifecycle::Started { call } => {
                w::ToolUpdate::Started(w::ToolCallStarted { function_call: call.clone() })
            }
            ToolLifecycle::Updated { id, call_id, message } => {
                w::ToolUpdate::Updated(w::ToolCallUpdated {
                    id: id.clone(),
                    call_id: call_id.clone(),
                    message: message.clone(),
                })
            }
            ToolLifecycle::Ended { id, call_id, message, status } => {
                w::ToolUpdate::Ended(w::ToolCallEnded {
                    id: id.clone(),
                    call_id: call_id.clone(),
                    message: message.clone(),
                    status: status.as_str().to_owned(),
                })
            }
            ToolLifecycle::Reply { update_ids, status, speech } => {
                w::ToolUpdate::Reply(w::ToolReplyUpdated {
                    update_ids: update_ids.clone(),
                    status: status.as_str().to_owned(),
                    speech_id: crate::speech_handle::SpeechWireId(*speech).to_string(),
                })
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolEndStatus {
    Done,
    Error,
    Cancelled,
}

impl ToolEndStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ToolEndStatus::Done => "done",
            ToolEndStatus::Error => "error",
            ToolEndStatus::Cancelled => "cancelled",
        }
    }
}

/// `ToolReplyUpdated.status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolReplyStatus {
    Scheduled,
    Completed,
    Interrupted,
    /// The model judged the results already covered and said nothing.
    Skipped,
}

impl ToolReplyStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ToolReplyStatus::Scheduled => "scheduled",
            ToolReplyStatus::Completed => "completed",
            ToolReplyStatus::Interrupted => "interrupted",
            ToolReplyStatus::Skipped => "skipped",
        }
    }
}

/// Machines hold no wall clock; the driver stamps.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[derive(lk_agents_abi::Vocabulary)]
pub enum SessionEvent {
    Diagnostic(DiagnosticEvent),
    EotPrediction(EotPredictionEvent),
    AgentBackchannelOpportunity(AgentBackchannelOpportunityEvent),
    UserTranscript(Box<UserTranscriptEvent>),
    StartOfSpeech(StartOfSpeechEvent),
    EndOfSpeech(EndOfSpeechEvent),
    PreemptiveGeneration(PreemptiveGenerationEvent),
    UserTurnExceeded(UserTurnExceededEvent),
    TranscriptionTimeout(TranscriptionTimeoutEvent),
    /// `on_backchannel_confirmed`: the driver clears the realtime model's buffered audio.
    BackchannelConfirmed,
    /// `on_interruption`: the overlap was a real interruption.
    OverlapInterruption(OverlapInterruptionEvent),
    /// `AgentFalseInterruptionEvent`; `resumed` is `false` when resolved without resuming.
    AgentFalseInterruption {
        resumed: bool,
    },
    /// The user's turn ended, emitted before the commit verdict (`on_end_of_turn`) so AMD's release
    /// gate hears it. It cannot veto the reply.
    UserTurnCompleted(UserTurnCompletedEvent),
    /// A speech was created, for every kind: the host's `speech_created` and
    /// `session.current_speech` for speeches it did not start itself.
    SpeechCreated {
        speech: Arc<crate::speech_handle::SpeechHandle>,
        kind: crate::agent_activity::scheduling::SpeechKind,
        /// The host asked for this one (`say()`, `generate_reply()`).
        user_initiated: bool,
    },
    /// `_update_agent_state(next)`: decided here, applied by the host's
    /// `AgentSession._update_agent_state`. `old` is always `Some`.
    AgentStateChanged {
        old: AgentState,
        new: AgentState,
    },
    /// Open the `agent_speaking` span, right before its `AgentStateChanged`.
    SpanStarted(lk_agents_abi::voice::session_effects::SpanStarted),
    /// Close it: the edge out of `speaking`.
    SpanEnded(lk_agents_abi::voice::session_effects::SpanEnded),
    /// `AgentSession.user_state` moved (the driver is the one writer). `at` is the edge's own
    /// back-dated instant, or the away timer's deadline.
    UserStateChanged {
        old: UserState,
        new: UserState,
        at: Ts,
    },
    /// `function_tools_executed`: one batch of tool calls, with their outputs.
    ToolsExecuted(Box<crate::tool_executor::ToolsExecutedEvent>),
    /// `tool_execution_updated`: one call's start, updates, terminal entry and deferred reply.
    ToolExecutionUpdated(ToolLifecycle),
    /// A speech ended, for every kind and cause, after every `ItemAdded` it produced; the handle is
    /// already `done()`.
    SpeechDone {
        speech: crate::agent_activity::scheduling::SpeechId,
        outcome: SpeechOutcome,
    },
    /// The floor changed hands (`None`: free); once per cycle, after the cycle's other effects.
    FloorChanged {
        holder: Option<crate::agent_activity::scheduling::SpeechId>,
    },
    /// The AMD machine released its verdict, once per machine.
    AmdResult {
        verdict: lk_agents_abi::voice::amd::AmdPrediction,
    },
    /// One provider metrics record (`metrics_collected`), computed by the driver or handed in by the
    /// host (`Command::HostMetrics`). `speech` is the speech it belongs to.
    MetricsCollected {
        metrics: Box<lk_agents_abi::metrics::base::AgentMetrics>,
        speech: Option<crate::agent_activity::scheduling::SpeechId>,
    },
    /// A provider failed (`AgentActivity._on_error`), one per failed attempt, before the close policy.
    Error(Box<ErrorEvent>),
    /// The session closed: the last event. `error` is set for a `CloseReason::Error` close.
    Close {
        reason: lk_agents_abi::voice::events::CloseReason,
        error: Option<lk_agents_abi::errors::ProviderError>,
    },
    ItemAdded {
        item: Box<lk_agents_abi::ChatItem>,
        /// Which agent produced it, as the host's opaque id. Stamped here because only the loop
        /// orders a handoff turn's items before the swap.
        agent: String,
        source: crate::agent::Source,
        /// Which of the host's two projections take it; the host decides nothing by item kind.
        visibility: crate::agent::Visibility,
        run: Option<RunId>,
    },
    /// A resumed frame took over the replaced frame's conversation (the reference's merge at an
    /// inline task's return). `items` already announced under `from`; no new `ItemAdded`.
    ContextCarried {
        from: String,
        to: String,
        items: Vec<String>,
    },
    /// `Command::Run` opened a run; everything asked for until `RunSettled` is the run's.
    RunStarted {
        run: RunId,
    },
    /// A tool of a speech the open run owns started: its call is the run's from now, not when the
    /// call is committed (`speech_handle._item_added([fnc_call])`, agent_activity.py:3416).
    RunItemAdded {
        run: RunId,
        item: Box<lk_agents_abi::ChatItem>,
    },
    /// `Command::Run` opened nothing; the host raises the reference's error for the cause.
    RunRefused {
        cause: RunRefusal,
    },
    /// The run is over. `speech` is the last owned speech (`__last_speech_handle`).
    RunSettled {
        run: RunId,
        outcome: RunOutcome,
        speech: Option<Arc<crate::speech_handle::SpeechHandle>>,
    },
}
