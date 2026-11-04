//! The voice session's span entities (`telemetry.md`, "Sans-I/O: the entity owns its span";
//! `telemetry-reference.md`, "Span trees"). Each entity a span describes owns it and the driver's
//! [`Trace`] (`self.telemetry`) opens and ends it: the user turns ([`user_turn`]), the agent turns
//! ([`agent_turn`]), and the session's own spans and its callbacks' ([`session`]). Ended spans
//! leave each step as records on `SessionEvent::Telemetry`; work that continues in the runtime
//! gets its parent's `SpanRef`.

pub(crate) mod agent_turn;
pub(crate) mod session;
pub(crate) mod user_turn;

use lk_agents_abi::telemetry::decl::Content;
use std::borrow::Cow;

use lk_agents_abi::telemetry::errors::ErrorType;
pub(crate) use lk_agents_abi::telemetry::errors::SpanError;
pub(crate) use lk_agents_abi::telemetry::spans::common::log_keys;
use lk_agents_abi::telemetry::record::JobKey;
use lk_agents_abi::telemetry::spans::common::ParticipantAttrs;
use lk_agents_abi::voice::room_io::ParticipantInfo;
use lk_agents_abi::voice::rtc::DisconnectReason;
use lk_agents_core::telemetry::span::IdSource;

pub(crate) use agent_turn::{AgentTurnSpan, Reply};
pub(crate) use lk_agents_abi::telemetry::spans::user_turn::EouSource;
pub(crate) use lk_agents_core::telemetry::span::{Reserved, Span, Trace, seeded_ids};
pub(crate) use session::{AmdDetection, Arriving, Blocking, Holding, HookSpan, ItemTurns, Under};
pub(crate) use user_turn::{Endpointing, Stt, UserTurnSpan};

// The declarations and records logic code names, under `telemetry::` (`telemetry.md`, rule 4).
pub(crate) use lk_agents_abi::telemetry::record::SpanRef;
pub(crate) use lk_agents_abi::telemetry::report::SessionReport;
pub(crate) use lk_agents_abi::telemetry::spans::session::{
    AgentSession, KeytermDetection, KeytermDetectionEnd, SessionClose, SessionStart,
    StartAgentActivity, UpdateAgent,
};
pub(crate) use lk_agents_abi::telemetry::spans::user_turn::{UserTurnOrigin, UserTurnOutcome};

pub fn participant_attrs(p: &ParticipantInfo) -> ParticipantAttrs {
    ParticipantAttrs {
        id: Some(p.sid.clone()),
        identity: Content(p.identity.as_str().to_owned()),
        kind: Some(p.kind.into()),
    }
}

/// `rtc.DisconnectReason.Name(..)`.
pub(crate) fn disconnect_reason(reason: DisconnectReason) -> String {
    match reason {
        DisconnectReason::Unknown => "UNKNOWN_REASON".to_owned(),
        reason => reason.as_str().to_ascii_uppercase(),
    }
}

/// What the runtime supplies so the session's spans join the job's trace.
#[derive(Default)]
pub struct SessionTelemetry {
    /// The span the session continues: the job's `job_entrypoint`, or the caller's current span.
    pub parent: Option<SpanRef>,
    pub job: Option<JobKey>,
    /// `gen_ai.conversation.id`: the call's thread, else the job's room session id; `None` (outside
    /// a job) records the session's own id.
    pub conversation: Option<String>,
    /// The job's ids, which `agent_session` repeats from `job_entrypoint`.
    pub job_ids: JobIds,
    /// Span and trace ids; `None` seeds them from the session id (tests, the DST).
    pub ids: Option<IdSource>,
    /// The spans `start()` reserved for its startup work; `None` draws them at the opening.
    pub opening: Option<Opening>,
}

/// A job's dispatch, worker and agent ids, as `job_entrypoint` and `agent_session` carry them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JobIds {
    pub dispatch_id: Option<String>,
    pub worker_id: Option<String>,
    pub agent_id: Option<String>,
}

/// The session's spans reserved at `start()`, before the driver exists: the room's connection
/// and the first agent's toolsets run under them, and the driver opens them as its own.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Opening {
    pub session: Reserved<AgentSession>,
    pub start: Reserved<SessionStart>,
    /// The first agent's activity, reserved when its toolsets were set up.
    pub activity: Option<Reserved<StartAgentActivity>>,
}

/// A handoff's spans, reserved when its toolsets were set up: the `update_agent` it opens, unless
/// one is open, and the arriving agent's activity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Activation {
    pub update: Reserved<UpdateAgent>,
    pub activity: Reserved<StartAgentActivity>,
}

/// How a hook failed, as its span records it.
pub(crate) enum HookFailure<'a> {
    Raised(&'a lk_agents_abi::errors::HostError),
    TimedOut,
    /// No exception: its agent was gone, its call undeliverable.
    Failed,
}

impl SpanError for HookFailure<'_> {
    fn error_type(&self) -> ErrorType {
        match self {
            HookFailure::Raised(error) => error.error_type(),
            HookFailure::TimedOut => ErrorType::Timeout,
            HookFailure::Failed => ErrorType::Unanswered,
        }
    }
    fn message(&self) -> String {
        match self {
            HookFailure::Raised(error) => error.message(),
            HookFailure::TimedOut => "the callback did not return within its bound".to_owned(),
            HookFailure::Failed => "the callback could not run".to_owned(),
        }
    }
    fn exception_type(&self) -> Cow<'static, str> {
        match self {
            HookFailure::Raised(error) => error.exception_type(),
            _ => self.error_type().as_str().into(),
        }
    }
    fn stacktrace(&self) -> Option<String> {
        match self {
            HookFailure::Raised(error) => error.stacktrace(),
            _ => None,
        }
    }
}

/// The default id seed: stable per session id.
pub(crate) fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
}
