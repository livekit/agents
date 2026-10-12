//! What `start()` decides and the trace-span edges the session state emits. The decision is in
//! `lk-agents-voice/src/agent_session/`.

use crate::Ts;
use crate::voice::session_options::RecordingOptions;
use lk_agents_host_derive::Host;

/// The decision `start()` gets back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Host)]
#[binding(emit)]
pub struct StartPlan {
    pub recording: RecordingOptions,
    pub is_primary: bool,
    /// `is_primary and (session_host if given else True)`: a second session must never host.
    pub hosting: bool,
    /// `records_anything` on the resolved options.
    pub records: bool,
}

/// Which span the host should open or close.
// An edge effect, not a state: a repeated transition cannot open two. No host target.
// The wire name: the OTel span the host opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Host)]
#[binding(fields_only)]
#[host(wire(msg = format!("unknown span {s:?}")))]
pub enum Span {
    #[host(rename = "agent_speaking")]
    AgentSpeaking,
    #[host(rename = "user_speaking")]
    UserSpeaking,
}

/// `start_time` is `None` for "now".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpanStarted {
    pub span: Span,
    pub start_time: Option<Ts>,
}

/// `end_time` is `None` for "now".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpanEnded {
    pub span: Span,
    pub end_time: Option<Ts>,
}
