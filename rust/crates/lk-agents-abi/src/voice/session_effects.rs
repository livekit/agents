//! What `start()` decides. The decision is in `lk-agents-voice/src/agent_session/`.

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

/// What a filler's callable answered: nothing, its text to say, or anything else (a speech it
/// scheduled itself).
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(extract, untagged)]
pub enum FillerReply {
    #[binding(none)]
    Declined,
    Text(String),
    #[binding(rest)]
    Spoke,
}
