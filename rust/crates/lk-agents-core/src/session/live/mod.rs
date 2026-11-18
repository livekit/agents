//! Transient ownership of live turns and identified host bodies. Saved state never recreates
//! these resources; a replacement owner reconciles uncertain work through its normal journal.

mod machine;
#[cfg(test)]
#[path = "live_tests.rs"]
mod tests;

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::agent_run::CallId;
use lk_agents_abi::agent::ArgumentIdentity;
use lk_agents_abi::llm::tool_executor::{ToolOutcome, ToolReport};
use lk_agents_abi::session::InputCandidate;
use lk_agents_abi::text_session::{FrameId, LiveTurnOrigin};
use lk_agents_abi::{ChatMessage, FunctionCall};

const MAX_QUEUED_TURNS: usize = 32;
const MAX_LIVE_BODIES: usize = 64;

pub(super) type BodyKey = (String, CallId);

pub(super) struct FrozenHuman {
    pub message: ChatMessage,
    pub candidates: Vec<InputCandidate>,
    pub offered: Vec<InputCandidate>,
    pub history: Vec<ChatMessage>,
}

pub(super) enum TurnInput {
    Initial,
    Cancelled,
    Human(Box<FrozenHuman>),
    Delivery(Vec<BodyKey>),
}

pub(super) struct Turn {
    pub id: u64,
    pub origin: LiveTurnOrigin,
    pub input: TurnInput,
}

enum Foreground {
    Running(Turn),
    Committing(Turn),
    Idle,
    Closed,
}

enum BodyState {
    Running,
    Detached,
    Ready(Option<ToolReport>),
    Delivering(Option<ToolReport>),
}

pub(super) struct Delivery {
    pub invocation_id: String,
    pub frame_id: FrameId,
    pub call: FunctionCall,
    /// Task death is not a final body report and cannot clear provisional progress.
    pub report: Option<ToolReport>,
}

struct Body {
    execution: u64,
    frame_id: FrameId,
    origin: LiveTurnOrigin,
    call: FunctionCall,
    identity: ArgumentIdentity,
    state: BodyState,
    cancellation_requested: bool,
}

pub(super) struct Observation {
    pub source: String,
    pub frame: FrameId,
    pub messages: Vec<ChatMessage>,
}

pub(super) struct Live {
    foreground: Foreground,
    queue: VecDeque<Turn>,
    bodies: BTreeMap<BodyKey, Body>,
    identities: BTreeMap<BodyKey, ArgumentIdentity>,
    next_id: u64,
    pub observation: Option<Observation>,
    cancelled: BTreeSet<String>,
}
