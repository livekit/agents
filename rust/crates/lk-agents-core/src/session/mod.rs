//! Conversation coordination shared by execution modes; no I/O or host resources.

pub mod effects;
pub mod input_routing;
mod machine;
mod resume;
mod routing;

pub use effects::SessionEffect;
pub use resume::validate_resume;

use crate::agent_run::{self, AgentRun, CallId, Opening, OutputVerdict, RunInput};
use crate::approval;
use crate::usage::Ledger;
use lk_agents_abi::Role;
use lk_agents_abi::agent::ArgumentIdentity;
use lk_agents_abi::agent::run::{
    Output, RunError, RunOutcome, RunResult, StreamEvent, ValidationId,
};
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::usage::Usage;
use lk_agents_abi::{ChatItem, ChatMessage, FunctionCall, Ts, WallAnchor};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

pub use lk_agents_abi::session::InputRoute;
pub use lk_agents_abi::text_session::{
    ChildCallStatus, FrameId, ParentCall, PendingChildCall, SessionFrame, SessionSnapshot,
    TextSessionError, TextSessionTurn,
};
pub use routing::{ChatHistory, LocalAgentTool, RoutingError, SessionAgent, SessionConfig};

// Checkpoints describe accepted progress; running invocation resources are never serialized.

/// Runtime inputs: identities and saved state; no services nor an instruction to load/save them.
pub struct SessionInput {
    pub session_id: String,
    /// The random value this request's runs mint their chat items under: drawn afresh
    /// per request, so nothing about it is saved.
    pub id_seed: u64,
    pub execution_id: String,
    pub task_id: Option<String>,
    pub state: Option<SessionSnapshot>,
    pub wall: WallAnchor,
    pub cleanup_timeout: Duration,
    pub delegated: bool,
}

/// Ephemeral detector input, resolved from canonical content and actual presentation evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct InputCandidate {
    pub reference: lk_agents_abi::session::InputRequestRef,
    pub question: String,
    pub presentation: ChatMessage,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InputRoutingRequest {
    pub message: ChatMessage,
    pub history: Vec<ChatMessage>,
    pub candidates: Vec<InputCandidate>,
}

/// Every event belonging to an invocation carries its identity. Late receipts cannot mutate
/// a new invocation of the same frame, nor a new request reconstructed from the returned state.
#[derive(Clone)]
pub enum SessionEvent {
    PersistenceFailed(lk_agents_abi::storage::StorageError),
    Start(ChatMessage),
    InputRouted {
        message_id: String,
        /// Dropping the producer reports task death; it cannot leave admission waiting.
        result: Option<Result<InputRoute, lk_agents_abi::errors::ProviderError>>,
        usage: Usage,
    },
    InputRoutingUnavailable,
    Run {
        invocation_id: String,
        event: agent_run::AgentRunEvent,
    },
    /// The host's accepted value serialized *after* validation/replacement. The runtime also
    /// retains the original typed host value for the public result.
    OutputValidated {
        invocation_id: String,
        validation: ValidationId,
        verdict: OutputVerdict,
        output: Option<Value>,
    },
    DialogueDelivered {
        invocation_id: String,
        message: ChatMessage,
    },
    /// All delivery receipts and invocation cleanup have been reported. A non-presenting
    /// consumer acknowledges immediately; opening a stream does not imply delivery. `delivered`:
    /// the text of each of the invocation's dialogue messages that reached the person, by id.
    InvocationReleased {
        invocation_id: String,
        delivered: BTreeMap<String, String>,
    },
    /// Return from a direct execution of the active frame, preserving its original parent call.
    /// As with AgentRun, the result must already have passed the agent's output validators.
    Reconcile {
        frame_id: FrameId,
        result: RunResult,
        output: Option<Value>,
    },
    Cancel,
    Wake,
}

struct PreparedChild {
    function_call: FunctionCall,
    arguments: Value,
    identity: String,
}

enum Child {
    Preparing(FunctionCall),
    Prepared(PreparedChild),
    Executing(PreparedChild),
}

enum InvocationRelease {
    Settled(Box<RunResult>, Option<Value>, Vec<PendingChildCall>),
    Parked,
}

struct Invocation {
    id: String,
    frame_id: FrameId,
    run: AgentRun,
    children: BTreeMap<CallId, Child>,
    outputs: BTreeMap<ValidationId, Value>,
}

enum Phase {
    NotStarted,
    Routing { message: ChatMessage, candidates: Vec<InputCandidate> },
    Running(Invocation),
    Releasing(Invocation, InvocationRelease),
    Done,
}

impl Phase {
    fn invocation(&self) -> Option<&Invocation> {
        match self {
            Self::Running(invocation) | Self::Releasing(invocation, _) => Some(invocation),
            Self::NotStarted | Self::Routing { .. } | Self::Done => None,
        }
    }

    fn invocation_mut(&mut self) -> Option<&mut Invocation> {
        match self {
            Self::Running(invocation) | Self::Releasing(invocation, _) => Some(invocation),
            Self::NotStarted | Self::Routing { .. } | Self::Done => None,
        }
    }
}

pub struct Session {
    failure_input_requirements: Vec<lk_agents_abi::agent::run::RunInputRequirement>,
    config: SessionConfig,
    state: SessionSnapshot,
    id_seed: u64,
    wall: WallAnchor,
    cleanup_timeout: Duration,
    delegated: bool,
    checkpoints: bool,
    persistence_failure: Option<lk_agents_abi::storage::StorageError>,
    closing: bool,
    recording: bool,
    checkpoint_due: Option<Ts>,
    checkpoint_bytes: usize,
    phase: Phase,
    /// The turn's ended invocations: the live one is added to it when read.
    usage: Ledger,
    sequence: u64,
}
