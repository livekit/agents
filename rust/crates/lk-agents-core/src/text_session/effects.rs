//! Work produced by [`super::machine::TextSession`], executed by the runtime text-session driver.
//! The `Run` variant contains canonical [`crate::effects::AgentRunEffect`] values.

use super::{FrameId, RoutingError, TextSessionError, TextSessionTurn};
use crate::agent_run;
use lk_agents_abi::ChatItem;
use lk_agents_abi::agent::run::{RunResult, StreamEvent};

#[derive(Debug, Clone, PartialEq)]
pub enum TextSessionEffect {
    /// The runtime binds this invocation to the supplied context scope and execution identity.
    InvocationStarted {
        frame_id: FrameId,
        agent_name: String,
        invocation_id: String,
        context_scope: String,
        /// The invocation reads its scope's application context; a separate child's own scope
        /// starts without one.
        reads_context: bool,
        execution_id: String,
        history: Vec<ChatItem>,
    },
    /// AgentRun has settled. Runtime finishes any selected delivery and reports its receipts
    /// before acknowledging release. Until then a child cannot merge or resume its parent.
    InvocationSettled {
        frame_id: FrameId,
        invocation_id: String,
        result: RunResult,
    },
    /// No host work remains in this parent invocation; its continuation is now routing data.
    InvocationParked {
        frame_id: FrameId,
        invocation_id: String,
    },
    Run {
        invocation_id: String,
        effect: agent_run::AgentRunEffect,
    },
    Emit {
        sequence: u64,
        frame_id: FrameId,
        agent_name: String,
        invocation_id: String,
        event: StreamEvent,
    },
    /// The call receives an error output and the parent may respond or choose another tool.
    RoutingRefused {
        frame_id: FrameId,
        call_id: String,
        error: RoutingError,
    },
    Finished {
        result: Result<TextSessionTurn, TextSessionError>,
        input_requirements: Vec<lk_agents_abi::agent::run::RunInputRequirement>,
    },
}
