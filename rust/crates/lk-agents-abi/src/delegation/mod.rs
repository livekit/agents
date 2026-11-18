//! Persistent remote communication is separate from local agent-call execution.

pub mod remote;
pub mod routing;

/// Conversation-wide delegation behavior, shared by every remote tool on an execution owner.
#[crate::host(record(
    module = "delegation",
    path = "livekit.agents",
    dataclass,
    frozen,
    kw_only,
    slots,
    extract
))]
#[derive(Clone)]
pub struct DelegationOptions {
    /// Separate routing model; None disables automatic routing and requires explicit task selection.
    #[host(default = None)]
    pub routing_model: Option<std::sync::Arc<dyn crate::Llm>>,
}
