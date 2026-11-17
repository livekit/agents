//! Persistent remote communication is separate from local agent-call execution.

pub mod remote;

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
    /// Separate model for recognizing responses to presented input requests.
    pub routing_model: std::sync::Arc<dyn crate::Llm>,
}
