pub mod attributes;
pub mod connect;
pub mod headers;
pub mod identity;
pub mod ids;
pub mod provider;
pub mod retry;

/// Ends the current spoken segment in an LLM node's output.
#[crate::host(record(module = "types", path = "livekit.agents.types", marker, emit))]
#[derive(Debug, Clone, Copy)]
pub struct FlushSentinel {}

/// Marks an omitted argument, independently of an explicit null value.
#[crate::host(record(
    module = "types",
    path = "livekit.agents.types",
    missing = "NOT_GIVEN",
    emit
))]
#[derive(Debug, Clone, Copy)]
pub struct NotGiven {}
