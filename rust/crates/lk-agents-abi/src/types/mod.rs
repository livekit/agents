//! `types.py`: the provider-facing vocabulary every provider shares (connect options, identity,
//! attempt policy).

pub mod connect;
pub mod identity;
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
