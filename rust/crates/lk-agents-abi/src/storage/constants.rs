#[crate::host(file(module = "storage_constants"))]
const _: () = ();

/// The protocol's partition key when an application omits one.
/// The framework resolves this default before issuing storage operations.
#[crate::host(constant(native, path = "livekit.agents.storage"))]
pub const DEFAULT_PARTITION_KEY: &str = livekit_protocol::agent::DEFAULT_PARTITION_KEY;
