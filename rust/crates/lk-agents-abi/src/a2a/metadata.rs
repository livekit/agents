//! LiveKit's wire metadata vocabulary. Short `livekit/` keys on the wire remain application data.

#[crate::host(file(module = "a2a_metadata"))]
const _: () = ();

#[crate::host(constant)]
pub const EXTENSION_URI: &str = "https://livekit.io/a2a/ext/agents/v1";
#[crate::host(constant)]
pub const EXTENSIONS_HEADER: &str = "A2A-Extensions";
/// The request header that carries a thread grant beside `Authorization`.
#[crate::host(constant)]
pub const GRANT_HEADER: &str = "X-LiveKit-Thread-Grant";
/// Where an endpoint publishes its agent card, under `/<endpoint>`.
#[crate::host(constant)]
pub const AGENT_CARD_PATH: &str = "/.well-known/agent-card.json";
/// Where an endpoint's JSON-RPC methods live, under `/<endpoint>`; the card's URL points here.
#[crate::host(constant)]
pub const RPC_PATH: &str = "/v1";
#[crate::host(constant(name = "CHANNEL"))]
pub const CHANNEL_KEY: &str = "https://livekit.io/a2a/ext/agents/v1/channel";
#[crate::host(constant(name = "KIND"))]
pub const KIND_KEY: &str = "https://livekit.io/a2a/ext/agents/v1/kind";
#[crate::host(constant(name = "INTERRUPTED"))]
pub const INTERRUPTED_KEY: &str = "https://livekit.io/a2a/ext/agents/v1/interrupted";
#[crate::host(constant(name = "INTERRUPTION_REASON"))]
pub const INTERRUPTION_REASON_KEY: &str = "https://livekit.io/a2a/ext/agents/v1/interruptionReason";
#[crate::host(constant(name = "MERGED_INTO"))]
pub const MERGED_INTO_KEY: &str = "https://livekit.io/a2a/ext/agents/v1/mergedInto";
#[crate::host(constant(name = "REASON"))]
pub const REASON_KEY: &str = "https://livekit.io/a2a/ext/agents/v1/reason";
#[crate::host(constant(name = "VERBATIM"))]
pub const VERBATIM_KEY: &str = "https://livekit.io/a2a/ext/agents/v1/verbatim";
#[crate::host(constant)]
pub const KIND_DELEGATION: &str = "delegation";
#[crate::host(constant)]
pub const KIND_CHAT_CTX: &str = "chat_ctx";
#[crate::host(constant)]
pub const ANSWER_ARTIFACT_NAME: &str = "answer";
