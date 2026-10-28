//! The answer streams' records: the A2A event frames a turn sent are kept as JSON records in a
//! stream (`AST_..`), read back from an offset. The runtime's streams client moves them.

use std::time::Duration;

use lk_agents_host_derive::Host;

pub const DEFAULT_TTL: Duration = Duration::from_secs(86_400);
pub const CONTENT_TYPE: &str = "application/json";
/// The beginning, as the server spells it.
pub const START: &str = "-1";

/// A created stream: its id (also the task id it answers) and when it expires.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Host)]
#[python(pydantic, module = "storage", config = "frozen=True")]
pub struct StreamInfo {
    pub stream_id: String,
    pub expires_at: i64,
}
