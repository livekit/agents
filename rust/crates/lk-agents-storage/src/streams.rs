//! The answer streams' records: the A2A event frames a turn sent are kept as JSON records in a
//! stream (`AST_..`), read back from an offset. The runtime's streams client moves them.

use std::time::Duration;

pub const DEFAULT_TTL: Duration = Duration::from_secs(86_400);
pub const CONTENT_TYPE: &str = "application/json";
/// The beginning, as the server spells it.
pub const START: &str = "-1";

pub use lk_agents_abi::storage::StreamInfo;
