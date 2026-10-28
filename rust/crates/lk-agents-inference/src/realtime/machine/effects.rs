//! Realtime operations decided by the core and executed by the runtime writer.

use super::{OptionsPatch, SessionOptions};
use crate::realtime::dialect::Dialect;
use lk_agents_abi::{AudioFrame, ChatItem, ToolChoice, ToolSpec};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum Outgoing {
    SessionUpdate { event_id: String, opts: SessionOptions, dialect: Dialect },
    InstructionsUpdate { event_id: String, instructions: String, dialect: Dialect },
    OptionsUpdate { event_id: String, patch: OptionsPatch, dialect: Dialect },
    ToolsUpdate { event_id: String, tools: Arc<Vec<ToolSpec>>, model: String, dialect: Dialect },
    ItemCreate { event_id: String, item: ChatItem, previous_item_id: Option<String> },
    ItemDelete { event_id: String, item_id: String },
    ItemTruncate { item_id: String, audio_end_ms: u64 },
    AudioAppend { frame: AudioFrame },
    AudioCommit,
    AudioClear,
    ResponseCreate { req_id: String, instructions: Option<String>, tool_choice: Option<ToolChoice> },
    ResponseCancel { response_id: Option<String> },
    ForceMessage { req_id: String, text: String },
}
