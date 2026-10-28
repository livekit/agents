//! Owned snapshots crossing the application load/save boundary. History uses the shared
//! ChatItem codec at that boundary; routing metadata contains no callbacks, tasks or host handles.

use std::collections::{BTreeMap, BTreeSet};

use lk_agents_abi::{ChatItem, ChatMessage};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParentCall {
    pub frame_id: String,
    pub call_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChildCallStatus {
    Approval,
    Ready,
    Running { frame_id: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct PendingChildCall {
    pub parent: ParentCall,
    pub invocation_id: String,
    pub execution_id: String,
    pub tool_name: String,
    pub agent_name: String,
    pub validated_arguments: Value,
    /// The shared approval gate's canonical validated identity, never the raw model arguments.
    pub identity: String,
    pub status: ChildCallStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SessionFrame {
    pub id: String,
    pub agent_name: String,
    pub history: Vec<ChatItem>,
    pub parent: Option<ParentCall>,
    pub context_scope: String,
    pub completed: bool,
    pub pending_calls: Vec<PendingChildCall>,
    pub answer_tool: Option<String>,
    /// Delivery receipts, in conversation order, excluding the initial tool task and fork.
    pub visible_dialogue: Vec<ChatMessage>,
    pub starting_history_ids: BTreeSet<String>,
    pub transferred_message_ids: BTreeSet<String>,
    /// Frozen at creation: reconfiguration on a later request cannot change the return policy.
    pub merge_chat_history: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextSessionState {
    pub session_id: String,
    pub execution_id: String,
    pub root_frame_id: String,
    pub active_frame_id: String,
    /// Root first, immediate parent last. Completed frames never occur here.
    pub ancestors: Vec<String>,
    pub frames: BTreeMap<String, SessionFrame>,
    pub next_id: u64,
}

impl TextSessionState {
    pub(super) fn mint(&mut self, kind: &str) -> String {
        self.next_id += 1;
        format!("{}/{kind}_{}", self.session_id, self.next_id)
    }
}
