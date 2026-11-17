//! Private shared conversation data. Host applications continue it through store binding.

use crate::{ChatItem, ChatMessage};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

/// A stable backend conversation frame, distinct from a call or invocation.
#[crate::host(record(module = "session_snapshot", emit, extract))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct FrameId {
    pub value: String,
}

#[crate::host(record(module = "session_snapshot", emit, extract))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParentCall {
    pub frame_id: FrameId,
    pub call_id: String,
}

#[crate::host(record(module = "session_snapshot", emit, extract))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", serde(tag = "status", rename_all = "snake_case"))]
pub enum ChildCallStatus {
    Approval,
    Ready,
    Running { frame_id: FrameId },
}

#[crate::host(record(module = "session_snapshot", emit, extract))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

/// A child's fixed policy. Reconfiguring its tool after a restart cannot change this choice.
#[crate::host(record(module = "session_snapshot", emit, extract))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FrameOptions {
    pub merge_chat_ctx: bool,
}

#[crate::host(record(module = "session_snapshot", emit, extract))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct SessionFrame {
    pub id: FrameId,
    pub agent_name: String,
    pub history: Vec<ChatItem>,
    pub parent: Option<ParentCall>,
    pub completed: bool,
    #[host(default = None)]
    pub output: Option<Value>,
    #[host(default = None)]
    pub error: Option<String>,
    #[host(default_factory = "dict")]
    pub input_requests: BTreeMap<String, crate::session::AgentInputRequest>,
    pub pending_calls: Vec<PendingChildCall>,
    pub answer_tool: Option<String>,
    /// Delivery receipts, in conversation order, excluding the initial tool task and fork.
    pub visible_dialogue: Vec<ChatMessage>,
    pub starting_history_ids: BTreeSet<String>,
    pub options: FrameOptions,
}

#[crate::host(record(module = "session_snapshot", name = "_SessionSnapshot", emit, extract))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct SessionSnapshot {
    pub session_id: String,
    pub execution_id: String,
    pub root_frame_id: FrameId,
    pub active_frame_id: FrameId,
    /// Root first, immediate parent last. Completed frames never occur here.
    pub ancestors: Vec<FrameId>,
    pub frames: BTreeMap<FrameId, SessionFrame>,
    pub next_id: u64,
    /// Accepted human input and delivered assistant content across every local agent.
    #[host(default_factory = "list")]
    pub chat_history: Vec<ChatMessage>,
    /// Remote communication spans local agent changes and execution modes.
    #[host(default_factory = "dict")]
    pub remote_agents: BTreeMap<String, crate::delegation::remote::RemoteAgentState>,
}
