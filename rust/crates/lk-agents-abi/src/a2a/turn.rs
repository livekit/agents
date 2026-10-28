//! A managed turn's typed host boundary. Protocol values cross as protobuf JSON; storage rows
//! cross as the same records returned by the thread's read APIs.

use serde_json::{Map, Value};

use crate::errors::ProviderError;
use crate::storage::{ItemRow, RunUsage, TaskRecord};
use crate::{StreamSink, StreamSource};

#[crate::host(record(module = "storage_turn", emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub message_id: String,
    pub parts: Vec<Value>,
    pub metadata: Map<String, Value>,
    pub channel: String,
    pub received_at: f64,
    pub task_id: Option<String>,
    pub context_id: Option<String>,
    pub raw: Value,
}

#[crate::host(record(module = "storage_turn", emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct TurnContext {
    pub thread_id: String,
    pub lane: String,
    pub message: Value,
    pub batch: Vec<Message>,
    pub history: Vec<ItemRow>,
    pub state: Option<Value>,
    pub read_only: bool,
    pub task_id: String,
    pub tasks: Vec<TaskRecord>,
    pub caller_history: Option<Value>,
    pub answer_tool: Option<String>,
}

#[crate::host(record(module = "storage_turn", extract))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingCall {
    pub name: String,
    pub call_id: String,
    #[host(default = None)]
    pub answer_tool: Option<String>,
}

#[crate::host(record(module = "storage_turn", extract))]
#[derive(Debug, Clone, PartialEq)]
pub struct HostEvent {
    pub r#type: String,
    pub payload: Value,
}

#[crate::host(record(module = "storage_turn", extract))]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HandlerOutput {
    pub output: Option<Value>,
    pub new_items: Vec<Value>,
    pub state: Option<Value>,
    pub usage: RunUsage,
    pub input_required: Option<String>,
    pub pending_call: Option<PendingCall>,
    pub interrupted: bool,
    pub metadata: Map<String, Value>,
    pub error: Option<String>,
    pub events: Vec<HostEvent>,
}

#[crate::host(record(module = "storage_turn", emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct TurnResult {
    pub task: TaskRecord,
    pub output: Option<Value>,
    pub thread_id: String,
    pub answered_by: String,
    pub run_id: Option<String>,
}

#[crate::host(record(module = "storage_turn", emit, extract))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    #[host(rename = "continue")]
    Continue,
    #[host(rename = "yield")]
    Yield,
}

#[crate::host(record(module = "storage_turn", extract))]
#[derive(Debug, Clone, PartialEq)]
pub enum TurnEvent {
    #[host(name = "Checkpoint")]
    Checkpoint,
    #[host(name = "Output")]
    Output { output: HandlerOutput },
}

#[crate::host(proxy(host = "livekit.agents.storage.context.TurnHost"))]
#[async_trait::async_trait]
pub trait TurnHost: Send + Sync {
    #[host(await, name = "_core_handle")]
    async fn handle(
        &self,
        ctx: TurnContext,
    ) -> Result<(Box<dyn StreamSink<Verdict>>, Box<dyn StreamSource<TurnEvent>>), ProviderError>;
}
