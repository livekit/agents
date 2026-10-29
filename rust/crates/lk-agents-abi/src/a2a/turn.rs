//! A managed turn's typed host boundary. Protocol values cross as protobuf JSON; storage rows
//! cross as the same records returned by the thread's read APIs.

use serde_json::{Map, Value};

use crate::errors::ProviderError;
use crate::storage::context::{AppStatement, ContextScope, ContextScopeWrite};
use crate::storage::{ItemRow, RunUsage, TaskRecord, TaskState};
use crate::{StreamSink, StreamSource};

pub use crate::a2a::client::Message;

#[crate::host(record(module = "storage_turn", emit))]
#[derive(Debug, Clone)]
pub struct TurnContext {
    pub thread_id: String,
    pub lane: String,
    pub message: Map<String, Value>,
    pub batch: Vec<Message>,
    pub history: Vec<ItemRow>,
    pub state: Option<Value>,
    pub context_scope_id: String,
    pub context_scopes: Vec<ContextScope>,
    pub read_only: bool,
    pub task_id: String,
    pub tasks: Vec<TaskRecord>,
    pub caller_history: Option<Map<String, Value>>,
    pub answer_tool: Option<String>,
    #[host(skip)]
    pub host_context: Option<crate::HostRef>,
}

/// A conversation tool's pending call, the one an `InputRequired` came from.
#[crate::host(record(module = "storage_turn", extract))]
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PendingCall {
    pub name: String,
    pub call_id: String,
    /// The tool that takes the answer, when the `InputRequired` named one.
    #[host(default = None)]
    pub answer_tool: Option<String>,
}

/// A journal event the handler recorded (docs/internals/storage.md#journal):
/// `tool_execution_updated` for a call's start, its progress and its end.
#[crate::host(record(module = "storage_turn", extract))]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    pub scope_writes: Vec<ContextScopeWrite>,
    /// What the handler's managed transactions staged, applied in the turn's commit.
    pub app_writes: Vec<AppStatement>,
    pub usage: RunUsage,
    pub input_required: Option<String>,
    pub pending_call: Option<PendingCall>,
    pub interrupted: bool,
    pub metadata: Map<String, Value>,
    pub error: Option<String>,
    pub events: Vec<HostEvent>,
    /// A `Rejected` or `Working` transition's state (docs/server.md#handler-outcomes); `output`
    /// is then its reason or status message.
    #[host(default = None)]
    pub status: Option<TaskState>,
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
    #[host(name = "Progress")]
    Progress { task_id: String, text: String },
    #[host(name = "Token")]
    Token { task_id: String, delta: String },
}

#[crate::host(proxy(host = "livekit.agents.storage.context._TurnHost"))]
pub trait TurnHost: Send + Sync {
    #[host(await, name = "_core_handle")]
    fn handle<'a>(
        &'a self,
        #[host(context)] ctx: &'a TurnContext,
    ) -> crate::BoxFuture<
        'a,
        Result<(Box<dyn StreamSink<Verdict>>, Box<dyn StreamSource<TurnEvent>>), ProviderError>,
    >;
}

impl crate::agent::HostContext for TurnContext {
    fn host_context(&self) -> Option<&crate::HostRef> {
        self.host_context.as_ref()
    }
}
