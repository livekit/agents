//! A managed turn's typed host boundary. Protocol values cross as protobuf JSON; storage rows
//! cross as the same records returned by the thread's read APIs.

use crate::storage::StorageError;
use crate::{ChatItem, usage::Usage};
use serde_json::{Map, Value};
use std::sync::Arc;

use crate::errors::ProviderError;
use crate::storage::context::{AppStatement, ContextScope, ContextScopeWrite};
use crate::storage::journal::ToolExecution;
use crate::storage::{ItemRow, TaskRecord, TaskState};

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
#[crate::host(record(module = "storage_turn", extract, emit))]
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PendingCall {
    pub name: String,
    pub call_id: String,
    /// The tool that takes the answer, when the `InputRequired` named one.
    #[host(default = None)]
    pub answer_tool: Option<String>,
}

#[crate::host(record(module = "storage_turn", extract, emit))]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HandlerOutput {
    pub output: Option<Value>,
    pub new_items: Vec<Value>,
    pub state: Option<Value>,
    pub scope_writes: Vec<ContextScopeWrite>,
    /// What the handler's managed transactions staged, applied in the turn's commit.
    pub app_writes: Vec<AppStatement>,
    /// The usage of each invocation the handler ran.
    pub usage: Vec<crate::usage::Usage>,
    pub input_required: Option<String>,
    pub pending_call: Option<PendingCall>,
    pub interrupted: bool,
    pub metadata: Map<String, Value>,
    pub error: Option<String>,
    /// The calls' starts, progress and ends the handler recorded, for the journal.
    pub tool_executions: Vec<ToolExecution>,
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
    Output { output: Box<HandlerOutput> },
    #[host(name = "Progress")]
    Progress { task_id: String, text: String },
    #[host(name = "Token")]
    Token { task_id: String, delta: String },
}

/// One handler operation: verdicts in and turn events out.
pub type TurnStream = crate::DuplexStream<
    Box<dyn crate::StreamInput<Verdict>>,
    Box<dyn crate::StreamOutput<TurnEvent>>,
>;

#[crate::host(proxy(host = "livekit.agents.storage.context._TurnHost"))]
pub trait TurnHost: Send + Sync {
    #[host(await, name = "_core_handle")]
    fn handle<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
        #[host(context)] ctx: &'a TurnContext,
    ) -> crate::BoxFuture<'a, Result<TurnStream, ProviderError>>;
}

impl crate::agent::HostContext for TurnContext {
    fn host_context(&self) -> Option<&crate::HostRef> {
        self.host_context.as_ref()
    }
}

#[crate::host(class(name = "Turn"))]
pub trait Turn: Send + Sync + Sized + 'static {
    #[host(env)]
    type Env;
    #[host(class(name = "Thread"))]
    type Thread: Send + Sync + 'static;
    #[host(new)]
    fn new(
        env: &Self::Env,
        thread: Arc<Self::Thread>,
        lane: String,
        host: Arc<dyn crate::a2a::turn::TurnHost>,
        #[host(default = None)] options: Option<crate::a2a::options::TurnOptions>,
    ) -> Self;
    #[host(getter)]
    fn lane(&self) -> String;
    #[host(call)]
    fn policy_for(&self, channel: String) -> String;
    #[host(await)]
    async fn run(
        &self,
        message: Value,
        #[host(default = None)] task_id: Option<String>,
        #[host(context)] context: crate::invoke::CallContext,
    ) -> Result<crate::a2a::turn::TurnResult, StorageError>;
}

#[crate::host(record(module = "storage_handler", extract))]
#[derive(Clone)]
pub enum HandlerOutcome {
    Value { value: Value },
    Completed { value: Value, verbatim: bool },
    InputRequired { question: String },
    Failed { error: String },
    Rejected { reason: String },
    Working { message: Option<String> },
}

#[crate::host(record(module = "storage_handler", extract))]
#[derive(Clone)]
pub struct HandlerRun {
    pub items: Vec<ChatItem>,
    pub usage: Usage,
    pub verbatim: Option<String>,
    pub question: Option<String>,
    pub pending_call: Option<PendingCall>,
}

#[crate::host(functions)]
pub trait HandlerOutcomes {
    #[host(function)]
    fn normalize_handler_output(
        output: HandlerOutput,
        outcome: HandlerOutcome,
        runs: Vec<HandlerRun>,
    ) -> Result<HandlerOutput, StorageError>;
}
