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

/// The turn's answer to a checkpoint the handler reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Continue,
    Yield,
}

/// What a handler reports while it runs, for the turn's response stream.
#[derive(Debug, Clone, PartialEq)]
pub enum TurnEvent {
    Progress { task_id: String, text: String },
    Token { task_id: String, delta: String },
}

/// The handler's side of one turn (docs/server.md "Progress, checkpoints and cancellation"):
/// what it reports, in the order it awaited each report.
#[crate::host(native(refused = "expected a native handler turn"))]
pub trait HandlerPort: std::any::Any + Send + Sync {
    /// The run reached a checkpoint; whether it yields. Waits for the turn's verdict.
    fn checkpoint(&self) -> crate::BoxFuture<'_, bool>;
    /// A progress message for the client; waits while the turn's queue is full.
    fn progress(&self, text: String) -> crate::BoxFuture<'_, ()>;
    /// One piece of the reply; waits while the turn's queue is full.
    fn token(&self, delta: String) -> crate::BoxFuture<'_, ()>;
}

/// One turn as its handler sees it: the handler awaits each report, and the turn reads them in
/// that order, publishing progress and tokens and answering checkpoints.
///
/// A report waits while the turn holds 64 unread ones, so a handler never runs ahead of the
/// turn. `checkpoint()` returns `True` when the run should yield (a newer message waits under
/// `interrupt`, or the client cancelled the task). Once the turn ended, or was closed, reports
/// are dropped and `checkpoint()` returns `True`.
#[crate::host(class(name = "_core_HandlerTurn", native = HandlerPort))]
pub trait HandlerTurn: Send + Sync + Sized + 'static {
    #[host(await)]
    async fn checkpoint(&self) -> bool;
    #[host(await)]
    async fn progress(&self, text: String);
    #[host(await)]
    async fn token(&self, delta: String);
}

/// The handler a turn runs on the host.
#[crate::host(proxy(host = "livekit.agents.storage.session._TurnHost"))]
pub trait TurnHost: Send + Sync {
    /// Run the handler on `ctx` to its output, reporting through `turn`. Cancelled when the turn
    /// closes it (a lost lease, a failed step); the turn then waits for its cleanup, within the
    /// host teardown bound.
    #[host(await, name = "_core_handle")]
    fn handle<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
        #[host(context)] ctx: &'a TurnContext,
        turn: Arc<dyn HandlerPort>,
        #[host(cancel)] cancel: Arc<dyn crate::invoke::Cancellation>,
    ) -> crate::BoxFuture<'a, Result<HandlerOutput, ProviderError>>;
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
