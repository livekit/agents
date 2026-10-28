//! The agent host: the application callbacks an agent invocation runs through `HostInvoker`
//! (docs/internals/rust-core.md "HostInvoker"), and the records that cross with them. The
//! decisions are `lk-agents-core`'s; the runtime executes these calls; the host binds signatures,
//! validates application types and runs the bodies.

pub mod assertions;
pub mod run;

use async_trait::async_trait;
use serde_json::{Map, Value};

use crate::errors::HostError;
use crate::llm::provider_format::tools::{DuplicateMode, DuplicateScope};
use crate::llm::tool_executor::{Approval, ToolReport};
use crate::utils::host_ref::HostRef;
use crate::{FunctionCall, ToolSpec};

/// What one host callback runs within, captured when the owning operation was created (a
/// `run()` call entry), never from whichever task dispatches it later.
#[crate::host(record(module = "agent_run", emit))]
#[derive(Debug, Clone)]
pub struct Invocation {
    /// This callback, unique within its agent invocation: what [`AgentHost::cancel`] names.
    pub operation: u64,
    /// The agent invocation this callback belongs to.
    pub invocation_id: String,
    pub agent_name: String,
    pub execution_id: Option<String>,
    pub frame_id: Option<crate::text_session::FrameId>,
    pub context_scope: Option<String>,
    pub application_context: Option<HostRef>,
    pub history: Vec<crate::ChatItem>,
    /// The tool call, for a tool's preparation, read-back and body.
    pub function_call: Option<FunctionCall>,
    /// The call's approval scope: the body runs on a granted read-back.
    pub approval: Option<Approval>,
    /// The host execution context captured at the operation's entry (Python: a
    /// `contextvars.Context`); the adapter enters a copy of it for the callback.
    #[host(skip)]
    pub host_context: Option<HostRef>,
}

/// What a `#[host(context)]` parameter carries: the host context a proxy call runs in.
pub trait HostContext {
    fn host_context(&self) -> Option<&HostRef>;
}

impl HostContext for Invocation {
    fn host_context(&self) -> Option<&HostRef> {
        self.host_context.as_ref()
    }
}

/// The validated arguments a call is identified by: their JSON, or a fingerprint the host
/// declares for a value JSON cannot represent.
#[crate::host(record(module = "agent_run", extract))]
#[derive(Debug, Clone, PartialEq)]
pub enum ArgumentIdentity {
    #[host(field = "value")]
    Json(Value),
    #[host(field = "value")]
    Fingerprint(String),
}

/// Arguments the host bound and validated: `prepared` holds them for the body.
#[crate::host(record(module = "agent_run", extract))]
#[derive(Debug, Clone)]
pub struct PreparedArguments {
    pub prepared: HostRef,
    pub identity: ArgumentIdentity,
}

/// Binding or validation of a call's arguments failed: feedback for the model. The host raises
/// it; its message is the feedback.
#[crate::host(error(raised = "livekit.agents.voice.agent_host.ToolArgumentError"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolArgumentError {
    pub message: String,
}

impl From<String> for ToolArgumentError {
    fn from(message: String) -> Self {
        ToolArgumentError { message }
    }
}

/// What a read-back function returned.
#[crate::host(record(module = "agent_run", extract))]
#[derive(Debug, Clone, PartialEq)]
pub enum ReadBackAnswer {
    #[host(field = "question")]
    Question(String),
    /// A detail is missing before there is anything to approve.
    InputRequired { question: String, verbatim: bool },
}

/// An output candidate for the host to validate.
#[crate::host(record(module = "agent_run", emit))]
#[derive(Debug, Clone, PartialEq)]
pub enum OutputCandidate {
    #[host(field = "text")]
    Text(String),
    /// `final_result`'s value, unwrapped.
    #[host(field = "value")]
    Value(Value),
}

/// An available JSON representation, including JSON null. Absence means the host value
/// has no persistent representation.
#[crate::host(record(module = "agent_run", extract))]
#[derive(Debug, Clone)]
pub struct SerializedOutput {
    pub value: Value,
}

/// The host's verdict on an output candidate.
#[crate::host(record(module = "agent_run", extract))]
#[derive(Debug, Clone)]
pub enum ValidatedOutput {
    /// Accepted: a `str` output's final text, or the validated value the host keeps.
    Accepted {
        text: Option<String>,
        value: Option<HostRef>,
        #[host(default = None)]
        serialized: Option<SerializedOutput>,
    },
    /// `ModelRetry` or a validation error: the model corrects its output.
    #[host(field = "reason")]
    Retry(String),
    /// A validator or output function failed outright.
    #[host(field = "error")]
    Failed(String),
}

/// The application callbacks of one agent definition, run by the host object of
/// `livekit.agents.voice.agent_host.AgentHost` (its methods carry these names). Every call carries
/// its [`Invocation`] and runs in a copy of the host context it captured; the caller's context is
/// never entered or changed.
#[crate::host(proxy(host = "livekit.agents.voice.agent_host.AgentHost"))]
#[async_trait]
pub trait AgentHost: Send + Sync {
    /// The surrounding managed turn's verdict; true yields before a model call or delivery.
    #[host(await)]
    async fn checkpoint(
        &self,
        #[host(context)] _invocation: &Invocation,
    ) -> Result<bool, HostError> {
        Ok(false)
    }

    /// The `@agent.instructions` functions, rendered and joined.
    #[host(await)]
    async fn render_instructions(
        &self,
        #[host(context)] invocation: &Invocation,
    ) -> Result<String, HostError>;

    /// Binds the call's arguments to the tool's signature and validates them, in the call's
    /// invocation scope.
    #[host(await)]
    async fn prepare_tool(
        &self,
        #[host(context)] invocation: &Invocation,
        arguments: Map<String, Value>,
    ) -> Result<Result<PreparedArguments, ToolArgumentError>, HostError>;

    /// The tool's read-back function on the prepared arguments.
    #[host(await)]
    async fn read_back(
        &self,
        #[host(context)] invocation: &Invocation,
        prepared: &HostRef,
    ) -> Result<ReadBackAnswer, HostError>;

    /// Runs the body with the prepared arguments and classifies what it returned or raised.
    #[host(await)]
    async fn run_tool(
        &self,
        #[host(context)] invocation: &Invocation,
        prepared: &HostRef,
    ) -> Result<(ToolReport, Map<String, Value>), HostError>;

    /// The output type, validators and output function on a candidate.
    #[host(await)]
    async fn validate_output(
        &self,
        #[host(context)] invocation: &Invocation,
        candidate: OutputCandidate,
    ) -> Result<ValidatedOutput, HostError>;

    /// Cancels the callback in flight for `invocation` (its tool call, when it has one). Its
    /// awaited call still settles, with the cancellation as its outcome.
    #[host(notify)]
    fn cancel(&self, #[host(context)] invocation: &Invocation);
}

/// The shared agent definition (agents.md "Agent"): reusable configuration every invocation reads
/// and none mutates. Host callables (tool bodies, validators, instruction functions, output
/// functions, read-backs) stay in the host; the definition says which of them exist.
#[crate::host(record(module = "agent_run", extract))]
#[derive(Debug, Clone, PartialEq)]
pub struct AgentDefinition {
    pub name: String,
    /// The static instructions.
    pub instructions: String,
    /// `@agent.instructions` functions are registered: the host renders them before each request.
    pub dynamic_instructions: bool,
    pub tools: Vec<FunctionToolInfo>,
    pub output: OutputSpec,
    pub retries: Retries,
}

/// `FunctionToolInfo`: one tool as the model and the gate see it.
#[crate::host(record(module = "agent_run", extract))]
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionToolInfo {
    pub spec: ToolSpec,
    /// Declared by a raw JSON schema (`raw_schema=`, MCP): no signature to validate against.
    pub raw_schema: bool,
    /// `needs_approval=`: the body runs only on an answered read-back.
    pub needs_approval: Option<ReadBack>,
    pub on_duplicate: DuplicateMode,
    pub duplicate_scope: DuplicateScope,
}

/// How a `needs_approval` tool asks.
#[crate::host(record(module = "agent_run", extract))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadBack {
    /// "Please confirm: <first sentence> (<arguments>). Yes or no?"
    Default,
    /// The application's read-back function, run by the host on the prepared arguments.
    Function,
}

/// The final output contract (`output_type`).
#[crate::host(record(module = "agent_run", extract))]
#[derive(Debug, Clone, PartialEq)]
pub enum OutputSpec {
    /// `str`: the model's text is the output. `validated` when output validators are registered.
    Text { validated: bool },
    /// Any other type, or an output function: the parameters' JSON schema. The host validates
    /// every candidate (type, validators, output function).
    Structured { schema: Value },
}

/// How many consecutive retries the model gets before the run fails: per tool for argument
/// errors and `ModelRetry`, and for the final output. pydantic-ai's defaults.
#[crate::host(record(module = "agent_run", extract))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Retries {
    #[host(default = 1)]
    pub tool: u32,
    #[host(default = 1)]
    pub output: u32,
}

impl Default for Retries {
    fn default() -> Self {
        Retries { tool: 1, output: 1 }
    }
}

/// The agent's `@agent.instructions` functions, rendered by the host for each model request of a
/// reply (`Agent._render_dynamic_instructions`). `speech` is the reply's, as the host's
/// `SpeechHandle` sees it.
#[crate::host(proxy(host = "livekit.agents.voice.agent_session._DynamicInstructions"))]
#[async_trait::async_trait]
pub trait HostInstructions: Send + Sync {
    #[host(await, name = "_core_render")]
    async fn render(&self, speech: &crate::CoreObject) -> Result<String, crate::errors::HostError>;
}
