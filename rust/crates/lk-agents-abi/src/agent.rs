//! The agent host: the application callbacks an agent invocation runs through `HostInvoker`
//! (docs/internals/rust-core.md "HostInvoker"), and the records that cross with them. The
//! decisions are `lk-agents-core`'s; the runtime executes these calls; the host binds signatures,
//! validates application types and runs the bodies.

pub mod assertions;
pub mod run;

use serde_json::{Map, Value};

use crate::errors::HostError;
use crate::invoke::Scope;
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

/// What a host call runs in: the host context captured at its operation's entry, and the
/// framework scopes that operation holds. A `#[host(context)]` proxy parameter is one; a proxy
/// call without one runs in the context captured when the proxy was taken, holding no scope.
pub trait HostContext {
    fn host_context(&self) -> Option<&HostRef>;

    fn holds(&self, scope: Scope) -> bool {
        let _ = scope;
        false
    }
}

impl HostContext for HostRef {
    fn host_context(&self) -> Option<&HostRef> {
        Some(self)
    }
}

impl HostContext for Invocation {
    fn host_context(&self) -> Option<&HostRef> {
        self.host_context.as_ref()
    }

    fn holds(&self, scope: Scope) -> bool {
        match scope {
            Scope::Agent => true,
            Scope::ToolCall => self.function_call.is_some(),
            Scope::Approval => self.approval.is_some(),
            Scope::Job | Scope::Speech => false,
        }
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
pub trait AgentHost: Send + Sync {
    /// The surrounding managed turn's verdict; true yields before a model call or delivery.
    #[host(await)]
    fn checkpoint<'a>(
        &'a self,
        #[host(context)] _invocation: &'a Invocation,
    ) -> crate::BoxFuture<'a, Result<bool, HostError>> {
        Box::pin(async move { Ok(false) })
    }

    /// The `@agent.instructions` functions, rendered and joined.
    #[host(await)]
    fn render_instructions<'a>(
        &'a self,
        #[host(context)] invocation: &'a Invocation,
    ) -> crate::BoxFuture<'a, Result<String, HostError>>;

    /// The tools available for this model step, after toolset discovery.
    #[host(await)]
    fn list_tools<'a>(
        &'a self,
        #[host(context)] invocation: &'a Invocation,
    ) -> crate::BoxFuture<'a, Result<Vec<FunctionToolInfo>, HostError>>;

    /// Binds the call's arguments to the tool's signature and validates them, in the call's
    /// invocation scope.
    #[host(await, requires(tool_call))]
    fn prepare_tool<'a>(
        &'a self,
        #[host(context)] invocation: &'a Invocation,
        arguments: Map<String, Value>,
    ) -> crate::BoxFuture<'a, Result<Result<PreparedArguments, ToolArgumentError>, HostError>>;

    /// The tool's read-back function on the prepared arguments.
    #[host(await, requires(tool_call))]
    fn read_back<'a>(
        &'a self,
        #[host(context)] invocation: &'a Invocation,
        prepared: &'a HostRef,
    ) -> crate::BoxFuture<'a, Result<ReadBackAnswer, HostError>>;

    /// Runs the body with the prepared arguments and classifies what it returned or raised.
    #[host(await, requires(tool_call))]
    fn run_tool<'a>(
        &'a self,
        #[host(context)] invocation: &'a Invocation,
        prepared: &'a HostRef,
    ) -> crate::BoxFuture<'a, Result<(ToolReport, Map<String, Value>), HostError>>;

    /// The output type, validators and output function on a candidate.
    #[host(await)]
    fn validate_output<'a>(
        &'a self,
        #[host(context)] invocation: &'a Invocation,
        candidate: OutputCandidate,
    ) -> crate::BoxFuture<'a, Result<ValidatedOutput, HostError>>;

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
pub trait HostInstructions: Send + Sync {
    #[host(await, name = "_core_render")]
    fn render<'a>(
        &'a self,
        speech: &'a crate::CoreObject,
    ) -> crate::BoxFuture<'a, Result<String, crate::errors::HostError>>;
}
