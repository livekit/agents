//! The agent host: the application callbacks an agent invocation runs through `HostInvoker`
//! (docs/internals/rust-core.md "HostInvoker"), and the records that cross with them. The
//! decisions are `lk-agents-core`'s; the runtime executes these calls; the host binds signatures,
//! validates application types and runs the bodies.

use async_trait::async_trait;
use serde_json::{Map, Value};

use crate::FunctionCall;
use crate::errors::HostError;
use crate::llm::tool_executor::{Approval, ToolReport};
use crate::utils::host_ref::HostRef;

/// What one host callback runs within, captured when the owning operation was created (a
/// `run()` call entry), never from whichever task dispatches it later.
#[derive(Debug, Clone)]
pub struct Invocation {
    /// This callback, unique within its agent invocation: what [`AgentHost::cancel`] names.
    pub operation: u64,
    /// The agent invocation this callback belongs to.
    pub invocation_id: String,
    pub agent_name: String,
    /// The tool call, for a tool's preparation, read-back and body.
    pub function_call: Option<FunctionCall>,
    /// The call's approval scope: the body runs on a granted read-back.
    pub approval: Option<Approval>,
    /// The host execution context captured at the operation's entry (Python: a
    /// `contextvars.Context`); the adapter enters a copy of it for the callback.
    pub host_context: Option<HostRef>,
}

/// The validated arguments a call is identified by: their JSON, or a fingerprint the host
/// declares for a value JSON cannot represent.
#[derive(Debug, Clone, PartialEq)]
pub enum ArgumentIdentity {
    Json(Value),
    Fingerprint(String),
}

/// Arguments the host bound and validated: `prepared` holds them for the body.
#[derive(Debug, Clone)]
pub struct PreparedArguments {
    pub prepared: HostRef,
    pub identity: ArgumentIdentity,
}

/// Binding or validation of a call's arguments failed: feedback for the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolArgumentError {
    pub message: String,
}

/// What a read-back function returned.
#[derive(Debug, Clone, PartialEq)]
pub enum ReadBackAnswer {
    Question(String),
    /// A detail is missing before there is anything to approve.
    InputRequired {
        question: String,
        verbatim: bool,
    },
}

/// An output candidate for the host to validate.
#[derive(Debug, Clone, PartialEq)]
pub enum OutputCandidate {
    Text(String),
    /// `final_result`'s value, unwrapped.
    Value(Value),
}

/// The host's verdict on an output candidate.
#[derive(Debug, Clone)]
pub enum ValidatedOutput {
    /// Accepted: a `str` output's final text, or the validated value the host keeps.
    Accepted { text: Option<String>, value: Option<HostRef> },
    /// `ModelRetry` or a validation error: the model corrects its output.
    Retry(String),
    /// A validator or output function failed outright.
    Failed(String),
}

/// The application callbacks of one agent definition. Every call carries its [`Invocation`];
/// the adapter enters that context, runs the callback, and restores the caller's on every exit.
// Its host proxy is generated once the proxy family moves to the declaration generator, with the
// invocation as the call's context argument.
#[async_trait]
pub trait AgentHost: Send + Sync {
    /// The `@agent.instructions` functions, rendered and joined.
    async fn render_instructions(&self, invocation: &Invocation) -> Result<String, HostError>;

    /// Binds the call's arguments to the tool's signature and validates them, in the call's
    /// invocation scope.
    async fn prepare_tool(
        &self,
        invocation: &Invocation,
        arguments: Map<String, Value>,
    ) -> Result<Result<PreparedArguments, ToolArgumentError>, HostError>;

    /// The tool's read-back function on the prepared arguments.
    async fn read_back(
        &self,
        invocation: &Invocation,
        prepared: &HostRef,
    ) -> Result<ReadBackAnswer, HostError>;

    /// Runs the body with the prepared arguments and classifies what it returned or raised.
    async fn run_tool(
        &self,
        invocation: &Invocation,
        prepared: &HostRef,
    ) -> Result<ToolReport, HostError>;

    /// The output type, validators and output function on a candidate.
    async fn validate_output(
        &self,
        invocation: &Invocation,
        candidate: OutputCandidate,
    ) -> Result<ValidatedOutput, HostError>;

    /// Cancels the callback in flight for `invocation` (its tool call, when it has one). Its
    /// awaited call still settles, with the cancellation as its outcome.
    fn cancel(&self, invocation: &Invocation);
}
