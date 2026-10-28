//! The agent host: the application callbacks an agent invocation runs through `HostInvoker`
//! (docs/internals/rust-core.md "HostInvoker"), and the records that cross with them. The
//! decisions are `lk-agents-core`'s; the runtime executes these calls; the host binds signatures,
//! validates application types and runs the bodies.

use async_trait::async_trait;
use lk_agents_host_derive::Host;
use serde_json::{Map, Value};

use crate::llm::provider_format::tools::{DuplicateMode, DuplicateScope};
use crate::{FunctionCall, ToolSpec};
use crate::errors::HostError;
use crate::llm::tool_executor::{Approval, ToolReport};
use crate::utils::host_ref::HostRef;

/// What one host callback runs within, captured when the owning operation was created (a
/// `run()` call entry), never from whichever task dispatches it later. The host receives it as a
/// dict of its fields, without the captured context.
#[derive(Debug, Clone, Host)]
#[binding(emit, from_items)]
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
    #[binding(skip)]
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
/// declares for a value JSON cannot represent. From the host: `{"kind": "json", "value": ..}` or
/// `{"kind": "fingerprint", "value": str}`.
#[crate::host(handwritten(extract, host = "dict[str, object]"))]
#[derive(Debug, Clone, PartialEq)]
pub enum ArgumentIdentity {
    Json(Value),
    Fingerprint(String),
}

/// Arguments the host bound and validated: `prepared` holds them for the body.
#[derive(Debug, Clone, Host)]
#[binding(extract, from_items)]
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

/// What a read-back function returned. From the host: `{"kind": "question", "question": str}`
/// or `{"kind": "input_required", "question": str, "verbatim": bool}`.
#[crate::host(handwritten(extract, host = "dict[str, object]"))]
#[derive(Debug, Clone, PartialEq)]
pub enum ReadBackAnswer {
    Question(String),
    /// A detail is missing before there is anything to approve.
    InputRequired {
        question: String,
        verbatim: bool,
    },
}

/// An output candidate for the host to validate. To the host: `{"kind": "text", "text": str}` or
/// `{"kind": "value", "value": ..}`.
#[crate::host(handwritten(emit, host = "dict[str, object]"))]
#[derive(Debug, Clone, PartialEq)]
pub enum OutputCandidate {
    Text(String),
    /// `final_result`'s value, unwrapped.
    Value(Value),
}

/// The host's verdict on an output candidate. From the host: `{"kind": "accepted", "text": str |
/// None, "value": object | None}`, `{"kind": "retry", "reason": str}` or `{"kind": "failed",
/// "error": str}`.
#[crate::host(handwritten(extract, host = "dict[str, object]"))]
#[derive(Debug, Clone)]
pub enum ValidatedOutput {
    /// Accepted: a `str` output's final text, or the validated value the host keeps.
    Accepted { text: Option<String>, value: Option<HostRef> },
    /// `ModelRetry` or a validation error: the model corrects its output.
    Retry(String),
    /// A validator or output function failed outright.
    Failed(String),
}

/// The application callbacks of one agent definition, run by the host object of
/// `livekit.agents.voice.agent_host.AgentHost` (its methods carry these names). Every call carries
/// its [`Invocation`] and runs in a copy of the host context it captured; the caller's context is
/// never entered or changed.
#[crate::host(proxy(host = "livekit.agents.voice.agent_host.AgentHost"))]
#[async_trait]
pub trait AgentHost: Send + Sync {
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
    ) -> Result<ToolReport, HostError>;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadBack {
    /// "Please confirm: <first sentence> (<arguments>). Yes or no?"
    Default,
    /// The application's read-back function, run by the host on the prepared arguments.
    Function,
}

/// The final output contract (`output_type`).
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Retries {
    pub tool: u32,
    pub output: u32,
}

impl Default for Retries {
    fn default() -> Self {
        Retries { tool: 1, output: 1 }
    }
}

// The host shapes of the callback records, read and written here until the records family
// declares them: a dict whose `kind` names the variant.
#[cfg(feature = "python")]
mod host_shapes {
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    use super::{ArgumentIdentity, OutputCandidate, ReadBackAnswer, ValidatedOutput};
    use crate::python::{value_from_py, value_to_py};

    fn kind(obj: &Bound<'_, PyAny>, owner: &str) -> PyResult<String> {
        obj.get_item("kind")
            .and_then(|k| k.extract())
            .map_err(|e| pyo3::exceptions::PyTypeError::new_err(format!("{owner}.kind: {e}")))
    }

    fn unknown(owner: &str, kind: &str) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(format!("{owner}: unknown kind {kind:?}"))
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for ArgumentIdentity {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            const OWNER: &str = "ArgumentIdentity";
            let obj = &*obj;
            let value = obj.get_item("value")?;
            match kind(obj, OWNER)?.as_str() {
                "json" => Ok(ArgumentIdentity::Json(value_from_py(&value, "ArgumentIdentity.value")?)),
                "fingerprint" => Ok(ArgumentIdentity::Fingerprint(value.extract()?)),
                other => Err(unknown(OWNER, other)),
            }
        }
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for ReadBackAnswer {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            const OWNER: &str = "ReadBackAnswer";
            let obj = &*obj;
            let question = obj.get_item("question")?.extract()?;
            match kind(obj, OWNER)?.as_str() {
                "question" => Ok(ReadBackAnswer::Question(question)),
                "input_required" => Ok(ReadBackAnswer::InputRequired {
                    question,
                    verbatim: obj.get_item("verbatim")?.extract()?,
                }),
                other => Err(unknown(OWNER, other)),
            }
        }
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for ValidatedOutput {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            const OWNER: &str = "ValidatedOutput";
            let obj = &*obj;
            match kind(obj, OWNER)?.as_str() {
                "accepted" => Ok(ValidatedOutput::Accepted {
                    text: obj.get_item("text")?.extract()?,
                    value: obj.get_item("value")?.extract()?,
                }),
                "retry" => Ok(ValidatedOutput::Retry(obj.get_item("reason")?.extract()?)),
                "failed" => Ok(ValidatedOutput::Failed(obj.get_item("error")?.extract()?)),
                other => Err(unknown(OWNER, other)),
            }
        }
    }

    impl<'py> IntoPyObject<'py> for OutputCandidate {
        type Target = PyDict;
        type Output = Bound<'py, PyDict>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
            let out = PyDict::new(py);
            match &self {
                OutputCandidate::Text(text) => {
                    out.set_item("kind", "text")?;
                    out.set_item("text", text)?;
                }
                OutputCandidate::Value(value) => {
                    out.set_item("kind", "value")?;
                    out.set_item("value", value_to_py(py, value)?)?;
                }
            }
            Ok(out)
        }
    }
}
