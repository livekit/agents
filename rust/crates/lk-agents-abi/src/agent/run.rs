//! One agent invocation as the host runs it (agents.md "Agent", "Streaming runs"): the request,
//! the updates it streams, its settled result, and [`AgentRunner`], the class that runs an
//! [`AgentDefinition`] against its model and [`AgentHost`]. `lk-agents-core`'s `AgentRun` decides;
//! the runtime implements the runner.

use std::sync::Arc;

use lk_agents_host_derive::Host;
use serde_json::Value;

use super::{AgentDefinition, AgentHost};
use crate::utils::stream::StreamSource;
use crate::{ApiConnectOptions, ChatItem, HostRef, Llm};

/// One output candidate's host validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValidationId(pub u64);

/// The settled invocation (`RunResult[Output]`): its status, then the history and usage both
/// branches expose.
#[derive(Debug, Clone, PartialEq)]
pub struct RunResult {
    pub outcome: RunOutcome,
    /// The incoming message and everything this invocation recorded, in order.
    pub new_messages: Vec<ChatItem>,
    /// The supplied history plus this invocation, without instructions.
    pub all_messages: Vec<ChatItem>,
    /// This invocation's usage alone; the caller adds it to an accumulator it supplied.
    pub usage: RunUsage,
    /// The part of the answer or question that reached the person exactly as written.
    pub verbatim: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RunOutcome {
    Completed {
        output: Output,
    },
    /// `answer_tool` takes the person's next message as its first string parameter
    /// (`InputRequired(tool=)`); the next invocation is told so.
    InputRequired {
        question: String,
        verbatim: bool,
        answer_tool: Option<String>,
    },
}

/// The accepted output of a completed invocation.
#[derive(Debug, Clone, PartialEq)]
pub enum Output {
    /// A `str` output.
    Text(String),
    /// The value the host validated for this candidate; the runtime retains it.
    Validated(ValidationId),
    /// `StopResponse`, or a turn ended by a tool that asks for no reply.
    Nothing,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RunUsage {
    pub requests: u32,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub tool_calls: u32,
}

/// Why an invocation settled without a result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunError {
    /// The agent has no model to run a step with.
    NoModel,
    Cancelled,
    /// The model request failed after the provider's own recovery.
    Model(String),
    /// A host callback the invocation depends on (instructions, output validation) failed.
    Host(String),
    /// The model kept calling `tool` wrongly past its retry budget.
    ToolRetriesExceeded {
        tool: String,
        retries: u32,
    },
    /// The model kept producing rejected final output past its retry budget.
    OutputRetriesExceeded {
        retries: u32,
    },
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunError::NoModel => f.write_str("the agent has no model"),
            RunError::Cancelled => f.write_str("the run was cancelled"),
            RunError::Model(e) => write!(f, "the model request failed: {e}"),
            RunError::Host(e) => write!(f, "{e}"),
            RunError::ToolRetriesExceeded { tool, retries } => {
                write!(f, "tool `{tool}` exceeded max retries count of {retries}")
            }
            RunError::OutputRetriesExceeded { retries } => {
                write!(f, "exceeded maximum retries ({retries}) for output validation")
            }
        }
    }
}

impl std::error::Error for RunError {}

/// The incremental view of an invocation (agents.md "Streaming runs"). The coordinator adds the
/// envelope (sequence, agent name, invocation and frame ids).
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    TextDelta {
        message_id: String,
        delta: String,
    },
    TextEnd {
        message_id: String,
        interrupted: bool,
    },
    /// A call whose arguments validated, with those arguments.
    ToolCall {
        call_id: String,
        tool_name: String,
        arguments: Value,
    },
    ToolResult {
        call_id: String,
        tool_name: String,
        status: ToolResultStatus,
        output: String,
    },
    /// `ctx.update(text)`.
    Progress {
        call_id: String,
        tool_name: String,
        message: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolResultStatus {
    Completed,
    Failed,
    Cancelled,
}

/// What one invocation is asked: the message, the history it continues, and the host context
/// captured at the `run()` entry (a `(loop, contextvars.Context)` pair) its callbacks run in.
#[derive(Debug, Clone, Host)]
#[binding(extract, from_items)]
pub struct RunRequest {
    /// Prefixes every id the invocation mints.
    pub invocation_id: String,
    /// A user message.
    pub message: ChatItem,
    pub history: Vec<ChatItem>,
    /// The previous invocation asked with `InputRequired(tool=)`: this message is that tool's.
    pub answer_tool: Option<String>,
    /// Another agent asked this run: questions pass up as written.
    pub delegated: bool,
    pub conn_options: ApiConnectOptions,
    pub host_context: HostRef,
}

/// A settled invocation, with the value the host validated when the output is one.
#[derive(Debug)]
pub struct RunSettled {
    pub result: RunResult,
    pub output: Option<HostRef>,
}

/// What the consumer of a run receives, in order. To the host: `{"kind": "stream", "event":
/// {"kind": <StreamEvent variant>, ..fields}}`, `{"kind": "incomplete_cleanup", "pending":
/// [str]}`, `{"kind": "finished", "result": {..}, "output": object | None}` or `{"kind": "failed",
/// "error": {"kind": <RunError variant>, "message": str, ..fields}}`.
#[crate::host(handwritten(emit, host = "dict[str, object]"))]
#[derive(Debug)]
pub enum RunUpdate {
    Stream(StreamEvent),
    /// Cancelled work missed its cleanup deadline; the runtime still owns it (decision 14).
    IncompleteCleanup(Vec<String>),
    /// The last update.
    Finished(Result<RunSettled, RunError>),
}

/// The definition or the request cannot run: what is wrong with it.
#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct InvalidRun(pub String);

/// Runs invocations of one agent definition: each `run` streams its updates, then its settled
/// result; closing the stream cancels the invocation, whose work still settles.
#[crate::host(class(name = "AgentRunner"))]
pub trait AgentRunner: Send + Sync + Sized + 'static {
    #[host(source(item = RunUpdate, class = "AgentRunUpdates"))]
    type Updates: StreamSource<RunUpdate> + 'static;

    /// The runner of `definition` on `llm`, its callbacks run by `host`.
    #[host(new)]
    fn new(
        definition: AgentDefinition,
        llm: Arc<dyn Llm>,
        host: Arc<dyn AgentHost>,
    ) -> Result<Self, InvalidRun>;

    /// Starts one invocation; nothing runs until the first pull.
    #[host(call)]
    fn run(&self, request: RunRequest) -> Result<Self::Updates, InvalidRun>;
}

// The updates' host shape, written here until the records family declares it: dicts whose
// `kind` names the variant.
#[cfg(feature = "python")]
mod host_shapes {
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    use super::{
        Output, RunError, RunOutcome, RunResult, RunUpdate, StreamEvent, ToolResultStatus,
    };
    use crate::python::value_to_py;

    fn tagged<'py>(py: Python<'py>, kind: &str) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("kind", kind)?;
        Ok(out)
    }

    fn event<'py>(py: Python<'py>, e: &StreamEvent) -> PyResult<Bound<'py, PyDict>> {
        Ok(match e {
            StreamEvent::TextDelta { message_id, delta } => {
                let d = tagged(py, "text_delta")?;
                d.set_item("message_id", message_id)?;
                d.set_item("delta", delta)?;
                d
            }
            StreamEvent::TextEnd { message_id, interrupted } => {
                let d = tagged(py, "text_end")?;
                d.set_item("message_id", message_id)?;
                d.set_item("interrupted", interrupted)?;
                d
            }
            StreamEvent::ToolCall { call_id, tool_name, arguments } => {
                let d = tagged(py, "tool_call")?;
                d.set_item("call_id", call_id)?;
                d.set_item("tool_name", tool_name)?;
                d.set_item("arguments", value_to_py(py, arguments)?)?;
                d
            }
            StreamEvent::ToolResult { call_id, tool_name, status, output } => {
                let d = tagged(py, "tool_result")?;
                d.set_item("call_id", call_id)?;
                d.set_item("tool_name", tool_name)?;
                let status = match status {
                    ToolResultStatus::Completed => "completed",
                    ToolResultStatus::Failed => "failed",
                    ToolResultStatus::Cancelled => "cancelled",
                };
                d.set_item("status", status)?;
                d.set_item("output", output)?;
                d
            }
            StreamEvent::Progress { call_id, tool_name, message } => {
                let d = tagged(py, "progress")?;
                d.set_item("call_id", call_id)?;
                d.set_item("tool_name", tool_name)?;
                d.set_item("message", message)?;
                d
            }
        })
    }

    fn result<'py>(py: Python<'py>, r: &RunResult) -> PyResult<Bound<'py, PyDict>> {
        let outcome = match &r.outcome {
            RunOutcome::Completed { output } => {
                let o = tagged(py, "completed")?;
                let out = match output {
                    Output::Text(text) => {
                        let t = tagged(py, "text")?;
                        t.set_item("text", text)?;
                        t
                    }
                    // The validated value is the update's `output`.
                    Output::Validated(_) => tagged(py, "validated")?,
                    Output::Nothing => tagged(py, "nothing")?,
                };
                o.set_item("output", out)?;
                o
            }
            RunOutcome::InputRequired { question, verbatim, answer_tool } => {
                let o = tagged(py, "input_required")?;
                o.set_item("question", question)?;
                o.set_item("verbatim", verbatim)?;
                o.set_item("answer_tool", answer_tool)?;
                o
            }
        };
        let usage = PyDict::new(py);
        usage.set_item("requests", r.usage.requests)?;
        usage.set_item("input_tokens", r.usage.input_tokens)?;
        usage.set_item("output_tokens", r.usage.output_tokens)?;
        usage.set_item("tool_calls", r.usage.tool_calls)?;
        let out = PyDict::new(py);
        out.set_item("outcome", outcome)?;
        out.set_item("new_messages", r.new_messages.clone())?;
        out.set_item("all_messages", r.all_messages.clone())?;
        out.set_item("usage", usage)?;
        out.set_item("verbatim", &r.verbatim)?;
        Ok(out)
    }

    fn error<'py>(py: Python<'py>, e: &RunError) -> PyResult<Bound<'py, PyDict>> {
        let d = tagged(
            py,
            match e {
                RunError::NoModel => "no_model",
                RunError::Cancelled => "cancelled",
                RunError::Model(_) => "model",
                RunError::Host(_) => "host",
                RunError::ToolRetriesExceeded { .. } => "tool_retries_exceeded",
                RunError::OutputRetriesExceeded { .. } => "output_retries_exceeded",
            },
        )?;
        d.set_item("message", e.to_string())?;
        match e {
            RunError::ToolRetriesExceeded { tool, retries } => {
                d.set_item("tool", tool)?;
                d.set_item("retries", retries)?;
            }
            RunError::OutputRetriesExceeded { retries } => d.set_item("retries", retries)?,
            _ => {}
        }
        Ok(d)
    }

    impl<'py> IntoPyObject<'py> for RunUpdate {
        type Target = PyDict;
        type Output = Bound<'py, PyDict>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
            Ok(match self {
                RunUpdate::Stream(e) => {
                    let d = tagged(py, "stream")?;
                    d.set_item("event", event(py, &e)?)?;
                    d
                }
                RunUpdate::IncompleteCleanup(pending) => {
                    let d = tagged(py, "incomplete_cleanup")?;
                    d.set_item("pending", pending)?;
                    d
                }
                RunUpdate::Finished(Ok(settled)) => {
                    let d = tagged(py, "finished")?;
                    d.set_item("result", result(py, &settled.result)?)?;
                    d.set_item("output", settled.output)?;
                    d
                }
                RunUpdate::Finished(Err(e)) => {
                    let d = tagged(py, "failed")?;
                    d.set_item("error", error(py, &e)?)?;
                    d
                }
            })
        }
    }
}
