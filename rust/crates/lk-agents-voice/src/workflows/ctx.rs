use crate::workflows::ToolBox;
use async_trait::async_trait;
use lk_agents_abi::{Modality, ToolSpec};
use lk_agents_voice::generation::Instructions;
use serde_json::Value;
use std::sync::Arc;

/// The task-specific part of an `AgentTask`. `tools()` is the initial, config-derived set; runtime
/// changes go through [`TaskCtx::tools`]`().replace(..)`.
#[async_trait]
pub trait Task: Send {
    type Output: Send + 'static;

    fn instructions(&self) -> Instructions;

    fn tools(&self) -> Vec<ToolSpec>;

    async fn call(&mut self, name: &str, args: Value, ctx: &mut TaskCtx<Self::Output>)
    -> ToolReply;
}

/// `RunContext` plus the completion slot. Construct once per turn: see
/// [`turn_tools_revision`](Self::turn_tools_revision).
pub struct TaskCtx<T> {
    modality: Modality,
    tools: ToolBox,
    turn_tools_revision: u64,
    outcome: Option<Result<T, ToolError>>,
}

impl<T> TaskCtx<T> {
    /// One per turn, like `RunContext` per speech handle: it snapshots the turn's tool list.
    pub fn new(modality: Modality, tools: ToolBox) -> Self {
        let turn_tools_revision = tools.revision();
        TaskCtx { modality, tools, turn_tools_revision, outcome: None }
    }

    /// The [`ToolBox::revision`] this turn's tool list was built from: a tool installed earlier in
    /// the same batch isn't visible to a later call (the workflows' "stale closure" comments).
    pub fn turn_tools_revision(&self) -> u64 {
        self.turn_tools_revision
    }

    pub fn modality(&self) -> Modality {
        self.modality
    }

    pub fn tools(&self) -> &ToolBox {
        &self.tools
    }

    /// `self.complete(value)`; `false` when already done (the reference raises instead, so every
    /// call site checks `done()`).
    pub fn complete(&mut self, value: T) -> bool {
        self.settle(Ok(value))
    }

    pub fn fail(&mut self, error: ToolError) -> bool {
        self.settle(Err(error))
    }

    fn settle(&mut self, outcome: Result<T, ToolError>) -> bool {
        if self.outcome.is_some() {
            return false;
        }
        self.outcome = Some(outcome);
        true
    }

    pub fn take_outcome(&mut self) -> Option<Result<T, ToolError>> {
        self.outcome.take()
    }
}

/// What the reference `logger.warning`s about, returned as data for the driver to log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Warning {
    /// "`extra_instructions` will be ignored when `instructions` is provided".
    ExtraInstructionsIgnored,
    /// "`target_phone_number` is deprecated, use `sip_call_to` instead".
    TargetPhoneNumberDeprecated,
}

impl Warning {
    /// The name a host logs it under; the host owns the wording.
    pub fn as_str(self) -> &'static str {
        match self {
            Warning::ExtraInstructionsIgnored => "extra_instructions_ignored",
            Warning::TargetPhoneNumberDeprecated => "target_phone_number_deprecated",
        }
    }
}

/// Everything a task is handed that is not workflow logic. Provider slots are carried for the
/// driver; user `tools` are merged ahead of the built-ins and survive every `update_tools`.
#[derive(Default)]
pub struct TaskOptions {
    /// `chat_ctx`. Carried; seeding the session with it is the driver's job.
    pub chat_ctx: Option<lk_agents_abi::ChatContext>,
    /// `tools: list[llm.Tool | llm.Toolset]`, already flattened to specs by the caller.
    pub tools: Vec<ToolSpec>,
    /// `allow_interruptions`. `None` is `NOT_GIVEN`.
    pub allow_interruptions: Option<bool>,
    /// `preserve_function_call_history`: decides `exclude_function_call` on the hand-back merge.
    pub preserve_function_call_history: bool,
    pub stt: Option<Arc<dyn lk_agents_abi::Stt>>,
    pub vad: Option<Arc<dyn lk_agents_abi::Vad>>,
    pub llm: Option<Arc<dyn lk_agents_abi::Llm>>,
    pub tts: Option<Arc<dyn lk_agents_abi::Tts>>,
}

impl std::fmt::Debug for TaskOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskOptions")
            .field("chat_ctx", &self.chat_ctx.is_some())
            .field("tools", &self.tools.len())
            .field("allow_interruptions", &self.allow_interruptions)
            .field("preserve_function_call_history", &self.preserve_function_call_history)
            .field("stt", &self.stt.is_some())
            .field("vad", &self.vad.is_some())
            .field("llm", &self.llm.is_some())
            .field("tts", &self.tts.is_some())
            .finish()
    }
}

/// Argument extraction for [`Task::call`], standing in for pydantic's validation: a mismatch is a
/// `ToolError` the model can retry.
pub mod args {
    use crate::workflows::ToolError;
    use serde_json::Value;

    fn bad(tool: &str, field: &str, want: &str) -> ToolError {
        ToolError::plain(format!("invalid arguments for `{tool}`: `{field}` must be {want}"))
    }

    pub fn string(args: &Value, tool: &str, field: &str) -> Result<String, ToolError> {
        match args.get(field) {
            Some(Value::String(s)) => Ok(s.clone()),
            _ => Err(bad(tool, field, "a string")),
        }
    }

    /// `str | None = None`: absent, `null` and non-strings all read as `None`.
    pub fn opt_string(args: &Value, field: &str) -> Option<String> {
        args.get(field).and_then(Value::as_str).map(str::to_owned)
    }

    /// An `int` parameter. A whole-valued float is accepted (pydantic lax mode); a fractional one
    /// is rejected rather than truncated.
    pub fn integer(args: &Value, tool: &str, field: &str) -> Result<i64, ToolError> {
        match args.get(field) {
            Some(Value::Number(n)) => n
                .as_i64()
                .or_else(|| match n.as_f64() {
                    // `as` saturates, so range-check first.
                    Some(f) if f.fract() == 0.0 && f >= -(2f64.powi(63)) && f < 2f64.powi(63) => {
                        Some(f as i64)
                    }
                    _ => None,
                })
                .ok_or_else(|| bad(tool, field, "an integer")),
            _ => Err(bad(tool, field, "an integer")),
        }
    }

    pub fn string_list(args: &Value, tool: &str, field: &str) -> Result<Vec<String>, ToolError> {
        match args.get(field) {
            Some(Value::Array(items)) => items
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| bad(tool, field, "an array of strings"))
                })
                .collect(),
            _ => Err(bad(tool, field, "an array of strings")),
        }
    }
}

/// `llm.tool_context.ToolError`: what a tool raises and `complete()` takes on failure. The class
/// stays Python; this is the data to rebuild one.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct ToolError {
    /// The whole `str(exc)` the reference produces, already formatted.
    pub message: String,
    pub kind: ToolErrorKind,
}

/// The subclass identity of a [`ToolError`]. Open (`class_name` string): workflows declare their
/// own subclasses and `except` on them, and a user `AgentTask` may too.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ToolErrorKind {
    #[default]
    Plain,
    /// `class_name` is `type(exc).__name__`; `reason` is the subclass's kept argument.
    Subclass { class_name: String, reason: String },
}

impl ToolError {
    pub fn plain(message: impl Into<String>) -> Self {
        ToolError { message: message.into(), kind: ToolErrorKind::Plain }
    }

    /// A subclass; `message` is the formatted `str(exc)`.
    pub fn subclass(
        class_name: impl Into<String>,
        message: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        ToolError {
            message: message.into(),
            kind: ToolErrorKind::Subclass { class_name: class_name.into(), reason: reason.into() },
        }
    }

    /// `(message, class_name, reason)` for a host to raise the right class.
    pub fn into_parts(self) -> (String, Option<String>, Option<String>) {
        match self.kind {
            ToolErrorKind::Plain => (self.message, None, None),
            ToolErrorKind::Subclass { class_name, reason } => {
                (self.message, Some(class_name), Some(reason))
            }
        }
    }
}

/// What a tool body returned (`-> str | None`, or a raise). [`Silent`](ToolReply::Silent) is
/// `return None`, not an empty text message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolReply {
    Silent,
    Text(String),
    Error(ToolError),
}

impl ToolReply {
    pub fn text(s: impl Into<String>) -> Self {
        ToolReply::Text(s.into())
    }
    pub fn error(s: impl Into<String>) -> Self {
        ToolReply::Error(ToolError::plain(s))
    }
    /// The reply to an unregistered name (a model can hallucinate one).
    pub fn unknown_tool(name: &str) -> Self {
        ToolReply::error(format!("unknown tool `{name}`"))
    }
}
