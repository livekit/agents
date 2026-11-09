use super::Retries;
use crate::llm::provider_format::tools::{DuplicateMode, DuplicateScope, ToolFlag};
use crate::{HostRef, Maybe, ToolChoice};
use serde_json::Value;

#[crate::host(file(module = "agent_authoring"))]
const _: () = ();
#[crate::host(constant)]
pub const DEFAULT_AGENT_NAME: &str = "default_agent";

/// A partial retry budget; omitted keys inherit the agent's configured budget.
#[crate::host(record(
    module = "agent_authoring",
    path = "livekit.agents.voice",
    typed_dict,
    extract
))]
#[derive(Clone)]
pub struct AgentRetries {
    pub tools: Maybe<u32>,
    pub output: Maybe<u32>,
}

#[crate::host(record(
    module = "agent_authoring",
    path = "livekit.agents.voice",
    dataclass,
    extract
))]
#[derive(Clone)]
pub struct ModelSettings {
    pub tool_choice: Maybe<ToolChoice>,
}

/// `Agent.tool()`'s options; an option left out keeps its default.
#[crate::host(record(module = "agent_authoring", dataclass, extract, kwargs))]
#[derive(Clone, Default)]
pub struct ToolOptions {
    /// The tool's name; the function's when `None`.
    pub name: Maybe<Option<String>>,
    /// What the model reads; the docstring when `None`.
    pub description: Maybe<Option<String>>,
    /// The tool's whole schema, for a function that takes its arguments as one raw dict.
    #[host(annotation = ".tool_context.RawFunctionDescription | dict[str, typing.Any] | None")]
    pub raw_schema: Maybe<Option<HostRef>>,
    #[host(annotation = ".tool_flags.ToolFlag")]
    pub flags: Maybe<ToolFlag>,
    pub on_duplicate: Maybe<DuplicateMode>,
    pub duplicate_scope: Maybe<DuplicateScope>,
    /// Hold each call for the person's approval of a read-back (tools.md "Approval"): `True`
    /// reads back the description and the arguments, a `ReadBack` function writes it.
    #[host(annotation = "bool | ..llm.tool_context.ReadBack")]
    pub needs_approval: Maybe<HostRef>,
    /// Consecutive retries the model gets for this tool; the agent's `retries['tools']` when
    /// `None`.
    pub retries: Maybe<Option<i64>>,
}

impl ToolOptions {
    /// The options given, by name.
    pub fn given(&self) -> Vec<String> {
        [
            ("name", self.name.is_given()),
            ("description", self.description.is_given()),
            ("raw_schema", self.raw_schema.is_given()),
            ("flags", self.flags.is_given()),
            ("on_duplicate", self.on_duplicate.is_given()),
            ("duplicate_scope", self.duplicate_scope.is_given()),
            ("needs_approval", self.needs_approval.is_given()),
            ("retries", self.retries.is_given()),
        ]
        .into_iter()
        .filter_map(|(name, given)| given.then(|| name.to_owned()))
        .collect()
    }
}

#[crate::host(record(module = "agent_authoring", enum, extract))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CallbackKind {
    #[host(rename = "tool")]
    Tool,
    #[host(rename = "instructions")]
    Instructions,
    #[host(rename = "validator")]
    Validator,
    #[host(rename = "on_enter")]
    OnEnter,
    #[host(rename = "on_exit")]
    OnExit,
}

/// Native configuration combined with SDK-derived tool and output schemas at the binding.
#[crate::host(record(module = "agent_authoring", typed_dict, emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct AgentConfiguration {
    pub name: String,
    pub instructions: String,
    pub dynamic_instructions: bool,
    pub retries: Retries,
}

#[crate::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AuthoringError {
    #[error("cannot register {what} on agent `{agent}`: it is already bound to a session")]
    Frozen { what: String, agent: String },
    #[host(kind = "ValueError")]
    #[error("retries must be non-negative integers, got {0}")]
    Retries(String),
    #[host(kind = "ValueError")]
    #[error("unknown agent callback kind {0:?}")]
    Kind(String),
    #[host(kind = "ValueError")]
    #[error("{0:?} apply to a function, not to a built tool")]
    BuiltToolOptions(Vec<String>),
}

#[crate::host(class(name = "_core_AgentConfig"))]
pub trait AgentConfig: Send + Sized + 'static {
    #[host(new)]
    fn new(
        instructions: String,
        name: Option<String>,
        retries: Option<Value>,
    ) -> Result<Self, AuthoringError>;
    #[host(call)]
    fn resolve_name(&mut self, discovered: Option<String>) -> String;
    #[host(getter)]
    fn instructions(&self) -> String;
    #[host(call)]
    fn freeze(&mut self);
    #[host(call)]
    fn check_mutable(&self, what: String) -> Result<(), AuthoringError>;
    #[host(call)]
    fn register(&mut self, kind: CallbackKind) -> Result<Registered, AuthoringError>;
    #[host(call)]
    fn callbacks(&self, kind: CallbackKind) -> Vec<u64>;
    #[host(call)]
    fn hook(&self, name: String) -> Option<u64>;
    #[host(call)]
    fn has_enter(&self, overridden: bool) -> bool;
    #[host(getter)]
    fn dynamic_instructions(&self) -> bool;
    #[host(getter)]
    fn validated_output(&self) -> bool;
    #[host(call)]
    fn configuration(&self, retries: Option<Value>) -> Result<AgentConfiguration, AuthoringError>;
    /// Refuses options given with an already built tool, which took its own when built.
    #[host(call)]
    fn check_tool_options(&self, options: ToolOptions) -> Result<(), AuthoringError>;
    #[host(call)]
    fn join_instructions(&self, parts: Vec<String>) -> String;
    #[host(getter)]
    fn retries(&self) -> Retries;
}

/// A callback's registration: its `token`, and the one it replaced (`retired`), an `on_enter` or
/// `on_exit` registered again.
#[crate::host(record(module = "agent_authoring", emit))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Registered {
    pub token: u64,
    pub retired: Option<u64>,
}
