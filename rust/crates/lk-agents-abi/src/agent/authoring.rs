//! Agent authoring state and generated Python configuration records. Callables stay in the host.

use super::Retries;
use crate::{Maybe, ToolChoice};
use serde_json::Value;

#[crate::host(file(module = "agent_authoring"))]
const _: () = ();
#[crate::host(constant)]
pub const DEFAULT_AGENT_NAME: &str = "default_agent";

/// A partial retry budget; omitted keys inherit the agent's configured budget.
#[crate::host(record(
    module = "agent_authoring",
    path = "livekit.agents.voice.agent",
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
    path = "livekit.agents.voice.agent",
    dataclass,
    extract
))]
#[derive(Clone)]
pub struct ModelSettings {
    pub tool_choice: Maybe<ToolChoice>,
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
    fn register(&mut self, kind: CallbackKind) -> Result<(u64, Option<u64>), AuthoringError>;
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
    #[host(call)]
    fn check_tool_options(&self, given: Vec<String>) -> Result<(), AuthoringError>;
    #[host(call)]
    fn join_instructions(&self, parts: Vec<String>) -> String;
    #[host(getter)]
    fn retries(&self) -> Retries;
}
