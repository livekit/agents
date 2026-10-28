//! The shared agent definition (agents.md "Agent"): reusable configuration every invocation reads
//! and none mutates. Host callables (tool bodies, validators, instruction functions, output
//! functions, read-backs) stay in the host; the definition says which of them exist.

use lk_agents_abi::ToolSpec;
use lk_agents_abi::llm::provider_format::tools::{DuplicateMode, DuplicateScope};
use lk_agents_abi::resources::LlmRef;
use serde_json::{Value, json};

/// The tool a non-`str` `output_type` is offered as.
pub const FINAL_RESULT_TOOL: &str = "final_result";

/// The final-result tool's description.
pub const FINAL_RESULT_DESCRIPTION: &str = "The final result of the run, once you have it.";

/// The key a non-object output schema is wrapped under, and unwrapped from.
pub const WRAPPED_RESULT_KEY: &str = "result";

#[derive(Debug, Clone, PartialEq)]
pub struct Agent {
    pub name: String,
    /// The static instructions.
    pub instructions: String,
    /// `@agent.instructions` functions are registered: the host renders them before each request.
    pub dynamic_instructions: bool,
    /// The reasoning model; an agent without one cannot run a model step.
    pub llm: Option<LlmRef>,
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

/// A definition no invocation can run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentError {
    /// A raw-schema tool has no signature to bind validated arguments to.
    ApprovalOnRawSchema { tool: String },
    DuplicateTool { tool: String },
    /// The name the structured output is offered under.
    ReservedToolName { tool: String },
}

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentError::ApprovalOnRawSchema { tool } => {
                write!(f, "tool `{tool}` has a raw schema and cannot require approval")
            }
            AgentError::DuplicateTool { tool } => write!(f, "tool `{tool}` is registered twice"),
            AgentError::ReservedToolName { tool } => {
                write!(f, "tool `{tool}` uses the name reserved for the structured output")
            }
        }
    }
}

impl std::error::Error for AgentError {}

impl Agent {
    /// The checks every invocation relies on.
    pub fn validate(&self) -> Result<(), AgentError> {
        let mut seen = std::collections::BTreeSet::new();
        for tool in &self.tools {
            let name = &tool.spec.name;
            if tool.raw_schema && tool.needs_approval.is_some() {
                return Err(AgentError::ApprovalOnRawSchema { tool: name.clone() });
            }
            if !seen.insert(name.as_str()) {
                return Err(AgentError::DuplicateTool { tool: name.clone() });
            }
            if name == FINAL_RESULT_TOOL && matches!(self.output, OutputSpec::Structured { .. }) {
                return Err(AgentError::ReservedToolName { tool: name.clone() });
            }
        }
        Ok(())
    }

    pub(crate) fn tool(&self, name: &str) -> Option<&FunctionToolInfo> {
        self.tools.iter().find(|t| t.spec.name == name)
    }

    /// The instructions for one request: the static ones, then the rendered dynamic ones.
    pub(crate) fn instructions_with(&self, dynamic: &str) -> String {
        match (self.instructions.is_empty(), dynamic.is_empty()) {
            (_, true) => self.instructions.clone(),
            (true, false) => dynamic.to_owned(),
            (false, false) => format!("{}\n\n{dynamic}", self.instructions),
        }
    }

    /// The `final_result` tool for a structured output, with a non-object schema wrapped.
    pub(crate) fn final_result_tool(&self) -> Option<ToolSpec> {
        let OutputSpec::Structured { schema } = &self.output else { return None };
        Some(ToolSpec {
            name: FINAL_RESULT_TOOL.to_owned(),
            description: FINAL_RESULT_DESCRIPTION.to_owned(),
            json_schema: wrapped_schema(schema).unwrap_or_else(|| schema.clone()),
            flags: Default::default(),
        })
    }

    /// Whether the structured output's schema is wrapped under [`WRAPPED_RESULT_KEY`].
    pub(crate) fn output_wrapped(&self) -> bool {
        matches!(&self.output, OutputSpec::Structured { schema } if wrapped_schema(schema).is_some())
    }
}

/// `{"result": <schema>}` for a schema that is not an object, its `$defs` kept at the root so
/// references still resolve; `None` for an object schema.
fn wrapped_schema(schema: &Value) -> Option<Value> {
    if schema.get("type").and_then(Value::as_str) == Some("object") {
        return None;
    }
    let mut inner = schema.clone();
    let defs = inner.as_object_mut().and_then(|o| o.remove("$defs"));
    let mut wrapped = json!({
        "type": "object",
        "properties": { WRAPPED_RESULT_KEY: inner },
        "required": [WRAPPED_RESULT_KEY],
    });
    if let Some(defs) = defs {
        wrapped["$defs"] = defs;
    }
    Some(wrapped)
}

/// The first string parameter of a tool, in schema order: what an `InputRequired(tool=)` answer
/// is handed to.
pub(crate) fn first_string_parameter(spec: &ToolSpec) -> Option<&str> {
    let properties = spec.json_schema.get("properties")?.as_object()?;
    properties
        .iter()
        .find(|(_, p)| p.get("type").and_then(Value::as_str) == Some("string"))
        .map(|(name, _)| name.as_str())
}
