//! What the core derives from an agent definition (`lk_agents_abi::agent::AgentDefinition`): its
//! validity, its tools, the instructions of a request and the structured output's tool.

use lk_agents_abi::ToolSpec;
use lk_agents_abi::agent::{AgentDefinition, FunctionToolInfo, OutputSpec};
use serde_json::{Value, json};

/// The tool a non-`str` `output_type` is offered as.
pub const FINAL_RESULT_TOOL: &str = "final_result";

/// The final-result tool's description.
pub const FINAL_RESULT_DESCRIPTION: &str = "The final result of the run, once you have it.";

/// The key a non-object output schema is wrapped under, and unwrapped from.
pub const WRAPPED_RESULT_KEY: &str = "result";

/// A definition no invocation can run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentError {
    /// A raw-schema tool has no signature to bind validated arguments to.
    ApprovalOnRawSchema {
        tool: String,
    },
    DuplicateTool {
        tool: String,
    },
    /// The name the structured output is offered under.
    ReservedToolName {
        tool: String,
    },
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

/// The checks every invocation relies on.
pub fn validate(agent: &AgentDefinition) -> Result<(), AgentError> {
    let mut seen = std::collections::BTreeSet::new();
    for tool in &agent.tools {
        let name = &tool.spec.name;
        if tool.raw_schema && tool.needs_approval.is_some() {
            return Err(AgentError::ApprovalOnRawSchema { tool: name.clone() });
        }
        if !seen.insert(name.as_str()) {
            return Err(AgentError::DuplicateTool { tool: name.clone() });
        }
        if name == FINAL_RESULT_TOOL && matches!(agent.output, OutputSpec::Structured { .. }) {
            return Err(AgentError::ReservedToolName { tool: name.clone() });
        }
    }
    Ok(())
}

pub(crate) fn tool<'a>(agent: &'a AgentDefinition, name: &str) -> Option<&'a FunctionToolInfo> {
    agent.tools.iter().find(|t| t.spec.provider_tool.is_none() && t.spec.name == name)
}

/// The instructions for one request: the static ones, then the rendered dynamic ones.
pub(crate) fn instructions_with(agent: &AgentDefinition, dynamic: &str) -> String {
    match (agent.instructions.is_empty(), dynamic.is_empty()) {
        (_, true) => agent.instructions.clone(),
        (true, false) => dynamic.to_owned(),
        (false, false) => format!("{}\n\n{dynamic}", agent.instructions),
    }
}

/// The `final_result` tool for a structured output, with a non-object schema wrapped.
pub(crate) fn final_result_tool(agent: &AgentDefinition) -> Option<ToolSpec> {
    let OutputSpec::Structured { schema } = &agent.output else { return None };
    Some(ToolSpec {
        provider_tool: None,
        name: FINAL_RESULT_TOOL.to_owned(),
        description: FINAL_RESULT_DESCRIPTION.to_owned(),
        json_schema: wrapped_schema(schema).unwrap_or_else(|| schema.clone()),
        flags: Default::default(),
    })
}

/// Whether the structured output's schema is wrapped under [`WRAPPED_RESULT_KEY`].
pub(crate) fn output_wrapped(agent: &AgentDefinition) -> bool {
    matches!(&agent.output, OutputSpec::Structured { schema } if wrapped_schema(schema).is_some())
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
