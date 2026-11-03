//! A remote tool's definition comes from its card. The extension's optional tool declaration
//! refines the ordinary text capability without changing what a plain A2A peer receives.

use a2a::AgentCard;
use lk_agents_abi::a2a::client::AgentCard as ToolCard;
use lk_agents_abi::a2a::metadata::EXTENSION_URI;
use lk_agents_abi::a2a_protocol as a2a;
use serde::Deserialize;
use serde_json::{Value, json};

use super::ClientError;

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToolDeclaration {
    tool_name: Option<String>,
    input_schema: Option<Value>,
}

pub fn tool_definition(card: &AgentCard) -> Result<ToolCard, ClientError> {
    let extensions = card.capabilities.extensions.as_deref().unwrap_or_default();
    let declaration = extensions
        .iter()
        .find(|extension| extension.uri == EXTENSION_URI)
        .and_then(|extension| extension.params.as_ref())
        .map(|params| {
            serde_json::to_value(params).and_then(serde_json::from_value::<ToolDeclaration>)
        })
        .transpose()
        .map_err(|error| ClientError::InvalidResponse(error.to_string()))?
        .unwrap_or_default();
    let name = declaration.tool_name.unwrap_or_else(|| card.name.clone());
    if name.is_empty() {
        return Err(ClientError::InvalidResponse("the remote tool has no name".into()));
    }
    let structured_input = declaration.input_schema.is_some();
    let input_schema = declaration.input_schema.unwrap_or_else(request_schema);
    let input_schema = input_schema.as_object().cloned().ok_or_else(|| {
        ClientError::InvalidResponse("the remote input schema must be an object".into())
    })?;
    Ok(ToolCard {
        name,
        description: card.description.clone(),
        extensions: extensions.iter().map(|extension| extension.uri.clone()).collect(),
        input_schema,
        structured_input,
    })
}

/// A delegation tool declared in code, without its card: the request, stated in full, goes to
/// the endpoint named `name`, which the parent model picks by `description`.
pub fn delegation_tool(name: String, description: String) -> Result<ToolCard, ClientError> {
    if name.trim().is_empty() || description.trim().is_empty() {
        return Err(ClientError::InvalidInput(
            "a remote agent needs the tool's name and description".into(),
        ));
    }
    let Value::Object(input_schema) = request_schema() else { unreachable!("a JSON object") };
    Ok(ToolCard {
        name,
        description,
        extensions: Vec::new(),
        input_schema,
        structured_input: false,
    })
}

/// The input of an unstructured delegation: the text the endpoint receives.
fn request_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "request": { "type": "string", "description": "The request, stated in full." }
        },
        "required": ["request"],
        "additionalProperties": false
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_delegation_tool_needs_a_name_and_description_and_takes_the_request() {
        let tool = delegation_tool("back_office".into(), "Books visits.".into()).unwrap();
        assert_eq!(tool.input_schema["required"], json!(["request"]));
        assert!(!tool.structured_input);
        assert!(delegation_tool(" ".into(), "Books visits.".into()).is_err());
        assert!(delegation_tool("back_office".into(), String::new()).is_err());
    }

    #[test]
    fn a_card_can_advertise_a_non_ask_tool_with_structured_input() {
        let card: AgentCard = serde_json::from_value(json!({
            "name": "insurance", "description": "Check insurance coverage", "version": "1",
            "supportedInterfaces": [],
            "capabilities": { "extensions": [{ "uri": EXTENSION_URI, "params": {
                "toolName": "verify_coverage", "inputSchema": {
                    "type": "object", "properties": {"payer": {"type": "string"}},
                    "required": ["payer"]
                }
            }}] }
        }))
        .unwrap();
        let definition = tool_definition(&card).unwrap();
        assert_eq!(definition.name, "verify_coverage");
        assert_eq!(definition.description, "Check insurance coverage");
        assert_eq!(definition.input_schema["required"], json!(["payer"]));
        assert!(definition.structured_input);

        let mut plain = card.clone();
        plain.capabilities.extensions = None;
        let definition = tool_definition(&plain).unwrap();
        assert_eq!(definition.name, "insurance");
        assert_eq!(definition.input_schema["required"], json!(["request"]));
        assert!(!definition.structured_input);
    }
}
