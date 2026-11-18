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
        .map_err(ClientError::invalid_response)?
        .unwrap_or_default();
    let name = declaration.tool_name.unwrap_or_else(|| card.name.clone());
    if name.is_empty() {
        return Err(ClientError::invalid_response("the remote tool has no name"));
    }
    let structured_input = declaration.input_schema.is_some();
    let input_schema = declaration.input_schema.unwrap_or_else(request_schema);
    let mut input_schema = input_schema.as_object().cloned().ok_or_else(|| {
        ClientError::invalid_response("the remote input schema must be an object")
    })?;
    add_task_argument(&mut input_schema)?;
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
    let Value::Object(mut input_schema) = request_schema() else { unreachable!("a JSON object") };
    add_task_argument(&mut input_schema)?;
    Ok(ToolCard {
        name,
        description,
        extensions: Vec::new(),
        input_schema,
        structured_input: false,
    })
}

/// Routing belongs to the caller. A remote's business schema cannot redefine its destination.
fn add_task_argument(schema: &mut serde_json::Map<String, Value>) -> Result<(), ClientError> {
    let properties = schema.entry("properties").or_insert_with(|| json!({}));
    let properties = properties.as_object_mut().ok_or_else(|| {
        ClientError::InvalidInput("the remote input schema properties must be an object".into())
    })?;
    if properties.contains_key("task_id") {
        return Err(ClientError::InvalidInput(
            "the remote input schema uses the reserved task_id routing argument".into(),
        ));
    }
    properties.insert(
        "task_id".into(),
        json!({
            "type": ["string", "null"],
            "description": "Continue a known task on this remote agent, for an answer, clarification or correction. Omit or use null to let the conversation route this request."
        }),
    );
    Ok(())
}

/// Remove framework routing before validating or transmitting the remote's business input.
pub fn take_task_id(arguments: &mut Value) -> Result<Option<String>, ClientError> {
    let arguments = arguments.as_object_mut().ok_or_else(|| {
        ClientError::InvalidInput("remote tool arguments must be a JSON object".into())
    })?;
    match arguments.get("task_id") {
        None | Some(Value::Null) => {
            arguments.remove("task_id");
            Ok(None)
        }
        Some(Value::String(id)) if !id.trim().is_empty() => {
            let id = id.clone();
            arguments.remove("task_id");
            Ok(Some(id))
        }
        Some(_) => {
            Err(ClientError::InvalidInput("task_id must be a nonempty string or null".into()))
        }
    }
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
        assert_eq!(tool.input_schema["properties"]["task_id"]["type"], json!(["string", "null"]));
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
        assert!(definition.input_schema["properties"].get("task_id").is_some());

        let mut collision = serde_json::to_value(&card).unwrap();
        collision["capabilities"]["extensions"][0]["params"]["inputSchema"]["properties"]["task_id"] =
            json!({"type": "string"});
        assert!(
            tool_definition(&serde_json::from_value(collision).unwrap()).is_err(),
            "a business field cannot become routing"
        );

        let mut plain = card.clone();
        plain.capabilities.extensions = None;
        let definition = tool_definition(&plain).unwrap();
        assert_eq!(definition.name, "insurance");
        assert_eq!(definition.input_schema["required"], json!(["request"]));
        assert!(!definition.structured_input);
    }

    #[test]
    fn routing_arguments_never_reach_the_remote_business_payload() {
        for (routing, expected) in [(json!("booking-42"), Some("booking-42")), (Value::Null, None)]
        {
            let mut arguments = json!({"payer": "insurance", "task_id": routing});
            assert_eq!(take_task_id(&mut arguments).unwrap().as_deref(), expected);
            assert_eq!(arguments, json!({"payer": "insurance"}));
        }
        let mut arguments = json!({"request": "New work"});
        assert_eq!(take_task_id(&mut arguments).unwrap(), None);
        assert_eq!(arguments, json!({"request": "New work"}));
        for malformed in [json!(" "), json!(3), json!({}), json!(["booking-42"])] {
            let mut arguments = json!({"request": "Friday", "task_id": malformed});
            let before = arguments.clone();
            assert!(take_task_id(&mut arguments).is_err());
            assert_eq!(arguments, before, "invalid routing cannot partially consume input");
        }
    }
}
