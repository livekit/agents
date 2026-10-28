//! MCP content interpretation uses the SDK's records and serializers.

use std::sync::Arc;

use rmcp::model::CallToolResult;
use serde_json::Value;

use super::McpError;

fn decode(result: &Value) -> Result<CallToolResult, McpError> {
    serde_json::from_value(result.clone()).map_err(|error| McpError::Decode(Arc::new(error)))
}

fn check(result: &CallToolResult) -> Result<(), McpError> {
    if result.is_error == Some(true) {
        let parts = result
            .content
            .iter()
            .map(|part| {
                part.as_text()
                    .map_or_else(|| serde_json::to_string(part), |text| Ok(text.text.clone()))
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| McpError::Decode(Arc::new(error)))?;
        return Err(McpError::ToolFailed(parts.join("\n")));
    }
    Ok(())
}

pub fn check_result(result: &Value) -> Result<(), McpError> {
    check(&decode(result)?)
}

pub fn render_result(tool_name: &str, result: &Value) -> Result<String, McpError> {
    let result = decode(result)?;
    check(&result)?;
    let rendered = match result.content.as_slice() {
        [] => return Err(McpError::EmptyResult(tool_name.into())),
        [one] => serde_json::to_string(one),
        many => serde_json::to_string(many),
    };
    rendered.map_err(|error| McpError::Decode(Arc::new(error)))
}
