use super::{BYTES_TAG, FormatError, ToolSchema, b64_encode};
use serde::{Deserialize, Serialize, Serializer, ser::SerializeMap};
use serde_json::Value;

pub fn serialize_wire<T: std::fmt::Display, S: Serializer>(
    value: &T,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.collect_str(value)
}

/// A bytes leaf materialized by the Python binding, rather than a base64 string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostBytes(pub Vec<u8>);

impl Serialize for HostBytes {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(1))?;
        map.serialize_entry(BYTES_TAG, &b64_encode(&self.0))?;
        map.end()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Content<P> {
    Text(String),
    Parts(Vec<P>),
}

/// The host supplies a flat definition for legacy/raw tools and a nested one for strict tools.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Deserialize)]
struct StrictTool {
    function: FunctionDefinition,
}

impl FunctionDefinition {
    pub fn from_tool(tool: &ToolSchema) -> Result<Self, FormatError> {
        let decode =
            |e: serde_json::Error| FormatError::MalformedToolSchema { detail: e.to_string() };
        match tool {
            ToolSchema::Raw { name, raw_schema } => {
                #[derive(Deserialize)]
                struct Raw {
                    description: Option<String>,
                    parameters: Option<Value>,
                    strict: Option<bool>,
                }
                let raw: Raw = serde_json::from_value(raw_schema.clone()).map_err(decode)?;
                Ok(Self {
                    name: name.clone(),
                    description: raw.description,
                    parameters: raw.parameters,
                    strict: raw.strict,
                })
            }
            ToolSchema::Legacy(schema) => serde_json::from_value(schema.clone()).map_err(decode),
            ToolSchema::Strict(schema) => {
                let mut definition =
                    serde_json::from_value::<StrictTool>(schema.clone()).map_err(decode)?.function;
                definition.strict = Some(true);
                Ok(definition)
            }
            ToolSchema::Provider(_) => Err(FormatError::MalformedToolSchema {
                detail: "a provider tool is not a function definition".into(),
            }),
        }
    }
}
