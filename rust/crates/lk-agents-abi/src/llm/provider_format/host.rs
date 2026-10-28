//! Chat and tool-format exports; implementations live in their owning cores and runtime.
use super::tools::{DuplicateKey, DuplicateMode, DuplicateScope};
use super::{SerializedImage, StrictError};
use crate::llm::schema::SchemaError;
use crate::{ChatItem, Role};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

#[crate::host(functions(pid = false))]
pub trait LlmAlgorithms {
    #[host(function, pid = true, name = "to_provider_format")]
    async fn prepare_provider_format(
        format: String,
        items: Vec<ChatItem>,
        #[host(kw, default = None)] images: Option<BTreeMap<String, SerializedImage>>,
        #[host(default = true)] inject_dummy_user_message: bool,
        #[host(default = false)] inject_trailing_user_message: bool,
        #[host(default = None)] thought_signatures: Option<BTreeMap<String, Vec<u8>>>,
    ) -> Result<(Vec<Value>, Value), LlmAlgorithmError>;
    #[host(function, pid = true)]
    fn convert_mid_conversation_instructions(
        items: Vec<ChatItem>,
        #[host(kw)] role: Role,
        template: &str,
    ) -> Vec<(usize, Option<ChatItem>)>;
    #[host(function, pid = true)]
    fn to_fnc_ctx(
        format: &str,
        tools: Vec<Value>,
        #[host(kw, default = None)] tool_behavior: Option<String>,
        #[host(default = false)] use_parameters_json_schema: bool,
    ) -> Result<Vec<Value>, LlmAlgorithmError>;
    #[host(function)]
    fn to_strict_json_schema(
        schema: &Value,
        #[host(kw, default = false)] null_sentinel_for_defaults: bool,
    ) -> Result<Map<String, Value>, StrictError>;
    #[host(function)]
    fn resolve_ref(
        #[host(kw)] root: &Map<String, Value>,
        r#ref: &str,
    ) -> Result<Map<String, Value>, StrictError>;
    #[host(function)]
    fn inject_schema_defaults(
        value: &Value,
        schema: &Value,
        #[host(default = None)] root: Option<Value>,
    ) -> Result<Value, SchemaError>;
    #[host(function)]
    fn build_legacy_openai_schema(
        name: &str,
        description: Option<String>,
        parameters: Value,
        #[host(kw, default = false)] internally_tagged: bool,
    ) -> Map<String, Value>;
    #[host(function)]
    fn build_strict_openai_schema(
        name: &str,
        description: Option<String>,
        parameters: &Value,
    ) -> Result<Map<String, Value>, SchemaError>;
    #[host(function)]
    fn to_openai_response_format(
        name: &str,
        schema: &Value,
    ) -> Result<Map<String, Value>, SchemaError>;
    #[host(function)]
    fn parse_function_arguments(raw: &str) -> Result<Map<String, Value>, LlmAlgorithmError>;
    #[host(function, pid = true)]
    fn duplicate_key(name: &str, args: Option<Value>, scope: DuplicateScope) -> DuplicateKey;
    #[host(function, name = "decide_duplicate")]
    fn decide_duplicate_wire(
        name: &str,
        mode: DuplicateMode,
        confirm_duplicate: Option<bool>,
        in_flight: Vec<(String, bool)>,
    ) -> (String, Option<Value>);
    #[host(function)]
    fn inject_confirm_duplicate(parameters: &Value) -> Map<String, Value>;
}

#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct LlmAlgorithmError(pub String);
