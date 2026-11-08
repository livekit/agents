//! Chat and tool-format exports; implementations live in their owning cores and runtime.
use super::vocabulary::GeminiToolBehavior;
use super::{ProviderFormat, SerializedImage, StrictError, ToolSchema};
use crate::errors::Invalid;
use crate::llm::schema::SchemaError;
use crate::{ChatItem, Role};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

#[crate::host(file(module = "provider_format"))]
const _: () = ();

/// `convert_mid_conversation_instructions`' default wrapping; `{content}` is the only
/// placeholder.
#[crate::host(constant)]
pub const DEFAULT_INLINE_INSTRUCTIONS_TEMPLATE: &str = "<instructions>\n{content}\n</instructions>";

#[crate::host(functions(pid = false))]
pub trait LlmAlgorithms {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(function, pid = true, name = "to_provider_format")]
    async fn prepare_provider_format(
        env: &Self::Env,
        format: ProviderFormat,
        items: Vec<ChatItem>,
        #[host(kw, default = None)] images: Option<BTreeMap<String, SerializedImage>>,
        #[host(default = true)] inject_dummy_user_message: bool,
        #[host(default = false)] inject_trailing_user_message: bool,
        #[host(default = None)] thought_signatures: Option<BTreeMap<String, Vec<u8>>>,
    ) -> Result<(Vec<Value>, Value), Invalid>;
    #[host(function, pid = true)]
    fn convert_mid_conversation_instructions(
        items: Vec<ChatItem>,
        #[host(kw)] role: Role,
        template: &str,
    ) -> Vec<ConvertedItem>;
    #[host(function, pid = true)]
    fn to_fnc_ctx(
        format: ProviderFormat,
        tools: Vec<ToolSchema>,
        #[host(kw, default = None)] tool_behavior: Option<GeminiToolBehavior>,
        #[host(default = false)] use_parameters_json_schema: bool,
    ) -> Result<Vec<Value>, Invalid>;
    #[host(function)]
    fn to_strict_json_schema(
        schema: &Value,
        #[host(kw, default = false)] null_sentinel_for_defaults: bool,
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
    fn parse_function_arguments(raw: &str) -> Result<Map<String, Value>, Invalid>;
    #[host(function)]
    fn inject_confirm_duplicate(parameters: &Value) -> Map<String, Value>;
}

/// One item of a converted conversation: the item at `index`, or `rewritten` in its place.
#[crate::host(record(module = "provider_format", dataclass, emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct ConvertedItem {
    pub index: usize,
    pub rewritten: Option<ChatItem>,
}
