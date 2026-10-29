//! The shared LLM algorithms plugins call (`llm.utils`, `_provider_format`): the core's
//! formatting, schemas and tool-argument rules, with provider formatting run on an admitted
//! media worker of the hosting process.
use lk_agents_abi::llm::provider_format::host::LlmAlgorithmError;
use lk_agents_abi::llm::provider_format::tools::{DuplicateKey, DuplicateMode, DuplicateScope};
use lk_agents_abi::llm::provider_format::{SerializedImage, StrictError};
use lk_agents_abi::llm::schema::SchemaError;
use lk_agents_abi::{ChatItem, Role};
use lk_agents_core::llm::{provider_format, schema, tool_arguments};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

use lk_agents_net::env::ProcessEnv;

pub struct LlmAlgorithms;

impl lk_agents_abi::llm::provider_format::host::LlmAlgorithms for LlmAlgorithms {
    type Env = ProcessEnv;

    async fn prepare_provider_format(
        env: &ProcessEnv,
        format: String,
        items: Vec<ChatItem>,
        images: Option<BTreeMap<String, SerializedImage>>,
        inject_dummy_user_message: bool,
        inject_trailing_user_message: bool,
        thought_signatures: Option<BTreeMap<String, Vec<u8>>>,
    ) -> Result<(Vec<Value>, Value), LlmAlgorithmError> {
        (env.workers)()
            .map_err(LlmAlgorithmError)?
            .try_spawn(lk_agents_media::workers::MediaWorkerClass::Formatting, move || {
                provider_format::host::to_provider_format(
                    &format,
                    items,
                    images,
                    inject_dummy_user_message,
                    inject_trailing_user_message,
                    thought_signatures,
                )
            })
            .map_err(|error| LlmAlgorithmError(error.to_string()))?
            .await
            .map_err(|error| LlmAlgorithmError(error.to_string()))?
            .map_err(LlmAlgorithmError)
    }
    fn convert_mid_conversation_instructions(
        items: Vec<ChatItem>,
        role: Role,
        template: &str,
    ) -> Vec<(usize, Option<ChatItem>)> {
        provider_format::host::convert_mid_conversation_instructions(items, role, template)
    }
    fn to_fnc_ctx(
        format: &str,
        tools: Vec<Value>,
        tool_behavior: Option<String>,
        use_parameters_json_schema: bool,
    ) -> Result<Vec<Value>, LlmAlgorithmError> {
        provider_format::host::to_fnc_ctx(
            format,
            tools,
            tool_behavior.as_deref(),
            use_parameters_json_schema,
        )
        .map_err(LlmAlgorithmError)
    }
    fn to_strict_json_schema(
        schema: &Value,
        null_sentinel_for_defaults: bool,
    ) -> Result<Map<String, Value>, StrictError> {
        provider_format::to_strict_json_schema(schema, null_sentinel_for_defaults)
    }
    fn resolve_ref(
        root: &Map<String, Value>,
        r#ref: &str,
    ) -> Result<Map<String, Value>, StrictError> {
        provider_format::resolve_ref(root, r#ref)
    }
    fn inject_schema_defaults(
        value: &Value,
        schema: &Value,
        root: Option<Value>,
    ) -> Result<Value, SchemaError> {
        schema::inject_schema_defaults_of(value, schema, root.as_ref())
    }
    fn build_legacy_openai_schema(
        name: &str,
        description: Option<String>,
        parameters: Value,
        internally_tagged: bool,
    ) -> Map<String, Value> {
        schema::build_legacy_openai_schema(
            name,
            description.as_deref(),
            parameters,
            internally_tagged,
        )
    }
    fn build_strict_openai_schema(
        name: &str,
        description: Option<String>,
        parameters: &Value,
    ) -> Result<Map<String, Value>, SchemaError> {
        schema::build_strict_openai_schema(name, description.as_deref(), parameters)
    }
    fn to_openai_response_format(
        name: &str,
        schema: &Value,
    ) -> Result<Map<String, Value>, SchemaError> {
        schema::to_openai_response_format(name, schema)
    }
    fn parse_function_arguments(raw: &str) -> Result<Map<String, Value>, LlmAlgorithmError> {
        tool_arguments::parse_logged(raw).map_err(LlmAlgorithmError)
    }
    fn duplicate_key(name: &str, args: Option<Value>, scope: DuplicateScope) -> DuplicateKey {
        provider_format::tools::duplicate_key(name, args.as_ref(), scope)
    }
    fn decide_duplicate_wire(
        name: &str,
        mode: DuplicateMode,
        confirm_duplicate: Option<bool>,
        in_flight: Vec<(String, bool)>,
    ) -> (String, Option<Value>) {
        let (decision, value) =
            provider_format::tools::decide_duplicate_wire(name, mode, confirm_duplicate, in_flight);
        (decision.to_owned(), value)
    }
    fn inject_confirm_duplicate(parameters: &Value) -> Map<String, Value> {
        provider_format::tools::inject_confirm_duplicate(parameters)
    }
}

#[cfg(test)]
mod tests {
    use super::LlmAlgorithms;
    use lk_agents_abi::llm::provider_format::host::LlmAlgorithms as _;
    // Catches: the repair warning losing the reference's `lk.pii.*` extras (the raw payload, the
    // repaired value, the strict error), which is all a reader of that warning has to go on.
    #[test]
    fn a_repair_is_logged_with_the_raw_payload_the_repair_and_the_error() {
        struct Keys(std::sync::Mutex<Vec<(String, String)>>);
        impl log::Log for Keys {
            fn enabled(&self, _: &log::Metadata<'_>) -> bool {
                true
            }
            fn log(&self, record: &log::Record<'_>) {
                struct Collect<'a>(&'a mut Vec<(String, String)>);
                impl<'kvs> log::kv::VisitSource<'kvs> for Collect<'_> {
                    fn visit_pair(
                        &mut self,
                        k: log::kv::Key<'kvs>,
                        v: log::kv::Value<'kvs>,
                    ) -> Result<(), log::kv::Error> {
                        self.0.push((k.to_string(), v.to_string()));
                        Ok(())
                    }
                }
                record.key_values().visit(&mut Collect(&mut self.0.lock().unwrap())).unwrap();
            }
            fn flush(&self) {}
        }
        let keys: &'static Keys = Box::leak(Box::new(Keys(Default::default())));
        log::set_logger(keys).unwrap();
        log::set_max_level(log::LevelFilter::Warn);
        LlmAlgorithms::parse_function_arguments("{'x':1}").unwrap();
        let seen = keys.0.lock().unwrap().clone();
        let get = |k: &str| seen.iter().find(|(key, _)| key == k).map(|(_, v)| v.as_str());
        assert_eq!(get("lk.pii.raw_arguments"), Some("{'x':1}"));
        assert_eq!(get("lk.pii.repaired"), Some(r#"{"x":1}"#));
        assert!(get("lk.pii.error").is_some_and(|e| !e.is_empty()), "{seen:?}");
    }
}
