//! `inference/llm.py`'s model tables: which sampling parameters a model refuses, the lowest
//! reasoning effort it admits, and which pair of think tags it wraps reasoning in.

/// Reasoning models don't support sampling parameters.
/// <https://platform.openai.com/docs/guides/reasoning>
const REASONING_UNSUPPORTED_PARAMS: &[&str] = &[
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "n",
];

/// xAI reasoning models only restrict these three. They still support temperature and top_p.
const XAI_REASONING_UNSUPPORTED_PARAMS: &[&str] =
    &["presence_penalty", "frequency_penalty", "stop"];

/// Model prefix -> the params to drop. Order is observable: the first matching prefix wins, as in
/// the reference's dict iteration.
const UNSUPPORTED_PARAMS: &[(&str, &[&str])] = &[
    ("o1", REASONING_UNSUPPORTED_PARAMS),
    ("o3", REASONING_UNSUPPORTED_PARAMS),
    ("o4", REASONING_UNSUPPORTED_PARAMS),
    ("gpt-5", REASONING_UNSUPPORTED_PARAMS),
    ("grok-4-1-fast-reasoning", XAI_REASONING_UNSUPPORTED_PARAMS),
    ("grok-4.20-0309-reasoning", XAI_REASONING_UNSUPPORTED_PARAMS),
    ("grok-4.20-multi-agent", XAI_REASONING_UNSUPPORTED_PARAMS),
];

/// Models that don't support `reasoning_effort` when function tools are present.
const REASONING_EFFORT_TOOL_INCOMPATIBLE_PREFIXES: &[&str] = &["gpt-5.2", "gpt-5.4"];

/// Lowest supported reasoning effort per model; `"none"` requires gpt-5.1+. Exact match, unlike
/// [`UNSUPPORTED_PARAMS`]'s prefix match.
const MIN_REASONING_EFFORT: &[(&str, &str)] = &[
    ("gpt-5.1", "none"),
    ("gpt-5.2", "none"),
    ("gpt-5.4", "none"),
    ("gpt-5.4-mini", "none"),
    ("gpt-5", "minimal"),
    ("gpt-5-mini", "minimal"),
    ("gpt-5-nano", "minimal"),
];

/// Models whose reasoning is delimited by something other than the default `<think>` pair.
const MODEL_THINK_TAGS: &[(&str, (&str, &str))] =
    &[("google/gemma-4-31b-it", ("<|channel>thought", "<channel|>"))];

/// Strip any provider prefix: `openai/o3-pro` -> `o3-pro`.
pub fn bare_model_name(model: &str) -> &str {
    model.rsplit('/').next().unwrap_or(model)
}

/// Remove parameters the given model does not support. `has_tools` is `bool(tools)`.
pub fn drop_unsupported_params(
    model: &str,
    params: &mut serde_json::Map<String, serde_json::Value>,
    has_tools: bool,
) {
    let name = bare_model_name(model);

    // `break` on the first matching prefix -- a model is never stripped by two rows.
    if let Some((_, unsupported)) =
        UNSUPPORTED_PARAMS.iter().find(|(prefix, _)| name.starts_with(prefix))
    {
        params.retain(|k, _| !unsupported.contains(&k.as_str()));
    }

    if has_tools && REASONING_EFFORT_TOOL_INCOMPATIBLE_PREFIXES.iter().any(|p| name.starts_with(p))
    {
        params.remove("reasoning_effort");
    }
}

/// The lowest reasoning effort the model supports, or `None` if it has no such control.
#[lk_agents_abi::host(export(pid = false))]
pub fn min_reasoning_effort(model: &str) -> Option<&'static str> {
    let name = bare_model_name(model);
    MIN_REASONING_EFFORT.iter().find(|(m, _)| *m == name).map(|(_, effort)| *effort)
}

/// The default think-tag pair, from `llm/utils.py`'s `THINK_TAG_START`/`THINK_TAG_END`.
pub const DEFAULT_THINK_TAGS: (&str, &str) = ("<think>", "</think>");

/// The think tags for a model, keyed on the full model string including the provider prefix.
pub fn think_tags(model: &str) -> (&'static str, &'static str) {
    MODEL_THINK_TAGS
        .iter()
        .find(|(m, _)| *m == model)
        .map(|(_, tags)| *tags)
        .unwrap_or(DEFAULT_THINK_TAGS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn params(keys: &[&str]) -> serde_json::Map<String, serde_json::Value> {
        keys.iter().map(|k| ((*k).to_owned(), json!(1))).collect()
    }

    fn keys(map: &serde_json::Map<String, serde_json::Value>) -> Vec<&str> {
        map.keys().map(String::as_str).collect()
    }

    // Catches: matching the prefixed model name against the table.
    #[test]
    fn the_provider_prefix_is_stripped_before_matching() {
        let mut p = params(&["temperature", "max_tokens"]);
        drop_unsupported_params("openai/o3-pro", &mut p, false);
        assert_eq!(keys(&p), ["max_tokens"]);

        assert_eq!(min_reasoning_effort("openai/gpt-5-mini"), Some("minimal"));
        assert_eq!(bare_model_name("gpt-4o"), "gpt-4o", "an unprefixed name is unchanged");
    }

    // Catches: applying the OpenAI reasoning set to xAI.
    #[test]
    fn xai_reasoning_models_keep_temperature_and_top_p() {
        let mut p = params(&["temperature", "top_p", "presence_penalty", "stop"]);
        drop_unsupported_params("xai/grok-4-1-fast-reasoning", &mut p, false);
        assert_eq!(keys(&p), ["temperature", "top_p"]);

        // ...and the *non*-reasoning sibling is not in the table at all, so nothing is stripped.
        let mut p = params(&["temperature", "presence_penalty"]);
        drop_unsupported_params("xai/grok-4-1-fast-non-reasoning", &mut p, false);
        assert_eq!(keys(&p), ["temperature", "presence_penalty"]);
    }

    // Catches: R2b on `if tools and ...` (an empty tool list keeps reasoning_effort).
    #[test]
    fn reasoning_effort_survives_when_the_tool_list_is_empty() {
        let mut p = params(&["reasoning_effort"]);
        drop_unsupported_params("openai/gpt-5.2", &mut p, false);
        assert_eq!(keys(&p), ["reasoning_effort"]);

        let mut p = params(&["reasoning_effort"]);
        drop_unsupported_params("openai/gpt-5.2", &mut p, true);
        assert!(p.is_empty());
    }

    // Catches: applying the prefix rule and the tools rule as if/else-if.
    #[test]
    fn both_drop_rules_can_apply_to_the_same_model() {
        let mut p = params(&["temperature", "reasoning_effort", "max_completion_tokens"]);
        drop_unsupported_params("openai/gpt-5.2", &mut p, true);
        assert_eq!(keys(&p), ["max_completion_tokens"]);
    }

    // Catches: turning MIN_REASONING_EFFORT into a prefix match.
    #[test]
    fn min_reasoning_effort_is_an_exact_match_not_a_prefix() {
        assert_eq!(min_reasoning_effort("openai/gpt-5.1"), Some("none"));
        assert_eq!(min_reasoning_effort("openai/gpt-5"), Some("minimal"));
        assert_eq!(min_reasoning_effort("openai/gpt-5-chat-latest"), None);
        assert_eq!(min_reasoning_effort("openai/gpt-5.5"), None);
        assert_eq!(min_reasoning_effort("openai/gpt-4o"), None);
    }

    // Catches: stripping the provider prefix before the think-tag lookup.
    #[test]
    fn think_tags_are_keyed_on_the_full_prefixed_model_string() {
        assert_eq!(think_tags("google/gemma-4-31b-it"), ("<|channel>thought", "<channel|>"));
        assert_eq!(think_tags("gemma-4-31b-it"), DEFAULT_THINK_TAGS);
        assert_eq!(think_tags("openai/gpt-4o"), DEFAULT_THINK_TAGS);
    }
}
