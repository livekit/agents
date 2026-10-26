//! The provider half of `AMD`: model selection. The classifier machine and the room wait live in
//! `lk_agents_voice::amd`.


use lk_agents_abi::voice::amd::AmdCategory;

pub use lk_agents_abi::voice::amd::{
    DEFAULT_LLM_MODEL, DEFAULT_STT_MODEL, EVALUATED_LLM_MODELS, EVALUATED_STT_MODELS,
};

/// `_warn_if_not_evaluated` (`detector.py:627`) as a predicate: `model == candidate ||
/// candidate.contains(model)`, lowercased; empty is accepted. A pinned snapshot (a superstring)
/// is rejected, as in the reference.
pub fn is_evaluated_model(model: &str, evaluated: &[&str]) -> bool {
    if model.is_empty() {
        return true;
    }
    let model = model.to_lowercase();
    evaluated.iter().any(|c| {
        let c = c.to_lowercase();
        model == c || c.contains(&model)
    })
}

/// [`is_evaluated_model`] against the `"llm"` or `"stt"` table; any other kind is refused.
#[lk_agents_abi::host(export(pid = false, name = "amd_is_evaluated_model"))]
pub fn is_evaluated_model_of_kind(model: &str, kind: &str) -> Result<bool, String> {
    let table = match kind {
        "llm" => EVALUATED_LLM_MODELS,
        "stt" => EVALUATED_STT_MODELS,
        other => return Err(format!("unknown model kind {other:?}")),
    };
    Ok(is_evaluated_model(model, table))
}

/// An [`AmdCategory`] by its wire value; an unknown one is refused by name.
fn parse_category(category: &str) -> Result<AmdCategory, String> {
    AmdCategory::from_wire(category).ok_or_else(|| format!("unknown AMD category {category:?}"))
}

/// `AMDPredictionEvent.is_human`, over [`AmdCategory::is_human`].
#[lk_agents_abi::host(export(pid = false, name = "amd_category_is_human"))]
pub fn category_is_human(category: &str) -> Result<bool, String> {
    Ok(parse_category(category)?.is_human())
}

/// `AMDPredictionEvent.is_machine`, over [`AmdCategory::is_machine`].
#[lk_agents_abi::host(export(pid = false, name = "amd_category_is_machine"))]
pub fn category_is_machine(category: &str) -> Result<bool, String> {
    Ok(parse_category(category)?.is_machine())
}

/// Whether AMD may fill in its own LLM and STT from the gateway (`detector.py:184-194`).
#[lk_agents_abi::host(export(pid = false, name = "amd_auto_select_models"))]
pub fn auto_select_models(livekit_url: &str, api_key: &str, api_secret: &str) -> bool {
    lk_agents_abi::is_cloud_url(livekit_url) && !api_key.is_empty() && !api_secret.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: flipping the substring direction (bare names must match prefixed ones); also pins the
    // reference's rejection of pinned snapshots.
    #[test]
    fn model_compatibility_matches_prefixes_but_not_pinned_snapshots() {
        assert!(is_evaluated_model("openai/gpt-4.1-mini", EVALUATED_LLM_MODELS), "exact");
        assert!(is_evaluated_model("OpenAI/GPT-4.1-Mini", EVALUATED_LLM_MODELS), "case-insensitive");
        assert!(is_evaluated_model("gpt-4.1-mini", EVALUATED_LLM_MODELS), "unprefixed substring");
        assert!(is_evaluated_model("", EVALUATED_LLM_MODELS), "no model reported -- never warn");

        assert!(
            !is_evaluated_model("openai/gpt-4.1-mini-2024-07-18", EVALUATED_LLM_MODELS),
            "a pinned snapshot is a superstring and is (reference-faithfully) rejected",
        );
        assert!(!is_evaluated_model("anthropic/claude-sonnet-4.5", EVALUATED_LLM_MODELS));

        assert!(is_evaluated_model(DEFAULT_LLM_MODEL, EVALUATED_LLM_MODELS), "our own default");
        assert!(is_evaluated_model(DEFAULT_STT_MODEL, EVALUATED_STT_MODELS));
        assert!(!is_evaluated_model(DEFAULT_STT_MODEL, EVALUATED_LLM_MODELS), "the two lists differ");
    }

    // Catches: auto-selecting the gateway for a self-hosted deployment, or a `contains` cloud check.
    #[test]
    fn the_gateway_is_only_auto_selected_for_cloud_with_both_credentials() {
        assert!(auto_select_models("wss://x.livekit.cloud", "k", "s"));
        assert!(!auto_select_models("wss://x.livekit.cloud", "", "s"));
        assert!(!auto_select_models("wss://x.livekit.cloud", "k", ""));
        assert!(!auto_select_models("wss://selfhosted.example.com", "k", "s"));
        assert!(!auto_select_models("wss://evil.com/?x=livekit.cloud", "k", "s"));
    }

    // Catches: a renamed category wire string.
    #[test]
    fn the_category_wire_strings_round_trip() {
        let all = [
            (AmdCategory::Human, "human"),
            (AmdCategory::MachineIvr, "machine-ivr"),
            (AmdCategory::MachineVm, "machine-vm"),
            (AmdCategory::MachineUnavailable, "machine-unavailable"),
            (AmdCategory::Uncertain, "uncertain"),
        ];
        for (cat, s) in all {
            assert_eq!(cat.as_str(), s);
            assert_eq!(AmdCategory::from_wire(s), Some(cat));
        }
        assert_eq!(AmdCategory::from_wire("Human"), None, "the LLM's label is matched exactly");
        assert_eq!(AmdCategory::from_wire("machine"), None);
    }
}
