//! The provider and room half of `AMD`: model selection and the setup anchor. The classifier
//! machine and the S-06 plan live in `lk_agents_voice::amd::classifier`.


use lk_agents_abi::voice::amd::AmdCategory;

pub use lk_agents_abi::voice::amd::{
    DEFAULT_LLM_MODEL, DEFAULT_STT_MODEL, EVALUATED_LLM_MODELS, EVALUATED_STT_MODELS, SIP_CALL_STATUS_ACTIVE,
    SIP_CALL_STATUS_ATTR, TRACK_PUBLICATION_TIMEOUT,
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

/// `_setup` / `_wait_for_sip_answer` (`detector.py:397`) as a stateless calculator over wire
/// names; the 5 s timeout is the driver's. `publisher` is read on `track_published` only: `"sip"`
/// (the track exists before the callee answers, so listening waits for `sip.callStatus ==
/// "active"`), `"other"` (the track arriving *is* the answer) or `None` (the publisher vanished
/// before lookup). A missing track, a vanished publisher and a SIP call that never goes active all
/// settle `participant_missing`, rather than listening for the full `no_speech_threshold`.
#[lk_agents_abi::host(export(pid = false, name = "amd_decide_setup_step"))]
pub fn decide_setup_step_named(event: &str, publisher: Option<&str>) -> Result<&'static str, String> {
    const LISTEN: &str = "start_listening";
    const SETTLE: &str = "settle_participant_missing";
    Ok(match (event, publisher) {
        // `session._room_io` is `None`: the reference warns and starts immediately.
        ("no_room_io", _) | ("sip_answered", _) => LISTEN,
        ("track_published", Some("sip")) => "wait_for_sip_answer",
        ("track_published", Some("other")) => LISTEN,
        ("track_published", None) | ("track_unavailable", _) | ("sip_answer_failed", _) => SETTLE,
        ("track_published", Some(o)) => return Err(format!("unknown publisher kind {o:?}")),
        (other, _) => return Err(format!("unknown setup event {other:?}")),
    })
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

    // Catches: an early end reaching `start_listening` instead of `settle`, listening when a SIP
    // track is published (before the callee answers), or a publisher kind read on an event that
    // carries none. With no room to wait on the reference warns and starts anyway.
    #[test]
    fn each_setup_event_takes_its_step() {
        for (event, publisher, want) in [
            ("track_unavailable", None, Ok("settle_participant_missing")),
            ("track_published", None, Ok("settle_participant_missing")),
            ("sip_answer_failed", None, Ok("settle_participant_missing")),
            ("track_published", Some("sip"), Ok("wait_for_sip_answer")),
            ("track_published", Some("other"), Ok("start_listening")),
            ("sip_answered", Some("robot"), Ok("start_listening")),
            ("no_room_io", None, Ok("start_listening")),
        ] {
            assert_eq!(decide_setup_step_named(event, publisher), want, "{event} {publisher:?}");
        }
        assert!(decide_setup_step_named("track_published", Some("robot")).is_err());
        assert!(decide_setup_step_named("dialled", None).is_err());
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
