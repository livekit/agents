//! `inference/stt.py`'s option resolution. Several are truthiness tests on values providers send
//! empty or `false`; getting one wrong silently flips a capability.

use lk_agents_abi::{AlignedTranscript, SttCapabilities};
use serde_json::{Map, Value, json};

/// Models verified to carry word timings. Unknown models stay disabled until verified.
const WORD_ALIGNED_MODELS: &[&str] = &[
    "deepgram/nova-3",
    "deepgram/nova-3-medical",
    "deepgram/nova-2",
    "deepgram/nova-2-medical",
    "deepgram/nova-2-conversationalai",
    "deepgram/nova-2-phonecall",
    "deepgram/flux-general",
    "deepgram/flux-general-en",
    "deepgram/flux-general-multi",
    "cartesia/ink-whisper",
    "assemblyai/universal-streaming",
    "assemblyai/universal-streaming-multilingual",
    "assemblyai/u3-rt-pro",
    "assemblyai/universal-3-5-pro",
    "elevenlabs/scribe_v2_realtime",
    "xai/stt-1",
    "speechmatics/enhanced",
    "speechmatics/standard",
];

/// Models that natively carry `agent_context` (the AssemblyAI U3 Pro family).
const ASSEMBLYAI_CARRYOVER_MODELS: &[&str] =
    &["assemblyai/u3-rt-pro", "assemblyai/universal-3-5-pro"];

const DIARIZATION_EXTRA_KEYS: &[&str] = &[
    "diarize",        // Deepgram, xAI
    "speaker_labels", // AssemblyAI
    "diarization",    // Speechmatics
];

pub const DEFAULT_ENCODING: &str = "pcm_s16le";
pub const DEFAULT_SAMPLE_RATE: u32 = 16_000;

/// `"provider/model:language"` -> `("provider/model", Some("language"))`, split at the last colon.
#[lk_agents_abi::host(export(name = "stt_parse_model_string"))]
pub fn parse_model_string(model: &str) -> (&str, Option<&str>) {
    match model.rfind(':') {
        Some(idx) => (&model[..idx], Some(&model[idx + 1..])),
        None => (model, None),
    }
}

/// True if any known provider diarization flag is truthy.
#[lk_agents_abi::host(export(name = "stt_diarization_enabled"))]
pub fn diarization_enabled(extra_kwargs: Option<&Map<String, Value>>) -> bool {
    let Some(extra) = extra_kwargs else { return false };
    for key in DIARIZATION_EXTRA_KEYS {
        let Some(value) = extra.get(*key) else { continue };
        // `if not value: continue` -- `false`, `0` and `""` do not turn diarization on.
        if is_falsy(value) {
            continue;
        }
        // Speechmatics' `diarization` accepts the string "none" to mean off.
        if value.as_str().is_some_and(|s| s.eq_ignore_ascii_case("none")) {
            continue;
        }
        return true;
    }
    false
}

/// Python truthiness for a JSON value: `null`, `false`, `0`, `""`, `[]`, `{}`.
pub(crate) fn is_falsy(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Bool(b) => !b,
        Value::Number(n) => n.as_f64() == Some(0.0),
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
    }
}

/// `None` means the model has no keyterm prompting (also the capability check).
pub fn keyterms_extra_for_model(
    model: Option<&str>,
    extra_kwargs: Option<&Map<String, Value>>,
    session_keyterms: &[String],
) -> Option<Map<String, Value>> {
    let model = model?;
    let empty = Map::new();
    let extra = extra_kwargs.unwrap_or(&empty);

    if model.starts_with("speechmatics/") {
        // Keep existing entries as-is (they may carry `sounds_like` etc.) and append new terms.
        let existing: Vec<Value> =
            extra.get("additional_vocab").and_then(Value::as_array).cloned().unwrap_or_default();
        let seen: Vec<&str> =
            existing.iter().filter_map(|v| v.get("content").and_then(Value::as_str)).collect();
        // The reference's `set(...) - seen` has unspecified order; kept in input order here.
        let mut additions: Vec<Value> = Vec::new();
        let mut added: Vec<&str> = Vec::new();
        for term in session_keyterms {
            if seen.contains(&term.as_str()) || added.contains(&term.as_str()) {
                continue;
            }
            added.push(term);
            additions.push(json!({"content": term}));
        }
        let mut out = Map::new();
        out.insert(
            "additional_vocab".into(),
            Value::Array(existing.into_iter().chain(additions).collect()),
        );
        return Some(out);
    }

    let key = if model.starts_with("deepgram/") {
        "keyterm"
    } else if model.starts_with("assemblyai/") {
        "keyterms_prompt"
    } else {
        return None;
    };

    // Deepgram's `keyterm` may be a bare string; it must become a one-element list.
    let mut merged: Vec<String> = match extra.get(key) {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect(),
        _ => Vec::new(),
    };
    // `dict.fromkeys([...])`: order-preserving deduplication.
    for term in session_keyterms {
        if !merged.contains(term) {
            merged.push(term.clone());
        }
    }
    let mut out = Map::new();
    out.insert(key.into(), Value::Array(merged.into_iter().map(Value::String).collect()));
    Some(out)
}

/// True if the model natively carries `agent_context`.
#[lk_agents_abi::host(export(name = "stt_supports_agent_context_carryover"))]
pub fn supports_agent_context_carryover(model: Option<&str>) -> bool {
    model.is_some_and(|m| ASSEMBLYAI_CARRYOVER_MODELS.contains(&m))
}

/// Word-level alignment for the model; `false` for `"auto"`, since claiming alignment is the costlier error.
#[lk_agents_abi::host(export(name = "stt_aligned_transcript_for_model"))]
pub fn aligned_transcript_for_model(model: Option<&str>) -> bool {
    model.is_some_and(|m| WORD_ALIGNED_MODELS.contains(&m))
}

pub use lk_agents_abi::inference::fallback::SttFallbackModel as FallbackModel;

/// `_normalize_fallback`: a bare string becomes `FallbackModel(model=<name>)`, with the
/// `:language` suffix parsed off and discarded.
#[lk_agents_abi::host(export(name = "stt_normalize_fallback_str"))]
pub fn normalize_fallback_str(model: &str) -> lk_agents_abi::inference::fallback::SttFallbackModel {
    let (name, _language) = parse_model_string(model);
    FallbackModel { model: name.to_owned(), extra_kwargs: None }
}

/// `aligned_transcript` requires every link of the chain to be aligned.
pub fn capabilities(
    model: Option<&str>,
    fallback: &[FallbackModel],
    extra_kwargs: Option<&Map<String, Value>>,
) -> SttCapabilities {
    let aligned = aligned_transcript_for_model(model)
        && fallback.iter().all(|f| aligned_transcript_for_model(Some(&f.model)));
    SttCapabilities {
        streaming: true,
        interim_results: true,
        diarization: diarization_enabled(extra_kwargs),
        // `Literal["word", False]` -- the gateway never reports chunk-level alignment.
        aligned_transcript: if aligned { AlignedTranscript::Word } else { AlignedTranscript::None },
        offline_recognize: false,
        keyterms: keyterms_extra_for_model(model, None, &[]).is_some(),
        chat_context: supports_agent_context_carryover(model),
    }
}

/// What `_resolve_vad_for_model` decides; the host performs it.
#[lk_agents_abi::host(wire_enum)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadResolution {
    /// Keep the caller's instance (Speechmatics, which endpoints client-side).
    Keep,
    /// Ignore the caller's instance and warn: the model endpoints server-side.
    DropAndWarn,
    /// Speechmatics with no VAD supplied: construct the default local one.
    CreateDefault,
    /// Nothing to do.
    None,
}

#[lk_agents_abi::host(export(name = "stt_resolve_vad_for_model", kw_only = "has_vad", defaults(has_vad = false)))]
pub fn resolve_vad_for_model(model: Option<&str>, has_vad: bool) -> VadResolution {
    let is_speechmatics = model.is_some_and(|m| m.starts_with("speechmatics/"));
    if has_vad && !is_speechmatics {
        return VadResolution::DropAndWarn;
    }
    if is_speechmatics && !has_vad {
        return VadResolution::CreateDefault;
    }
    if has_vad { VadResolution::Keep } else { VadResolution::None }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(&str, Value)]) -> Map<String, Value> {
        pairs.iter().map(|(k, v)| ((*k).to_owned(), v.clone())).collect()
    }

    // Catches: Python truthiness lost (`false`, `0`, `""`, `[]`, Speechmatics' `"none"` in any case
    // must not advertise diarization).
    #[test]
    fn only_a_truthy_diarization_flag_enables_diarization() {
        for (key, value, want) in [
            ("diarize", json!(false), false),
            ("speaker_labels", json!(false), false),
            ("diarize", json!(0), false),
            ("diarize", json!(""), false),
            ("speaker_labels", json!([]), false),
            ("diarization", json!("none"), false),
            ("diarization", json!("NONE"), false),
            ("diarize", json!(true), true),
            ("diarize", json!(1), true),
            ("diarization", json!("speaker"), true),
        ] {
            assert_eq!(diarization_enabled(Some(&map(&[(key, value.clone())]))), want, "{key}={value}");
        }
        assert!(!diarization_enabled(None));
        assert!(!diarization_enabled(Some(&Map::new())));
    }

    // Catches: advertising `chat_context` for a model that cannot consume it, or the reverse.
    #[test]
    fn only_the_assemblyai_u3_pro_family_carries_agent_context() {
        assert!(supports_agent_context_carryover(Some("assemblyai/u3-rt-pro")));
        assert!(supports_agent_context_carryover(Some("assemblyai/universal-3-5-pro")));
        assert!(!supports_agent_context_carryover(Some("assemblyai/universal-streaming")));
        assert!(!supports_agent_context_carryover(Some("deepgram/nova-3")));
        assert!(!supports_agent_context_carryover(None));
        assert!(capabilities(Some("assemblyai/u3-rt-pro"), &[], None).chat_context);
        assert!(!capabilities(Some("deepgram/nova-3"), &[], None).chat_context);
    }

    // Catches: claiming word alignment for "auto" or an unlisted model.
    #[test]
    fn auto_and_unknown_models_are_not_word_aligned() {
        assert!(aligned_transcript_for_model(Some("deepgram/nova-3")));
        assert!(!aligned_transcript_for_model(Some("auto")));
        assert!(!aligned_transcript_for_model(Some("inworld/inworld-stt-1")));
        assert!(!aligned_transcript_for_model(None));
    }

    // Catches: computing `aligned_transcript` from the primary model alone.
    #[test]
    fn one_unaligned_fallback_disables_alignment_for_the_whole_chain() {
        let aligned = FallbackModel { model: "cartesia/ink-whisper".into(), extra_kwargs: None };
        let unaligned = FallbackModel { model: "inworld/inworld-stt-1".into(), extra_kwargs: None };
        assert_eq!(
            capabilities(Some("deepgram/nova-3"), std::slice::from_ref(&aligned), None)
                .aligned_transcript,
            AlignedTranscript::Word
        );
        assert_eq!(
            capabilities(Some("deepgram/nova-3"), &[unaligned], None).aligned_transcript,
            AlignedTranscript::None
        );
        assert_eq!(
            capabilities(Some("deepgram/nova-3"), &[], None).aligned_transcript,
            AlignedTranscript::Word
        );
    }

    // Catches: a bare-string `keyterm` splatted into characters.
    #[test]
    fn a_bare_string_keyterm_becomes_a_one_element_list() {
        let extra = map(&[("keyterm", json!("livekit"))]);
        let got =
            keyterms_extra_for_model(Some("deepgram/nova-3"), Some(&extra), &["agents".into()])
                .unwrap();
        assert_eq!(got["keyterm"], json!(["livekit", "agents"]));
    }

    // Catches: dropping the order-preserving dedup.
    #[test]
    fn keyterms_are_deduplicated_in_first_seen_order() {
        let extra = map(&[("keyterm", json!(["b", "a"]))]);
        let got = keyterms_extra_for_model(
            Some("deepgram/nova-3"),
            Some(&extra),
            &["a".into(), "c".into(), "c".into()],
        )
        .unwrap();
        assert_eq!(got["keyterm"], json!(["b", "a", "c"]));
    }

    // Catches: rebuilding Speechmatics' additional_vocab from scratch (losing `sounds_like`).
    #[test]
    fn speechmatics_keeps_existing_vocabulary_entries_verbatim() {
        let extra = map(&[(
            "additional_vocab",
            json!([{"content": "gnocchi", "sounds_like": ["nyohki"]}]),
        )]);
        let got = keyterms_extra_for_model(
            Some("speechmatics/enhanced"),
            Some(&extra),
            &["gnocchi".into(), "livekit".into()],
        )
        .unwrap();
        assert_eq!(
            got["additional_vocab"],
            json!([{"content": "gnocchi", "sounds_like": ["nyohki"]}, {"content": "livekit"}]),
            "the already-present term is not re-added"
        );
    }

    // Catches: advertising keyterms (a `Some(empty map)`) for a provider that has none.
    #[test]
    fn a_provider_without_keyterm_prompting_returns_none() {
        assert!(
            keyterms_extra_for_model(Some("cartesia/ink-whisper"), None, &["x".into()]).is_none()
        );
        assert!(keyterms_extra_for_model(None, None, &[]).is_none());
        assert!(keyterms_extra_for_model(Some("assemblyai/u3-rt-pro"), None, &[]).is_some());
    }

    // Catches: `find(':')` instead of `rfind(':')`, or dropping the trailing-colon case.
    #[test]
    fn the_language_suffix_is_taken_from_the_last_colon() {
        assert_eq!(parse_model_string("deepgram/nova-3:en"), ("deepgram/nova-3", Some("en")));
        assert_eq!(parse_model_string("deepgram/nova-3"), ("deepgram/nova-3", None));
        assert_eq!(parse_model_string("a:b:c"), ("a:b", Some("c")));
        assert_eq!(parse_model_string("deepgram/nova-3:"), ("deepgram/nova-3", Some("")));
    }

    // Catches: ignoring a caller-supplied VAD, or constructing one for a server-endpointing model.
    #[test]
    fn only_speechmatics_uses_a_client_side_vad() {
        assert_eq!(
            resolve_vad_for_model(Some("speechmatics/enhanced"), false),
            VadResolution::CreateDefault
        );
        assert_eq!(resolve_vad_for_model(Some("speechmatics/enhanced"), true), VadResolution::Keep);
        assert_eq!(
            resolve_vad_for_model(Some("deepgram/nova-3"), true),
            VadResolution::DropAndWarn
        );
        assert_eq!(resolve_vad_for_model(Some("deepgram/nova-3"), false), VadResolution::None);
        assert_eq!(resolve_vad_for_model(None, false), VadResolution::None);
    }

    // Catches: keeping the `:language` suffix on a fallback entry.
    #[test]
    fn a_fallback_given_as_a_string_drops_its_language_suffix() {
        assert_eq!(
            normalize_fallback_str("deepgram/nova-3:fr"),
            FallbackModel { model: "deepgram/nova-3".into(), extra_kwargs: None }
        );
    }
}
