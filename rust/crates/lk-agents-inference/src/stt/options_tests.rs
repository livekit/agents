use super::*;
use serde_json::json;

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
    assert!(supports_agent_context_carryover(Some("assemblyai/universal-3-6-pro")));
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
    assert!(aligned_transcript_for_model(Some("assemblyai/universal-3-6-pro")));
    assert!(!aligned_transcript_for_model(Some("speechmatics/linden-1")));
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
    let got = keyterms_extra_for_model(Some("deepgram/nova-3"), Some(&extra), &["agents".into()])
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
    let extra =
        map(&[("additional_vocab", json!([{"content": "gnocchi", "sounds_like": ["nyohki"]}]))]);
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
    assert!(keyterms_extra_for_model(Some("cartesia/ink-whisper"), None, &["x".into()]).is_none());
    assert!(keyterms_extra_for_model(None, None, &[]).is_none());
    assert!(keyterms_extra_for_model(Some("speechmatics/linden-1"), None, &["x".into()]).is_none());
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
fn speechmatics_and_openai_use_a_client_side_vad() {
    assert_eq!(
        resolve_vad_for_model(Some("openai/gpt-live-transcribe"), false),
        VadResolution::CreateDefault
    );
    assert_eq!(
        resolve_vad_for_model(Some("openai/gpt-live-transcribe"), true),
        VadResolution::Keep
    );
    assert_eq!(
        resolve_vad_for_model(Some("speechmatics/enhanced"), false),
        VadResolution::CreateDefault
    );
    assert_eq!(resolve_vad_for_model(Some("speechmatics/enhanced"), true), VadResolution::Keep);
    assert_eq!(resolve_vad_for_model(Some("deepgram/nova-3"), true), VadResolution::DropAndWarn);
    assert_eq!(resolve_vad_for_model(Some("deepgram/nova-3"), false), VadResolution::None);
    assert_eq!(resolve_vad_for_model(None, false), VadResolution::None);
    // linden-1 detects turns server-side.
    assert_eq!(resolve_vad_for_model(Some("speechmatics/linden-1"), false), VadResolution::None);
    assert_eq!(
        resolve_vad_for_model(Some("speechmatics/linden-1"), true),
        VadResolution::DropAndWarn
    );
}

// Catches: keeping the `:language` suffix on a fallback entry.
#[test]
fn a_fallback_given_as_a_string_drops_its_language_suffix() {
    assert_eq!(
        normalize_fallback_str("deepgram/nova-3:fr"),
        FallbackModel { model: "deepgram/nova-3".into(), extra_kwargs: None }
    );
}

#[test]
fn keyterms_reach_openai_and_cartesia_turns_with_the_server_key() {
    for (model, key) in [("openai/gpt-live-transcribe", "keywords"), ("cartesia/ink-2", "keyterm")]
    {
        let extra = keyterms_extra_for_model(Some(model), None, &["LiveKit".into()]).unwrap();
        assert_eq!(extra[key], json!(["LiveKit"]));
        assert!(capabilities(Some(model), &[], None).keyterms);
    }
}

/// Speechmatics `linden-1` takes session keyterms as `additional_vocab` after the caller's own.
/// https://github.com/livekit/agents/pull/7853
#[test]
#[ignore = "https://github.com/livekit/agents/pull/7853 reproduces: linden-1 gets no keyterms (a_provider_without_keyterm_prompting_returns_none pins the old behaviour)"]
fn linden_takes_session_keyterms_as_additional_vocab() {
    let extra =
        map(&[("additional_vocab", json!([{"content": "gnocchi", "sounds_like": ["nyohki"]}]))]);
    let got = keyterms_extra_for_model(
        Some("speechmatics/linden-1"),
        Some(&extra),
        &["gnocchi".into(), "livekit".into()],
    )
    .expect("linden-1 accepts additional_vocab");
    assert_eq!(
        got["additional_vocab"],
        json!([{"content": "gnocchi", "sounds_like": ["nyohki"]}, {"content": "livekit"}])
    );
}
