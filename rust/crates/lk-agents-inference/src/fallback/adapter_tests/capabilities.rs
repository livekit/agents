use super::*;

// Catches: merging STT capabilities with the wrong quantifier (`stt:79-93`), or inverting the
// `aligned_transcript` test.
#[test]
fn merging_stt_capabilities_uses_a_different_rule_per_field() {
    let rich = SttCapabilities {
        streaming: true,
        interim_results: true,
        diarization: true,
        aligned_transcript: AlignedTranscript::Word,
        offline_recognize: true,
        keyterms: true,
        chat_context: false,
    };
    let plain = SttCapabilities {
        streaming: true,
        interim_results: false,
        diarization: true,
        aligned_transcript: AlignedTranscript::Chunk,
        offline_recognize: true,
        keyterms: false,
        chat_context: true,
    };

    let merged = crate::fallback::stt::merge_capabilities(&[rich, plain]);
    assert!(merged.streaming, "the adapter always streams; every entry is required to");
    assert!(!merged.interim_results, "one entry without them loses them for the whole chain");
    assert!(merged.diarization, "both have it");
    assert_eq!(
        merged.aligned_transcript,
        AlignedTranscript::Word,
        "the primary's granularity, because every entry has some alignment"
    );
    assert!(merged.keyterms, "any");
    assert!(merged.chat_context, "any");
    assert!(merged.offline_recognize, "not merged -- the reference takes the field default");

    let unaligned = SttCapabilities { aligned_transcript: AlignedTranscript::None, ..plain };
    assert_eq!(
        crate::fallback::stt::merge_capabilities(&[rich, unaligned]).aligned_transcript,
        AlignedTranscript::None,
        "one entry with no alignment drops it for everyone"
    );
}

// Catches: one quantifier for both TTS capabilities (`streaming` any, `aligned_transcript` all,
// `tts:92-95`).
#[test]
fn merging_tts_capabilities_is_any_for_streaming_and_all_for_alignment() {
    let duplex = TtsCapabilities { streaming: true, aligned_transcript: true };
    let batch = TtsCapabilities { streaming: false, aligned_transcript: true };
    let unaligned = TtsCapabilities { streaming: true, aligned_transcript: false };

    let merged = crate::fallback::tts::merge_capabilities(&[duplex, batch]);
    assert!(merged.streaming, "any -- see the function's own docs for why this over-promises");
    assert!(merged.aligned_transcript, "both align");

    assert!(
        !crate::fallback::tts::merge_capabilities(&[duplex, unaligned]).aligned_transcript,
        "all"
    );
    assert!(!crate::fallback::tts::merge_capabilities(&[batch]).streaming, "no entry streams");
}

// Catches: `FallbackLlm` inheriting `Llm::model()`'s `"unknown"` default instead of following the
// active entry (`_evaluate_with_llm` excludes `temperature` on gpt-5).
#[tokio::test]
async fn the_chain_reports_the_active_entrys_model_not_its_own() {
    // The primary raises on `chat()`, so the failover is real.
    let primary = FakeLlm::build("p", Some("gpt-5"), None, false);
    let backup = FakeLlm::build("b", Some("gpt-4o-mini"), Some(vec![chunk("hi")]), false);
    let fb = llm_chain(vec![primary.clone(), backup.clone()]);

    assert_eq!(fb.model(), "gpt-5", "before any traffic the primary's model is the answer");

    let mut stream =
        within("chat", fb.chat(&ChatRequest::default())).await.expect("the backup serves the turn");
    while within("next", stream.next()).await.expect("drain").is_some() {}

    assert_eq!(fb.model(), "gpt-4o-mini", "after a failover it must follow the active entry");
}
