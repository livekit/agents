use lk_agents_abi::TtsCapabilities;

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
