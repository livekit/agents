use super::provider::ProviderMetrics;
use lk_agents_abi::metrics::provider::ProviderMetrics as _;
use lk_agents_abi::{SpeechEvent, SpeechEventType};

/// Streaming STT reports audio usage without stream-lifetime latency. https://github.com/livekit/agents/pull/1927
#[test]
fn streaming_recognition_never_reports_the_sessions_lifetime_as_request_duration() {
    for timestamp in [1.0, 3_600.0, 86_400.0] {
        let mut event = SpeechEvent::new(SpeechEventType::RecognitionUsage);
        event.recognition_usage = Some(Box::new(lk_agents_abi::stt::events::RecognitionUsage {
            audio_duration: Some(2.5),
            ..Default::default()
        }));
        let metrics = ProviderMetrics::stt_stream_metrics(
            event,
            timestamp,
            "stt".into(),
            "model".into(),
            "provider".into(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(metrics.duration, None);
        assert_eq!(metrics.audio_duration, Some(2.5));
        assert!(metrics.streamed);
    }
}

/// A streamed synthesis cut before or during its audio is metered once, as cancelled, for all the
/// text sent to the provider. https://github.com/livekit/agents/issues/3397
#[test]
fn cancelled_streamed_synthesis_counts_all_submitted_text() {
    use super::provider::TtsMetricState;
    let sent = "Sure, let me check. One moment please.";
    let frame = || lk_agents_abi::voice::rtc::silence_frame(240, 24_000, 1);
    for audio_before_cut in [0, 2] {
        let mut fold = TtsMetricState::new(true, "tts".into(), None);
        for _ in 0..audio_before_cut {
            fold.observe(&lk_agents_abi::SynthesizedAudio::new(frame(), "req"), || 0.1);
        }
        let record = fold.finish(2.0, 1.0, sent, 9.0, true).expect("the cut synthesis is metered");
        assert!(record.cancelled);
        assert_eq!(record.characters_count, sent.chars().count() as u32);
        assert!(fold.finish(3.0, 1.0, "", 9.0, true).is_none(), "metered once");
    }
}
