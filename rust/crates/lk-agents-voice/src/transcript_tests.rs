use lk_agents_abi::TranscriptChunk;
use lk_agents_voice::transcription::effects::TranscriptSyncEffect;
use lk_agents_voice::transcription::machine::lifecycle::{
    TranscriptPacerOptions, TranscriptSync, TranscriptSyncEvent,
};
use std::time::Duration;

fn synchronized() -> TranscriptSync {
    let mut sync = TranscriptSync::new(TranscriptPacerOptions::default());
    sync.handle(TranscriptSyncEvent::AttachText);
    sync.handle(TranscriptSyncEvent::AttachmentChanged { audio: Some(true), text: Some(true) });
    sync
}

fn capture(raw: &str, start_time: Option<f64>, end_time: Option<f64>) -> TranscriptSyncEvent {
    TranscriptSyncEvent::Capture {
        text: TranscriptChunk { start_time, end_time, ..TranscriptChunk::new(raw) },
        aligned: start_time.is_some() || end_time.is_some(),
    }
}

fn words(effects: &[TranscriptSyncEffect]) -> String {
    effects
        .iter()
        .filter_map(|effect| match effect {
            TranscriptSyncEffect::PublishWord { word, .. } => Some(word.as_str()),
            _ => None,
        })
        .collect()
}

#[test]
fn expressive_markup_adds_no_spoken_pacing_delay() {
    let text = concat!(
        "<expr type=\"expression\" label=\"speak with warm surprise and bright energy\"/> ",
        "Hello there my friend! ",
        "<expr type=\"sound\" label=\"laugh\"/> ",
        "<expr type=\"expression\" label=\"speak calmly and evenly, unhurried\"/> ",
        "How are you today?"
    );
    let mut sync = synchronized();
    for event in [
        capture(text, None, None),
        TranscriptSyncEvent::FlushText,
        TranscriptSyncEvent::AudioCaptureStarted { secs: 3.0 },
        TranscriptSyncEvent::AudioEnded,
    ] {
        sync.handle(event);
    }
    let effects = sync.handle(TranscriptSyncEvent::PlaybackProgressed(Duration::from_secs(3)));
    assert_eq!(words(&effects), text);
}

#[test]
fn a_timed_transcripts_last_word_is_shown_before_text_input_ends() {
    let mut sync = synchronized();
    for event in [
        TranscriptSyncEvent::AudioCaptureStarted { secs: 0.4 },
        TranscriptSyncEvent::AudioEnded,
        capture("It is", Some(0.0), Some(0.2)),
        capture(" sunny", Some(0.2), Some(0.4)),
    ] {
        sync.handle(event);
    }
    let effects = sync.handle(TranscriptSyncEvent::PlaybackProgressed(Duration::from_millis(400)));
    assert_eq!(words(&effects), "It is sunny");
}

// The owner must publish the final paced words and flush before reporting heard text, with no
// second terminal output if a sink repeats its finish notification.
#[test]
fn playback_finish_publishes_words_before_flush_and_heard_once() {
    let mut sync = synchronized();
    for event in [
        capture("hello world", Some(0.0), Some(1.0)),
        TranscriptSyncEvent::FlushText,
        TranscriptSyncEvent::AudioCaptureStarted { secs: 1.0 },
        TranscriptSyncEvent::AudioEnded,
    ] {
        sync.handle(event);
    }
    let finish = || TranscriptSyncEvent::PlaybackFinished {
        position: Duration::from_secs(1),
        interrupted: false,
    };
    let effects = sync.handle(finish());
    assert_eq!(words(&effects), "hello world");
    let flush = effects
        .iter()
        .position(|e| matches!(e, TranscriptSyncEffect::FlushTranscript))
        .expect("completed transcript flushes");
    let report = effects
        .iter()
        .position(|e| matches!(e, TranscriptSyncEffect::ReportHeard(_)))
        .expect("completed playback reports what was heard");
    assert!(effects[..flush].iter().any(|e| matches!(e, TranscriptSyncEffect::PublishWord { .. })));
    assert!(flush < report);
    let TranscriptSyncEffect::ReportHeard(heard) = &effects[report] else { unreachable!() };
    assert_eq!(heard.text, "hello world");
    assert!(heard.complete);
    assert!(sync.handle(finish()).is_empty());
}

// Closing must deliver pending transcript output before stopping its physical forwarder.
#[test]
fn closing_flushes_transcript_before_sender_closes_and_waits_once() {
    let mut sync = synchronized();
    sync.handle(capture("hello world", None, None));
    sync.handle(TranscriptSyncEvent::AudioCaptureStarted { secs: 1.0 });
    let effects = sync.handle(TranscriptSyncEvent::Close);
    let flush = effects
        .iter()
        .position(|e| matches!(e, TranscriptSyncEffect::FlushTranscript))
        .expect("closing flushes the live transcript");
    let closed = effects
        .iter()
        .position(|e| matches!(e, TranscriptSyncEffect::CloseSender))
        .expect("closing ends forwarding admission");
    let wait = effects
        .iter()
        .position(|e| matches!(e, TranscriptSyncEffect::BeginWait))
        .expect("closing waits for forwarding");
    assert!(flush < closed && closed < wait);
    assert!(sync.handle(TranscriptSyncEvent::Close).is_empty());
}
