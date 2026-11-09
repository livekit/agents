use super::*;
use serde_json::{Value, json};

fn msg(json: Value) -> Incoming {
    crate::stt::wire::decode(&json.to_string()).unwrap()
}

/// The final transcript, which is never `events[0]` -- the inferred START_OF_SPEECH is.
fn final_of(p: &Processed) -> &SpeechEvent {
    p.events.iter().find(|e| e.kind == SpeechEventType::FinalTranscript).expect("a final")
}

fn kinds(p: &Processed) -> Vec<SpeechEventType> {
    p.events.iter().map(|e| e.kind).collect()
}

// Catches: dropping an empty final transcript along with the empty interims.
#[test]
fn an_empty_final_transcript_still_ends_the_turn() {
    let mut m = TranscriptMachine::new("stt_request_x", None);
    m.record_audio(1.5);
    m.process(msg(json!({"type": "start_of_speech"})));
    let out = m.process(msg(json!({"type": "final_transcript", "transcript": ""})));
    assert_eq!(
        kinds(&out),
        [
            SpeechEventType::RecognitionUsage,
            SpeechEventType::FinalTranscript,
            SpeechEventType::EndOfSpeech
        ]
    );
    assert!(out.end_of_speech);
    assert!(!m.is_speaking());

    // ...while an empty interim produces nothing at all.
    let mut m = TranscriptMachine::new("r", None);
    assert!(
        m.process(msg(json!({"type": "interim_transcript", "transcript": ""}))).events.is_empty()
    );
}

// Catches: swallowing the gateway's `error` frame, or copying its message (which can carry
// customer content) into the error the application logs; the code is the diagnostic.
#[test]
fn a_gateway_error_frame_becomes_a_terminal_error() {
    let mut m = TranscriptMachine::new("r", None);
    let out = m.process(msg(json!({"type": "error", "message": "user words", "code": 400})));
    let err = out.error.expect("an error");
    assert_eq!(err.status_code, Some(400));
    assert_eq!(err.message, "LiveKit Inference STT returned an error");
    assert!(!err.retryable, "a 400 is not retried");
    assert!(out.events.is_empty());

    // ...and one with no message still reports, rather than being dropped for lack of detail.
    let out = m.process(msg(json!({"type": "error"})));
    let err = out.error.expect("an error");
    assert_eq!(err.status_code, Some(-1));
}

// Catches: emitting START_OF_SPEECH twice (missing `if self._speaking: return`).
#[test]
fn start_of_speech_is_emitted_once_per_turn() {
    let mut m = TranscriptMachine::new("r", None);
    assert_eq!(
        kinds(&m.process(msg(json!({"type": "start_of_speech"})))),
        [SpeechEventType::StartOfSpeech]
    );
    assert!(m.process(msg(json!({"type": "start_of_speech"}))).events.is_empty());
    let out = m.process(msg(json!({"type": "interim_transcript", "transcript": "hi"})));
    assert_eq!(kinds(&out), [SpeechEventType::InterimTranscript], "no second onset");
}

// Catches: the first transcript not opening the turn, or opening it after the transcript.
#[test]
fn the_first_transcript_opens_the_turn_when_the_provider_sends_no_onset() {
    let mut m = TranscriptMachine::new("r", None);
    let out = m.process(msg(json!({"type": "interim_transcript", "transcript": "hel"})));
    assert_eq!(kinds(&out), [SpeechEventType::StartOfSpeech, SpeechEventType::InterimTranscript]);
}

// Catches: forwarding a preflight transcript outside a turn.
#[test]
fn a_preflight_outside_a_turn_is_dropped() {
    let mut m = TranscriptMachine::new("r", None);
    assert!(
        m.process(msg(json!({"type": "preflight_transcript", "transcript": "hi"})))
            .events
            .is_empty()
    );
    m.process(msg(json!({"type": "start_of_speech"})));
    let out = m.process(msg(json!({"type": "preflight_transcript", "transcript": "hi"})));
    assert_eq!(kinds(&out), [SpeechEventType::PreflightTranscript]);
    // ...and an empty one is dropped even inside a turn.
    assert!(
        m.process(msg(json!({"type": "preflight_transcript", "transcript": ""}))).events.is_empty()
    );
}

// Catches: a zero-duration usage record, or not resetting the accumulator.
#[test]
fn the_usage_record_is_emitted_once_per_turn_and_then_reset() {
    let mut m = TranscriptMachine::new("r", None);
    m.record_audio(0.05);
    m.record_audio(0.05);
    let out = m.process(msg(json!({"type": "final_transcript", "transcript": "hi"})));
    let usage = out.events.iter().find(|e| e.kind == SpeechEventType::RecognitionUsage).unwrap();
    assert!((usage.recognition_usage.as_ref().unwrap().audio_duration.unwrap() - 0.1).abs() < 1e-9);

    let out = m.process(msg(json!({"type": "final_transcript", "transcript": "again"})));
    assert!(
        !out.events.iter().any(|e| e.kind == SpeechEventType::RecognitionUsage),
        "no audio was pushed since the last final"
    );
}

// Catches: `end_time` other than `start + duration`, or word confidence defaulting to 1.0.
#[test]
fn timings_add_up_and_word_confidence_defaults_to_zero() {
    let mut m = TranscriptMachine::new("r", None);
    let out = m.process(msg(json!({
        "type": "final_transcript",
        "transcript": "hi there",
        "start": 1.0,
        "duration": 2.0,
        "words": [{"word": "hi", "start": 1.0, "end": 1.5}]
    })));
    let data = out.events.iter().find(|e| e.kind == SpeechEventType::FinalTranscript).unwrap();
    let alt = data.best_alternative().unwrap();
    assert_eq!(alt.start_time, 1.0);
    assert_eq!(alt.end_time, 3.0, "start + duration");
    assert_eq!(alt.confidence, 1.0, "the utterance default");
    let word = &alt.words.as_ref().unwrap()[0];
    assert_eq!((word.start_time, word.end_time), (Some(1.0), Some(1.5)));
    assert_eq!(word.confidence, Some(0.0), "the *word* default differs from the utterance's");
    assert_eq!(word.start_time_offset, Some(0.0));
}

// Catches: R2b on the metadata guard.
#[test]
fn an_empty_extra_object_is_not_metadata() {
    let mut m = TranscriptMachine::new("r", None);
    let out = m.process(msg(json!({"type": "final_transcript", "transcript": "x", "extra": {}})));
    assert_eq!(final_of(&out).best_alternative().unwrap().metadata, None);
    let out =
        m.process(msg(json!({"type": "final_transcript", "transcript": "x", "extra": {"a": 1}})));
    assert!(final_of(&out).best_alternative().unwrap().metadata.is_some());
}

// Catches: using the connection's request id when the message carries its own.
#[test]
fn a_message_request_id_wins_over_the_connections() {
    let mut m = TranscriptMachine::new("stt_request_local", None);
    let out = m.process(msg(json!({"type": "final_transcript", "transcript": "x"})));
    assert_eq!(final_of(&out).request_id, "stt_request_local");
    let out = m.process(msg(
        json!({"type": "final_transcript", "transcript": "y", "request_id": "gw_42"}),
    ));
    assert_eq!(final_of(&out).request_id, "gw_42");
}

// Catches: R2b on the language fallback.
#[test]
fn an_empty_configured_language_falls_back_to_english() {
    let mut m = TranscriptMachine::new("r", Some(""));
    let out = m.process(msg(json!({"type": "final_transcript", "transcript": "x"})));
    assert_eq!(final_of(&out).best_alternative().unwrap().language.as_str(), "en");

    let mut m = TranscriptMachine::new("r", Some("fr"));
    let out = m.process(msg(json!({"type": "final_transcript", "transcript": "x"})));
    assert_eq!(final_of(&out).best_alternative().unwrap().language.as_str(), "fr");
}
