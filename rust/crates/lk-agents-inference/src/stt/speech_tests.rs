use super::*;
use SpeechEventType::{EndOfSpeech, FinalTranscript, InterimTranscript, StartOfSpeech};

fn heard(text: &str, is_final: bool) -> Transcript {
    Transcript { r#final: is_final, ..Transcript::new("en", text) }
}

fn kinds(out: &[SpeechEvent]) -> Vec<SpeechEventType> {
    out.iter().map(|e| e.kind).collect()
}

// Catches: a provider without its own VAD never starting or ending speech, or starting it twice.
#[test]
fn transcripts_start_the_speech_once_and_a_final_ends_it() {
    let mut speech = Speech::new(false);
    let mut out = Vec::new();
    speech.transcript(heard("hel", false), &mut out);
    speech.transcript(heard("hello", false), &mut out);
    assert!(speech.transcript(heard("hello", true), &mut out), "the final ends the speech");
    assert_eq!(
        kinds(&out),
        [StartOfSpeech, InterimTranscript, InterimTranscript, FinalTranscript, EndOfSpeech]
    );
}

// Catches: a final mid-utterance ending the speech of a provider that endpoints itself, or its
// repeated start and end reaching the reader twice.
#[test]
fn a_provider_vad_owns_the_end_and_its_repeats_are_dropped() {
    let mut speech = Speech::new(true);
    let mut out = Vec::new();
    speech.started(&mut out);
    speech.started(&mut out);
    speech.transcript(heard("hello", true), &mut out);
    assert!(speech.speaking(), "a final is not the provider's end");
    speech.ended(&mut out);
    speech.ended(&mut out);
    assert_eq!(kinds(&out), [StartOfSpeech, FinalTranscript, EndOfSpeech]);
}

// Catches: an interim the provider withdrew left on screen, or the empty final ending a speech
// the provider still holds open.
#[test]
fn an_empty_final_retracts_the_interim_it_follows() {
    let mut speech = Speech::new(true);
    let mut out = Vec::new();
    speech.transcript(heard("uh", false), &mut out);
    speech.transcript(heard("", true), &mut out);
    assert_eq!(kinds(&out), [StartOfSpeech, InterimTranscript, InterimTranscript]);
    assert_eq!(out[2].alternatives[0].text, "");
    assert!(speech.speaking());

    // ...while one after a start no words followed ends that speech
    let mut speech = Speech::new(true);
    let mut out = Vec::new();
    speech.started(&mut out);
    assert!(speech.transcript(heard("", true), &mut out));
    assert_eq!(kinds(&out), [StartOfSpeech, FinalTranscript, EndOfSpeech]);
}

// Catches: a reconnect leaving the speech open, so the next connection's words land inside it.
#[test]
fn a_connection_boundary_ends_the_open_speech_once() {
    let mut speech = Speech::new(true);
    let mut out = Vec::new();
    speech.transcript(heard("hello", false), &mut out);
    assert!(speech.ended(&mut out));
    assert!(!speech.ended(&mut out));
    speech.transcript(heard("again", false), &mut out);
    assert_eq!(
        kinds(&out),
        [StartOfSpeech, InterimTranscript, EndOfSpeech, StartOfSpeech, InterimTranscript]
    );
}
