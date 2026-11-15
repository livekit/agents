use super::*;
use SpeechEventType::{
    EndOfSpeech, EndOfSpeechPending, FinalTranscript, InterimTranscript, StartOfSpeech,
};
use lk_agents_abi::tokenize::language::LanguageCode;

fn heard(text: &str, is_final: bool) -> Transcript {
    Transcript { r#final: is_final, ..Transcript::new("en", text) }
}

fn kinds(out: &[SpeechEvent]) -> Vec<SpeechEventType> {
    out.iter().map(|e| e.kind).collect()
}

// Catches: a provider without its own VAD never starting or ending speech, or starting it twice.
#[test]
fn transcripts_start_the_speech_once_and_a_final_ends_it() {
    let mut speech = SpeechNormalizer::new(false, false);
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
    let mut speech = SpeechNormalizer::new(true, false);
    let mut out = Vec::new();
    speech.started(None, &mut out);
    speech.started(None, &mut out);
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
    let mut speech = SpeechNormalizer::new(true, false);
    let mut out = Vec::new();
    speech.transcript(heard("uh", false), &mut out);
    speech.transcript(heard("", true), &mut out);
    assert_eq!(kinds(&out), [StartOfSpeech, InterimTranscript, InterimTranscript]);
    assert_eq!(out[2].alternatives[0].text, "");
    assert!(speech.speaking());

    // ...while one after a start no words followed ends that speech
    let mut speech = SpeechNormalizer::new(true, false);
    let mut out = Vec::new();
    speech.started(None, &mut out);
    assert!(speech.transcript(heard("", true), &mut out));
    assert_eq!(kinds(&out), [StartOfSpeech, FinalTranscript, EndOfSpeech]);
}

// Catches: a provider's end arriving before the final of its words, so the turn ends with no
// final and the final lands in a speech of its own, or a pause timing out the words it waits for.
#[test]
fn a_provider_end_waits_for_the_final_of_its_words() {
    let mut speech = SpeechNormalizer::new(true, false);
    let mut out = Vec::new();
    speech.transcript(heard("book a flight to Par", false), &mut out);
    assert!(!speech.ended(&mut out), "the end waits for the final");
    assert!(speech.speaking(), "however long the final takes");
    assert!(speech.transcript(heard("book a flight to Paris", true), &mut out));
    assert_eq!(
        kinds(&out),
        [StartOfSpeech, InterimTranscript, EndOfSpeechPending, FinalTranscript, EndOfSpeech]
    );
    assert!(out.iter().all(|e| e.utterance == Some(1)), "the late final is the same utterance's");

    // a speech no words were heard in ends at once, and the next words are a new utterance
    let mut out = Vec::new();
    speech.started(None, &mut out);
    assert!(speech.ended(&mut out));
    speech.transcript(heard("on Monday", true), &mut out);
    assert_eq!(kinds(&out), [StartOfSpeech, EndOfSpeech, StartOfSpeech, FinalTranscript]);
    let utterances: Vec<_> = out.iter().map(|e| e.utterance).collect();
    assert_eq!(utterances, [Some(2), Some(2), Some(3), Some(3)]);
}

// Catches: a provider that transcribes after the speech ends having its no-words end close the
// speech, so the words it then sends land in a speech of their own; or a provider without that
// capability holding the end of a speech it heard nothing in.
#[test]
fn a_no_words_end_waits_for_a_provider_that_transcribes_after_the_speech() {
    let held = || {
        let mut speech = SpeechNormalizer::new(true, true);
        let mut out = Vec::new();
        speech.started(None, &mut out);
        assert!(!speech.ended(&mut out), "the words are still owed");
        assert!(speech.speaking());
        (speech, out)
    };
    let utterances = |out: &[SpeechEvent]| out.iter().map(|e| e.utterance).collect::<Vec<_>>();

    // a final: emitted, then the end
    let (mut speech, mut out) = held();
    assert!(speech.transcript(heard("hello", true), &mut out));
    assert_eq!(kinds(&out), [StartOfSpeech, EndOfSpeechPending, FinalTranscript, EndOfSpeech]);
    assert_eq!(utterances(&out), [Some(1); 4]);

    // an empty final: noise, the speech ends
    let (mut speech, mut out) = held();
    assert!(speech.transcript(heard("", true), &mut out));
    assert_eq!(kinds(&out), [StartOfSpeech, EndOfSpeechPending, FinalTranscript, EndOfSpeech]);

    // the next start: the held speech ends first, its pending interim left that utterance's
    let (mut speech, mut out) = held();
    speech.transcript(heard("hel", false), &mut out);
    speech.started(None, &mut out);
    assert_eq!(
        kinds(&out),
        [StartOfSpeech, EndOfSpeechPending, InterimTranscript, EndOfSpeech, StartOfSpeech]
    );
    assert_eq!(utterances(&out), [Some(1), Some(1), Some(1), Some(1), Some(2)]);

    // a connection boundary: the pending interim made final, then the end
    let (mut speech, mut out) = held();
    speech.transcript(heard("hel", false), &mut out);
    assert!(speech.boundary(&mut out));
    assert_eq!(
        kinds(&out),
        [StartOfSpeech, EndOfSpeechPending, InterimTranscript, FinalTranscript, EndOfSpeech]
    );
    assert_eq!(out[3].alternatives[0].text, "hel");

    // words outside the provider's speech are of one it already ended: their final ends it
    let mut speech = SpeechNormalizer::new(true, true);
    let mut out = Vec::new();
    speech.transcript(heard("bye", false), &mut out);
    assert!(speech.transcript(heard("bye", true), &mut out));
    assert_eq!(kinds(&out), [StartOfSpeech, InterimTranscript, FinalTranscript, EndOfSpeech]);

    // once its words are final, the end is at once
    let mut speech = SpeechNormalizer::new(true, true);
    let mut out = Vec::new();
    speech.started(None, &mut out);
    speech.transcript(heard("hello", true), &mut out);
    assert!(speech.ended(&mut out));

    // a provider without the capability ends a no-words speech at once
    let mut speech = SpeechNormalizer::new(true, false);
    let mut out = Vec::new();
    speech.started(None, &mut out);
    assert!(speech.ended(&mut out));
    assert_eq!(kinds(&out), [StartOfSpeech, EndOfSpeech]);
}

// Catches: the next speech's start landing inside a held one, or its interim made final while
// its final may still come.
#[test]
fn a_new_speech_ends_the_held_one_first() {
    let mut speech = SpeechNormalizer::new(true, false);
    let mut out = Vec::new();
    speech.transcript(heard("hel", false), &mut out);
    speech.ended(&mut out);
    speech.started(None, &mut out);
    assert_eq!(
        kinds(&out),
        [StartOfSpeech, InterimTranscript, EndOfSpeechPending, EndOfSpeech, StartOfSpeech]
    );
    let utterances: Vec<_> = out.iter().map(|e| e.utterance).collect();
    assert_eq!(utterances, [Some(1), Some(1), Some(1), Some(1), Some(2)]);
}

// Catches: a dropped connection losing the words of its pending interim, or leaving the speech
// open so the next connection's words land inside it.
#[test]
fn a_connection_boundary_makes_the_interim_final_and_ends_the_speech_once() {
    for provider_vad in [false, true] {
        let mut speech = SpeechNormalizer::new(provider_vad, false);
        let mut out = Vec::new();
        speech.transcript(heard("hello", false), &mut out);
        assert!(speech.boundary(&mut out));
        assert!(!speech.boundary(&mut out));
        speech.transcript(heard("again", false), &mut out);
        assert_eq!(
            kinds(&out),
            [
                StartOfSpeech,
                InterimTranscript,
                FinalTranscript,
                EndOfSpeech,
                StartOfSpeech,
                InterimTranscript
            ]
        );
        assert_eq!(out[2].alternatives[0].text, "hello");
    }
}

// Catches: the provider's onset time dropped, so the speech starts when its message arrived.
#[test]
fn a_provider_start_carries_its_onset_time() {
    let mut speech = SpeechNormalizer::new(true, false);
    let mut out = Vec::new();
    speech.started(Some(12.5), &mut out);
    assert_eq!(out[0].speech_start_time, Some(12.5));
}

// Catches: a translating provider's source and target segments dropped from the transcript.
#[test]
fn a_transcript_keeps_its_translation() {
    let mut speech = SpeechNormalizer::new(false, false);
    let mut out = Vec::new();
    speech.transcript(
        Transcript {
            source_texts: Some(vec!["hola".into()]),
            target_languages: Some(vec![LanguageCode::new("en")]),
            target_texts: Some(vec!["hello".into()]),
            ..heard("hello", true)
        },
        &mut out,
    );
    let data = &out[1].alternatives[0];
    assert_eq!(data.source_texts.as_deref(), Some(&["hola".to_owned()][..]));
    assert_eq!(data.target_languages.as_deref(), Some(&[LanguageCode::new("en")][..]));
    assert_eq!(data.target_texts.as_deref(), Some(&["hello".to_owned()][..]));
}

// Catches: the provider's request id missing from the speech boundaries its words bound.
#[test]
fn the_speech_events_carry_the_request_id_of_its_words() {
    let mut speech = SpeechNormalizer::new(false, false);
    let mut out = Vec::new();
    speech.transcript(Transcript { request_id: "r1".into(), ..heard("hello", true) }, &mut out);
    assert_eq!(kinds(&out), [StartOfSpeech, FinalTranscript, EndOfSpeech]);
    assert!(out.iter().all(|e| e.request_id == "r1"));
}

// Catches: an empty final after a connection's promotion taken for words, or a final-only
// utterance after it numbered as the promoted one, so its words would replace them.
#[test]
fn a_final_only_utterance_after_a_promotion_is_its_own() {
    // the connection's request id is on every transcript: provenance, not the utterance
    let said =
        |text: &str, is_final| Transcript { request_id: "r1".into(), ..heard(text, is_final) };
    let mut speech = SpeechNormalizer::new(false, false);
    let mut out = Vec::new();
    speech.transcript(said("hel", false), &mut out);
    speech.boundary(&mut out);
    let promoted = out.len();
    assert!(!speech.transcript(said("", true), &mut out), "no speech is open");
    assert_eq!(out.len(), promoted, "an empty final says nothing");
    speech.transcript(said("yes", true), &mut out);
    assert_eq!(kinds(&out[promoted..]), [StartOfSpeech, FinalTranscript, EndOfSpeech]);
    assert_eq!(out[promoted - 2].utterance, Some(1));
    assert!(out[promoted..].iter().all(|e| e.utterance == Some(2)));
}
