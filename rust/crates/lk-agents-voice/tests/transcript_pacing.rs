use lk_agents_voice::transcription::synchronizer::{PaceEffect, PaceEvent, Pacer, PacerOptions};
use std::time::Duration;

#[test]
fn expressive_markup_adds_no_spoken_pacing_delay() {
    let text = concat!(
        "<expr type=\"expression\" label=\"speak with warm surprise and bright energy\"/> ",
        "Hello there my friend! ",
        "<expr type=\"sound\" label=\"laugh\"/> ",
        "<expr type=\"expression\" label=\"speak calmly and evenly, unhurried\"/> ",
        "How are you today?"
    );
    let mut pacer = Pacer::new(PacerOptions::default());
    for event in [
        PaceEvent::Text { raw: text.into(), start_time: None, end_time: None },
        PaceEvent::TextEnd,
        PaceEvent::Audio { secs: 3.0 },
        PaceEvent::AudioEnd,
    ] {
        pacer = pacer.step(event).0;
    }
    let (_, effects) = pacer.step(PaceEvent::Position(Duration::from_secs(3)));
    let shown: String = effects
        .iter()
        .filter_map(|effect| match effect {
            PaceEffect::Forward { word, .. } => Some(word.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(shown, text);
}

#[test]
fn a_timed_transcripts_last_word_is_shown_before_text_input_ends() {
    let mut pacer = Pacer::new(PacerOptions::default());
    for event in [
        PaceEvent::Audio { secs: 0.4 },
        PaceEvent::AudioEnd,
        PaceEvent::Text { raw: "It is".into(), start_time: Some(0.0), end_time: Some(0.2) },
        PaceEvent::Text { raw: " sunny".into(), start_time: Some(0.2), end_time: Some(0.4) },
    ] {
        pacer = pacer.step(event).0;
    }
    let (_, effects) = pacer.step(PaceEvent::Position(Duration::from_millis(400)));
    let shown: String = effects
        .iter()
        .filter_map(|effect| match effect {
            PaceEffect::Forward { word, .. } => Some(word.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(shown, "It is sunny");
}
