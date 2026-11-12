use super::*;
use crate::transcription::SpeechViews;

fn stamp(word: &str, start: f64, end: f64) -> TranscriptChunk {
    TranscriptChunk { start_time: Some(start), end_time: Some(end), ..TranscriptChunk::new(word) }
}

/// `(text, start, end, spoken)` of every piece the transcript is shown, stamps then the end.
fn aligned(
    reply: &str,
    stamps: &[TranscriptChunk],
) -> Vec<(String, Option<f64>, Option<f64>, bool)> {
    let said = Visible::of(reply, &SpeechViews::default());
    let mut state = AlignedTranscript::default();
    let mut effects: Vec<_> = stamps.iter().flat_map(|s| state.word(s, &said)).collect();
    effects.extend(state.end(&said));
    effects
        .into_iter()
        .flat_map(|e| match e {
            AlignedTranscriptEffect::CaptureTimed(c) => vec![c],
            AlignedTranscriptEffect::Capture(cs) => cs,
            _ => Vec::new(),
        })
        .map(|c| (c.text, c.start_time, c.end_time, c.spoken))
        .collect()
}

fn shown(pieces: &[(String, Option<f64>, Option<f64>, bool)]) -> String {
    pieces.iter().map(|p| p.0.as_str()).collect()
}

#[test]
fn the_transcript_keeps_its_text_and_takes_the_voices_timings() {
    let reply = "Say **very** important 🎉";
    let got = aligned(
        reply,
        &[stamp("Say", 0.0, 0.2), stamp("very", 0.2, 0.4), stamp("important", 0.4, 0.8)],
    );
    assert_eq!(shown(&got), reply);
    let at = |text: &str| got.iter().find(|p| p.0 == text).cloned().expect(text);
    assert_eq!(at("very"), ("very".into(), Some(0.2), Some(0.4), true));
    // syntax shows the moment its word starts, at no cost
    assert_eq!(at("**"), ("**".into(), Some(0.2), Some(0.2), false));
    // what no stamp reached is shown at the end
    assert_eq!(got.last().map(|p| p.0.as_str()), Some("🎉"));
}

#[test]
fn a_word_the_voice_expanded_takes_its_span_and_nothing_drifts() {
    let got = aligned(
        "It costs $5 now",
        &[
            stamp("It", 0.0, 0.1),
            stamp("costs", 0.1, 0.3),
            stamp("five", 0.3, 0.5),
            stamp("dollars", 0.5, 0.8),
            stamp("now", 0.8, 1.0),
        ],
    );
    assert_eq!(shown(&got), "It costs $5 now");
    let five = got.iter().find(|p| p.0.contains("$5")).expect("$5");
    assert_eq!((five.1, five.2), (Some(0.3), Some(0.8)));
    let now = got.iter().find(|p| p.0.contains("now")).expect("now");
    assert_eq!((now.1, now.2), (Some(0.8), Some(1.0)));
}

#[test]
fn a_word_the_voice_rewrote_keeps_its_place() {
    let got = aligned(
        "Buy 15 apples",
        &[stamp("Buy", 0.0, 0.2), stamp("fifteen", 0.2, 0.6), stamp("apples", 0.6, 0.9)],
    );
    let fifteen = got.iter().find(|p| p.0.contains("15")).expect("15");
    assert_eq!((fifteen.1, fifteen.2), (Some(0.2), Some(0.6)));
}

#[test]
fn stamps_that_never_match_map_by_position_rather_than_stall() {
    let words: Vec<_> = (0..8).map(|i| stamp("zzz", f64::from(i), f64::from(i) + 1.0)).collect();
    let got = aligned("one two three four five six seven eight", &words);
    assert_eq!(shown(&got), "one two three four five six seven eight");
    // held stamps beyond the lookahead release words in order
    assert_eq!(got[0].1, Some(0.0));
}

#[test]
fn an_unstamped_segment_shows_the_transcript_whole() {
    let said = Visible::of("**Hi** there", &SpeechViews::default());
    let mut state = AlignedTranscript::default();
    let effects = state.end(&said);
    assert!(matches!(effects[0], AlignedTranscriptEffect::WarnUnaligned));
    let AlignedTranscriptEffect::Capture(chunks) = &effects[1] else { panic!() };
    assert_eq!(chunks, &said.shown);
    assert!(state.end(&said).is_empty(), "closing flushes once");
}
