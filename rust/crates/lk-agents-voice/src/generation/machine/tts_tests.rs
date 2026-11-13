use super::*;
use lk_agents_core::lexer::Vocabulary;

fn stamp(word: &str, start: f64, end: f64) -> TranscriptChunk {
    TranscriptChunk { start_time: Some(start), end_time: Some(end), ..TranscriptChunk::new(word) }
}

/// `(text, start, end, spoken)` of every piece the transcript is shown, stamps then the end.
fn aligned(
    reply: &str,
    stamps: &[TranscriptChunk],
) -> Vec<(String, Option<f64>, Option<f64>, bool)> {
    let said = Visible::of(reply, &Vocabulary::LiveKit);
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

/// Seed 283's "Wednesday 9-1 ☕": no word follows the emoji, so no stamp releases it. It is the
/// voice's text all the same and shows with the word before it, never untimed as the model's.
#[test]
fn silent_text_after_the_last_word_shows_with_it() {
    let got = aligned("Open 9-1 \u{2615}", &[stamp("Open", 0.0, 0.3), stamp("9-1", 0.3, 0.7)]);
    assert_eq!(got.last(), Some(&("\u{2615}".into(), Some(0.3), Some(0.3), false)));
    // a word no stamp reached is still shown at the end, untimed
    let got = aligned("Open now", &[stamp("Open", 0.0, 0.3)]);
    assert_eq!(got.last().map(|p| (p.0.trim(), p.1)), Some(("now", None)));
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
    let said = Visible::of("**Hi** there", &Vocabulary::LiveKit);
    let mut state = AlignedTranscript::default();
    let effects = state.end(&said);
    assert!(matches!(effects[0], AlignedTranscriptEffect::WarnUnaligned));
    let AlignedTranscriptEffect::Capture(chunks) = &effects[1] else { panic!() };
    assert_eq!(chunks, &said.shown);
    assert!(state.end(&said).is_empty(), "closing flushes once");
}

// Catches: a word the reply's deltas split (`Re` + `f`) read as two words, so the voice's stamp for
// the whole word matched neither and every word after was mapped by position (survey seed 152).
#[test]
fn a_word_split_across_deltas_is_one_word() {
    let mut said = Visible::of("Re", &Vocabulary::LiveKit);
    said.extend(Visible::of("f #A1 and **Up", &Vocabulary::LiveKit));
    said.extend(Visible::of("grade** now", &Vocabulary::LiveKit));
    let mut state = AlignedTranscript::default();
    let stamps = [
        stamp("Ref ", 0.0, 0.3),
        stamp("#A1 ", 0.3, 0.6),
        stamp("and ", 0.6, 0.7),
        stamp("Upgrade ", 0.7, 1.0),
        stamp("now", 1.0, 1.2),
    ];
    let pieces: Vec<_> = stamps
        .iter()
        .flat_map(|s| state.word(s, &said))
        .filter_map(|e| match e {
            AlignedTranscriptEffect::CaptureTimed(c) => Some((c.text, c.start_time)),
            _ => None,
        })
        .collect();
    let at = |text: &str| pieces.iter().find(|p| p.0 == text).map(|p| p.1).expect(text);
    assert_eq!((at("Re"), at("f")), (Some(0.0), Some(0.0)), "{pieces:?}");
    assert_eq!((at("**Up"), at("grade**")), (Some(0.7), Some(0.7)), "{pieces:?}");
    assert_eq!(at(" now"), Some(1.0), "every word after keeps its own stamp: {pieces:?}");
}

// Catches: a stamp mapped by position onto a word the reply had not finished writing, so the
// transcript showed half a word (`Tuesda`) and a cut there kept it in the history (survey seed 152).
#[test]
fn a_word_still_being_written_waits_for_the_rest_of_it() {
    let mut said = Visible::of("Monday. Tuesda", &Vocabulary::LiveKit);
    let mut state = AlignedTranscript::default();
    let shown = |effects: Vec<AlignedTranscriptEffect>| -> Vec<String> {
        effects
            .into_iter()
            .flat_map(|e| match e {
                AlignedTranscriptEffect::CaptureTimed(c) => vec![c.text],
                AlignedTranscriptEffect::Capture(cs) => cs.into_iter().map(|c| c.text).collect(),
                _ => Vec::new(),
            })
            .collect()
    };
    let mut got = shown(state.word(&stamp("Monday. ", 0.0, 0.5), &said));
    for (i, word) in ["Tuesday ", "nine ", "to ", "five ", "sharp "].iter().enumerate() {
        let at = 0.5 + i as f64 * 0.3;
        got.extend(shown(state.word(&stamp(word, at, at + 0.3), &said)));
    }
    assert!(!got.iter().any(|t| t.contains("Tuesda")), "a half word was shown: {got:?}");
    said.extend(Visible::of("y 9-5 sharp", &Vocabulary::LiveKit));
    got.extend(shown(state.end(&said)));
    assert_eq!(got.concat(), "Monday. Tuesday 9-5 sharp", "{got:?}");
}
