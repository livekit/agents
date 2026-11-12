use super::*;

/// `[word, line break after it, last word, syllables, weight]`, from the fitted model in Python.
const VECTORS: &str = include_str!("estimate_vectors.json");

// Catches: the port drifting from the model it carries the weights of, which would misplace every
// word by however much a feature is miscounted.
#[test]
fn reproduces_the_fitted_model() {
    let vectors: Vec<(String, bool, bool, f64, f64)> =
        serde_json::from_str(VECTORS).expect("the vectors parse");
    let mut wrong = Vec::new();
    for (word, line_break, last, syl, expected) in &vectors {
        let (got_syl, got) = (syllables(word), weight(word, *line_break, *last));
        if (got_syl - syl).abs() > 1e-9 || (got - expected).abs() > 1e-6 * expected.max(1.0) {
            wrong.push(format!(
                "{word:?}: syllables {got_syl} vs {syl}, weight {got:.6} vs {expected:.6}"
            ));
        }
    }
    assert!(wrong.is_empty(), "{} of {} differ:\n{}", wrong.len(), vectors.len(), wrong.join("\n"));
}

// Catches: numbers and codes counted by their characters, so a price or an order number got a
// fraction of the time a voice takes to read it.
#[test]
fn numbers_and_codes_are_counted_as_read() {
    assert_eq!(syllables("$2850.32"), 15.0);
    assert_eq!(syllables("9973034634"), 12.0);
    assert_eq!(syllables("#W9571698"), 11.0);
    assert_eq!(syllables("2024"), 5.0);
    assert!(weight("#W9571698", false, false) > 2.0 * weight("order", false, false));
    assert_eq!(weight("—", false, false), 0.0);
}

fn words(spec: &[(f64, bool)]) -> Vec<Weighed> {
    spec.iter()
        .map(|&(weight, breaks)| Weighed { weight, breaks: breaks.then_some(Break::Sentence) })
        .collect()
}

// Catches: a pause matched to the wrong break, or the words after it not re-anchored at its end.
#[test]
fn a_pause_anchors_the_words_after_the_break_it_follows() {
    // two sentences of equal words at 5 weight/s: 1.2 s, a 0.6 s pause, 1.2 s
    let ws =
        words(&[(2.0, false), (2.0, false), (2.0, true), (2.0, false), (2.0, false), (2.0, false)]);
    let pauses = [Span { start: 1.2, end: 1.8 }];
    let got = place_weighed(&ws, &pauses, 0.0, End::Known(3.0), 1.0);
    let want = [0.0, 0.4, 0.8, 1.8, 2.2, 2.6];
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < 1e-6, "{got:?}");
    }
}

// Catches: a hesitation inside a phrase taken for a sentence break, pulling the next sentence early.
#[test]
fn a_pause_with_no_break_near_it_is_skipped_over() {
    let ws =
        words(&[(2.0, false), (2.0, false), (2.0, false), (2.0, false), (2.0, true), (2.0, false)]);
    // a short hesitation after the second word, the real pause after the fifth
    let pauses = [Span { start: 0.8, end: 1.0 }, Span { start: 2.2, end: 2.8 }];
    let got = place_weighed(&ws, &pauses, 0.0, End::Known(3.2), 1.0);
    assert!((got[5] - 2.8).abs() < 1e-6, "the sixth word starts after the long pause: {got:?}");
    assert!((got[2] - 1.0).abs() < 1e-6, "the third word starts after the hesitation: {got:?}");
}

// Catches: an open end paced as if the audio stopped where it was received, squeezing the rest.
#[test]
fn an_open_end_paces_the_words_past_the_audio_received_at_the_learnt_rate() {
    let ws = words(&[(6.0, true), (6.0, true), (6.0, false), (6.0, false)]);
    let pauses = [Span { start: 1.0, end: 1.3 }, Span { start: 2.3, end: 2.6 }];
    let got = place_weighed(&ws, &pauses, 0.0, End::Open { received: 2.7 }, 1.0);
    assert!((got[1] - 1.3).abs() < 1e-6 && (got[2] - 2.6).abs() < 1e-6, "{got:?}");
    // the matched stretches ran at 6 weight/s, not the prior: the rest goes on at it
    assert!((got[3] - 3.6).abs() < 1e-6, "{got:?}");
}

// Catches: every break taken as equally likely to pause, so a long pause halfway between a comma
// and a sentence end anchors at the comma as readily as at the sentence end.
#[test]
fn a_long_pause_goes_to_the_sentence_end_rather_than_the_comma() {
    let breaks = [None, Some(Break::Clause), None, Some(Break::Sentence), None, None];
    let ws: Vec<Weighed> = breaks.iter().map(|&breaks| Weighed { weight: 2.0, breaks }).collect();
    // 0.4 s a word: the comma is due at 0.8 s, the sentence end at 1.6 s; the pause sits between
    let pauses = [Span { start: 1.2, end: 1.9 }];
    let got = place_weighed(&ws, &pauses, 0.0, End::Known(2.7), 1.0);
    assert!(
        (got[4] - 1.9).abs() < 1e-6,
        "the word after the sentence end follows the pause: {got:?}"
    );
}

// Catches: a pause the text allows not taken for one, or taken for the wrong kind: punctuation
// inside a closing quote or bracket, an ellipsis or a dash, and other scripts' marks.
#[test]
fn a_break_may_hide_inside_a_closing_quote_or_bracket() {
    for word in ["said.\"", "(below).", "oui.»", "「はい。」", "wait…", "؟"] {
        assert_eq!(breaks(word, false), Some(Break::Sentence), "{word}");
    }
    for word in ["well—", "نعم،", "ναι\u{37e}", "first,"] {
        assert_eq!(breaks(word, false), Some(Break::Clause), "{word}");
    }
    for word in ["word", "\"quoted\"", "(aside)", "3.5"] {
        assert_eq!(breaks(word, false), None, "{word}");
    }
    assert_eq!(breaks("item", true), Some(Break::Line));
}
