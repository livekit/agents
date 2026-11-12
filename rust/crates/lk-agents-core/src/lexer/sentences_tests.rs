//! The sentence splitter in front of the TTS reads the synthesis view: it must never split a tag,
//! never lose a word, and never hold a sentence for prose that only looks like markup.

use super::*;
use crate::tokenize::sentences::{SentenceOptions, SentenceStream};
use crate::tts::markup::Provider;

const SYNTHESIS: View = View::Synthesis { keep_markdown: false, keep_emoji: false };

/// The TTS's sentences for `deltas`, split as a stream.
fn sentences(deltas: &[&str]) -> Vec<String> {
    let markup = Some(Vocabulary::of(Some(Provider::Cartesia)));
    let mut stream = SentenceStream::new(SentenceOptions { markup, ..SentenceOptions::default() });
    let mut out: Vec<String> = deltas.iter().flat_map(|d| stream.push(d)).collect();
    out.extend(stream.flush());
    out
}

/// What the TTS is sent for `reply`, as the voice streams it.
fn synthesis(provider: Provider, reply: &str) -> String {
    let tokens = Lexer::all(Rules::speech(Vocabulary::of(Some(provider))), reply);
    Projector::new(SYNTHESIS).text(&tokens)
}

fn words(s: &str) -> Vec<&str> {
    s.split_whitespace().collect()
}

#[test]
fn no_sentence_splits_a_marker_and_no_word_is_lost() {
    let reply = "<expr type=\"expression\" label=\"warm. calm! sure?\"/> Hello there. \
How are you? <expr type=\"sound\" label=\"laugh\"/> Great. Next one.";
    let text = synthesis(Provider::Cartesia, reply);
    let out = sentences(&[&text]);
    assert_eq!(words(&out.join(" ")), words(&text));
    for s in &out {
        // each sentence lexed alone has no fragment
        let tokens = Lexer::all(Rules::speech(Vocabulary::LiveKit), s);
        assert!(tokens.iter().all(|t| t.kind != Kind::Hidden(Hidden::Markup)), "{s:?} in {out:?}");
    }
}

#[test]
fn prose_that_looks_like_markup_does_not_hold_sentences() {
    let text = synthesis(Provider::Cartesia, "If a<b then stop. Then go home now. And rest a bit.");
    let deltas: Vec<String> = text.chars().map(String::from).collect();
    let deltas: Vec<&str> = deltas.iter().map(String::as_str).collect();
    let out = sentences(&deltas);
    assert!(out.len() >= 2, "one held sentence: {out:?}");
}
