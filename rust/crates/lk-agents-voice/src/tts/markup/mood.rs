//! `tts/_mood.py`.

use crate::tts::markup::mood_data::keywords;

pub use lk_agents_abi::tts::markup::mood::Mood;

/// Tie-break order, most specific first.
pub const MOOD_PRIORITY: [Mood; 11] = [
    Mood::Angry,
    Mood::Sad,
    Mood::Anxious,
    Mood::Surprised,
    Mood::Playful,
    Mood::Empathetic,
    Mood::Excited,
    Mood::Curious,
    Mood::Hopeful,
    Mood::Happy,
    Mood::Calm,
];

/// The most recessive mood: an unmatched label asserts no feeling.
pub const DEFAULT_MOOD: Mood = Mood::Calm;

/// Stems must not match mid-word (`irate` in "pirate", #6698).
fn matches_word(text: &str, keyword: &str) -> bool {
    let mut start = 0usize;
    while let Some(rel) = text[start..].find(keyword) {
        let at = start + rel;
        if at == 0 || !text[..at].chars().next_back().is_some_and(char::is_alphabetic) {
            return true;
        }
        // advance one *character*, as CPython's `str.find(kw, at + 1)` does over code points
        start = at + text[at..].chars().next().map_or(1, char::len_utf8);
    }
    false
}

/// Match a raw delivery label to a mood; an unrecognized label resolves to `fallback`.
///
/// ```
/// # use crate::tts::markup::{match_mood, Mood, DEFAULT_MOOD};
/// assert_eq!(match_mood("soft, with genuine care", Some(DEFAULT_MOOD)), Some(Mood::Empathetic));
/// assert_eq!(match_mood("like a pirate", Some(DEFAULT_MOOD)), Some(Mood::Calm));
/// ```
pub fn match_mood(label: &str, fallback: Option<Mood>) -> Option<Mood> {
    let text = label.to_lowercase();
    let mut best: Option<Mood> = None;
    let mut best_score = 0u32;
    for mood in MOOD_PRIORITY {
        let score: u32 =
            keywords(mood).iter().filter(|(kw, _)| matches_word(&text, kw)).map(|(_, w)| *w).sum();
        // strictly greater, so MOOD_PRIORITY breaks ties
        if score > best_score {
            best = Some(mood);
            best_score = score;
        }
    }
    best.or(fallback)
}
