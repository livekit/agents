//! Emoji: a run of pictographs with their joiners, selectors and keycap combiner.

use super::{Context, INLINE_WINDOW, Kind, Scan, Token};

pub(super) fn is_emoji(c: char) -> bool {
    matches!(c,
        '\u{1F000}'..='\u{1FBFF}'   // Mahjong Tiles .. Symbols for Legacy Computing, flags, tones
        | '\u{2600}'..='\u{26FF}'   // Miscellaneous Symbols
        | '\u{2700}'..='\u{27BF}'   // Dingbats
        | '\u{2B00}'..='\u{2BFF}'   // Miscellaneous Symbols and Arrows
        | '\u{FE00}'..='\u{FE0F}'   // Variation selectors
        | '\u{200D}'                // Zero width joiner
        | '\u{20E3}'                // Combining enclosing keycap
    )
}

/// One run, held while it may still grow (a joiner sequence split across deltas is one emoji), and
/// cut where a character starts past the inline window.
pub(super) fn run(s: &str, ctx: Context, at_end: bool) -> Scan {
    let len = s
        .char_indices()
        .find(|(i, c)| *i >= INLINE_WINDOW || !is_emoji(*c))
        .map_or(s.len(), |(i, _)| i);
    if len == s.len() && !at_end {
        return Scan::NeedMore;
    }
    Scan::Match { len, tokens: vec![Token { kind: Kind::Emoji, raw: s[..len].to_owned(), ctx }] }
}
