//! Keeps a TTS markup tag from being split across synthesis segments (`tokenize/token_stream.py`).
//! `<(/?)([A-Za-z]\w*)[^>]*?(/?)\s*>` scanned by hand. Offsets are characters, not bytes.

use crate::tokenize::token_buffer::{Span, Tokenize};

/// **Character** indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tag {
    pub start: usize,
    pub end: usize,
    pub is_closing: bool,
    pub is_self_closing: bool,
}

/// The name must start with a letter (`<5>` is text), the same test the incomplete-tag check uses.
pub fn tags(text: &str) -> Vec<Tag> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] != '<' {
            i += 1;
            continue;
        }
        let start = i;
        let mut j = i + 1;
        let is_closing = chars.get(j) == Some(&'/');
        if is_closing {
            j += 1;
        }
        // `[A-Za-z]` -- ASCII, as the pattern says, and not `is_alphabetic`.
        if !chars.get(j).is_some_and(|c| c.is_ascii_alphabetic()) {
            i += 1;
            continue;
        }
        j += 1;
        // `\w*`: the rest of the name.
        while chars.get(j).is_some_and(|c| c.is_alphanumeric() || *c == '_') {
            j += 1;
        }
        // `[^>]*?...>`: lazy, so the tag ends at the first `>` after the name.
        let Some(close) = (j..chars.len()).find(|k| chars[*k] == '>') else {
            // No `>`: not a tag; the tail check decides whether it holds streaming.
            i += 1;
            continue;
        };
        // `(/?)\s*>`: trailing whitespace, then an optional `/`.
        let mut k = close;
        while k > j && chars[k - 1].is_whitespace() {
            k -= 1;
        }
        let is_self_closing = k > j && chars[k - 1] == '/';
        out.push(Tag { start, end: close + 1, is_closing, is_self_closing });
        i = close + 1;
    }
    out
}

/// Is there an incomplete or unclosed tag in `text`? A `<` after the last `>` is tag-shaped when
/// followed by `/`, a letter, or nothing.
#[crate::host(export(pid = false, name = "xml_has_unclosed_tags"))]
pub fn has_unclosed_tags(text: &str) -> bool {
    if !text.contains('<') {
        return false;
    }

    let chars: Vec<char> = text.chars().collect();
    let last_open = chars.iter().rposition(|c| *c == '<');
    let last_close = chars.iter().rposition(|c| *c == '>');
    if let Some(open) = last_open
        && last_close.is_none_or(|close| open > close)
    {
        match chars.get(open + 1) {
            None => return true,
            Some('/') => return true,
            Some(c) if c.is_ascii_alphabetic() => return true,
            _ => {}
        }
    }

    // Unbalanced open/close pairs; a self-closing tag counts as neither.
    let mut depth: i32 = 0;
    for tag in tags(text) {
        if tag.is_self_closing {
            continue;
        } else if tag.is_closing {
            depth -= 1;
        } else {
            depth += 1;
        }
    }
    depth > 0
}

/// Does `text` contain tags but no words? Such a segment is merged into the next.
pub fn is_xml_only(text: &str) -> bool {
    if !text.contains('<') {
        return false;
    }
    strip_tags(text).trim().is_empty()
}

pub fn strip_tags(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let spans = tags(text);
    let mut out = String::new();
    let mut cursor = 0;
    for tag in &spans {
        out.extend(&chars[cursor..tag.start]);
        cursor = tag.end;
    }
    out.extend(&chars[cursor..]);
    out
}

/// A tag exactly at the boundary goes to the next sentence (`<`, not `<=`).
pub fn clean_to_orig(clean_pos: usize, tag_spans: &[Tag]) -> usize {
    let mut orig = clean_pos;
    for tag in tag_spans {
        if tag.start < orig {
            orig += tag.end - tag.start;
        } else {
            break;
        }
    }
    orig
}

/// The span is in the **original**.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub text: String,
    pub start: usize,
    pub end: usize,
}

/// Glues a segment with an unclosed tag or only markup to the next one.
pub fn remap(text: &str, clean_ends: &[usize], tag_spans: &[Tag]) -> Vec<Segment> {
    let chars: Vec<char> = text.chars().collect();
    let slice = |a: usize, b: usize| -> String {
        chars[a.min(chars.len())..b.min(chars.len())].iter().collect()
    };

    let mut result: Vec<Segment> = Vec::new();
    let mut start = 0;
    for clean_end in clean_ends {
        let orig_end = clean_to_orig(*clean_end, tag_spans);
        let sentence = slice(start, orig_end);
        let trimmed = sentence.trim();
        if !trimmed.is_empty() {
            result.push(Segment { text: trimmed.to_owned(), start, end: orig_end });
        }
        start = orig_end;
    }
    if start < chars.len() {
        let sentence = slice(start, chars.len());
        let trimmed = sentence.trim();
        if !trimmed.is_empty() {
            result.push(Segment { text: trimmed.to_owned(), start, end: chars.len() });
        }
    }

    if result.is_empty() {
        return result;
    }
    let mut merged: Vec<Segment> = vec![result[0].clone()];
    for seg in &result[1..] {
        let prev = merged.last().expect("seeded above");
        if has_unclosed_tags(&prev.text) || is_xml_only(&prev.text) {
            let prev_start = prev.start;
            let text = slice(prev_start, seg.end).trim().to_owned();
            *merged.last_mut().expect("seeded above") =
                Segment { text, start: prev_start, end: seg.end };
        } else {
            merged.push(seg.clone());
        }
    }
    merged
}

/// Tag-free text goes to the tokenizer untouched; tag-only text comes back as one span.
pub fn tokenize_aware<T: Tokenize + ?Sized>(tok: &T, text: &str) -> Vec<Span> {
    let tag_spans = tags(text);
    if tag_spans.is_empty() {
        return tok.tokenize(text);
    }
    let clean = strip_tags(text);
    if clean.trim().is_empty() {
        return if text.trim().is_empty() { Vec::new() } else { vec![Span::whole(text)] };
    }
    let clean_ends: Vec<usize> = tok.tokenize(&clean).into_iter().map(|s| s.end).collect();
    if clean_ends.is_empty() {
        return Vec::new();
    }
    remap(text, &clean_ends, &tag_spans)
        .into_iter()
        .map(|s| Span { text: s.text, start: s.start, end: s.end })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;



    // Catches: counting a self-closing tag as an open.
    #[test]
    fn a_self_closing_tag_changes_the_depth_by_nothing() {
        assert!(!has_unclosed_tags("hello <break/> world"));
        assert!(!has_unclosed_tags("hello <break time=\"1s\" /> world"));
        assert!(!has_unclosed_tags("<emphasis>hi</emphasis>"));
        assert!(has_unclosed_tags("<emphasis>hi"));
    }

    // Catches: a markup-only segment emitted on its own.
    #[test]
    fn markup_with_no_words_is_recognised() {
        assert!(is_xml_only("<break/>"));
        assert!(is_xml_only("  <break/> <break/> "));
        assert!(!is_xml_only("<emphasis>hi</emphasis>"));
        assert!(!is_xml_only("plain text"));
    }

    // Catches: `<=` in the boundary test.
    #[test]
    fn a_tag_exactly_at_the_boundary_goes_to_the_next_sentence() {
        let text = "Hello.<break/>World.";
        let spans = tags(text);
        // "Hello." is 6 clean characters; the tag starts at 6 in the original.
        assert_eq!(clean_to_orig(6, &spans), 6);
        // One character further in the clean text is past the tag.
        assert_eq!(clean_to_orig(7, &spans), 6 + (spans[0].end - spans[0].start) + 1);
    }

    // Catches: byte offsets.
    #[test]
    fn offsets_are_character_indices() {
        let text = "héllo <break/> wörld";
        let spans = tags(text);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].start, 6, "byte offset would be 7");
        assert_eq!(strip_tags(text), "héllo  wörld");
    }

    // Catches: emitting a markup-only sentence rather than gluing it forward.
    #[test]
    fn a_markup_only_sentence_is_merged_into_the_one_after_it() {
        let text = "<break/>Hello there.";
        let spans = tags(text);
        // The tokenizer saw the clean text "Hello there." and ended it at 12.
        let out = remap(text, &[0, 12], &spans);
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!(out[0].text, "<break/>Hello there.");
    }

    // Catches: splitting a sentence in the middle of a tag pair.
    #[test]
    fn a_sentence_with_an_unclosed_tag_takes_the_next_one() {
        let text = "<emphasis>One. Two.</emphasis>";
        let spans = tags(text);
        let out = remap(text, &[4, 9], &spans);
        assert_eq!(out.len(), 1, "{out:?}");
        assert!(out[0].text.starts_with("<emphasis>"));
    }

    // Catches: dropping the tail.
    #[test]
    fn text_after_the_last_offset_is_still_emitted() {
        let text = "One. Two.";
        let out = remap(text, &[4], &[]);
        assert_eq!(out.len(), 2, "{out:?}");
        assert_eq!(out[1].text, "Two.");
    }
}
