//! `replace_words` and `split_paragraphs`, with the reference's arithmetic. Characters, not bytes.

use std::collections::HashMap;

use crate::tokenize::py_strip;
use crate::tokenize::word::{WordTokenizerOptions, is_punctuation, split_words};

/// `replace_words`, case-insensitively, over one string. Returns the text and `processed_index`.
// `hold_last_word` holds the final word for the next chunk and returns `None` below two words.
// Reference details kept: an empty replacement is skipped (`if replacement:`); only trailing
// punctuation is stripped for the lookup and it survives; the signed offset accumulates;
// `processed_index` advances on every word.
#[crate::host(export(pid = false, defaults(hold_last_word = false)))]
pub fn replace_words(
    text: &str,
    replacements: HashMap<String, String>,
    hold_last_word: bool,
) -> Option<(String, usize)> {
    // The reference lowercases the keys once.
    let lookup: HashMap<String, &str> =
        replacements.iter().map(|(k, v)| (k.to_lowercase(), v.as_str())).collect();

    // Punctuated token text, so `rstrip` can measure the punctuation to keep.
    let opts = WordTokenizerOptions {
        ignore_punctuation: false,
        split_character: false,
        retain_format: false,
    };
    let mut words: Vec<(Vec<char>, usize, usize)> = split_words(text, opts)
        .into_iter()
        .map(|w| (w.text.chars().collect(), w.start, w.end))
        .collect();

    if hold_last_word {
        if words.len() <= 1 {
            return None;
        }
        words.pop();
    }

    let mut chars: Vec<char> = text.chars().collect();
    let mut offset: isize = 0;
    let mut processed_index: usize = 0;

    for (word, start, end) in words {
        // `word.rstrip(PUNCTUATIONS)`: the surviving length.
        let kept = word.iter().rposition(|c| !is_punctuation(*c)).map_or(0, |i| i + 1);
        let punctuation_off = word.len() - kept;
        let key: String = word[..kept].iter().collect::<String>().to_lowercase();

        // `if replacement:` -- see (1). A missing key and an empty value take the same branch.
        if let Some(rep) = lookup.get(&key).copied().filter(|r| !r.is_empty()) {
            let rep_chars: Vec<char> = rep.chars().collect();
            let lo = (start as isize + offset) as usize;
            let hi = (end as isize + offset - punctuation_off as isize) as usize;
            let grew = rep_chars.len() as isize - word.len() as isize + punctuation_off as isize;
            chars.splice(lo..hi, rep_chars);
            offset += grew;
        }

        processed_index = (end as isize + offset) as usize;
    }

    Some((chars.into_iter().collect(), processed_index))
}

/// `split_paragraphs`: the text between blank lines, each with its span in the original.
// Offsets use `text.index(stripped)` (the first occurrence), as the reference does. Empty
// paragraphs are dropped.
#[crate::host(export(pid = false))]
pub fn split_paragraphs(text: &str) -> Vec<(String, usize, usize)> {
    let chars: Vec<char> = text.chars().collect();

    // `r"\n\s*\n"` by hand, leftmost-first and non-overlapping.
    let mut splits: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\n' {
            let mut j = i + 1;
            while j < chars.len() && crate::tokenize::is_py_space(chars[j]) && chars[j] != '\n' {
                j += 1;
            }
            if j < chars.len() && chars[j] == '\n' {
                splits.push((i, j + 1));
                i = j + 1;
                continue;
            }
        }
        i += 1;
    }

    let mut out: Vec<(String, usize, usize)> = Vec::new();
    let push = |slice_start: usize, slice_end: usize, out: &mut Vec<_>| {
        let slice: String = chars[slice_start..slice_end].iter().collect();
        let stripped = py_strip(&slice);
        if stripped.is_empty() {
            return;
        }
        // `.index(stripped)` -- the first occurrence, in characters. See the note above.
        let within = char_find(&slice, stripped).unwrap_or(0);
        let start = slice_start + within;
        out.push((stripped.to_owned(), start, start + stripped.chars().count()));
    };

    if splits.is_empty() {
        push(0, chars.len(), &mut out);
        return out;
    }

    let mut start = 0;
    for (split_start, split_end) in &splits {
        push(start, *split_start, &mut out);
        start = *split_end;
    }
    push(start, chars.len(), &mut out);
    out
}

/// In **characters**.
fn char_find(haystack: &str, needle: &str) -> Option<usize> {
    haystack.find(needle).map(|byte| haystack[..byte].chars().count())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The non-streaming call, which cannot return `None`.
    fn replace_words_all(text: &str, replacements: &HashMap<String, String>) -> (String, usize) {
        replace_words(text, replacements.clone(), false).expect("only the streaming mode holds back")
    }

    fn map(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    // Catches: applying an empty replacement, stripping punctuation from both ends before the
    // lookup, or byte offsets.
    #[test]
    fn replacements_skip_empties_strip_trailing_punctuation_and_count_characters() {
        for (text, pairs, want) in [
            ("hello world", &[("hello", "")][..], "hello world"),
            ("hello, (hello", &[("hello", "hi")][..], "hi, (hello"),
            ("héllo wörld héllo", &[("héllo", "hi")][..], "hi wörld hi"),
        ] {
            assert_eq!(replace_words_all(text, &map(pairs)).0, want);
        }
    }

    // Catches: emitting on a buffer with one word.
    #[test]
    fn the_streaming_mode_holds_back_until_there_are_two_words() {
        let r = map(&[("hello", "hi")]);
        assert!(replace_words("hel", r.clone(), true).is_none());
        assert!(replace_words("", r.clone(), true).is_none());
        let (out, idx) = replace_words("hello world", r.clone(), true).unwrap();
        // "world" is held back, so only "hello" was considered and the emit stops after "hi".
        assert_eq!(out, "hi world");
        assert_eq!(idx, 2);
    }




    // Catches: "fixing" `text.index(stripped)` (first occurrence).
    #[test]
    fn the_offset_is_the_first_occurrence_as_the_reference_finds_it() {
        let out = split_paragraphs("a a");
        assert_eq!(out, vec![("a a".to_owned(), 0, 3)]);

        // The recovered index is relative to the slice.
        let out = split_paragraphs("x\n\n  a  ");
        assert_eq!(out, vec![("x".to_owned(), 0, 1), ("a".to_owned(), 5, 6)]);
    }

}
