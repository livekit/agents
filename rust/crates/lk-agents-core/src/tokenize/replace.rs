use std::collections::BTreeMap;

use crate::tokenize::word::{WordTokenizerOptions, is_punctuation, split_words};

// Offsets follow the reference's arithmetic, in characters, not bytes.

/// `replace_words`, case-insensitively, over one string. Returns the text and `processed_index`.
// `hold_last_word` holds the final word for the next chunk and returns `None` below two words.
// An empty replacement deletes the word. Trailing punctuation is stripped one character at a
// time for the lookup (a key may end in punctuation, e.g. Hebrew "ג׳"), the longest matching key
// wins, and the stripped punctuation survives. `processed_index` advances on every word.
pub fn replace_words(
    text: &str,
    replacements: impl IntoIterator<Item = (String, String)>,
    hold_last_word: bool,
) -> Option<(String, usize)> {
    // The reference lowercases the keys once.
    let lookup: BTreeMap<String, String> =
        replacements.into_iter().map(|(key, value)| (key.to_lowercase(), value)).collect();

    // Punctuated token text, so `rstrip` can measure the punctuation to keep.
    let opts = WordTokenizerOptions { ignore_punctuation: false, retain_format: false };
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
        let lookup_at = |kept: usize| {
            lookup.get(&word[..kept].iter().collect::<String>().to_lowercase()).map(String::as_str)
        };
        let mut kept = word.len();
        let mut replacement = lookup_at(kept);
        while replacement.is_none() && kept > 0 && is_punctuation(word[kept - 1]) {
            kept -= 1;
            replacement = lookup_at(kept);
        }
        let punctuation_off = word.len() - kept;

        if let Some(rep) = replacement {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The non-streaming call, which cannot return `None`.
    fn replace_words_all(text: &str, replacements: &BTreeMap<String, String>) -> (String, usize) {
        replace_words(text, replacements.clone(), false)
            .expect("only the streaming mode holds back")
    }

    fn map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    // Catches: skipping an empty replacement, stripping punctuation from both ends or all at once
    // before the lookup, or byte offsets.
    #[test]
    fn replacements_delete_on_empty_strip_trailing_punctuation_and_count_characters() {
        for (text, pairs, want) in [
            ("hello world", &[("hello", "")][..], " world"),
            ("hello, (hello", &[("hello", "hi")][..], "hi, (hello"),
            ("héllo wörld héllo", &[("héllo", "hi")][..], "hi wörld hi"),
            // A key ending in punctuation matches before the bare word does.
            ("ג׳ ג", &[("ג׳", "gimel"), ("ג", "x")][..], "gimel x"),
            ("ok?! no", &[("ok?", "fine")][..], "fine! no"),
            ("مرحبا، صديقي", &[("مرحبا", "hello")][..], "hello، صديقي"),
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
}
