//! `tokenize/_basic_word.py::split_words`. `retain_format` (#3334) makes the tokens concatenate
//! back to the source for the transcript synchroniser.

use std::borrow::Cow;

/// `tokenize/tokenizer.py:10`. `str.translate` deletes these anywhere in the token.
pub const PUNCTUATIONS: [char; 39] = [
    '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=',
    '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '±', '—', '‘', '’', '“', '”',
    '…',
];

/// Python's `str.isspace()`, which also counts `\x1c`-`\x1f`, unlike `char::is_whitespace()`.
#[inline]
pub fn is_py_space(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
}

/// `str.strip()`, over Python's whitespace set.
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(is_py_space)
}

/// `char_based_codes`, verbatim. Deliberately no Hangul or Arabic/Hebrew/Indic (#2366).
#[inline]
pub fn is_char_based(c: char) -> bool {
    matches!(c,
        '\u{4e00}'..='\u{9fff}'   // CJK unified ideographs
        | '\u{3040}'..='\u{30ff}' // hiragana + katakana
        | '\u{3400}'..='\u{4dbf}' // CJK extension A
        | '\u{0e00}'..='\u{0e7f}' // Thai
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WordTokenizerOptions {
    /// Offsets still span the source.
    pub ignore_punctuation: bool,
    /// One token per [`is_char_based`] character; the interruption guards turn it on for CJK.
    pub split_character: bool,
    /// Whitespace is glued to the front of the following token.
    pub retain_format: bool,
}

impl Default for WordTokenizerOptions {
    /// `WordTokenizer()` (`audio_recognition.py:338`). Not derived: `ignore_punctuation` is true.
    fn default() -> Self {
        Self { ignore_punctuation: true, split_character: false, retain_format: false }
    }
}

/// The half-open character span of the source (`replace_words` splices with it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Word<'a> {
    pub text: Cow<'a, str>,
    pub start: usize,
    pub end: usize,
}

/// Characters (`ch`, reported) and bytes (`by`, for slicing), kept together.
#[derive(Debug, Clone, Copy)]
struct Pos {
    ch: usize,
    by: usize,
}

/// `split_words` for the host: `(word, start_char, end_char)`, with character offsets
/// (`replace_words` splices with them).
#[crate::host(export(
    pid = false,
    name = "split_words",
    defaults(ignore_punctuation = true, split_character = false, retain_format = false)
))]
pub fn split_words_of(
    text: &str,
    ignore_punctuation: bool,
    split_character: bool,
    retain_format: bool,
) -> Vec<(String, usize, usize)> {
    let opts = WordTokenizerOptions { ignore_punctuation, split_character, retain_format };
    split_words(text, opts).into_iter().map(|w| (w.text.into_owned(), w.start, w.end)).collect()
}

/// The public `basic.split_words` is this with `retain_format: false`.
pub fn split_words(text: &str, opts: WordTokenizerOptions) -> Vec<Word<'_>> {
    let mut out = Vec::new();
    scan(text, opts, |w| out.push(w));
    out
}

/// Without building the tokens.
pub fn count_words(text: &str, opts: WordTokenizerOptions) -> usize {
    let mut n = 0;
    scan(text, opts, |_| n += 1);
    n
}

fn scan<'a>(text: &'a str, opts: WordTokenizerOptions, mut sink: impl FnMut(Word<'a>)) {
    let mut emit = |start: Pos, end: Pos| {
        let raw = &text[start.by..end.by];
        let word = if opts.ignore_punctuation && raw.chars().any(is_punctuation) {
            Cow::Owned(raw.chars().filter(|c| !is_punctuation(*c)).collect::<String>())
        } else {
            Cow::Borrowed(raw)
        };
        // All-punctuation and empty tokens are dropped ("..." does not count).
        if !word.is_empty() {
            sink(Word { text: word, start: start.ch, end: end.ch });
        }
    };

    let mut word_start = Pos { ch: 0, by: 0 };
    let mut ch = 0usize;

    for (by, c) in text.char_indices() {
        let pos = Pos { ch, by };
        ch += 1;
        let next = Pos { ch: pos.ch + 1, by: by + c.len_utf8() };

        if is_py_space(c) {
            // `retain_format`: hold a whitespace-only run for the front of the next token.
            if opts.retain_format && text[word_start.by..pos.by].chars().all(is_py_space) {
                continue;
            }
            emit(word_start, pos);
            word_start = if opts.retain_format { pos } else { next };
        } else if opts.split_character && is_char_based(c) {
            // Flush whatever preceded the character (possibly pure whitespace, as the reference
            // does). `<=` is an equivalent mutant.
            if word_start.ch < pos.ch {
                emit(word_start, pos);
            }
            emit(pos, next);
            word_start = next;
        }
    }

    emit(word_start, Pos { ch, by: text.len() });
}

#[inline]
pub(crate) fn is_punctuation(c: char) -> bool {
    PUNCTUATIONS.contains(&c)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(text, start, end)` triples, for array-literal expectations.
    fn spans<'a>(w: &'a [Word<'_>]) -> Vec<(&'a str, usize, usize)> {
        w.iter().map(|t| (t.text.as_ref(), t.start, t.end)).collect()
    }

    const DEFAULTS: WordTokenizerOptions = WordTokenizerOptions {
        ignore_punctuation: true,
        split_character: false,
        retain_format: false,
    };

    // Catches: `ignore_punctuation` as an edge trim (`str.translate` deletes everywhere; "don't"
    // and "dont" must match in `_transcripts_equivalent`).
    #[test]
    fn punctuation_is_deleted_from_inside_the_token_not_trimmed() {
        assert_eq!(
            spans(&split_words("don't well-known", DEFAULTS)),
            [("dont", 0, 5), ("wellknown", 6, 16)],
            "the span stays that of the punctuated source"
        );
        assert_eq!(spans(&split_words("\"quoted\",", DEFAULTS)), [("quoted", 0, 9)]);
    }

    // Catches: offsets recomputed from stripped text, or an all-punctuation token kept.
    #[test]
    fn an_all_punctuation_token_is_dropped_and_its_neighbours_keep_source_offsets() {
        assert_eq!(spans(&split_words("hi ... there", DEFAULTS)), [("hi", 0, 2), ("there", 7, 12)]);
        assert!(split_words("!!!", DEFAULTS).is_empty());
        assert_eq!(
            spans(&split_words(
                "!!!",
                WordTokenizerOptions { ignore_punctuation: false, ..DEFAULTS }
            )),
            [("!!!", 0, 3)],
            "without the flag it is an ordinary token"
        );
    }

    // Catches: reporting byte offsets.
    #[test]
    fn offsets_are_character_indices_not_byte_indices() {
        assert_eq!(
            spans(&split_words("héllo wörld", DEFAULTS)),
            [("héllo", 0, 5), ("wörld", 6, 11)]
        );
        let split = WordTokenizerOptions { split_character: true, ..DEFAULTS };
        assert_eq!(spans(&split_words("é 世界", split)), [("é", 0, 1), ("世", 2, 3), ("界", 3, 4)]);
    }

    // Catches: the character branch always on (#2366); both interruption guards opt in.
    #[test]
    fn cjk_splits_per_character_only_when_the_flag_is_set() {
        assert_eq!(spans(&split_words("你好吗", DEFAULTS)), [("你好吗", 0, 3)]);
        let split = WordTokenizerOptions { split_character: true, ..DEFAULTS };
        assert_eq!(
            spans(&split_words("你好吗", split)),
            [("你", 0, 1), ("好", 1, 2), ("吗", 2, 3)]
        );
    }

    // Catches: porting the docstring's language list instead of the character class.
    #[test]
    fn only_the_four_ranges_in_the_class_split_hangul_and_arabic_do_not() {
        let split = WordTokenizerOptions { split_character: true, ..DEFAULTS };
        assert_eq!(split_words("สวัสดี", split).len(), 6, "Thai is in the class");
        assert_eq!(split_words("カナ", split).len(), 2, "katakana is in the class");
        assert_eq!(
            spans(&split_words("안녕하세요", split)),
            [("안녕하세요", 0, 5)],
            "Hangul is not"
        );
        assert_eq!(spans(&split_words("مرحبا", split)), [("مرحبا", 0, 5)], "Arabic: gone in #2366");
    }

    // Catches: dropping the whitespace run under `retain_format`, or attaching it to the previous
    // token (`synchronizer.py:502` concatenates the tokens).
    #[test]
    fn retain_format_tokens_concatenate_back_to_the_source() {
        let opts =
            WordTokenizerOptions { ignore_punctuation: false, retain_format: true, ..DEFAULTS };
        for src in ["  a  b ", "a\n\nb", "hello world", "   ", "", "a"] {
            let joined: String = split_words(src, opts).iter().map(|w| w.text.as_ref()).collect();
            assert_eq!(joined, src, "retain_format lost or duplicated whitespace in {src:?}");
        }
        assert_eq!(spans(&split_words("a  b ", opts)), [("a", 0, 1), ("  b", 1, 4), (" ", 4, 5)]);
    }

    // Catches: "cleaning up" the whitespace-only token emitted with `retain_format` and
    // `split_character` (the synchroniser's configuration).
    #[test]
    fn retain_format_emits_a_whitespace_only_token_before_a_cjk_character() {
        let opts = WordTokenizerOptions {
            ignore_punctuation: false,
            split_character: true,
            retain_format: true,
        };
        assert_eq!(spans(&split_words("a 世", opts)), [("a", 0, 1), (" ", 1, 2), ("世", 2, 3)]);
    }

    // Catches: using `char::is_whitespace`.
    #[test]
    fn python_isspace_covers_the_c0_separators() {
        assert!(!'\u{1e}'.is_whitespace(), "the premise: Rust does not call this whitespace");
        assert_eq!(spans(&split_words("a\u{1e}b", DEFAULTS)), [("a", 0, 1), ("b", 2, 3)]);
        assert_eq!(
            spans(&split_words(
                "a\u{1e}b",
                WordTokenizerOptions { retain_format: true, ..DEFAULTS }
            )),
            [("a", 0, 1), ("\u{1e}b", 1, 3)]
        );
    }

    // Catches: an empty token for trailing whitespace, or empty input treated as an error.
    #[test]
    fn empty_and_whitespace_only_input_produce_no_tokens() {
        assert!(split_words("", DEFAULTS).is_empty());
        assert!(split_words("   ", DEFAULTS).is_empty());
        assert_eq!(spans(&split_words("hi ", DEFAULTS)), [("hi", 0, 2)]);
        assert_eq!(spans(&split_words(" hi", DEFAULTS)), [("hi", 1, 3)]);
        assert!(
            split_words("", WordTokenizerOptions { retain_format: true, ..DEFAULTS }).is_empty(),
            "not even retain_format invents a token for empty input"
        );
    }

    // Catches: `count_words` drifting from `split_words`.
    #[test]
    fn count_words_agrees_with_split_words() {
        let cases = ["", "  ", "hello, world!", "你好 world", "a\u{1e}b...", "  a  b "];
        for &text in &cases {
            for flags in 0..8u8 {
                let opts = WordTokenizerOptions {
                    ignore_punctuation: flags & 1 != 0,
                    split_character: flags & 2 != 0,
                    retain_format: flags & 4 != 0,
                };
                assert_eq!(
                    count_words(text, opts),
                    split_words(text, opts).len(),
                    "{text:?} with {opts:?}"
                );
            }
        }
    }
}
