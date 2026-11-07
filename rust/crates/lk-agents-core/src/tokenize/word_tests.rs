use super::*;

/// `(text, start, end)` triples, for array-literal expectations.
fn spans<'a>(w: &'a [Word<'_>]) -> Vec<(&'a str, usize, usize)> {
    w.iter().map(|t| (t.text.as_ref(), t.start, t.end)).collect()
}

const DEFAULTS: WordTokenizerOptions =
    WordTokenizerOptions { ignore_punctuation: true, split_character: false, retain_format: false };

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

// Catches: non-ASCII punctuation (Arabic, Devanagari, CJK) kept on the word.
#[test]
fn unicode_punctuation_is_dropped_like_ascii() {
    assert_eq!(
        spans(&split_words("مرحبا، كيف؟ नमस्ते। 你好，", DEFAULTS)),
        [("مرحبا", 0, 6), ("كيف", 7, 11), ("नमस्ते", 12, 19), ("你好", 20, 23)]
    );
    assert!(is_punctuation('،') && is_punctuation('。') && !is_punctuation('a'));
    assert!(!is_punctuation('€'), "a symbol outside the list is not punctuation");
}

// Catches: offsets recomputed from stripped text, or an all-punctuation token kept.
#[test]
fn an_all_punctuation_token_is_dropped_and_its_neighbours_keep_source_offsets() {
    assert_eq!(spans(&split_words("hi ... there", DEFAULTS)), [("hi", 0, 2), ("there", 7, 12)]);
    assert!(split_words("!!!", DEFAULTS).is_empty());
    assert_eq!(
        spans(&split_words("!!!", WordTokenizerOptions { ignore_punctuation: false, ..DEFAULTS })),
        [("!!!", 0, 3)],
        "without the flag it is an ordinary token"
    );
}

// Catches: reporting byte offsets.
#[test]
fn offsets_are_character_indices_not_byte_indices() {
    assert_eq!(spans(&split_words("héllo wörld", DEFAULTS)), [("héllo", 0, 5), ("wörld", 6, 11)]);
    let split = WordTokenizerOptions { split_character: true, ..DEFAULTS };
    assert_eq!(spans(&split_words("é 世界", split)), [("é", 0, 1), ("世", 2, 3), ("界", 3, 4)]);
}

// Catches: the character branch always on (#2366); both interruption guards opt in.
#[test]
fn cjk_splits_per_character_only_when_the_flag_is_set() {
    assert_eq!(spans(&split_words("你好吗", DEFAULTS)), [("你好吗", 0, 3)]);
    let split = WordTokenizerOptions { split_character: true, ..DEFAULTS };
    assert_eq!(spans(&split_words("你好吗", split)), [("你", 0, 1), ("好", 1, 2), ("吗", 2, 3)]);
}

// Catches: porting the docstring's language list instead of the character class.
#[test]
fn only_the_four_ranges_in_the_class_split_hangul_and_arabic_do_not() {
    let split = WordTokenizerOptions { split_character: true, ..DEFAULTS };
    assert_eq!(split_words("สวัสดี", split).len(), 6, "Thai is in the class");
    assert_eq!(split_words("カナ", split).len(), 2, "katakana is in the class");
    assert_eq!(spans(&split_words("안녕하세요", split)), [("안녕하세요", 0, 5)], "Hangul is not");
    assert_eq!(spans(&split_words("مرحبا", split)), [("مرحبا", 0, 5)], "Arabic: gone in #2366");
}

// Catches: dropping the whitespace run under `retain_format`, or attaching it to the previous
// token (`synchronizer.py:502` concatenates the tokens).
#[test]
fn retain_format_tokens_concatenate_back_to_the_source() {
    let opts = WordTokenizerOptions { ignore_punctuation: false, retain_format: true, ..DEFAULTS };
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
        spans(&split_words("a\u{1e}b", WordTokenizerOptions { retain_format: true, ..DEFAULTS })),
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
