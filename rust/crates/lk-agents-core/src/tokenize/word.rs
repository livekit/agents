//! `tokenize/_basic_word.py::split_words`. `retain_format` (#3334) makes the tokens concatenate
//! back to the source for the transcript synchroniser.

use std::borrow::Cow;

/// `tokenize/tokenizer.py:10`. `str.translate` deletes these anywhere in the token.
pub const PUNCTUATIONS: [char; 39] = [
    '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=',
    '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '±', '—', '‘', '’', '“', '”',
    '…',
];

pub use crate::text::{is_py_space, py_strip};

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

/// `tokenizer.is_punctuation`: [`PUNCTUATIONS`] or any Unicode punctuation (`P*`), e.g. Arabic
/// "،" and "؟", Devanagari "।", CJK "，" and "。".
#[inline]
pub fn is_punctuation(c: char) -> bool {
    static UNICODE_PUNCTUATION: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"^\p{P}$").expect("constant pattern"));
    PUNCTUATIONS.contains(&c)
        || (!c.is_ascii() && UNICODE_PUNCTUATION.is_match(c.encode_utf8(&mut [0; 4])))
}

#[cfg(test)]
#[path = "word_tests.rs"]
mod tests;

/// Really naive implementation of a word tokenizer: words split on whitespace, punctuation
/// stripped unless asked to keep it.
#[derive(Debug, Clone)]
pub struct BasicWordTokenizer {
    pub opts: WordTokenizerOptions,
}

impl lk_agents_abi::tokenize::bindings::WordTokenizer for BasicWordTokenizer {
    fn new(ignore_punctuation: bool, split_character: bool, retain_format: bool) -> Self {
        BasicWordTokenizer {
            opts: WordTokenizerOptions { ignore_punctuation, split_character, retain_format },
        }
    }

    /// The words of `text`; `language` is accepted and unused.
    fn tokenize(&self, text: String, _language: Option<String>) -> Vec<String> {
        split_words(&text, self.opts).into_iter().map(|w| w.text.into_owned()).collect()
    }

    /// `(word, start, end)` per word: what a stream batches.
    fn split(&self, text: String) -> Vec<(String, usize, usize)> {
        split_words(&text, self.opts)
            .into_iter()
            .map(|w| (w.text.into_owned(), w.start, w.end))
            .collect()
    }

    /// The shortest token a stream emits: each word once the next proves it whole.
    fn min_token_len(&self) -> usize {
        1
    }

    /// The text a stream buffers before it splits.
    fn min_ctx_len(&self) -> usize {
        1
    }
}
