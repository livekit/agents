use crate::tokenize::token_buffer::{NativeTokenBuffer, TokenBufferOptions, Words};
use lk_agents_abi::tokenize::TokenSpan;
use regex::Regex;
use std::borrow::Cow;
use std::sync::LazyLock;

// `retain_format` makes the tokens concatenate back to the source for the transcript synchroniser.

/// `tokenize/tokenizer.py:10`. `str.translate` deletes these anywhere in the token.
pub const PUNCTUATIONS: [char; 39] = [
    '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=',
    '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '±', '—', '‘', '’', '“', '”',
    '…',
];

pub use crate::text::{is_py_space, py_strip};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WordTokenizerOptions {
    /// Offsets still span the source.
    pub ignore_punctuation: bool,
    /// Whitespace is glued to the front of the following token.
    pub retain_format: bool,
}

impl Default for WordTokenizerOptions {
    /// `WordTokenizer()` (`audio_recognition.py:338`). Not derived: `ignore_punctuation` is true.
    fn default() -> Self {
        Self { ignore_punctuation: true, retain_format: false }
    }
}

/// The half-open character span of the source (`replace_words` splices with it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Word<'a> {
    pub text: Cow<'a, str>,
    pub start: usize,
    pub end: usize,
}

impl Word<'_> {
    fn into_span(self) -> TokenSpan {
        TokenSpan { token: self.text.into_owned(), start: self.start, end: self.end }
    }
}

/// Characters (`ch`, reported) and bytes (`by`, for slicing), kept together.
#[derive(Debug, Clone, Copy)]
struct Pos {
    ch: usize,
    by: usize,
}

/// `split_words` for the host: `(word, start_char, end_char)`, with character offsets
/// (`replace_words` splices with them).
pub fn split_words_of(text: &str, ignore_punctuation: bool, retain_format: bool) -> Vec<TokenSpan> {
    let opts = WordTokenizerOptions { ignore_punctuation, retain_format };
    split_words(text, opts).into_iter().map(Word::into_span).collect()
}

/// Words split at whitespace, with each Han, Hiragana and Katakana character a word of its own.
/// Punctuation and combining marks join the word in progress.
pub fn split_words(text: &str, opts: WordTokenizerOptions) -> Vec<Word<'_>> {
    let mut out = Vec::new();
    scan(text, opts, |w| out.push(w));
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    Space,
    /// Punctuation and combining marks: they join the token in progress.
    Attached,
    /// Han, Hiragana and Katakana: one token per character.
    Ideograph,
    Other,
}

fn classify(c: char) -> Class {
    // Marks, variation selectors, emoji modifiers, ZWJ and the half-width (han)dakuten.
    static JOINER: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"^[\p{M}\p{gcb=Extend}\p{gcb=ZWJ}]$").expect("constant pattern")
    });
    // Script extensions, so the prolonged sound mark counts; punctuation is classified first.
    static IDEOGRAPH: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"^[\p{scx=Han}\p{scx=Hiragana}\p{scx=Katakana}]$").expect("constant pattern")
    });
    let matches = |re: &Regex| !c.is_ascii() && re.is_match(c.encode_utf8(&mut [0; 4]));
    if is_py_space(c) {
        Class::Space
    } else if is_punctuation(c) || matches(&JOINER) {
        Class::Attached
    } else if matches(&IDEOGRAPH) {
        Class::Ideograph
    } else {
        Class::Other
    }
}

/// What the token in progress holds, past any glued whitespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Run {
    Empty,
    /// Only punctuation or marks so far: the next character joins them.
    Lead,
    Word,
    Ideograph,
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

    let mut start = Pos { ch: 0, by: 0 };
    let mut end = start;
    let mut run = Run::Empty;
    for (ch, (by, c)) in text.char_indices().enumerate() {
        let pos = Pos { ch, by };
        let next = Pos { ch: ch + 1, by: by + c.len_utf8() };
        end = next;
        let (breaks, then) = match (classify(c), run) {
            (Class::Space, Run::Empty) => {
                if !opts.retain_format {
                    start = next;
                }
                continue;
            }
            (Class::Space, _) => {
                emit(start, pos);
                start = if opts.retain_format { pos } else { next };
                run = Run::Empty;
                continue;
            }
            (Class::Attached, Run::Empty) => (false, Run::Lead),
            (Class::Attached, run) => (false, run),
            (Class::Ideograph, run) => (matches!(run, Run::Word | Run::Ideograph), Run::Ideograph),
            (Class::Other, run) => (run == Run::Ideograph, Run::Word),
        };
        if breaks {
            emit(start, pos);
            start = pos;
        }
        run = then;
    }

    // Under `retain_format`, trailing whitespace is a token of its own.
    emit(start, end);
}

/// `tokenizer.is_punctuation`: [`PUNCTUATIONS`] or any Unicode punctuation (`P*`), e.g. Arabic
/// "،" and "؟", Devanagari "।", CJK "，" and "。".
#[inline]
pub fn is_punctuation(c: char) -> bool {
    static UNICODE_PUNCTUATION: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"^\p{P}$").expect("constant pattern"));
    PUNCTUATIONS.contains(&c)
        || (!c.is_ascii() && UNICODE_PUNCTUATION.is_match(c.encode_utf8(&mut [0; 4])))
}

#[cfg(test)]
#[path = "word_tests.rs"]
mod tests;

/// `tokenize.WordTokenizer`: see [`split_words`].
#[derive(Debug, Clone)]
pub struct SpeechWordTokenizer(WordTokenizerOptions);

impl lk_agents_abi::tokenize::bindings::SpeechWordTokenizer for SpeechWordTokenizer {
    type Stream = WordStream;

    fn new(ignore_punctuation: bool, retain_format: bool) -> Self {
        SpeechWordTokenizer(WordTokenizerOptions { ignore_punctuation, retain_format })
    }

    /// The words of `text`; `language` is accepted and unused.
    fn tokenize(&self, text: String, _language: Option<String>) -> Vec<String> {
        split_words(&text, self.0).into_iter().map(|w| w.text.into_owned()).collect()
    }

    fn stream(&self) -> WordStream {
        WordStream::new(self.0)
    }
}

/// Words cut as their text streams in: each goes out once the next proves it whole.
#[derive(Debug, Clone)]
pub struct WordStream(NativeTokenBuffer<Words>);

impl WordStream {
    pub fn new(opts: WordTokenizerOptions) -> Self {
        WordStream(NativeTokenBuffer::new(Words(opts), TokenBufferOptions::WORDS))
    }

    pub fn push(&mut self, text: &str) -> Vec<String> {
        self.0.push(text)
    }

    pub fn flush(&mut self) -> Vec<String> {
        self.0.flush()
    }
}

impl lk_agents_abi::tokenize::bindings::SpeechWordStream for WordStream {
    fn push(&mut self, text: String) -> Vec<String> {
        self.0.push(&text)
    }

    fn flush(&mut self) -> Vec<String> {
        self.0.flush()
    }
}
