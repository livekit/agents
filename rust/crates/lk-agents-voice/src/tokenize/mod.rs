//! `tokenize/`, the parts on the default path. Python keeps the tokenizer ABCs (plugins subclass and
//! `isinstance` them). Offsets are characters, and `str.isspace()` is not `char::is_whitespace()`.

pub use lk_agents_abi::tokenize::language;
mod replace;

mod sentence;

mod token_buffer;
mod word;
pub mod xml;

pub use replace::{replace_words, split_paragraphs, tokenize_paragraphs};
pub use sentence::{BasicSentenceTokenizer, basic_split_sentences};

pub use token_buffer::{
    NativeTokenBuffer, Span, TokenBuffer, TokenBufferOptions, TokenBufferStep, Tokenize, Words,
    spans_from_tokens,
};
pub use word::{
    BasicWordTokenizer, PUNCTUATIONS, Word, WordTokenizerOptions, count_words, is_char_based,
    is_punctuation, is_py_space, py_strip, split_words, split_words_of,
};

pub mod blingfire;

/// The host's text tokenization functions (`tokenize.basic`, `tokenize.utils`).
pub struct SpeechTokenizer;

impl lk_agents_abi::tokenize::bindings::TokenizeFunctions for SpeechTokenizer {
    fn split_words(
        text: String,
        ignore_punctuation: bool,
        split_character: bool,
        retain_format: bool,
    ) -> Vec<(String, usize, usize)> {
        split_words_of(&text, ignore_punctuation, split_character, retain_format)
    }
    fn basic_split_sentences(
        text: String,
        min_sentence_len: usize,
        retain_format: bool,
    ) -> Vec<(String, usize, usize)> {
        basic_split_sentences(&text, min_sentence_len, retain_format)
    }
    fn replace_words(
        text: String,
        replacements: indexmap::IndexMap<String, String>,
        hold_last_word: bool,
    ) -> Option<(String, usize)> {
        replace_words(&text, replacements, hold_last_word)
    }
    fn split_paragraphs(text: String) -> Vec<(String, usize, usize)> {
        split_paragraphs(&text)
    }
    fn tokenize_paragraphs(text: String) -> Vec<String> {
        tokenize_paragraphs(&text)
    }
    fn xml_has_unclosed_tags(text: String) -> bool {
        xml::has_unclosed_tags(&text)
    }
}
