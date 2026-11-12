pub use lk_agents_abi::tokenize::language;
mod replace;
#[cfg(test)]
mod replace_tests;

pub mod sentences;

mod token_buffer;
mod word;

pub use replace::replace_words;
pub use sentences::spoken_words;
pub use token_buffer::BufferedTokenStream;
pub use word::{
    PUNCTUATIONS, SpeechWordTokenizer, Word, WordStream, WordTokenizerOptions, is_punctuation,
    split_words, split_words_of,
};

/// The host's text tokenization functions (`tokenize.split_words`, `tokenize.replace_words`).
pub struct SpeechTokenizer;

impl lk_agents_abi::tokenize::bindings::TokenizeFunctions for SpeechTokenizer {
    fn split_words(
        text: String,
        ignore_punctuation: bool,
        retain_format: bool,
    ) -> Vec<lk_agents_abi::tokenize::TokenSpan> {
        split_words_of(&text, ignore_punctuation, retain_format)
    }

    fn replace_words(
        text: String,
        replacements: indexmap::IndexMap<String, String>,
        hold_last_word: bool,
    ) -> Option<lk_agents_abi::tokenize::bindings::ReplacedWords> {
        replace_words(&text, replacements, hold_last_word).map(|(text, processed)| {
            lk_agents_abi::tokenize::bindings::ReplacedWords { text, processed }
        })
    }
}
