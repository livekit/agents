//! The host's tokenization functions; the tokenizers are `lk_agents_core::tokenize`'s.

/// The host's text tokenization functions (`tokenize.basic`, `tokenize.utils`).
pub struct SpeechTokenizer;

impl lk_agents_abi::tokenize::bindings::TokenizeFunctions for SpeechTokenizer {
    fn split_words(
        text: String,
        ignore_punctuation: bool,
        split_character: bool,
        retain_format: bool,
    ) -> Vec<(String, usize, usize)> {
        lk_agents_core::tokenize::split_words_of(
            &text,
            ignore_punctuation,
            split_character,
            retain_format,
        )
    }
    fn replace_words(
        text: String,
        replacements: indexmap::IndexMap<String, String>,
        hold_last_word: bool,
    ) -> Option<(String, usize)> {
        lk_agents_core::tokenize::replace_words(&text, replacements, hold_last_word)
    }
    fn tokenize_paragraphs(text: String) -> Vec<String> {
        lk_agents_core::tokenize::tokenize_paragraphs(&text)
    }
    fn hyphenate_word(word: String) -> Vec<String> {
        crate::transcription::synchronizer::hyphenate_word(&word)
    }
}
