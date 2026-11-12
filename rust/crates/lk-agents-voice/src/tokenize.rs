/// The host's text tokenization functions (`tokenize.split_words`, `tokenize.replace_words`).
pub struct SpeechTokenizer;

impl lk_agents_abi::tokenize::bindings::TokenizeFunctions for SpeechTokenizer {
    fn split_words(
        text: String,
        ignore_punctuation: bool,
        retain_format: bool,
    ) -> Vec<lk_agents_abi::tokenize::TokenSpan> {
        lk_agents_core::tokenize::split_words_of(&text, ignore_punctuation, retain_format)
    }
    fn replace_words(
        text: String,
        replacements: indexmap::IndexMap<String, String>,
        hold_last_word: bool,
    ) -> Option<lk_agents_abi::tokenize::bindings::ReplacedWords> {
        lk_agents_core::tokenize::replace_words(&text, replacements, hold_last_word).map(
            |(text, processed)| lk_agents_abi::tokenize::bindings::ReplacedWords {
                text,
                processed,
            },
        )
    }
}
