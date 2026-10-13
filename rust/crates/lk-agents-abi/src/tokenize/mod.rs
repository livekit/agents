//! `tokenize/`, the parts on the default path. Python keeps the tokenizer ABCs (plugins subclass and
//! `isinstance` them). Offsets are characters, and `str.isspace()` is not `char::is_whitespace()`.

pub mod language;
mod replace;
pub mod search;
pub mod thinking;
mod token_buffer;
mod word;
pub mod xml;

pub use replace::{replace_words, split_paragraphs};
pub use thinking::{FilterState, Stripped, strip_thinking_tokens};
pub use token_buffer::{Span, TokenBuffer, TokenBufferOptions, Tokenize, Words};
pub use word::{
    PUNCTUATIONS, Word, WordTokenizerOptions, count_words, is_char_based, is_py_space, py_strip, split_words,
    split_words_of,
};
