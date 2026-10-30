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

pub use token_buffer::sealed;
pub use token_buffer::{
    NativeTokenBuffer, Span, TokenBuffer, TokenBufferOptions, TokenBufferStep, Tokenize, Words,
    spans_from_tokens,
};
pub use word::{
    BasicWordTokenizer, PUNCTUATIONS, Word, WordTokenizerOptions, count_words, is_char_based,
    is_punctuation, is_py_space, py_strip, split_words, split_words_of,
};

mod host;
pub use host::{BufferedTokenStream, HostTokenizer};
