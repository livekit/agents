use lk_agents_abi::errors::HostError;
use lk_agents_abi::tokenize::bindings::{BufferedTokens, TokenizerCallback};
use std::sync::Arc;

pub use lk_agents_abi::tokenize::language;
mod replace;
#[cfg(test)]
mod replace_tests;

pub mod sentences;

mod token_buffer;
mod word;

pub use replace::replace_words;
pub use sentences::spoken_words;

use token_buffer::{Span, TokenBuffer, TokenBufferOptions, TokenBufferStep, spans_from_tokens};
pub use word::{
    PUNCTUATIONS, SpeechWordTokenizer, Word, WordStream, WordTokenizerOptions, is_punctuation,
    is_py_space, py_strip, split_words, split_words_of,
};

// Offsets are characters, and `str.isspace()` is not `char::is_whitespace()`.

fn supplied_spans(tokenizer: &dyn TokenizerCallback, text: &str) -> Result<Vec<Span>, HostError> {
    tokenizer.tokenize(text).map(|tokens| spans_from_tokens(text, tokens))
}

fn fallback(text: &str) -> Vec<Span> {
    if text.trim().is_empty() { Vec::new() } else { vec![Span::whole(text)] }
}

pub struct BufferedTokenStream {
    inner: TokenBuffer,
    tokenizer: Arc<dyn TokenizerCallback>,
}

impl BufferedTokens for BufferedTokenStream {
    fn new(
        tokenize_fnc: Arc<dyn TokenizerCallback>,
        min_token_len: usize,
        min_ctx_len: usize,
        max_token_len: Option<usize>,
    ) -> Self {
        let opts = TokenBufferOptions { min_token_len, min_ctx_len, max_token_len };
        Self { inner: TokenBuffer::new(opts), tokenizer: tokenize_fnc }
    }
    fn push(&mut self, text: String) -> Result<Vec<String>, HostError> {
        let step = self.inner.push(&text);
        self.drive(step)
    }
    fn flush(&mut self) -> Result<Vec<String>, HostError> {
        let step = self.inner.flush();
        self.drive(step)
    }
}

impl BufferedTokenStream {
    /// The callback's error is reported once the step completes, the buffer left consistent.
    fn drive(&mut self, mut step: TokenBufferStep) -> Result<Vec<String>, HostError> {
        let mut failed = None;
        loop {
            step = match step {
                TokenBufferStep::Tokenize(text) => {
                    let spans =
                        supplied_spans(self.tokenizer.as_ref(), &text).unwrap_or_else(|error| {
                            failed = Some(error);
                            fallback(&text)
                        });
                    self.inner.accept(spans)
                }
                TokenBufferStep::Done(out) => {
                    return match failed {
                        Some(error) => Err(error),
                        None => Ok(out),
                    };
                }
            };
        }
    }
}
