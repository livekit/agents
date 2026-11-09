use std::borrow::Cow;

use crate::tokenize::word::{WordTokenizerOptions, split_words};
use crate::tokenize::xml;
use lk_agents_abi::SentenceSplitter;
use lk_agents_abi::tokenize::TokenSpan;
use lk_agents_abi::tokenize::bindings::TokenizerToken;

// Lengths and offsets are characters, not bytes, as in Python.

/// Character indices. Spans must tile the input start-to-start, or the text in a gap is dropped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    pub text: String,
    pub start: usize,
    pub end: usize,
}

impl From<Span> for TokenSpan {
    fn from(span: Span) -> Self {
        TokenSpan { token: span.text, start: span.start, end: span.end }
    }
}

impl Span {
    /// `push` holds it, `flush` says it whole.
    pub fn whole(text: &str) -> Span {
        Span { text: text.to_owned(), start: 0, end: text.chars().count() }
    }
}

/// Text-only callback tokens retain the reference's first-match offsets and following whitespace.
pub fn spans_from_tokens(text: &str, tokens: Vec<TokenizerToken>) -> Vec<Span> {
    tokens
        .into_iter()
        .map(|token| match token {
            TokenizerToken::Span(TokenSpan { token, start, end }) => {
                Span { text: token, start, end }
            }
            TokenizerToken::Text(token) => {
                let start = text.find(&token).map_or(0, |byte| text[..byte].chars().count());
                let mut end = start + token.chars().count();
                end += text.chars().skip(end).take_while(|c| c.is_whitespace()).count();
                Span { text: token, start, end }
            }
        })
        .collect()
}

/// Implemented only by native tokenizers: a host callback cannot run under a native buffer.
pub mod sealed {
    pub trait NativeTokenizer {}
}

/// Infallible: on failure return [`Span::whole`].
pub trait Tokenize: sealed::NativeTokenizer {
    fn tokenize(&self, text: &str) -> Vec<Span>;
}

/// Minus the tokenizer (`retain_format` is a tokenizer setting).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenBufferOptions {
    /// Batch complete tokens until the running token reaches this many characters.
    pub min_token_len: usize,
    /// Consult the tokenizer only once this much text is buffered.
    pub min_ctx_len: usize,
    /// Emit before a token would take the batch past this (an oversized token goes whole).
    pub max_token_len: Option<usize>,
    /// Never split a tag; hold a token with an unclosed one.
    pub xml_aware: bool,
}

impl TokenBufferOptions {
    /// `basic.WordTokenizer.stream()`: emit each word once a following one proves it whole.
    pub const WORDS: TokenBufferOptions = TokenBufferOptions {
        min_token_len: 1,
        min_ctx_len: 1,
        max_token_len: None,
        xml_aware: false,
    };
}

/// One tokenization requested from the queue owner, or the completed operation's text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenBufferStep {
    Tokenize(String),
    Done(Vec<String>),
}

#[derive(Debug, Clone, Copy)]
enum Operation {
    Push,
    Flush,
}

#[derive(Debug, Clone)]
enum Phase {
    Idle,
    Awaiting { operation: Operation, plan: xml::Tokenization, emitted: Vec<String> },
}

/// Pure buffering state. Its owner supplies the spans of each requested text.
#[derive(Debug, Clone)]
pub struct TokenBuffer {
    opts: TokenBufferOptions,
    in_buf: String,
    out_buf: String,
    phase: Phase,
}

impl TokenBuffer {
    pub fn new(opts: TokenBufferOptions) -> Self {
        Self { opts, in_buf: String::new(), out_buf: String::new(), phase: Phase::Idle }
    }

    pub fn push(&mut self, text: &str) -> TokenBufferStep {
        assert!(matches!(self.phase, Phase::Idle), "tokenization is already pending");
        if text.is_empty() {
            return TokenBufferStep::Done(Vec::new());
        }
        self.in_buf.push_str(text);
        if self.in_buf.chars().count() < self.opts.min_ctx_len {
            return TokenBufferStep::Done(Vec::new());
        }
        self.request(Operation::Push, Vec::new())
    }

    /// Ignores the minimum token length; the cap still applies.
    pub fn flush(&mut self) -> TokenBufferStep {
        assert!(matches!(self.phase, Phase::Idle), "tokenization is already pending");
        if self.in_buf.is_empty() && self.out_buf.is_empty() {
            return TokenBufferStep::Done(Vec::new());
        }
        self.request(Operation::Flush, Vec::new())
    }

    pub fn accept(&mut self, spans: Vec<Span>) -> TokenBufferStep {
        let Phase::Awaiting { operation, plan, emitted } =
            std::mem::replace(&mut self.phase, Phase::Idle)
        else {
            panic!("no tokenization is pending");
        };
        self.apply(operation, emitted, plan.finish(spans))
    }

    fn request(&mut self, operation: Operation, emitted: Vec<String>) -> TokenBufferStep {
        let plan = if self.opts.xml_aware {
            xml::Tokenization::new(&self.in_buf)
        } else {
            xml::Tokenization::Plain(self.in_buf.clone())
        };
        match plan.input() {
            Some(text) => {
                let text = text.to_owned();
                self.phase = Phase::Awaiting { operation, plan, emitted };
                TokenBufferStep::Tokenize(text)
            }
            None => self.apply(operation, emitted, plan.finish(Vec::new())),
        }
    }

    fn apply(
        &mut self,
        operation: Operation,
        mut emitted: Vec<String>,
        spans: Vec<Span>,
    ) -> TokenBufferStep {
        match operation {
            Operation::Push => {
                let mut spans = spans.into_iter();
                let (Some(first), Some(_second)) = (spans.next(), spans.next()) else {
                    return TokenBufferStep::Done(emitted);
                };
                if self.opts.xml_aware && xml::has_unclosed_tags(&first.text) {
                    return TokenBufferStep::Done(emitted);
                }
                self.absorb(&first.text, &mut emitted);
                if self.out_buf.chars().count() >= self.opts.min_token_len {
                    emitted.push(std::mem::take(&mut self.out_buf));
                }
                let cut = self
                    .in_buf
                    .char_indices()
                    .nth(first.end)
                    .map_or(self.in_buf.len(), |(byte, _)| byte);
                self.in_buf.drain(..cut);
                self.request(operation, emitted)
            }
            Operation::Flush => {
                self.in_buf.clear();
                for span in spans {
                    self.absorb(&span.text, &mut emitted);
                }
                if !self.out_buf.is_empty() {
                    emitted.push(std::mem::take(&mut self.out_buf));
                }
                TokenBufferStep::Done(emitted)
            }
        }
    }

    fn absorb(&mut self, token: &str, out: &mut Vec<String>) {
        if let Some(max) = self.opts.max_token_len
            && !self.out_buf.is_empty()
            && self.out_buf.chars().count() + 1 + token.chars().count() > max
        {
            out.push(std::mem::take(&mut self.out_buf));
        }
        if !self.out_buf.is_empty() {
            self.out_buf.push(' ');
        }
        self.out_buf.push_str(token);
    }
}

/// Convenience owner for built-in native tokenizers. The sealed tokenizer cannot call a host.
#[derive(Debug, Clone)]
pub struct NativeTokenBuffer<T: Tokenize> {
    tokenizer: T,
    state: TokenBuffer,
}

impl<T: Tokenize> NativeTokenBuffer<T> {
    pub fn new(tokenizer: T, opts: TokenBufferOptions) -> Self {
        Self { tokenizer, state: TokenBuffer::new(opts) }
    }

    pub fn push(&mut self, text: &str) -> Vec<String> {
        let step = self.state.push(text);
        self.drive(step)
    }

    pub fn flush(&mut self) -> Vec<String> {
        let step = self.state.flush();
        self.drive(step)
    }

    fn drive(&mut self, mut step: TokenBufferStep) -> Vec<String> {
        loop {
            step = match step {
                TokenBufferStep::Tokenize(text) => {
                    self.state.accept(self.tokenizer.tokenize(&text))
                }
                TokenBufferStep::Done(out) => return out,
            };
        }
    }
}

impl<T: Tokenize + Send> SentenceSplitter for NativeTokenBuffer<T> {
    fn push(&mut self, text: &str) -> Vec<String> {
        NativeTokenBuffer::push(self, text)
    }
    fn flush(&mut self) -> Vec<String> {
        NativeTokenBuffer::flush(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Words(pub WordTokenizerOptions);

impl sealed::NativeTokenizer for Words {}

impl Tokenize for Words {
    fn tokenize(&self, text: &str) -> Vec<Span> {
        split_words(text, self.0)
            .into_iter()
            .map(|w| Span {
                text: match w.text {
                    Cow::Borrowed(s) => s.to_owned(),
                    Cow::Owned(s) => s,
                },
                start: w.start,
                end: w.end,
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "token_buffer_tests.rs"]
mod tests;
