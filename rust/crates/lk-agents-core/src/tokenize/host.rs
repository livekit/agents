//! Tokenization through an application's tokenizer callback: the host executes the voice core's
//! tokenization steps, and a failing callback falls back to the whole text.

use std::sync::Arc;

use super::{Span, TokenBuffer, TokenBufferOptions, TokenBufferStep, spans_from_tokens, xml};
use lk_agents_abi::errors::HostError;
use lk_agents_abi::tokenize::bindings::{BufferedTokens, HostTokenization, TokenizerCallback};

pub struct HostTokenizer;

impl HostTokenization for HostTokenizer {
    fn xml_tokenize(
        tokenize_fnc: Arc<dyn TokenizerCallback>,
        text: String,
    ) -> Result<Vec<(String, usize, usize)>, HostError> {
        let plan = xml::Tokenization::new(&text);
        let spans = plan
            .input()
            .map(|input| {
                supplied_spans(tokenize_fnc.as_ref(), input).unwrap_or_else(|_| fallback(input))
            })
            .unwrap_or_default();
        Ok(plan.finish(spans).into_iter().map(|span| (span.text, span.start, span.end)).collect())
    }
}

fn supplied_spans(tokenizer: &dyn TokenizerCallback, text: &str) -> Result<Vec<Span>, HostError> {
    tokenizer.tokenize(text).map(|tokens| spans_from_tokens(text, tokens))
}

fn fallback(text: &str) -> Vec<Span> {
    if text.trim().is_empty() { Vec::new() } else { vec![Span::whole(text)] }
}

pub struct BufferedTokenStream {
    inner: TokenBuffer,
    tokenizer: Arc<dyn TokenizerCallback>,
    xml_aware: bool,
}

impl BufferedTokens for BufferedTokenStream {
    fn new(
        tokenize_fnc: Arc<dyn TokenizerCallback>,
        min_token_len: usize,
        min_ctx_len: usize,
        max_token_len: Option<usize>,
        xml_aware: bool,
    ) -> Self {
        let opts = TokenBufferOptions { min_token_len, min_ctx_len, max_token_len, xml_aware };
        Self { inner: TokenBuffer::new(opts), tokenizer: tokenize_fnc, xml_aware }
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
    /// An XML-aware stream keeps going on the fallback; a plain one reports the callback's error
    /// once the step completes.
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
                        Some(error) if !self.xml_aware => Err(error),
                        _ => Ok(out),
                    };
                }
            };
        }
    }
}
