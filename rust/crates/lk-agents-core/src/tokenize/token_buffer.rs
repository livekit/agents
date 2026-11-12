use lk_agents_abi::errors::HostError;
use lk_agents_abi::tokenize::TokenSpan;
use lk_agents_abi::tokenize::bindings::{BufferedTokens, TokenizerCallback, TokenizerToken};
use std::sync::Arc;

// Lengths and offsets are characters, not bytes, as in Python.

/// An application tokenizer's tokens, batched, each emitted once the next one proves it complete.
pub struct BufferedTokenStream {
    tokenizer: Arc<dyn TokenizerCallback>,
    /// Batch complete tokens until the running token reaches this many characters.
    min_token_len: usize,
    /// Consult the tokenizer only once this much text is buffered.
    min_ctx_len: usize,
    /// Emit before a token would take the batch past this (an oversized token goes whole).
    max_token_len: Option<usize>,
    in_buf: String,
    out_buf: String,
}

impl BufferedTokens for BufferedTokenStream {
    fn new(
        tokenize_fnc: Arc<dyn TokenizerCallback>,
        min_token_len: usize,
        min_ctx_len: usize,
        max_token_len: Option<usize>,
    ) -> Self {
        BufferedTokenStream {
            tokenizer: tokenize_fnc,
            min_token_len,
            min_ctx_len,
            max_token_len,
            in_buf: String::new(),
            out_buf: String::new(),
        }
    }

    /// The callback's error is reported once the push completes, the buffer left consistent.
    fn push(&mut self, text: String) -> Result<Vec<String>, HostError> {
        let mut out = Vec::new();
        if text.is_empty() {
            return Ok(out);
        }
        self.in_buf.push_str(&text);
        if self.in_buf.chars().count() < self.min_ctx_len {
            return Ok(out);
        }
        let mut failed = None;
        loop {
            let spans = self.spans(&mut failed);
            // The last token may still grow; a first token that covers no text proves nothing.
            let [first, _, ..] = spans.as_slice() else { break };
            if first.end == 0 {
                break;
            }
            self.absorb(&first.token, &mut out);
            if self.out_buf.chars().count() >= self.min_token_len {
                out.push(std::mem::take(&mut self.out_buf));
            }
            let cut =
                self.in_buf.char_indices().nth(first.end).map_or(self.in_buf.len(), |(by, _)| by);
            self.in_buf.drain(..cut);
        }
        failed.map_or(Ok(out), Err)
    }

    /// Ignores the minimum token length; the cap still applies.
    fn flush(&mut self) -> Result<Vec<String>, HostError> {
        let mut out = Vec::new();
        if self.in_buf.is_empty() && self.out_buf.is_empty() {
            return Ok(out);
        }
        let mut failed = None;
        let spans = self.spans(&mut failed);
        self.in_buf.clear();
        for span in spans {
            self.absorb(&span.token, &mut out);
        }
        if !self.out_buf.is_empty() {
            out.push(std::mem::take(&mut self.out_buf));
        }
        failed.map_or(Ok(out), Err)
    }
}

impl BufferedTokenStream {
    /// The buffered text's tokens. Spans must tile the text start-to-start, or the text in a gap
    /// is dropped; a text token takes its first match and the whitespace after it. If the
    /// tokenizer fails, its error is kept and the text is one token.
    fn spans(&self, failed: &mut Option<HostError>) -> Vec<TokenSpan> {
        let text = &self.in_buf;
        match self.tokenizer.tokenize(text) {
            Ok(tokens) => tokens
                .into_iter()
                .map(|token| match token {
                    TokenizerToken::Span(span) => span,
                    TokenizerToken::Text(token) => {
                        let start =
                            text.find(&token).map_or(0, |byte| text[..byte].chars().count());
                        let mut end = start + token.chars().count();
                        end += text.chars().skip(end).take_while(|c| c.is_whitespace()).count();
                        TokenSpan { token, start, end }
                    }
                })
                .collect(),
            Err(error) => {
                *failed = Some(error);
                if text.trim().is_empty() {
                    Vec::new()
                } else {
                    vec![TokenSpan { token: text.clone(), start: 0, end: text.chars().count() }]
                }
            }
        }
    }

    fn absorb(&mut self, token: &str, out: &mut Vec<String>) {
        if let Some(max) = self.max_token_len
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

#[cfg(test)]
#[path = "token_buffer_tests.rs"]
mod tests;
