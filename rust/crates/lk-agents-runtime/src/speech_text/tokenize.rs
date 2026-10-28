//! Speech tokenizer exports and host-authored tokenizer callbacks.

use lk_agents_abi::errors::HostError;
use lk_agents_abi::tokenize::bindings::{
    BufferedTokens, SentenceTokenizer, TokenizeFunctions, TokenizerCallback, WordTokenizer,
};
use lk_agents_voice::tokenize::{
    self as voice, Span, TokenBuffer, TokenBufferOptions, TokenBufferStep,
};
use std::sync::Arc;

pub struct SpeechTokenizer;

impl TokenizeFunctions for SpeechTokenizer {
    fn split_words(
        text: String,
        ignore_punctuation: bool,
        split_character: bool,
        retain_format: bool,
    ) -> Vec<(String, usize, usize)> {
        voice::split_words_of(&text, ignore_punctuation, split_character, retain_format)
    }
    fn basic_split_sentences(
        text: String,
        min_sentence_len: usize,
        retain_format: bool,
    ) -> Vec<(String, usize, usize)> {
        voice::basic_split_sentences(&text, min_sentence_len, retain_format)
    }
    fn replace_words(
        text: String,
        replacements: indexmap::IndexMap<String, String>,
        hold_last_word: bool,
    ) -> Option<(String, usize)> {
        voice::replace_words(&text, replacements, hold_last_word)
    }
    fn split_paragraphs(text: String) -> Vec<(String, usize, usize)> {
        voice::split_paragraphs(&text)
    }
    fn tokenize_paragraphs(text: String) -> Vec<String> {
        voice::tokenize_paragraphs(&text)
    }
    fn xml_has_unclosed_tags(text: String) -> bool {
        voice::xml::has_unclosed_tags(&text)
    }
    fn strip_thinking(
        content: Option<String>,
        start_tag: String,
        end_tag: String,
        buffer: String,
        end_marker: Option<String>,
        is_final: bool,
    ) -> (Option<String>, String, Option<String>) {
        lk_agents_inference::llm::thinking::strip_thinking(
            content.as_deref(),
            &start_tag,
            &end_tag,
            &buffer,
            end_marker.as_deref(),
            is_final,
        )
    }
    fn flatten_delta_text(parts: Vec<String>) -> Option<String> {
        lk_agents_inference::llm::thinking::flatten_text_parts(&parts)
    }
    fn xml_tokenize(
        tokenize_fnc: Arc<dyn TokenizerCallback>,
        text: String,
    ) -> Result<Vec<(String, usize, usize)>, HostError> {
        let plan = voice::xml::Tokenization::new(&text);
        let spans = plan
            .input()
            .map(|input| {
                supplied_spans(tokenize_fnc.as_ref(), input).unwrap_or_else(|_| fallback(input))
            })
            .unwrap_or_default();
        Ok(plan.finish(spans).into_iter().map(|span| (span.text, span.start, span.end)).collect())
    }
}

pub struct BasicWordTokenizer(voice::BasicWordTokenizer);
impl WordTokenizer for BasicWordTokenizer {
    fn new(ignore_punctuation: bool, split_character: bool, retain_format: bool) -> Self {
        Self(voice::BasicWordTokenizer::new(ignore_punctuation, split_character, retain_format))
    }
    fn tokenize(&self, text: String, language: Option<String>) -> Vec<String> {
        self.0.tokenize(&text, language.as_deref())
    }
    fn split(&self, text: String) -> Vec<(String, usize, usize)> {
        self.0.split(&text)
    }
    fn min_token_len(&self) -> usize {
        self.0.min_token_len()
    }
    fn min_ctx_len(&self) -> usize {
        self.0.min_ctx_len()
    }
}

pub struct BasicSentenceTokenizer(voice::BasicSentenceTokenizer);
impl SentenceTokenizer for BasicSentenceTokenizer {
    fn new(
        language: String,
        min_sentence_len: usize,
        stream_context_len: usize,
        retain_format: bool,
    ) -> Self {
        Self(voice::BasicSentenceTokenizer::new(
            &language,
            min_sentence_len,
            stream_context_len,
            retain_format,
        ))
    }
    fn tokenize(&self, text: String, language: Option<String>) -> Vec<String> {
        self.0.tokenize(&text, language.as_deref())
    }
    fn split(&self, text: String) -> Vec<(String, usize, usize)> {
        self.0.split(&text)
    }
    fn min_token_len(&self) -> usize {
        self.0.min_token_len()
    }
    fn min_ctx_len(&self) -> usize {
        self.0.min_ctx_len()
    }
}

fn supplied_spans(tokenizer: &dyn TokenizerCallback, text: &str) -> Result<Vec<Span>, HostError> {
    tokenizer.tokenize(text).map(|tokens| voice::spans_from_tokens(text, tokens))
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
    fn drive(&mut self, mut step: TokenBufferStep) -> Result<Vec<String>, HostError> {
        let mut failed = None;
        loop {
            step = match step {
                TokenBufferStep::Tokenize(text) => {
                    let spans = match supplied_spans(self.tokenizer.as_ref(), &text) {
                        Ok(spans) => spans,
                        Err(error) => {
                            failed = Some(error);
                            fallback(&text)
                        }
                    };
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
