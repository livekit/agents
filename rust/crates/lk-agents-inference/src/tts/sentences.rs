use lk_agents_media::blingfire::{Error, split_sentences};

use lk_agents_abi::tts::markup::SentenceTokenizerConfig;
use lk_agents_core::tokenize::{
    NativeTokenBuffer as TokenBuffer, Span as TextSpan, TokenBufferOptions, Tokenize,
};

/// `tokenize/blingfire.py::SentenceTokenizer`'s two model knobs; batching is [`TokenBuffer`]'s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Blingfire {
    pub min_sentence_len: usize,
    pub retain_format: bool,
}

impl lk_agents_core::tokenize::sealed::NativeTokenizer for Blingfire {}

impl Tokenize for Blingfire {
    fn tokenize(&self, text: &str) -> Vec<TextSpan> {
        match split_sentences(text, self.min_sentence_len, self.retain_format) {
            Ok(spans) => {
                spans.into_iter().map(|(text, start, end)| TextSpan { text, start, end }).collect()
            }
            // A NUL or a C++ failure is not worth failing a turn: the text comes back whole.
            Err(Error::InteriorNul) | Err(Error::Failed) => vec![TextSpan::whole(text)],
        }
    }
}

/// `tokenize.blingfire.SentenceTokenizer`: [`Blingfire`] sentences, batched by the host's
/// buffered stream with these options.
pub struct BlingfireSentenceTokenizer {
    blingfire: Blingfire,
    stream_context_len: usize,
    max_token_len: Option<usize>,
    min_token_len: Option<usize>,
    xml_aware: bool,
}

impl lk_agents_abi::tokenize::bindings::BlingfireSentenceTokenizer for BlingfireSentenceTokenizer {
    fn new(
        min_sentence_len: usize,
        stream_context_len: usize,
        retain_format: bool,
        max_token_len: Option<usize>,
        min_token_len: Option<usize>,
        xml_aware: bool,
    ) -> Self {
        BlingfireSentenceTokenizer {
            blingfire: Blingfire { min_sentence_len, retain_format },
            stream_context_len,
            max_token_len,
            min_token_len,
            xml_aware,
        }
    }

    fn tokenize(&self, text: String, _language: Option<String>) -> Vec<String> {
        if !self.xml_aware {
            return self.blingfire.tokenize(&text).into_iter().map(|s| s.text).collect();
        }
        let plan = lk_agents_core::tokenize::xml::Tokenization::new(&text);
        let spans = plan.input().map(|input| self.blingfire.tokenize(input)).unwrap_or_default();
        plan.finish(spans).into_iter().map(|s| s.text).collect()
    }

    fn split(&self, text: String) -> Vec<lk_agents_abi::tokenize::TokenSpan> {
        self.blingfire.tokenize(&text).into_iter().map(Into::into).collect()
    }

    fn min_token_len(&self) -> usize {
        self.min_token_len.unwrap_or(self.blingfire.min_sentence_len)
    }

    fn min_ctx_len(&self) -> usize {
        self.stream_context_len
    }

    fn max_token_len(&self) -> Option<usize> {
        self.max_token_len
    }

    fn xml_aware(&self) -> bool {
        self.xml_aware
    }
}

/// `TokenBuffer<Blingfire>` with the reference's `BufferedSentenceStream` defaults. The
/// parameters are audible: without batching, short utterances reach the TTS alone and sound clipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SentenceBufferOptions {
    pub min_sentence_len: usize,
    /// `BufferedTokenStream(min_token_len=)`; the reference defaults it to `min_sentence_len`.
    pub min_token_len: usize,
    /// `BufferedTokenStream(min_ctx_len=)`, i.e. `stream_context_len`.
    pub min_ctx_len: usize,
    /// `None` is the reference's default: no cap.
    pub max_token_len: Option<usize>,
    /// `true` is what `StreamAdapter` asks for.
    pub retain_format: bool,
    /// Never split a markup tag; on only under expressive.
    pub xml_aware: bool,
}

impl Default for SentenceBufferOptions {
    /// `SentenceTokenizer(retain_format=True).stream()`, as `tts/stream_adapter.py:40` builds it.
    fn default() -> Self {
        SentenceBufferOptions {
            min_sentence_len: 20,
            min_token_len: 20,
            min_ctx_len: 10,
            max_token_len: None,
            retain_format: true,
            xml_aware: false,
        }
    }
}

impl SentenceBufferOptions {
    /// As `tts.py` applies `Markup.sentence_tokenizer(expressive)`: the cap is the provider's max,
    /// the floor is that cap under expressive, and `xml_aware` follows expressive.
    pub fn from_config(cfg: SentenceTokenizerConfig) -> Self {
        let base = SentenceBufferOptions::default();
        SentenceBufferOptions {
            max_token_len: cfg.max_token_len,
            min_token_len: cfg.min_token_len.unwrap_or(base.min_sentence_len),
            xml_aware: cfg.xml_aware,
            ..base
        }
    }

    fn split(self) -> (Blingfire, TokenBufferOptions) {
        (
            Blingfire {
                min_sentence_len: self.min_sentence_len,
                retain_format: self.retain_format,
            },
            TokenBufferOptions {
                min_token_len: self.min_token_len,
                min_ctx_len: self.min_ctx_len,
                max_token_len: self.max_token_len,
                xml_aware: self.xml_aware,
            },
        )
    }
}

#[derive(Debug, Clone)]
pub struct SentenceBuffer(TokenBuffer<Blingfire>);

impl Default for SentenceBuffer {
    fn default() -> Self {
        SentenceBuffer::new()
    }
}

impl SentenceBuffer {
    pub fn new() -> Self {
        SentenceBuffer::with_options(SentenceBufferOptions::default())
    }

    pub fn with_options(opts: SentenceBufferOptions) -> Self {
        let (tok, buf) = opts.split();
        SentenceBuffer(TokenBuffer::new(tok, buf))
    }
}

impl lk_agents_abi::SentenceSplitter for SentenceBuffer {
    fn push(&mut self, text: &str) -> Vec<String> {
        self.0.push(text)
    }

    fn flush(&mut self) -> Vec<String> {
        self.0.flush()
    }
}

/// The sentence splitting a synthesis applies: `None` selects the sentence adapter's default; a
/// provider selects its gateway limits.
pub fn sentence_splitter(provider: Option<&str>, expressive: bool) -> SentenceBuffer {
    let opts = match provider {
        Some(provider) => {
            let (max_token_len, min_token_len, xml_aware) =
                lk_agents_core::tts::markup::sentence_tokenizer_of(provider, expressive);
            SentenceBufferOptions {
                retain_format: false,
                ..SentenceBufferOptions::from_config(SentenceTokenizerConfig {
                    max_token_len,
                    min_token_len,
                    xml_aware,
                })
            }
        }
        None => SentenceBufferOptions { xml_aware: expressive, ..Default::default() },
    };
    SentenceBuffer::with_options(opts)
}

#[cfg(test)]
#[path = "sentences_tests.rs"]
mod splitter_tests;
