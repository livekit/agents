use lk_agents_abi::tokenize::bindings::SpeechSentenceTokenizer as SentenceTokenizerClass;
use lk_agents_abi::tts::markup::Provider;
use lk_agents_core::lexer::Vocabulary;
use lk_agents_core::tokenize::sentences::{SentenceOptions, SentenceSplitter, SentenceStream};

/// `tokenize.SentenceTokenizer`.
pub struct SpeechSentenceTokenizer(SentenceOptions);

impl SentenceTokenizerClass for SpeechSentenceTokenizer {
    type Stream = SentenceBuffer;

    fn new(
        min_words: usize,
        clause_words: Option<usize>,
        retain_format: bool,
        max_token_len: Option<usize>,
    ) -> Self {
        SpeechSentenceTokenizer(SentenceOptions {
            min_words,
            clause_words,
            max_token_len,
            retain_format,
            markup: None,
        })
    }

    fn tokenize(&self, text: String, _language: Option<String>) -> Vec<String> {
        let sentences = SentenceSplitter::default().sentences(&text, &self.0);
        sentences.into_iter().map(|(text, ..)| text).collect()
    }

    fn stream(&self) -> SentenceBuffer {
        SentenceBuffer::with_options(self.0.clone())
    }

    fn max_token_len(&self) -> Option<usize> {
        self.0.max_token_len
    }
}

/// The sentences a synthesis speaks, cut as its text streams in.
pub struct SentenceBuffer(SentenceStream);

impl Default for SentenceBuffer {
    /// `SentenceTokenizer(retain_format=True).stream()`, what `StreamAdapter` builds. Batching is
    /// audible: a short sentence reaching the TTS alone sounds clipped.
    fn default() -> Self {
        SentenceBuffer::with_options(SentenceOptions {
            retain_format: true,
            ..SentenceOptions::default()
        })
    }
}

impl SentenceBuffer {
    pub fn with_options(opts: SentenceOptions) -> Self {
        SentenceBuffer(SentenceStream::new(opts))
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

impl lk_agents_abi::tokenize::bindings::SpeechSentenceStream for SentenceBuffer {
    fn push(&mut self, text: String) -> Vec<String> {
        self.0.push(&text)
    }

    fn flush(&mut self) -> Vec<String> {
        self.0.flush()
    }
}

/// The markup a voice speaking `markup_provider`'s dialect keeps whole, while it flows.
pub fn markup_vocabulary(markup_provider: &str, expressive: bool) -> Option<Vocabulary> {
    expressive.then(|| Vocabulary::of(Provider::from_wire(markup_provider)))
}

/// The sentence splitting a synthesis applies: `None` selects the sentence adapter's default; a
/// provider selects its gateway limits. `markup` is what is never split; under it, sentences
/// batch up to the provider's limit.
pub fn sentence_splitter(provider: Option<&str>, markup: Option<Vocabulary>) -> SentenceBuffer {
    let Some(provider) = provider else {
        return SentenceBuffer::with_options(SentenceOptions {
            retain_format: true,
            markup,
            ..SentenceOptions::default()
        });
    };
    let cfg = lk_agents_core::tts::markup::sentence_tokenizer_of(provider, markup.is_some());
    let base = SentenceOptions::default();
    SentenceBuffer::with_options(SentenceOptions {
        min_words: if cfg.batch_to_max { usize::MAX } else { base.min_words },
        max_token_len: cfg.max_token_len,
        markup,
        ..base
    })
}

#[cfg(test)]
#[path = "sentences_tests.rs"]
mod splitter_tests;
