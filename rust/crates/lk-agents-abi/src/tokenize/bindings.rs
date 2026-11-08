//! Export contracts for speech tokenization, implemented by the voice core.

use crate::errors::HostError;
use std::sync::Arc;

/// A tokenizer may supply text or character-indexed spans.
#[derive(Debug, Clone, PartialEq, Eq)]
#[crate::host(record(module = "tokenizer", extract, transparent))]
pub enum TokenizerToken {
    Text(String),
    Span(super::TokenSpan),
}

/// `replace_words` over a chunk: the text with its replacements, and how much of it is final
/// (`processed`, in characters); the rest may still change with the next chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
#[crate::host(record(module = "tokenizer", dataclass, emit))]
pub struct ReplacedWords {
    pub text: String,
    pub processed: usize,
}

/// A synchronous, application-supplied tokenizer, invoked by the runtime.
#[crate::host(proxy(host = "collections.abc.Callable"))]
pub trait TokenizerCallback: Send + Sync {
    #[host(call, name = "__call__")]
    fn tokenize(&self, text: &str) -> Result<Vec<TokenizerToken>, HostError>;
}
#[crate::host(functions(pid = false))]
pub trait TokenizeFunctions {
    #[host(function, path = "livekit.agents.tokenize.basic")]
    fn split_words(
        text: String,
        #[host(kw, default = true)] ignore_punctuation: bool,
        #[host(default = false)] split_character: bool,
        #[host(default = false)] retain_format: bool,
    ) -> Vec<super::TokenSpan>;

    #[host(function)]
    fn replace_words(
        text: String,
        replacements: indexmap::IndexMap<String, String>,
        #[host(default = false)] hold_last_word: bool,
    ) -> Option<ReplacedWords>;

    #[host(function, path = "livekit.agents.tokenize.basic")]
    fn tokenize_paragraphs(text: String) -> Vec<String>;

    /// `hyphenate_word(word)`: the pieces, broken at the possible hyphenation points.
    #[host(function, path = "livekit.agents.tokenize.basic")]
    fn hyphenate_word(word: String) -> Vec<String>;
}

/// Tokenization through an application tokenizer: the host calls the voice core's steps request.
#[crate::host(functions(pid = false))]
pub trait HostTokenization {
    #[host(function)]
    fn xml_tokenize(
        tokenize_fnc: Arc<dyn TokenizerCallback>,
        text: String,
    ) -> Result<Vec<super::TokenSpan>, HostError>;
}

/// `tokenize.basic.WordTokenizer`'s engine.
#[crate::host(class(name = "BasicWordTokenizer", pid = false))]
pub trait WordTokenizer: Send + Sized + 'static {
    #[host(new)]
    fn new(
        #[host(kw, default = true)] ignore_punctuation: bool,
        #[host(default = false)] split_character: bool,
        #[host(default = false)] retain_format: bool,
    ) -> Self;
    #[host(call)]
    fn tokenize(
        &self,
        text: String,
        #[host(kw, default = None)] language: Option<String>,
    ) -> Vec<String>;
    #[host(call, name = "_core_split")]
    fn split(&self, text: String) -> Vec<super::TokenSpan>;
    #[host(getter, name = "_core_min_token_len")]
    fn min_token_len(&self) -> usize;
    #[host(getter, name = "_core_min_ctx_len")]
    fn min_ctx_len(&self) -> usize;
}

/// `tokenize.basic.SentenceTokenizer`'s engine.
#[crate::host(class(name = "BasicSentenceTokenizer", pid = false))]
pub trait SentenceTokenizer: Send + Sized + 'static {
    #[host(new)]
    fn new(
        #[host(kw, default = "english")] language: String,
        #[host(default = 20)] min_sentence_len: usize,
        #[host(default = 10)] stream_context_len: usize,
        #[host(default = false)] retain_format: bool,
    ) -> Self;
    #[host(call)]
    fn tokenize(
        &self,
        text: String,
        #[host(kw, default = None)] language: Option<String>,
    ) -> Vec<String>;
    #[host(call, name = "_core_split")]
    fn split(&self, text: String) -> Vec<super::TokenSpan>;
    #[host(getter, name = "_core_min_token_len")]
    fn min_token_len(&self) -> usize;
    #[host(getter, name = "_core_min_ctx_len")]
    fn min_ctx_len(&self) -> usize;
    #[host(getter, name = "_core_max_token_len")]
    fn max_token_len(&self) -> Option<usize>;
    #[host(getter, name = "_core_xml_aware")]
    fn xml_aware(&self) -> bool;
}

/// `tokenize.blingfire.SentenceTokenizer`'s engine.
#[crate::host(class(name = "BlingfireSentenceTokenizer", pid = false))]
pub trait BlingfireSentenceTokenizer: Send + Sized + 'static {
    #[host(new)]
    fn new(
        #[host(kw, default = 20)] min_sentence_len: usize,
        #[host(default = 10)] stream_context_len: usize,
        #[host(default = false)] retain_format: bool,
        #[host(default = None)] max_token_len: Option<usize>,
        #[host(default = None)] min_token_len: Option<usize>,
        #[host(default = false)] xml_aware: bool,
    ) -> Self;
    /// The sentences of `text`, markup kept whole when `xml_aware`.
    #[host(call)]
    fn tokenize(
        &self,
        text: String,
        #[host(kw, default = None)] language: Option<String>,
    ) -> Vec<String>;
    #[host(call, name = "_core_split")]
    fn split(&self, text: String) -> Vec<super::TokenSpan>;
    #[host(getter, name = "_core_min_token_len")]
    fn min_token_len(&self) -> usize;
    #[host(getter, name = "_core_min_ctx_len")]
    fn min_ctx_len(&self) -> usize;
    #[host(getter, name = "_core_max_token_len")]
    fn max_token_len(&self) -> Option<usize>;
    #[host(getter, name = "_core_xml_aware")]
    fn xml_aware(&self) -> bool;
}

/// Really naive implementation of a word tokenizer: words split on whitespace, punctuation
/// stripped unless asked to keep it.
#[crate::host(facade(
    path = "livekit.agents.tokenize.basic",
    name = "WordTokenizer",
    base = "tokenize._CoreWordTokenizer",
    native = WordTokenizer
))]
pub trait BasicWords {
    #[host(new)]
    fn new(
        #[host(kw, default = true)] ignore_punctuation: bool,
        #[host(default = false)] split_character: bool,
        #[host(default = false)] retain_format: bool,
    ) -> Self;
    fn tokenize(
        &self,
        text: String,
        #[host(kw, default = None)] language: Option<String>,
    ) -> Vec<String>;
}

/// Really naive implementation of a sentence tokenizer: rule-based, and only English is really
/// tested.
#[crate::host(facade(
    path = "livekit.agents.tokenize.basic",
    name = "SentenceTokenizer",
    base = "tokenize._CoreSentenceTokenizer",
    native = SentenceTokenizer
))]
pub trait BasicSentences {
    #[host(new)]
    fn new(
        #[host(kw, default = "english")] language: String,
        #[host(default = 20)] min_sentence_len: usize,
        #[host(default = 10)] stream_context_len: usize,
        #[host(default = false)] retain_format: bool,
    ) -> Self;
    fn tokenize(
        &self,
        text: String,
        #[host(kw, default = None)] language: Option<String>,
    ) -> Vec<String>;
}

/// Sentences split by the BlingFire model, batched for speech.
///
/// Args:
///     min_sentence_len: minimum length for a span to be treated as its own sentence; shorter
///         spans are merged forward into the next one.
///     stream_context_len: minimum buffered text before the stream emits.
///     retain_format: keep original whitespace/formatting in emitted tokens.
///     max_token_len: hard cap on emitted token length; a token is flushed before appending a
///         sentence that would exceed it.
///     min_token_len: minimum length a token must reach before it is emitted. Sentences are
///         batched until the running token reaches this length, so raising it (e.g. toward
///         ``max_token_len``) yields larger, fewer chunks. Defaults to ``min_sentence_len``.
///     xml_aware: treat XML markup as atomic: never split a tag across tokens and keep tags
///         attached to the following sentence. Only enable when the input carries markup (e.g.
///         expressive TTS): a stray "<" in plain text can otherwise hold back streaming.
#[crate::host(facade(
    path = "livekit.agents.tokenize.blingfire",
    name = "SentenceTokenizer",
    base = "tokenize._CoreSentenceTokenizer",
    native = BlingfireSentenceTokenizer
))]
pub trait BlingfireSentences {
    #[host(new)]
    fn new(
        #[host(kw, default = 20)] min_sentence_len: usize,
        #[host(default = 10)] stream_context_len: usize,
        #[host(default = false)] retain_format: bool,
        #[host(default = None)] max_token_len: Option<usize>,
        #[host(default = None)] min_token_len: Option<usize>,
        #[host(default = false)] xml_aware: bool,
    ) -> Self;
    /// The sentences of `text`, markup kept whole when `xml_aware`.
    fn tokenize(
        &self,
        text: String,
        #[host(kw, default = None)] language: Option<String>,
    ) -> Vec<String>;
}

#[crate::host(class(name = "TokenBuffer", pid = false))]
pub trait BufferedTokens: Send + Sized + 'static {
    #[host(new)]
    fn new(
        tokenize_fnc: Arc<dyn TokenizerCallback>,
        #[host(kw)] min_token_len: usize,
        min_ctx_len: usize,
        #[host(default = None)] max_token_len: Option<usize>,
        #[host(default = false)] xml_aware: bool,
    ) -> Self;
    /// The tokens ``text`` proves complete, batched.
    #[host(call)]
    fn push(&mut self, text: String) -> Result<Vec<String>, HostError>;
    #[host(call)]
    fn flush(&mut self) -> Result<Vec<String>, HostError>;
}
