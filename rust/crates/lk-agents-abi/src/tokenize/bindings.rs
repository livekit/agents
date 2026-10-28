//! Export contracts for speech tokenization, implemented by the runtime over the voice core.

use crate::errors::HostError;
use std::sync::Arc;

/// A tokenizer may supply text or character-indexed spans.
#[derive(Debug, Clone, PartialEq, Eq)]
#[crate::host(record(module = "tokenizer", extract, transparent))]
pub enum TokenizerToken {
    Text(String),
    Span((String, usize, usize)),
}

/// A synchronous, application-supplied tokenizer, invoked by the runtime.
#[crate::host(proxy(host = "collections.abc.Callable"))]
pub trait TokenizerCallback: Send + Sync {
    #[host(call, name = "__call__")]
    fn tokenize(&self, text: &str) -> Result<Vec<TokenizerToken>, HostError>;
}
#[crate::host(functions(pid = false))]
pub trait TokenizeFunctions {
    #[host(function)]
    #[python(forward(module = "tokenize_basic"))]
    fn split_words(
        text: String,
        #[host(kw, default = true)] ignore_punctuation: bool,
        #[host(default = false)] split_character: bool,
        #[host(default = false)] retain_format: bool,
    ) -> Vec<(String, usize, usize)>;

    #[host(function)]
    fn basic_split_sentences(
        text: String,
        #[host(default = 20)] min_sentence_len: usize,
        #[host(kw, default = false)] retain_format: bool,
    ) -> Vec<(String, usize, usize)>;

    #[host(function)]
    fn replace_words(
        text: String,
        replacements: indexmap::IndexMap<String, String>,
        #[host(default = false)] hold_last_word: bool,
    ) -> Option<(String, usize)>;

    #[host(function)]
    fn split_paragraphs(text: String) -> Vec<(String, usize, usize)>;

    #[host(function)]
    #[python(forward(module = "tokenize_basic"))]
    fn tokenize_paragraphs(text: String) -> Vec<String>;

    #[host(function)]
    fn xml_has_unclosed_tags(text: String) -> bool;

    #[host(function)]
    fn strip_thinking(
        content: Option<String>,
        start_tag: String,
        end_tag: String,
        buffer: String,
        end_marker: Option<String>,
        is_final: bool,
    ) -> (Option<String>, String, Option<String>);

    #[host(function, pid = true)]
    fn flatten_delta_text(parts: Vec<String>) -> Option<String>;

    #[host(function)]
    fn xml_tokenize(
        tokenize_fnc: Arc<dyn TokenizerCallback>,
        text: String,
    ) -> Result<Vec<(String, usize, usize)>, HostError>;
}

/// Really naive implementation of a word tokenizer: words split on whitespace, punctuation
/// stripped unless asked to keep it.
#[crate::host(class(name = "BasicWordTokenizer", pid = false))]
#[python(wrapper(
    module = "tokenize_basic",
    path = "livekit.agents.tokenize.basic",
    name = "WordTokenizer",
    base = "..tokenize:_CoreWordTokenizer"
))]
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
    fn split(&self, text: String) -> Vec<(String, usize, usize)>;
    #[host(getter, name = "_core_min_token_len")]
    fn min_token_len(&self) -> usize;
    #[host(getter, name = "_core_min_ctx_len")]
    fn min_ctx_len(&self) -> usize;
}

/// Really naive implementation of a sentence tokenizer: rule-based, and only English is really
/// tested.
#[crate::host(class(name = "BasicSentenceTokenizer", pid = false))]
#[python(wrapper(
    module = "tokenize_basic",
    path = "livekit.agents.tokenize.basic",
    name = "SentenceTokenizer",
    base = "..tokenize:_CoreSentenceTokenizer"
))]
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
    fn split(&self, text: String) -> Vec<(String, usize, usize)>;
    #[host(getter, name = "_core_min_token_len")]
    fn min_token_len(&self) -> usize;
    #[host(getter, name = "_core_min_ctx_len")]
    fn min_ctx_len(&self) -> usize;
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
