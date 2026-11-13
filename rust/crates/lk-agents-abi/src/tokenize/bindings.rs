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
    /// The words of `text`, each with its character offsets: what `WordTokenizer` splits.
    #[host(function, path = "livekit.agents.tokenize")]
    fn split_words(
        text: String,
        #[host(kw, default = true)] ignore_punctuation: bool,
        #[host(default = false)] retain_format: bool,
    ) -> Vec<super::TokenSpan>;

    #[host(function)]
    fn replace_words(
        text: String,
        replacements: indexmap::IndexMap<String, String>,
        #[host(default = false)] hold_last_word: bool,
    ) -> Option<ReplacedWords>;
}

/// A built-in `tokenize.SentenceTokenizer`'s settings, which a TTS's stage splits by natively.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SentenceSettings {
    pub min_words: usize,
    pub clause_words: Option<usize>,
    pub retain_format: bool,
    pub max_token_len: Option<usize>,
}

impl Default for SentenceSettings {
    /// `SentenceTokenizer()`'s.
    fn default() -> Self {
        SentenceSettings {
            min_words: 4,
            clause_words: None,
            retain_format: false,
            max_token_len: None,
        }
    }
}

/// A built-in `tokenize.SentenceTokenizer`, as a TTS's stage reads it.
#[crate::host(native(refused = "expected a tokenize.SentenceTokenizer"))]
pub trait NativeSentences: Send + Sync {
    fn settings(&self) -> SentenceSettings;
}

impl NativeSentences for SentenceSettings {
    fn settings(&self) -> SentenceSettings {
        *self
    }
}

/// What splits a TTS's text: a built-in `tokenize.SentenceTokenizer`, which the stage runs
/// natively with its settings, or a custom one, whose sentences it reads from the host.
#[crate::host(record(module = "tts", extract, transparent))]
#[derive(Clone)]
pub enum SentenceTokenizerLeg {
    Native(Arc<dyn NativeSentences>),
    Host(Arc<dyn crate::tts::stream_adapter::HostSentenceTokenizer>),
}

/// `tokenize.SentenceTokenizer`'s engine.
#[crate::host(class(name = "SpeechSentenceTokenizer", pid = false, native = NativeSentences))]
pub trait SpeechSentenceTokenizer: Send + Sized + 'static {
    #[host(class(name = "SpeechSentenceStream"))]
    type Stream;
    #[host(init)]
    fn new(
        #[host(kw, default = 4)] min_words: usize,
        #[host(default = None)] clause_words: Option<usize>,
        #[host(default = false)] retain_format: bool,
        #[host(default = None)] max_token_len: Option<usize>,
    ) -> Self;
    /// The sentences of `text`, batched as the stream batches them.
    #[host(call)]
    fn tokenize(
        &self,
        text: String,
        #[host(kw, default = None)] language: Option<String>,
    ) -> Vec<String>;
    #[host(call, name = "_core_stream")]
    fn stream(&self) -> Self::Stream;
    #[host(getter, name = "_core_max_token_len")]
    fn max_token_len(&self) -> Option<usize>;
}

/// `tokenize.SentenceTokenizer`'s stream: a sentence goes out once no text to come could
/// change it.
#[crate::host(class(name = "SpeechSentenceStream", pid = false))]
pub trait SpeechSentenceStream: Send + Sized + 'static {
    /// The sentences ``text`` completes.
    #[host(call)]
    fn push(&mut self, text: String) -> Vec<String>;
    #[host(call)]
    fn flush(&mut self) -> Vec<String>;
}

/// `tokenize.WordTokenizer`'s engine.
#[crate::host(class(name = "SpeechWordTokenizer", pid = false))]
pub trait SpeechWordTokenizer: Send + Sized + 'static {
    #[host(class(name = "SpeechWordStream"))]
    type Stream;
    #[host(init)]
    fn new(
        #[host(kw, default = true)] ignore_punctuation: bool,
        #[host(default = false)] retain_format: bool,
    ) -> Self;
    #[host(call)]
    fn tokenize(
        &self,
        text: String,
        #[host(kw, default = None)] language: Option<String>,
    ) -> Vec<String>;
    #[host(call, name = "_core_stream")]
    fn stream(&self) -> Self::Stream;
}

/// `tokenize.WordTokenizer`'s stream: a word goes out once the next one proves it whole.
#[crate::host(class(name = "SpeechWordStream", pid = false))]
pub trait SpeechWordStream: Send + Sized + 'static {
    /// The words ``text`` completes.
    #[host(call)]
    fn push(&mut self, text: String) -> Vec<String>;
    #[host(call)]
    fn flush(&mut self) -> Vec<String>;
}

/// Splits text into words: at whitespace, with each Chinese or Japanese character a word of its
/// own. Thai, Lao, Khmer and Burmese split only at whitespace.
///
/// The stream emits a word once the next one proves it whole. Subclass it for a custom
/// tokenizer: override `tokenize`, and the inherited `stream` emits each of its words once the
/// next one starts; override `stream` too to stream them yourself.
///
/// Args:
///     ignore_punctuation: drop punctuation from the words; a word of punctuation alone is
///         dropped.
///     retain_format: keep the whitespace before each word, so the words join back to the text.
#[crate::host(facade(
    path = "livekit.agents.tokenize",
    name = "WordTokenizer",
    base = "tokenize._WordTokenizerBase",
    native = SpeechWordTokenizer,
    proxied(tokenize, stream)
))]
pub trait SpeechWords {
    #[host(new)]
    fn new(
        #[host(kw, default = true)] ignore_punctuation: bool,
        #[host(default = false)] retain_format: bool,
    ) -> Self;
    fn tokenize(
        &self,
        text: String,
        #[host(kw, default = None)] language: Option<String>,
    ) -> Vec<String>;
}

/// Splits text into sentences for speech, batched so short ones are not spoken alone.
///
/// The stream emits a sentence as soon as no text to come could change where it ends. Subclass
/// it for a custom tokenizer: override `tokenize`, and the inherited `stream` emits each of its
/// sentences once the next one starts; override `stream` too to stream them yourself.
///
/// Args:
///     min_words: sentences are batched until a chunk holds this many spoken words. A Chinese or
///         Japanese character counts as a word, and Thai about every three characters.
///     clause_words: a sentence running past this many words without an end breaks at its next
///         clause mark (``,`` ``;`` ``:`` and their fullwidth forms, never inside a number such as
///         ``1,000``). ``None`` breaks only at sentence ends.
///     retain_format: keep original whitespace/formatting in emitted tokens.
///     max_token_len: cap on a chunk's length in characters: batching stops before a sentence
///         that would pass it. A longer sentence is cut at its last clause mark or word
///         boundary before the cap (1000 characters when unset).
#[crate::host(facade(
    path = "livekit.agents.tokenize",
    name = "SentenceTokenizer",
    base = "tokenize._SentenceTokenizerBase",
    native = SpeechSentenceTokenizer,
    proxied(tokenize, stream)
))]
pub trait SpeechSentences {
    #[host(new)]
    fn new(
        #[host(kw, default = 4)] min_words: usize,
        #[host(default = None)] clause_words: Option<usize>,
        #[host(default = false)] retain_format: bool,
        #[host(default = None)] max_token_len: Option<usize>,
    ) -> Self;
    /// The sentences of `text`, batched as the stream batches them.
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
    ) -> Self;
    /// The tokens ``text`` proves complete, batched.
    #[host(call)]
    fn push(&mut self, text: String) -> Result<Vec<String>, HostError>;
    #[host(call)]
    fn flush(&mut self) -> Result<Vec<String>, HostError>;
}
