//! Sentence boundaries for speech: BlingFire's sentence breaker, ported rule for rule, and the
//! chunks a voice speaks, batched to a spoken length and cut as the text streams in.

mod chunks;
mod lexer;
mod rules;
mod stream;

#[cfg(test)]
mod mutants_tests;
#[cfg(test)]
mod sentences_tests;
#[cfg(test)]
mod stream_tests;

pub use stream::SentenceStream;

use crate::lexer::Vocabulary;
use chunks::{Chunker, View};

/// Four words: about twenty characters of English.
pub const DEFAULT_MIN_WORDS: usize = 4;

/// How text is cut into the chunks a voice speaks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SentenceOptions {
    /// Sentences batch until a chunk holds this many [`spoken_words`].
    pub min_words: usize,
    /// A sentence past this many words without an end breaks at its next clause mark.
    pub clause_words: Option<usize>,
    /// Batching never grows a chunk past this many characters; a longer sentence goes whole.
    pub max_token_len: Option<usize>,
    /// Keep the raw text; otherwise newlines collapse to spaces and each chunk is trimmed.
    pub retain_format: bool,
    /// The tags no chunk splits or leaves open.
    pub markup: Option<Vocabulary>,
}

impl Default for SentenceOptions {
    fn default() -> Self {
        SentenceOptions {
            min_words: DEFAULT_MIN_WORDS,
            clause_words: None,
            max_token_len: None,
            retain_format: false,
            markup: None,
        }
    }
}

/// The sentence splitting a synthesis applies: without a provider, the sentence adapter's
/// default; with one, its gateway limits. `markup` is what is never split; under it, sentences
/// batch up to the provider's limit.
pub fn sentence_splitter(provider: Option<&str>, markup: Option<Vocabulary>) -> SentenceStream {
    let Some(provider) = provider else {
        return SentenceStream::new(SentenceOptions {
            retain_format: true,
            markup,
            ..SentenceOptions::default()
        });
    };
    let cfg = crate::tts::markup::sentence_tokenizer_of(provider, markup.is_some());
    let base = SentenceOptions::default();
    SentenceStream::new(SentenceOptions {
        min_words: if cfg.batch_to_max { usize::MAX } else { base.min_words },
        max_token_len: cfg.max_token_len,
        markup,
        ..base
    })
}

/// `tokenize.SentenceTokenizer`.
pub struct SpeechSentenceTokenizer(SentenceOptions);

impl lk_agents_abi::tokenize::bindings::SpeechSentenceTokenizer for SpeechSentenceTokenizer {
    type Stream = SentenceStream;

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
        SentenceSplitter::default().sentences(&text, &self.0).into_iter().map(|s| s.text).collect()
    }

    fn stream(&self) -> SentenceStream {
        SentenceStream::new(self.0.clone())
    }

    fn max_token_len(&self) -> Option<usize> {
        self.0.max_token_len
    }
}

/// Splits text into sentences. Holds the rule automata's caches, so keep one per stream.
pub struct SentenceSplitter {
    caches: lexer::Caches<'static>,
}

impl Default for SentenceSplitter {
    fn default() -> Self {
        SentenceSplitter { caches: lexer::Caches::new(rules::rule_sets()) }
    }
}

impl SentenceSplitter {
    /// The sentences of `text` as character spans, end exclusive. A span starts at the
    /// sentence's first non-whitespace character; whitespace-only stretches are not sentences.
    pub fn spans(&mut self, text: &str) -> Vec<(usize, usize)> {
        spans(&mut self.caches, text)
    }

    /// The chunks of `text` a voice speaks, each with its character span. Chunks run end to end,
    /// so the whitespace after a sentence belongs to the next; whitespace at the end of the text
    /// belongs to none. A [`SentenceStream`] emits the same chunks.
    pub fn sentences(&mut self, text: &str, opts: &SentenceOptions) -> Vec<Sentence> {
        let view = View::new(text, opts.markup.as_ref());
        let ends = self.caches.sentence_ends(&view.clean);
        let mut chunker = Chunker::default();
        let mut cuts = Vec::new();
        chunker.advance(&view, opts, &ends, view.clean.len(), &mut cuts);
        chunker.finish(&view, opts, &mut cuts);
        cuts.into_iter()
            .map(|r| Sentence {
                text: view.render(r.clone(), opts.retain_format),
                start: r.start,
                end: r.end,
            })
            .collect()
    }
}

/// Whether `text`, under the voice's `markup`, can go to the voice on its own: it ends outside
/// every tag and says something besides markup. A chunk that cannot joins the next.
pub fn speakable_alone(text: &str, markup: &Vocabulary) -> bool {
    let view = chunks::View::new(text, Some(markup));
    let said: String = view.clean.iter().collect();
    spoken_words(&said) > 0 && !chunks::has_unclosed_tags(text, markup)
}

/// How long `text` takes to say, in words: a Han, Hiragana or Katakana character is a word, a
/// script written without spaces (Thai, Lao, Khmer, Burmese) has a word about every three
/// characters, and elsewhere words are separated by whitespace. Punctuation alone is not said.
pub fn spoken_words(text: &str) -> usize {
    let mut words = 0;
    let said = |run: &&str| !run.chars().all(crate::tokenize::is_punctuation);
    for run in text.split(is_space).filter(said) {
        // `spaced` counts the stretches between ideographs that hold a letter or a digit.
        let (mut ideographs, mut unspaced, mut spaced, mut counted) = (0, 0usize, 0, false);
        for c in run.chars() {
            match script(c) {
                Script::Ideographic => (ideographs, counted) = (ideographs + 1, false),
                Script::Unspaced => (unspaced, counted) = (unspaced + 1, false),
                Script::Spaced if c.is_alphanumeric() && !counted => {
                    (spaced, counted) = (spaced + 1, true)
                }
                Script::Spaced => {}
            }
        }
        words += (ideographs + unspaced.div_ceil(3) + spaced).max(1);
    }
    words
}

enum Script {
    Ideographic,
    Unspaced,
    Spaced,
}

fn script(c: char) -> Script {
    match c {
        '\u{3040}'..='\u{30FF}'
        | '\u{31F0}'..='\u{31FF}'
        | '\u{3400}'..='\u{4DBF}'
        | '\u{4E00}'..='\u{9FFF}'
        | '\u{F900}'..='\u{FAFF}'
        | '\u{FF66}'..='\u{FF9F}'
        | '\u{20000}'..='\u{3134F}' => Script::Ideographic,
        '\u{0E00}'..='\u{0EFF}' | '\u{1000}'..='\u{109F}' | '\u{1780}'..='\u{17FF}' => {
            Script::Unspaced
        }
        _ => Script::Spaced,
    }
}

/// A chunk a voice speaks, with its character span of the source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sentence {
    pub text: String,
    pub start: usize,
    pub end: usize,
}

/// Runs of whitespace that contain a newline become one space; the result is trimmed. Python's
/// `\s` (`re.sub(r"\s*\n+\s*", " ", s).strip()`), so the C0 separators count as whitespace.
fn collapse_newlines(s: &str) -> String {
    let is_space = |c| lk_agents_abi::utils::strings::PYTHON_WHITESPACE.contains(&c);
    let mut out = String::with_capacity(s.len());
    let mut run = String::new();
    let mut run_has_newline = false;
    for c in s.chars() {
        if is_space(c) {
            run_has_newline |= c == '\n' || c == '\r';
            run.push(c);
            continue;
        }
        if !run.is_empty() {
            if run_has_newline {
                out.push(' ');
            } else {
                out.push_str(&run);
            }
            run.clear();
            run_has_newline = false;
        }
        out.push(c);
    }
    if !run.is_empty() && !run_has_newline {
        out.push_str(&run);
    }
    out.trim().to_owned()
}

fn spans(caches: &mut lexer::Caches<'_>, text: &str) -> Vec<(usize, usize)> {
    let chars: Vec<char> = text.chars().collect();
    let mut spans = Vec::new();
    let mut push = |from: usize, to: usize| {
        if let Some(start) = (from..=to).find(|&i| !is_space(chars[i])) {
            spans.push((start, to + 1));
        }
    };
    let mut next = 0;
    for end in caches.sentence_ends(&chars) {
        if end >= next {
            push(next, end);
        }
        next = end + 1;
    }
    if next < chars.len() {
        push(next, chars.len() - 1);
    }
    spans
}

/// `__FAIsWhiteSpace__`: what a sentence may not start with.
fn is_space(c: char) -> bool {
    matches!(c, '\0'..=' ' | '\u{a0}' | '\u{2000}'..='\u{200f}' | '\u{202f}' | '\u{205f}'
        | '\u{2060}' | '\u{2420}' | '\u{2424}' | '\u{3000}' | '\u{feff}')
}
