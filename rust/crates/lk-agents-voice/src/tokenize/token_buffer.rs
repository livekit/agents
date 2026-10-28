//! `tokenize/token_stream.py`'s `BufferedTokenStream` (`:87`). Lengths and offsets are characters,
//! not bytes, as in Python.

use std::borrow::Cow;

use crate::tokenize::word::{WordTokenizerOptions, split_words};
use crate::tokenize::xml;
use lk_agents_abi::SentenceSplitter;
use lk_agents_abi::tokenize::bindings::TokenizerToken;

/// Character indices. Spans must tile the input start-to-start, or the text in a gap is dropped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    pub text: String,
    pub start: usize,
    pub end: usize,
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
            TokenizerToken::Span((text, start, end)) => Span { text, start, end },
            TokenizerToken::Text(token) => {
                let start = text.find(&token).map_or(0, |byte| text[..byte].chars().count());
                let mut end = start + token.chars().count();
                end += text.chars().skip(end).take_while(|c| c.is_whitespace()).count();
                Span { text: token, start, end }
            }
        })
        .collect()
}

pub(crate) mod sealed {
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
mod tests {
    use super::NativeTokenBuffer as TokenBuffer;
    use super::*;

    /// A model-free sentence tokenizer: ends at `.`, `!` or `?` before whitespace or end; spans
    /// tile, text trimmed.
    struct Dots;

    impl sealed::NativeTokenizer for Dots {}

    impl Tokenize for Dots {
        fn tokenize(&self, text: &str) -> Vec<Span> {
            let chars: Vec<char> = text.chars().collect();
            let mut out = Vec::new();
            let mut start = 0;
            for i in 0..chars.len() {
                let terminal = matches!(chars[i], '.' | '!' | '?');
                let boundary = chars.get(i + 1).is_none_or(|c| c.is_whitespace());
                if terminal && boundary {
                    let s: String = chars[start..=i].iter().collect();
                    out.push(Span { text: s.trim().to_owned(), start, end: i + 1 });
                    start = i + 1;
                }
            }
            if start < chars.len() {
                let s: String = chars[start..].iter().collect();
                if !s.trim().is_empty() {
                    out.push(Span { text: s.trim().to_owned(), start, end: chars.len() });
                }
            }
            out
        }
    }

    fn sentences(opts: TokenBufferOptions) -> TokenBuffer<Dots> {
        TokenBuffer::new(Dots, opts)
    }

    const PER_SENTENCE: TokenBufferOptions = TokenBufferOptions {
        min_token_len: 1,
        min_ctx_len: 1,
        max_token_len: None,
        xml_aware: false,
    };

    fn drive<T: Tokenize>(buf: &mut TokenBuffer<T>, chunks: &[String]) -> Vec<String> {
        let mut out: Vec<String> = chunks.iter().flat_map(|c| buf.push(c)).collect();
        out.extend(buf.flush());
        out
    }

    fn drive_str<T: Tokenize>(buf: &mut TokenBuffer<T>, chunks: &[&str]) -> Vec<String> {
        drive(buf, &chunks.iter().map(|c| (*c).to_owned()).collect::<Vec<_>>())
    }

    #[test]
    fn supplied_spans_drive_requests_without_retaining_a_tokenizer() {
        let mut state = super::TokenBuffer::new(TokenBufferOptions {
            min_ctx_len: 4,
            ..TokenBufferOptions::WORDS
        });
        assert_eq!(state.push("é "), TokenBufferStep::Done(vec![]));
        assert_eq!(state.push("word "), TokenBufferStep::Tokenize("é word ".into()));
        assert_eq!(
            state.accept(vec![
                Span { text: "é".into(), start: 0, end: 2 },
                Span { text: "word".into(), start: 2, end: 7 },
            ]),
            TokenBufferStep::Tokenize("word ".into())
        );
        assert_eq!(
            state.accept(vec![Span { text: "word".into(), start: 0, end: 5 }]),
            TokenBufferStep::Done(vec!["é".into()])
        );
        assert_eq!(state.flush(), TokenBufferStep::Tokenize("word ".into()));
        assert_eq!(
            state.accept(vec![Span { text: "word".into(), start: 0, end: 5 }]),
            TokenBufferStep::Done(vec!["word".into()])
        );
        assert_eq!(state.flush(), TokenBufferStep::Done(vec![]));
    }

    // --- The hold-back rules, one at a time.

    // Catches: `push` emitting the last span.
    #[test]
    fn the_final_token_is_held_until_flush() {
        let mut buf = sentences(PER_SENTENCE);
        assert_eq!(buf.push("First one."), Vec::<String>::new(), "nothing follows it yet");
        assert_eq!(buf.push(" Second one."), vec!["First one."]);
        assert_eq!(buf.flush(), vec!["Second one."]);
        assert_eq!(buf.flush(), Vec::<String>::new(), "flush is not repeatable");
    }

    // Catches: skipping `min_token_len` batching, or `>` for `>=` at the floor exactly.
    #[test]
    fn short_tokens_are_batched_until_the_floor_is_reached() {
        let chunks = ["Sure. ", "Let me look. ", "And one more here. Tail."];
        let mut buf = sentences(TokenBufferOptions { min_token_len: 20, ..PER_SENTENCE });
        assert_eq!(
            drive_str(&mut buf, &chunks),
            vec!["Sure. Let me look. And one more here.", "Tail."]
        );

        // "Sure. Let me look." is 18 characters: the floor is inclusive.
        let mut buf = sentences(TokenBufferOptions { min_token_len: 18, ..PER_SENTENCE });
        assert_eq!(
            drive_str(&mut buf, &chunks),
            vec!["Sure. Let me look.", "And one more here.", "Tail."]
        );
    }

    // Catches: `min_ctx_len` being ignored.
    #[test]
    fn nothing_is_emitted_below_the_context_floor() {
        let mut buf = sentences(TokenBufferOptions { min_ctx_len: 10, ..PER_SENTENCE });
        assert_eq!(buf.push("Hi. Yo."), Vec::<String>::new());
        assert_eq!(buf.flush(), vec!["Hi. Yo."], "flush batches both, under no floor");

        // Exactly at the floor (7 chars) the tokenizer is consulted; `<=` would hold it.
        let mut buf = sentences(TokenBufferOptions { min_ctx_len: 7, ..PER_SENTENCE });
        assert_eq!(buf.push("Hi. Yo."), vec!["Hi."], "the floor is exclusive");
    }

    // Catches: the cap checked after the join rather than before, or not at all.
    #[test]
    fn the_max_token_len_cap_splits_a_batch() {
        let mut buf = sentences(TokenBufferOptions {
            min_token_len: 100,
            max_token_len: Some(25),
            ..PER_SENTENCE
        });
        assert_eq!(
            drive_str(&mut buf, &["Sure. Let me look. One more bit here. And more."]),
            vec!["Sure. Let me look.", "One more bit here.", "And more."]
        );
    }

    // Catches: the cap cutting a token (an oversized one goes whole).
    #[test]
    fn a_single_token_longer_than_the_cap_is_emitted_whole() {
        let mut buf = sentences(TokenBufferOptions { max_token_len: Some(10), ..PER_SENTENCE });
        assert_eq!(
            drive_str(&mut buf, &["This sentence is well past ten. Ok."]),
            vec!["This sentence is well past ten.", "Ok."]
        );
    }

    // Catches: `flush` honouring `min_token_len` (the cap still applies).
    #[test]
    fn flush_ignores_the_floor_but_keeps_the_cap() {
        let mut buf = sentences(TokenBufferOptions {
            min_token_len: 100,
            max_token_len: Some(12),
            ..PER_SENTENCE
        });
        // The cap, not the floor, is what releases the first one during the push.
        assert_eq!(buf.push("One two. Three four. Five."), vec!["One two."]);
        assert_eq!(buf.flush(), vec!["Three four.", "Five."]);
    }

    // Catches: `flush` on an empty buffer inventing a token, or `push("")` consulting the tokenizer.
    #[test]
    fn flushing_nothing_emits_nothing() {
        let mut buf = sentences(PER_SENTENCE);
        assert_eq!(buf.flush(), Vec::<String>::new());
        assert_eq!(buf.push(""), Vec::<String>::new());
        assert_eq!(buf.flush(), Vec::<String>::new());
    }

    // Catches: slicing or counting by bytes instead of characters.
    #[test]
    fn multibyte_text_is_sliced_and_counted_by_character() {
        let mut buf = sentences(TokenBufferOptions { min_token_len: 12, ..PER_SENTENCE });
        assert_eq!(
            drive_str(&mut buf, &["Готово. ", "Дальше что? ", "Ещё."]),
            vec!["Готово. Дальше что?", "Ещё."]
        );
    }

    // Catches: chunking changing the output.
    #[test]
    fn the_tokens_do_not_depend_on_how_the_text_was_chunked() {
        let text = "Hello there, my 友人! How are you? I am well, thanks.";
        let opts = TokenBufferOptions { min_token_len: 15, ..PER_SENTENCE };
        let whole = drive_str(&mut sentences(opts), &[text]);
        let per_char: Vec<String> = text.chars().map(String::from).collect();
        assert_eq!(drive(&mut sentences(opts), &per_char), whole);
        assert_eq!(whole, vec!["Hello there, my 友人!", "How are you? I am well, thanks."]);
    }

    // --- The tokenizer implementations.

    // Catches: `Words` dropping whitespace under `retain_format`, or spans with gaps.
    #[test]
    fn words_spans_tile_the_input_and_reassemble_it() {
        let text = "  Hello there,  wörld! 你好 ok ";
        let opts = WordTokenizerOptions {
            ignore_punctuation: false,
            split_character: true,
            retain_format: true,
        };
        let spans = Words(opts).tokenize(text);
        assert_eq!(spans[0].start, 0);
        for pair in spans.windows(2) {
            assert_eq!(pair[0].end, pair[1].start, "gap between spans: {spans:?}");
        }
        assert_eq!(spans.last().expect("tokens").end, text.chars().count());
        assert_eq!(spans.iter().map(|s| s.text.as_str()).collect::<String>(), text);
    }

    // Catches: a boundary tag left on the previous sentence, or emitted alone.
    #[test]
    fn xml_aware_keeps_a_boundary_tag_with_the_following_sentence() {
        let out = xml::tokenize_aware(&Dots, "Hello. <break/> World. Bye.");
        let texts: Vec<&str> = out.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(texts, vec!["Hello.", "<break/> World.", "Bye."]);
        assert_eq!(out[1].start, 6);
        assert_eq!(out[1].end, 22, "offsets index the original, tag included");
    }

    // --- xml-aware streaming (the rest of `TestStreamingTokenizer` runs in the reference suite).

    const XML_STREAM: TokenBufferOptions = TokenBufferOptions {
        min_token_len: 1,
        min_ctx_len: 5,
        max_token_len: None,
        xml_aware: true,
    };

    // A-31. Catches: a `<break time="1s"/>` arriving one character at a time being cut. At the
    // reference's floor of 20 it rides inside its sentence.
    #[test]
    fn a_break_tag_mid_sentence_never_splits() {
        let text = "Hello <break time=\"1s\"/> there my friend. Second sentence here.";
        let per_char: Vec<String> = text.chars().map(String::from).collect();
        let got = drive(
            &mut sentences(TokenBufferOptions { min_token_len: 20, ..XML_STREAM }),
            &per_char,
        );
        assert_eq!(
            got,
            vec!["Hello <break time=\"1s\"/> there my friend.", "Second sentence here."]
        );

        let got = drive(&mut sentences(XML_STREAM), &per_char);
        assert_eq!(got.iter().filter(|t| t.contains("<break time=\"1s\"/>")).count(), 1, "{got:?}");
        for token in &got {
            assert!(!xml::has_unclosed_tags(token), "a token split a tag: {token:?}");
            assert!(!xml::is_xml_only(token), "a tag was emitted alone: {token:?}");
        }
    }
}
