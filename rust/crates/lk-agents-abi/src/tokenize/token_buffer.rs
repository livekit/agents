//! `tokenize/token_stream.py`'s `BufferedTokenStream` (`:87`). Lengths and offsets are characters,
//! not bytes, as in Python.

use std::borrow::Cow;

use crate::SentenceSplitter;
use crate::tokenize::word::{WordTokenizerOptions, split_words};
use crate::tokenize::xml;

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

/// Infallible: on failure return [`Span::whole`].
pub trait Tokenize {
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
    pub const WORDS: TokenBufferOptions =
        TokenBufferOptions { min_token_len: 1, min_ctx_len: 1, max_token_len: None, xml_aware: false };
}

#[derive(Debug, Clone)]
pub struct TokenBuffer<T: Tokenize> {
    tok: T,
    opts: TokenBufferOptions,
    in_buf: String,
    out_buf: String,
}

impl<T: Tokenize> TokenBuffer<T> {
    pub fn new(tok: T, opts: TokenBufferOptions) -> Self {
        TokenBuffer { tok, opts, in_buf: String::new(), out_buf: String::new() }
    }

    fn spans(&self, text: &str) -> Vec<Span> {
        if self.opts.xml_aware { xml::tokenize_aware(&self.tok, text) } else { self.tok.tokenize(text) }
    }

    pub fn push(&mut self, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        if text.is_empty() {
            return out;
        }
        self.in_buf.push_str(text);
        if self.in_buf.chars().count() < self.opts.min_ctx_len {
            return out;
        }

        loop {
            let mut spans = self.spans(&self.in_buf).into_iter();
            let (Some(first), Some(_second)) = (spans.next(), spans.next()) else {
                // One span is a fragment: something has to follow a token before it is complete.
                return out;
            };
            // A token that would split a tag waits for the chunk that closes it.
            if self.opts.xml_aware && xml::has_unclosed_tags(&first.text) {
                return out;
            }
            self.absorb(&first.text, &mut out);
            if self.out_buf.chars().count() >= self.opts.min_token_len {
                out.push(std::mem::take(&mut self.out_buf));
            }
            // Character offsets, so `chars()` and not a byte range.
            let cut = self.in_buf.char_indices().nth(first.end).map_or(self.in_buf.len(), |(b, _)| b);
            self.in_buf.drain(..cut);
        }
    }

    /// Ignores `min_token_len`.
    pub fn flush(&mut self) -> Vec<String> {
        let mut out = Vec::new();
        if self.in_buf.is_empty() && self.out_buf.is_empty() {
            return out;
        }
        let in_buf = std::mem::take(&mut self.in_buf);
        for span in self.spans(&in_buf) {
            self.absorb(&span.text, &mut out);
        }
        if !self.out_buf.is_empty() {
            out.push(std::mem::take(&mut self.out_buf));
        }
        out
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

impl<T: Tokenize + Send> SentenceSplitter for TokenBuffer<T> {
    fn push(&mut self, text: &str) -> Vec<String> {
        TokenBuffer::push(self, text)
    }

    fn flush(&mut self) -> Vec<String> {
        TokenBuffer::flush(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Words(pub WordTokenizerOptions);

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

// The `SentenceTokenizer`/`WordTokenizer` ABCs stay Python (subclassed and `isinstance`-checked).
#[cfg(feature = "python")]
mod python {
    use std::sync::Mutex;

    use pyo3::prelude::*;

    use super::{TokenBuffer, TokenBufferOptions};
    use crate::tokenize::{Span, Tokenize};

    /// An exception is parked and re-raised by the calling method, unless `xml_aware` (the
    /// reference swallows it there); either way the text comes back as one span.
    pub struct HostTokenizer {
        fnc: Py<PyAny>,
        failed: Mutex<Option<PyErr>>,
    }

    impl HostTokenizer {
        fn call(&self, py: Python<'_>, text: &str) -> PyResult<Vec<Span>> {
            let tokens = self.fnc.bind(py).call1((text,))?;
            let mut out = Vec::new();
            for item in tokens.try_iter()? {
                let item = item?;
                if let Ok((tok, start, end)) = item.extract::<(String, usize, usize)>() {
                    out.push(Span { text: tok, start, end });
                    continue;
                }
                // No offsets: locate with `str.find` (0 when absent) and absorb the following
                // whitespace, matching the reference's `.lstrip()`.
                let tok: String = item.extract()?;
                let start = text.find(&tok).map_or(0, |b| text[..b].chars().count());
                let mut end = start + tok.chars().count();
                end += text.chars().skip(end).take_while(|c| c.is_whitespace()).count();
                out.push(Span { text: tok, start, end });
            }
            Ok(out)
        }
    }

    impl Tokenize for HostTokenizer {
        fn tokenize(&self, text: &str) -> Vec<Span> {
            Python::attach(|py| match self.call(py, text) {
                Ok(spans) => spans,
                Err(err) => {
                    *self.failed.lock().unwrap() = Some(err);
                    if text.trim().is_empty() { Vec::new() } else { vec![Span::whole(text)] }
                }
            })
        }
    }

    /// `BufferedTokenStream`'s batching loop over a host tokenizer. The Python class keeps the
    /// channel, the segment ids and the `closed` rule; this keeps the two buffers and the five rules.
    #[pyclass(name = "TokenBuffer", module = "livekit.agents._core")]
    pub struct PyTokenBuffer {
        inner: TokenBuffer<HostTokenizer>,
    }

    #[pymethods]
    impl PyTokenBuffer {
        #[new]
        #[pyo3(signature = (tokenize_fnc, *, min_token_len, min_ctx_len, max_token_len=None, xml_aware=false))]
        pub(super) fn new(
            tokenize_fnc: Py<PyAny>,
            min_token_len: usize,
            min_ctx_len: usize,
            max_token_len: Option<usize>,
            xml_aware: bool,
        ) -> Self {
            let tok = HostTokenizer { fnc: tokenize_fnc, failed: Mutex::new(None) };
            let opts = TokenBufferOptions { min_token_len, min_ctx_len, max_token_len, xml_aware };
            PyTokenBuffer { inner: TokenBuffer::new(tok, opts) }
        }

        /// `push_text`: the tokens now proven complete, batched.
        pub(super) fn push(&mut self, text: &str) -> PyResult<Vec<String>> {
            let out = self.inner.push(text);
            self.settle(out)
        }

        /// `flush`: everything held, batched under the cap but with no `min_token_len` gate.
        pub(super) fn flush(&mut self) -> PyResult<Vec<String>> {
            let out = self.inner.flush();
            self.settle(out)
        }
    }

    impl PyTokenBuffer {
        /// Re-raise a parked tokenizer exception, unless the XML wrap swallows it (as the
        /// reference's does: worse prosody, not a lost reply).
        fn settle(&self, out: Vec<String>) -> PyResult<Vec<String>> {
            match self.inner.tok.failed.lock().unwrap().take() {
                Some(err) if !self.inner.opts.xml_aware => Err(err),
                _ => Ok(out),
            }
        }
    }

    /// `SentenceTokenizer.tokenize` with `xml_aware`: [`crate::tokenize::xml::tokenize_aware`] over a
    /// host tokenizer. A tokenizer error leaves the text as one span, as on the streaming path.
    #[pyfunction]
    pub fn xml_tokenize(tokenize_fnc: Py<PyAny>, text: &str) -> Vec<(String, usize, usize)> {
        let tok = HostTokenizer { fnc: tokenize_fnc, failed: Mutex::new(None) };
        crate::tokenize::xml::tokenize_aware(&tok, text).into_iter().map(|s| (s.text, s.start, s.end)).collect()
    }

    crate::register_host!(class PyTokenBuffer);
    crate::register_host!(fn xml_tokenize);
}

#[cfg(all(test, feature = "python"))]
mod python_tests {
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    use super::python::PyTokenBuffer;

    fn attach<T>(f: impl FnOnce(Python<'_>) -> T) -> T {
        Python::initialize();
        Python::attach(f)
    }

    /// A sentence tokenizer that returns `(text, start, end)` triples, as `_core.split_sentences`
    /// does: a sentence ends at a `.` followed by whitespace or the end of the text.
    fn by_sentence(py: Python<'_>) -> Py<PyAny> {
        let locals = PyDict::new(py);
        py.run(
            c"def by_sentence(text):
    out = []
    start = 0
    for i, ch in enumerate(text):
        if ch == '.' and (i + 1 == len(text) or text[i + 1].isspace()):
            out.append((text[start:i + 1].strip(), start, i + 1))
            start = i + 1
    if text[start:].strip():
        out.append((text[start:].strip(), start, len(text)))
    return out
",
            None,
            Some(&locals),
        )
        .expect("defines");
        locals.get_item("by_sentence").expect("lookup").expect("defined").unbind()
    }

    fn buffer(fnc: Py<PyAny>, min_token_len: usize) -> PyTokenBuffer {
        PyTokenBuffer::new(fnc, min_token_len, 1, None, false)
    }

    // Catches: the crossing changing the batching, e.g. a byte/char offset mix-up.
    #[test]
    fn a_host_tokenizer_with_offsets_batches_like_the_reference() {
        attach(|py| {
            let mut buf = buffer(by_sentence(py), 20);
            let mut out = buf.push("Sure. ").unwrap();
            out.extend(buf.push("Let me look. ").unwrap());
            out.extend(buf.push("And one more here. Tail.").unwrap());
            assert_eq!(out, vec!["Sure. Let me look. And one more here."]);
            assert_eq!(buf.flush().unwrap(), vec!["Tail."]);
            assert_eq!(buf.flush().unwrap(), Vec::<String>::new());
        });
    }

    // Catches: a plain-string tokenizer advanced by the wrong amount.
    #[test]
    fn a_host_tokenizer_without_offsets_is_located_by_find() {
        attach(|py| {
            let split = py.eval(c"lambda text: text.split()", None, None).unwrap().unbind();
            let mut buf = buffer(split, 1);
            assert_eq!(buf.push("Hello wor").unwrap(), vec!["Hello"]);
            assert_eq!(buf.push("ld  ok").unwrap(), vec!["world"]);
            assert_eq!(buf.flush().unwrap(), vec!["ok"]);
        });
    }

    // Catches: the host's exception swallowed outside the XML wrap (which swallows it, pinned in
    // `tests/test_token_stream_xml.py`).
    #[test]
    fn a_raising_tokenizer_raises_from_push() {
        attach(|py| {
            let boom = py.eval(c"lambda text: 1 / 0", None, None).unwrap().unbind();
            let err = buffer(boom, 1).push("abc def. ghi").unwrap_err();
            assert!(err.is_instance_of::<pyo3::exceptions::PyZeroDivisionError>(py), "{err}");
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model-free sentence tokenizer: ends at `.`, `!` or `?` before whitespace or end; spans
    /// tile, text trimmed.
    struct Dots;

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

    const PER_SENTENCE: TokenBufferOptions =
        TokenBufferOptions { min_token_len: 1, min_ctx_len: 1, max_token_len: None, xml_aware: false };

    fn drive<T: Tokenize>(buf: &mut TokenBuffer<T>, chunks: &[String]) -> Vec<String> {
        let mut out: Vec<String> = chunks.iter().flat_map(|c| buf.push(c)).collect();
        out.extend(buf.flush());
        out
    }

    fn drive_str<T: Tokenize>(buf: &mut TokenBuffer<T>, chunks: &[&str]) -> Vec<String> {
        drive(buf, &chunks.iter().map(|c| (*c).to_owned()).collect::<Vec<_>>())
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
        let opts = WordTokenizerOptions { ignore_punctuation: false, split_character: true, retain_format: true };
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

    const XML_STREAM: TokenBufferOptions =
        TokenBufferOptions { min_token_len: 1, min_ctx_len: 5, max_token_len: None, xml_aware: true };

    // A-31. Catches: a `<break time="1s"/>` arriving one character at a time being cut. At the
    // reference's floor of 20 it rides inside its sentence.
    #[test]
    fn a_break_tag_mid_sentence_never_splits() {
        let text = "Hello <break time=\"1s\"/> there my friend. Second sentence here.";
        let per_char: Vec<String> = text.chars().map(String::from).collect();
        let got = drive(&mut sentences(TokenBufferOptions { min_token_len: 20, ..XML_STREAM }), &per_char);
        assert_eq!(got, vec!["Hello <break time=\"1s\"/> there my friend.", "Second sentence here."]);

        let got = drive(&mut sentences(XML_STREAM), &per_char);
        assert_eq!(got.iter().filter(|t| t.contains("<break time=\"1s\"/>")).count(), 1, "{got:?}");
        for token in &got {
            assert!(!xml::has_unclosed_tags(token), "a token split a tag: {token:?}");
            assert!(!xml::is_xml_only(token), "a tag was emitted alone: {token:?}");
        }
    }

}
