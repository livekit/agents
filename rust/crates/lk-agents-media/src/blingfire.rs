//! Sentence and word boundaries from `microsoft/BlingFire`, the same pinned upstream and C++
//! wrapper as the Python wheel and Node addon, so boundaries (where the TTS breathes) agree.
//! Offsets are codepoints, not bytes; the empty string is guarded before the C call.

use std::ffi::{CStr, CString, c_char, c_int};

#[repr(C)]
struct RawResult {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn lk_bf_tokenize_plain(input: *const c_char, mode: c_int) -> *mut c_char;
    fn lk_bf_free_string(p: *mut c_char);
    fn lk_bf_tokenize_with_offsets(
        input: *const c_char,
        mode: c_int,
        offsets: c_int,
    ) -> *mut RawResult;
    fn lk_bf_result_text(r: *const RawResult) -> *const c_char;
    fn lk_bf_result_span_count(r: *const RawResult) -> usize;
    fn lk_bf_result_span(r: *const RawResult, i: usize, start: *mut c_int, end: *mut c_int);
    fn lk_bf_result_free(r: *mut RawResult);
}

/// A "character index" is a codepoint to Python and a UTF-16 unit to JavaScript.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Offsets {
    Codepoints,
    Utf16,
}

impl Offsets {
    fn as_int(self) -> c_int {
        match self {
            Offsets::Codepoints => 0,
            Offsets::Utf16 => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tokenized {
    pub text: String,
    pub spans: Vec<Span>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("input contains a NUL byte")]
    InteriorNul,
    #[error("blingfire tokenization failed")]
    Failed,
}

const MODE_SENTENCES: c_int = 0;
const MODE_WORDS: c_int = 1;

fn tokenize_plain(text: &str, mode: c_int) -> Result<String, Error> {
    let input = CString::new(text).map_err(|_| Error::InteriorNul)?;

    // SAFETY: `input` outlives the call; the returned pointer is either null or a malloc'd
    // NUL-terminated string that we free below.
    unsafe {
        let raw = lk_bf_tokenize_plain(input.as_ptr(), mode);
        if raw.is_null() {
            return Err(Error::Failed);
        }
        let out = CStr::from_ptr(raw).to_string_lossy().into_owned();
        lk_bf_free_string(raw);
        Ok(out)
    }
}

fn tokenize_with_offsets(text: &str, mode: c_int, offsets: Offsets) -> Result<Tokenized, Error> {
    let input = CString::new(text).map_err(|_| Error::InteriorNul)?;

    // SAFETY: the handle is owned by C++ for the duration of this block and freed before
    // returning; every accessor is called with a non-null handle and an in-range index.
    unsafe {
        let handle = lk_bf_tokenize_with_offsets(input.as_ptr(), mode, offsets.as_int());
        if handle.is_null() {
            return Err(Error::Failed);
        }

        let text_ptr = lk_bf_result_text(handle);
        let text = if text_ptr.is_null() {
            String::new()
        } else {
            CStr::from_ptr(text_ptr).to_string_lossy().into_owned()
        };

        let count = lk_bf_result_span_count(handle);
        let mut spans = Vec::with_capacity(count);
        for i in 0..count {
            let (mut start, mut end) = (0 as c_int, 0 as c_int);
            lk_bf_result_span(handle, i, &mut start, &mut end);
            spans.push(Span { start: start.max(0) as usize, end: end.max(0) as usize });
        }

        lk_bf_result_free(handle);
        Ok(Tokenized { text, spans })
    }
}

pub fn text_to_sentences(text: &str) -> Result<String, Error> {
    tokenize_plain(text, MODE_SENTENCES)
}

pub fn text_to_words(text: &str) -> Result<String, Error> {
    tokenize_plain(text, MODE_WORDS)
}

pub fn text_to_sentences_with_offsets(text: &str, offsets: Offsets) -> Result<Tokenized, Error> {
    tokenize_with_offsets(text, MODE_SENTENCES, offsets)
}

pub fn text_to_words_with_offsets(text: &str, offsets: Offsets) -> Result<Tokenized, Error> {
    tokenize_with_offsets(text, MODE_WORDS, offsets)
}

// ==================================================================================================
// The shape the host wants
// ==================================================================================================

/// `(text, start, end)`, with a **character** span.
pub type Sentence = (String, usize, usize);

/// `tokenize/blingfire.py::_split_sentences`, exactly.
///
/// Not a convenience wrapper -- it is the contract. Four behaviours here are load-bearing and
/// every one of them is a line in the Python this replaces:
///
/// * **blank input short-circuits**, before the C call. Upstream sizes its output buffer at
///   `len * 2`, so an empty input hands the C API a zero-length buffer; the Python binding raises
///   `ValueError: basic_string` on it and this one does not, which is undefined behaviour
///   resolving two ways rather than a difference of opinion.
/// * **spans run start-to-start.** Each sentence's `start` is the previous one's `end`, not the
///   span BlingFire reported -- so the inter-sentence whitespace belongs to the sentence that
///   follows it and no character of the input is unaccounted for. `BufferedTokenStream` advances
///   its input buffer by `tok[2]`, so a gap here would silently drop text.
/// * **`min_sentence_len` merges forward by *skipping*.** A sentence shorter than the floor is not
///   emitted and `start` is not advanced, so it is absorbed into the next one. Emitting it instead
///   makes the synthesiser breathe after "Sure."
/// * **`retain_format` keeps the raw slice**; otherwise newlines collapse to single spaces and the
///   result is trimmed. A TTS reading `"...end.\n\nNext..."` aloud must not pause twice.
#[lk_agents_abi::host(export(
    pid = false,
    kw_only = "retain_format",
    defaults(min_sentence_len = 20, retain_format = false)
))]
pub fn split_sentences(
    text: &str,
    min_sentence_len: usize,
    retain_format: bool,
) -> Result<Vec<Sentence>, Error> {
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }

    let chars: Vec<char> = text.chars().collect();
    let tokenized = text_to_sentences_with_offsets(text, Offsets::Codepoints)?;

    let mut out = Vec::new();
    let mut start = 0usize;
    for span in &tokenized.spans {
        let end = span.end.min(chars.len());
        if end <= start {
            continue;
        }
        let raw: String = chars[start..end].iter().collect();
        let cleaned = collapse_newlines(&raw);
        if cleaned.is_empty() || cleaned.chars().count() < min_sentence_len {
            continue;
        }
        out.push((if retain_format { raw } else { cleaned }, start, end));
        start = end;
    }

    if start < chars.len() {
        let raw: String = chars[start..].iter().collect();
        if retain_format {
            out.push((raw, start, chars.len()));
        } else {
            // `.strip()`, not the loop's newline collapse: the reference's asymmetry, kept.
            let trimmed = raw.trim();
            if !trimmed.is_empty() {
                out.push((trimmed.to_owned(), start, chars.len()));
            }
        }
    }

    Ok(out)
}

/// `re.sub(r"\s*\n+\s*", " ", s).strip()`: only runs containing a newline collapse. Python's `\s`
/// also counts the C0 separators `\x1c`-`\x1f`.
fn collapse_newlines(s: &str) -> String {
    let is_space = lk_agents_abi::tokenize::is_py_space;
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

#[cfg(test)]
mod tests {
    use super::*;

    // **The reason this crate exists rather than a hand-rolled splitter**, pinned as data.
    //
    // These are the exact spans `livekit.blingfire.text_to_sentences_with_offsets` returns for the
    // same inputs -- read off the installed Python wheel, not derived. If this crate and that
    // wheel ever disagree, one of the two hosts breathes in a different place from the other and
    // nothing else in either tree would notice.
    //
    // Note the third case: BlingFire does **not** split `"Héllo thère. Ça va?"`, and the Python
    // binding does not either. An expectation that it "should" is the test being wrong about
    // English orthography, which is exactly the class of disagreement a shared model removes.
    #[test]
    fn the_spans_match_the_python_binding_exactly() {
        for (text, expected) in [
            ("Hello there. How are you?", vec![(0usize, 12usize), (13, 25)]),
            ("One.   Two.\n\nThree.", vec![(0, 4), (7, 11), (13, 19)]),
            ("Héllo thère. Ça va?", vec![(0, 19)]),
            ("Готово. Дальше что?", vec![(0, 7), (8, 19)]),
        ] {
            let out = text_to_sentences_with_offsets(text, Offsets::Codepoints).unwrap();
            let got: Vec<(usize, usize)> = out.spans.iter().map(|s| (s.start, s.end)).collect();
            assert_eq!(got, expected, "{text:?}");
        }
    }

    // **The equivalence check that decides whether this crate may replace the Python.**
    //
    // Every expectation here is `tokenize/blingfire.py::_split_sentences`'s actual output for the
    // same input, read off the installed `livekit-blingfire` wheel rather than derived. Four of
    // the eight cases pin a behaviour that a tidier implementation gets wrong:
    //
    //   * `min_sentence_len=20` MERGES FORWARD BY SKIPPING -- "Hello there." is under the floor,
    //     so it is absorbed and the emitted span runs 0..25, not 12..25;
    //   * `"One.   Two."` KEEPS ITS THREE SPACES in the span (4..11) even though the text is
    //     trimmed, because the span is start-to-start;
    //   * `retain_format=True` returns `"\n\n   Second line."` -- leading whitespace and all;
    //   * a trailing fragment with no terminator is still a sentence.
    #[test]
    fn split_sentences_matches_the_python_implementation() {
        let cases: Vec<(&str, usize, bool, Vec<(&str, usize, usize)>)> = vec![
            (
                "Hello there. How are you? I am well.",
                1,
                false,
                vec![("Hello there.", 0, 12), ("How are you?", 12, 25), ("I am well.", 25, 36)],
            ),
            (
                "Hello there. How are you? I am well.",
                20,
                false,
                vec![("Hello there. How are you?", 0, 25), ("I am well.", 25, 36)],
            ),
            (
                "First line.\n\n   Second line.",
                1,
                false,
                vec![("First line.", 0, 11), ("Second line.", 11, 28)],
            ),
            (
                "First line.\n\n   Second line.",
                1,
                true,
                vec![("First line.", 0, 11), ("\n\n   Second line.", 11, 28)],
            ),
            ("One.   Two.", 1, false, vec![("One.", 0, 4), ("Two.", 4, 11)]),
            (
                "Sure. Let me look that up for you right away.",
                20,
                false,
                vec![("Sure. Let me look that up for you right away.", 0, 45)],
            ),
            ("Готово. Дальше что?", 1, false, vec![("Готово.", 0, 7), ("Дальше что?", 7, 19)]),
            // Three bytes per codepoint: a byte offset sliced as a char index corrupts this one.
            ("一つ目です。二つ目です。", 1, false, vec![("一つ目です。", 0, 6), ("二つ目です。", 6, 12)]),
            (
                "trailing fragment with no stop",
                1,
                false,
                vec![("trailing fragment with no stop", 0, 30)],
            ),
        ];
        for (text, min_len, retain, expected) in cases {
            let got = split_sentences(text, min_len, retain).unwrap();
            let got: Vec<(&str, usize, usize)> =
                got.iter().map(|(s, a, b)| (s.as_str(), *a, *b)).collect();
            assert_eq!(got, expected, "{text:?} min={min_len} retain={retain}");
        }
    }

    // Catches: passing the empty string through. Upstream sizes its buffer at `len * 2`, so this
    // hands the C API a zero-length buffer -- undefined, and the Python binding raises
    // `ValueError: basic_string` where this one does not.
    #[test]
    fn blank_input_short_circuits_before_the_c_call() {
        assert_eq!(split_sentences("", 1, false).unwrap(), Vec::<Sentence>::new());
        assert_eq!(split_sentences("   \n\t ", 1, false).unwrap(), Vec::<Sentence>::new());
    }

    // Catches: panicking on host-supplied text (§1.6 #6). A NUL cannot cross the C boundary and
    // the answer is an error, not an abort.
    #[test]
    fn an_interior_nul_is_an_error_and_not_a_panic() {
        assert_eq!(text_to_sentences("a\0b").unwrap_err(), Error::InteriorNul);
        assert_eq!(split_sentences("a\0b", 1, false).unwrap_err(), Error::InteriorNul);
    }

    // Catches: the codepoint/UTF-16 distinction collapsing. An emoji is one codepoint and two
    // UTF-16 code units, so the two units disagree after it -- which is the whole reason the enum
    // exists, and the Node binding is the caller that needs the other one.
    #[test]
    fn codepoint_and_utf16_offsets_differ_past_astral_text() {
        let text = "🙂 alpha beta. Gamma delta.";
        let cp = text_to_sentences_with_offsets(text, Offsets::Codepoints).unwrap();
        let u16 = text_to_sentences_with_offsets(text, Offsets::Utf16).unwrap();
        assert_eq!(cp.spans.len(), u16.spans.len());
        assert!(u16.spans.last().unwrap().end > cp.spans.last().unwrap().end);
    }
}

// ==================================================================================================
// The incremental splitter
// ==================================================================================================

use lk_agents_abi::tts::markup::SentenceTokenizerConfig;
use lk_agents_abi::tokenize::{Span as TextSpan, TokenBuffer, TokenBufferOptions, Tokenize};

/// `tokenize/blingfire.py::SentenceTokenizer`'s two model knobs; batching is [`TokenBuffer`]'s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Blingfire {
    pub min_sentence_len: usize,
    pub retain_format: bool,
}

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
            Blingfire { min_sentence_len: self.min_sentence_len, retain_format: self.retain_format },
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

    /// One splitter per stream, for [`StreamAdapter`](lk_agents_abi::StreamAdapter).
    pub fn factory() -> lk_agents_abi::SplitterFactory {
        SentenceBuffer::factory_with(SentenceBufferOptions::default())
    }

    pub fn factory_with(opts: SentenceBufferOptions) -> lk_agents_abi::SplitterFactory {
        std::sync::Arc::new(move || {
            Box::new(SentenceBuffer::with_options(opts)) as Box<dyn lk_agents_abi::SentenceSplitter>
        })
    }

    pub fn into_inner(self) -> TokenBuffer<Blingfire> {
        self.0
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

#[cfg(test)]
mod splitter_tests {
    use super::*;
    use lk_agents_abi::SentenceSplitter as _;

    // Every expectation below is **measured against the reference**, not reasoned about: each was
    // produced by running the same chunks through
    // `tokenize.blingfire.SentenceTokenizer(retain_format=True).stream()`, which is what the
    // reference's `StreamAdapter` builds. PLAN.md G3's "audible one" was that this file and that
    // stream disagreed, so agreeing with it is the property under test.

    fn drive(chunks: &[&str]) -> Vec<String> {
        let mut s = SentenceBuffer::new();
        let mut out = Vec::new();
        for c in chunks {
            out.extend(s.push(c));
        }
        out.extend(s.flush());
        out
    }

    // Catches: losing text at a chunk boundary. The model writes tokens, not sentences, so a
    // sentence usually arrives across several pushes -- and every character has to come back out.
    //
    // Note the leading space on the tail: `retain_format` is on, so the whitespace between
    // sentences belongs to the one that follows it and is *not* trimmed away.
    #[test]
    fn a_sentence_split_across_chunks_comes_back_whole() {
        assert_eq!(
            drive(&["Hel", "lo the", "re. How ", "are you? I am", " well."]),
            vec!["Hello there. How are you?", " I am well."]
        );
    }

    // Catches the bug this file was fixed for, in the form it actually took. The old test here was
    // named `a_short_sentence_is_emitted_rather_than_merged_forward` and its comment argued that
    // "the floor belongs to the tokenizer the caller configured, not here" -- which was true when
    // this type was one splitter among several, and false once `DriverBuilder` made it *the*
    // tokenizer on the agent-speech path. `min_sentence_len = 20` merges "Sure." forward, which is
    // what stops the synthesiser breathing after it.
    #[test]
    fn a_short_sentence_is_merged_forward_not_spoken_alone() {
        assert_eq!(drive(&["Sure. ", "Let me look."]), vec!["Sure. Let me look."]);
    }

    // Catches: the codepoint spans this crate converts from BlingFire's byte offsets disagreeing
    // with the character drain in `TokenBuffer`. Two long Cyrillic sentences (two bytes per
    // letter), so the first is emitted on `push` and the buffer is cut after it -- a byte-based
    // cut on either side lands mid-codepoint (a panic on the audio path) or leaves half a
    // sentence behind, and the second push then resolves against a corrupted remainder. The old
    // form of this test (`"Готово. "` + `"Дальше что?"`, 19 characters) never reached the cut:
    // it was one span held to the flush, and indistinguishable from the "Sure." test above.
    #[test]
    fn a_multibyte_buffer_is_cut_at_the_codepoint_and_carried_over_correctly() {
        let mut s = SentenceBuffer::new();
        assert_eq!(
            s.push("Это первое достаточно длинное предложение."),
            Vec::<String>::new(),
            "nothing follows it yet"
        );
        assert_eq!(
            s.push(" А это второе достаточно длинное предложение."),
            vec!["Это первое достаточно длинное предложение."]
        );
        assert_eq!(s.flush(), vec![" А это второе достаточно длинное предложение."]);
    }

    // Catches: the last span being emitted early. A sentence is only complete when something
    // follows it, so `push` alone never releases the tail -- `flush` is what does.
    #[test]
    fn the_final_sentence_is_held_until_flush() {
        let mut s = SentenceBuffer::new();
        assert_eq!(
            s.push("This is a sufficiently long first sentence."),
            Vec::<String>::new(),
            "nothing follows it yet"
        );
        let out = s.push(" And this is a sufficiently long second one.");
        assert_eq!(out, vec!["This is a sufficiently long first sentence."]);
        assert_eq!(s.flush(), vec![" And this is a sufficiently long second one."]);
        assert_eq!(s.flush(), Vec::<String>::new(), "flush is not repeatable");
    }

    // Catches: the `retain_format` knob not reaching the tokenizer through `SentenceBufferOptions`.
    // The default (`true`, what the reference's `StreamAdapter` asks for) is pinned by the leading
    // space `a_sentence_split_across_chunks_comes_back_whole` expects; this is the other setting.
    #[test]
    fn retain_format_off_collapses_newlines_through_the_buffer() {
        let opts = SentenceBufferOptions { retain_format: false, ..Default::default() };
        let mut s = SentenceBuffer::with_options(opts);
        let mut out = s.push("This is a sufficiently long first sentence.\n\n");
        out.extend(s.push("And this is a sufficiently long second one."));
        out.extend(s.flush());
        assert!(!out.iter().any(|t| t.contains('\n')), "newlines survived: {out:?}");
    }

    // -------------------------------------------------------------------------------------
    // `tests/test_tokenizer.py`, ported: `blingfire.SentenceTokenizer(min_sentence_len=20)` fed
    // `TEXT` in chunks of 1, 2, 4, ... characters, expected `EXPECTED_MIN_20_BLINGFIRE`.
    // -------------------------------------------------------------------------------------

    const TEXT: &str = concat!(
        "Hi! ",
        "LiveKit is a platform for live audio and video applications and services. \n\n",
        "R.T.C stands for Real-Time Communication... again R.T.C. ",
        "Mr. Theo is testing the sentence tokenizer. ",
        "\nThis is a test. Another test. ",
        "A short sentence.\n",
        "A longer sentence that is longer than the previous sentence. ",
        "f(x) = x * 2.54 + 42. ",
        "Hey!\n Hi! Hello! ",
        "\n\n",
        "This is a sentence. 这是一个中文句子。これは日本語の文章です。",
        "你好！LiveKit是一个直播音频和视频应用程序和服务的平台。",
        "\nThis is a sentence contains   consecutive spaces."
    );

    const EXPECTED_MIN_20_BLINGFIRE: [&str; 8] = [
        "Hi! LiveKit is a platform for live audio and video applications and services.",
        "R.T.C stands for Real-Time Communication... again R.T.C. Mr. Theo is testing the sentence tokenizer.",
        "This is a test. Another test.",
        "A short sentence. A longer sentence that is longer than the previous sentence. f(x) = x * 2.54 + 42.",
        "Hey! Hi! Hello! This is a sentence.",
        "这是一个中文句子。これは日本語の文章です。",
        "你好！LiveKit是一个直播音频和视频应用程序和服务的平台。",
        "This is a sentence contains   consecutive spaces.",
    ];

    /// "divide text by chunks of arbitrary length (1-4)": the reference's `[1, 2, 4]` pattern.
    fn reference_chunks(text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut out = Vec::new();
        let mut i = 0;
        for size in [1usize, 2, 4].iter().cycle() {
            if i >= chars.len() {
                break;
            }
            let end = (i + size).min(chars.len());
            out.push(chars[i..end].iter().collect());
            i = end;
        }
        out
    }

    // `test_streamed_sent_tokenizer[blingfire]`. Same text, same chunking, same eight tokens: the
    // one buffer over the one model reproduces the Python stream exactly.
    #[test]
    fn streamed_sentence_tokenizer_matches_the_reference() {
        let opts = SentenceBufferOptions {
            min_sentence_len: 20,
            min_token_len: 20,
            min_ctx_len: 10,
            retain_format: false,
            ..Default::default()
        };
        let mut s = SentenceBuffer::with_options(opts);
        let mut out = Vec::new();
        for chunk in reference_chunks(TEXT) {
            out.extend(s.push(&chunk));
        }
        out.extend(s.flush());
        assert_eq!(out, EXPECTED_MIN_20_BLINGFIRE);
    }

    // Catches: the tokenizer leaf failing a turn. A NUL cannot cross the C boundary; the answer is
    // the text whole -- held by the buffer, said at the flush -- not a panic and not silence.
    #[test]
    fn blingfire_as_a_tokenizer_hands_back_what_it_cannot_split() {
        use lk_agents_abi::tokenize::Tokenize as _;
        let tok = Blingfire { min_sentence_len: 1, retain_format: false };
        let spans = tok.tokenize("Hello there. How are you?");
        assert_eq!(spans.iter().map(|s| s.text.as_str()).collect::<Vec<_>>(), ["Hello there.", "How are you?"]);
        assert_eq!((spans[1].start, spans[1].end), (12, 25));
        let held = tok.tokenize("a\0b");
        assert_eq!(held.len(), 1);
        assert_eq!(held[0].text, "a\0b");
        assert_eq!(tok.tokenize("   "), Vec::<lk_agents_abi::tokenize::Span>::new());
    }

    // A-31, under the real model. Catches: a `<break time="1s"/>` arriving one character at a
    // time being cut at a chunk boundary, or split off its sentence. This is what
    // `SentenceBufferOptions::from_config(markup.sentence_tokenizer(true))` turns on. The floor
    // is the reference's default; at `min_token_len = 1` the reference emits `"Hello"` alone.
    #[test]
    fn a_break_tag_mid_sentence_never_splits() {
        let opts = SentenceBufferOptions {
            min_sentence_len: 1,
            min_token_len: 20,
            min_ctx_len: 5,
            retain_format: false,
            xml_aware: true,
            ..Default::default()
        };
        let mut s = SentenceBuffer::with_options(opts);
        let text = "Hello <break time=\"1s\"/> there my friend. Second sentence here.";
        let mut out = Vec::new();
        for ch in text.chars() {
            out.extend(s.push(&ch.to_string()));
        }
        out.extend(s.flush());
        assert_eq!(out, vec!["Hello <break time=\"1s\"/> there my friend.", "Second sentence here."]);
        for token in &out {
            assert!(!lk_agents_abi::tokenize::xml::has_unclosed_tags(token), "a token split a tag: {token:?}");
        }
    }

    // Catches: the provider knobs not reaching the buffer. Expressive: the floor is the cap and
    // the wrap is on; plain: per-sentence emission and no wrap.
    #[test]
    fn from_config_applies_the_three_provider_knobs() {
        let expressive = SentenceTokenizerConfig { max_token_len: Some(900), min_token_len: Some(900), xml_aware: true };
        let got = SentenceBufferOptions::from_config(expressive);
        assert_eq!((got.max_token_len, got.min_token_len, got.xml_aware), (Some(900), 900, true));
        assert_eq!((got.min_sentence_len, got.min_ctx_len, got.retain_format), (20, 10, true));
        let plain = SentenceTokenizerConfig { max_token_len: Some(900), min_token_len: None, xml_aware: false };
        let got = SentenceBufferOptions::from_config(plain);
        assert_eq!((got.max_token_len, got.min_token_len, got.xml_aware), (Some(900), 20, false));
    }
}
