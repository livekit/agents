//! Sentence boundaries from `microsoft/BlingFire`, the same pinned upstream and C++
//! wrapper as the Python wheel and Node addon, so boundaries (where the TTS breathes) agree.
//! Offsets are codepoints, not bytes; the empty string is guarded before the C call.

use std::ffi::{CStr, CString, c_char, c_int};

#[repr(C)]
struct RawResult {
    _private: [u8; 0],
}

unsafe extern "C" {
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

/// Offsets are codepoints for Python and UTF-16 units for Node.
pub fn text_to_sentences_with_offsets(text: &str, offsets: Offsets) -> Result<Tokenized, Error> {
    let input = CString::new(text).map_err(|_| Error::InteriorNul)?;

    // SAFETY: the handle is owned by C++ for the duration of this block and freed before
    // returning; every accessor is called with a non-null handle and an in-range index.
    unsafe {
        let handle = lk_bf_tokenize_with_offsets(input.as_ptr(), MODE_SENTENCES, offsets.as_int());
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

pub struct MediaTokenization;

impl lk_agents_abi::media::codecs::MediaTokenization for MediaTokenization {
    fn split_sentences(
        text: &str,
        min_sentence_len: usize,
        retain_format: bool,
    ) -> Result<Vec<(String, usize, usize)>, lk_agents_abi::media::codecs::CodecFailure> {
        split_sentences(text, min_sentence_len, retain_format)
            .map_err(|error| lk_agents_abi::media::codecs::CodecFailure(error.to_string()))
    }
}

/// `re.sub(r"\s*\n+\s*", " ", s).strip()`: only runs containing a newline collapse. Python's `\s`
/// also counts the C0 separators `\x1c`-`\x1f`.
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
            (
                "一つ目です。二つ目です。",
                1,
                false,
                vec![("一つ目です。", 0, 6), ("二つ目です。", 6, 12)],
            ),
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
        assert_eq!(
            text_to_sentences_with_offsets("a\0b", Offsets::Codepoints).unwrap_err(),
            Error::InteriorNul
        );
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
