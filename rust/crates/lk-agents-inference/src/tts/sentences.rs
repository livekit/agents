//! BlingFire-backed speech tokenization and incremental sentence batching.

use lk_agents_media::blingfire::{Error, split_sentences};

// ==================================================================================================
// The incremental splitter
// ==================================================================================================

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
        use lk_agents_core::tokenize::Tokenize as _;
        let tok = Blingfire { min_sentence_len: 1, retain_format: false };
        let spans = tok.tokenize("Hello there. How are you?");
        assert_eq!(
            spans.iter().map(|s| s.text.as_str()).collect::<Vec<_>>(),
            ["Hello there.", "How are you?"]
        );
        assert_eq!((spans[1].start, spans[1].end), (12, 25));
        let held = tok.tokenize("a\0b");
        assert_eq!(held.len(), 1);
        assert_eq!(held[0].text, "a\0b");
        assert_eq!(tok.tokenize("   "), Vec::<lk_agents_core::tokenize::Span>::new());
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
        assert_eq!(
            out,
            vec!["Hello <break time=\"1s\"/> there my friend.", "Second sentence here."]
        );
        for token in &out {
            assert!(
                !lk_agents_core::tokenize::xml::has_unclosed_tags(token),
                "a token split a tag: {token:?}"
            );
        }
    }

    // Catches: the provider knobs not reaching the buffer. Expressive: the floor is the cap and
    // the wrap is on; plain: per-sentence emission and no wrap.
    #[test]
    fn from_config_applies_the_three_provider_knobs() {
        let expressive = SentenceTokenizerConfig {
            max_token_len: Some(900),
            min_token_len: Some(900),
            xml_aware: true,
        };
        let got = SentenceBufferOptions::from_config(expressive);
        assert_eq!((got.max_token_len, got.min_token_len, got.xml_aware), (Some(900), 900, true));
        assert_eq!((got.min_sentence_len, got.min_ctx_len, got.retain_format), (20, 10, true));
        let plain = SentenceTokenizerConfig {
            max_token_len: Some(900),
            min_token_len: None,
            xml_aware: false,
        };
        let got = SentenceBufferOptions::from_config(plain);
        assert_eq!((got.max_token_len, got.min_token_len, got.xml_aware), (Some(900), 20, false));
    }
}
