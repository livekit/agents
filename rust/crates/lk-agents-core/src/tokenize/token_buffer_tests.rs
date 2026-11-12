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

const PER_SENTENCE: TokenBufferOptions =
    TokenBufferOptions { min_token_len: 1, min_ctx_len: 1, max_token_len: None };

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
    let mut state =
        super::TokenBuffer::new(TokenBufferOptions { min_ctx_len: 4, ..TokenBufferOptions::WORDS });
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

// Catches: re-requesting the same text forever when a tokenizer's first token is empty.
#[test]
fn an_empty_first_token_holds_the_text_until_flush() {
    let mut state = super::TokenBuffer::new(TokenBufferOptions::WORDS);
    assert_eq!(state.push("-c"), TokenBufferStep::Tokenize("-c".into()));
    let spans = vec![
        Span { text: String::new(), start: 0, end: 0 },
        Span { text: "c".into(), start: 1, end: 2 },
    ];
    assert_eq!(state.accept(spans.clone()), TokenBufferStep::Done(vec![]));
    assert_eq!(state.flush(), TokenBufferStep::Tokenize("-c".into()));
    assert_eq!(state.accept(spans), TokenBufferStep::Done(vec!["c".into()]));
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
    let whole = drive_str(&mut sentences(opts.clone()), &[text]);
    let per_char: Vec<String> = text.chars().map(String::from).collect();
    assert_eq!(drive(&mut sentences(opts), &per_char), whole);
    assert_eq!(whole, vec!["Hello there, my 友人!", "How are you? I am well, thanks."]);
}

// --- The tokenizer implementations.

// Catches: `Words` dropping whitespace under `retain_format`, or spans with gaps.
#[test]
fn words_spans_tile_the_input_and_reassemble_it() {
    let text = "  Hello there,  wörld! 你好 ok ";
    let opts = WordTokenizerOptions { ignore_punctuation: false, retain_format: true };
    let spans = Words(opts).tokenize(text);
    assert_eq!(spans[0].start, 0);
    for pair in spans.windows(2) {
        assert_eq!(pair[0].end, pair[1].start, "gap between spans: {spans:?}");
    }
    assert_eq!(spans.last().expect("tokens").end, text.chars().count());
    assert_eq!(spans.iter().map(|s| s.text.as_str()).collect::<String>(), text);
}
