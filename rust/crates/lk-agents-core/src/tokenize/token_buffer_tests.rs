use super::*;

/// An application tokenizer: sentences end at `.`, `!` or `?` before whitespace or the end;
/// spans tile, text trimmed.
struct Dots;

impl TokenizerCallback for Dots {
    fn tokenize(&self, text: &str) -> Result<Vec<TokenizerToken>, HostError> {
        let chars: Vec<char> = text.chars().collect();
        let mut out = Vec::new();
        let mut span = |start: usize, end: usize| {
            let token: String = chars[start..end].iter().collect::<String>().trim().to_owned();
            if !token.is_empty() {
                out.push(TokenizerToken::Span(TokenSpan { token, start, end }));
            }
        };
        let mut start = 0;
        for i in 0..chars.len() {
            let terminal = matches!(chars[i], '.' | '!' | '?');
            if terminal && chars.get(i + 1).is_none_or(|c| c.is_whitespace()) {
                span(start, i + 1);
                start = i + 1;
            }
        }
        span(start, chars.len());
        Ok(out)
    }
}

/// Tokens given as text alone, split at whitespace.
struct Whitespace;

impl TokenizerCallback for Whitespace {
    fn tokenize(&self, text: &str) -> Result<Vec<TokenizerToken>, HostError> {
        Ok(text.split_whitespace().map(|w| TokenizerToken::Text(w.to_owned())).collect())
    }
}

fn buffer(
    tokenizer: impl TokenizerCallback + 'static,
    min_token_len: usize,
    min_ctx_len: usize,
    max_token_len: Option<usize>,
) -> BufferedTokenStream {
    BufferedTokenStream::new(Arc::new(tokenizer), min_token_len, min_ctx_len, max_token_len)
}

fn sentences(min_token_len: usize, max_token_len: Option<usize>) -> BufferedTokenStream {
    buffer(Dots, min_token_len, 1, max_token_len)
}

fn push(buf: &mut BufferedTokenStream, text: &str) -> Vec<String> {
    buf.push(text.to_owned()).expect("the tokenizer does not fail")
}

fn flush(buf: &mut BufferedTokenStream) -> Vec<String> {
    buf.flush().expect("the tokenizer does not fail")
}

fn drive(buf: &mut BufferedTokenStream, chunks: &[&str]) -> Vec<String> {
    let mut out: Vec<String> = chunks.iter().flat_map(|c| push(buf, c)).collect();
    out.extend(flush(buf));
    out
}

// Catches: `push` emitting the last token.
#[test]
fn the_final_token_is_held_until_flush() {
    let mut buf = sentences(1, None);
    assert_eq!(push(&mut buf, "First one."), Vec::<String>::new(), "nothing follows it yet");
    assert_eq!(push(&mut buf, " Second one."), vec!["First one."]);
    assert_eq!(flush(&mut buf), vec!["Second one."]);
    assert_eq!(flush(&mut buf), Vec::<String>::new(), "flush is not repeatable");
}

// Catches: text tokens placed by bytes, or without the whitespace that follows them.
#[test]
fn text_tokens_are_found_in_the_text() {
    let mut buf = buffer(Whitespace, 1, 1, None);
    assert_eq!(push(&mut buf, "é  word "), vec!["é"]);
    assert_eq!(push(&mut buf, "next"), vec!["word"]);
    assert_eq!(flush(&mut buf), vec!["next"]);
}

// Catches: re-requesting the same text forever when a tokenizer's first token is empty.
#[test]
fn an_empty_first_token_holds_the_text_until_flush() {
    struct EmptyFirst;
    impl TokenizerCallback for EmptyFirst {
        fn tokenize(&self, _: &str) -> Result<Vec<TokenizerToken>, HostError> {
            Ok(vec![
                TokenizerToken::Span(TokenSpan { token: String::new(), start: 0, end: 0 }),
                TokenizerToken::Span(TokenSpan { token: "c".into(), start: 1, end: 2 }),
            ])
        }
    }
    let mut buf = buffer(EmptyFirst, 1, 1, None);
    assert_eq!(push(&mut buf, "-c"), Vec::<String>::new());
    assert_eq!(flush(&mut buf), vec!["c"]);
}

// Catches: skipping `min_token_len` batching, or `>` for `>=` at the floor exactly.
#[test]
fn short_tokens_are_batched_until_the_floor_is_reached() {
    let chunks = ["Sure. ", "Let me look. ", "And one more here. Tail."];
    assert_eq!(
        drive(&mut sentences(20, None), &chunks),
        vec!["Sure. Let me look. And one more here.", "Tail."]
    );
    // "Sure. Let me look." is 18 characters: the floor is inclusive.
    assert_eq!(
        drive(&mut sentences(18, None), &chunks),
        vec!["Sure. Let me look.", "And one more here.", "Tail."]
    );
}

// Catches: `min_ctx_len` being ignored.
#[test]
fn nothing_is_emitted_below_the_context_floor() {
    let mut buf = buffer(Dots, 1, 10, None);
    assert_eq!(push(&mut buf, "Hi. Yo."), Vec::<String>::new());
    assert_eq!(flush(&mut buf), vec!["Hi. Yo."], "flush batches both, under no floor");

    // Exactly at the floor (7 chars) the tokenizer is consulted; `<=` would hold it.
    let mut buf = buffer(Dots, 1, 7, None);
    assert_eq!(push(&mut buf, "Hi. Yo."), vec!["Hi."], "the floor is exclusive");
}

// Catches: the cap checked after the join rather than before, or not at all.
#[test]
fn the_max_token_len_cap_splits_a_batch() {
    assert_eq!(
        drive(&mut sentences(100, Some(25)), &["Sure. Let me look. One more bit here. And more."]),
        vec!["Sure. Let me look.", "One more bit here.", "And more."]
    );
}

// Catches: the cap cutting a token (an oversized one goes whole).
#[test]
fn a_single_token_longer_than_the_cap_is_emitted_whole() {
    assert_eq!(
        drive(&mut sentences(1, Some(10)), &["This sentence is well past ten. Ok."]),
        vec!["This sentence is well past ten.", "Ok."]
    );
}

// Catches: `flush` honouring `min_token_len` (the cap still applies).
#[test]
fn flush_ignores_the_floor_but_keeps_the_cap() {
    let mut buf = sentences(100, Some(12));
    // The cap, not the floor, is what releases the first one during the push.
    assert_eq!(push(&mut buf, "One two. Three four. Five."), vec!["One two."]);
    assert_eq!(flush(&mut buf), vec!["Three four.", "Five."]);
}

// Catches: `flush` on an empty buffer inventing a token, or `push("")` consulting the tokenizer.
#[test]
fn flushing_nothing_emits_nothing() {
    let mut buf = sentences(1, None);
    assert_eq!(flush(&mut buf), Vec::<String>::new());
    assert_eq!(push(&mut buf, ""), Vec::<String>::new());
    assert_eq!(flush(&mut buf), Vec::<String>::new());
}

// Catches: slicing or counting by bytes instead of characters.
#[test]
fn multibyte_text_is_sliced_and_counted_by_character() {
    assert_eq!(
        drive(&mut sentences(12, None), &["Готово. ", "Дальше что? ", "Ещё."]),
        vec!["Готово. Дальше что?", "Ещё."]
    );
}

// Catches: chunking changing the output.
#[test]
fn the_tokens_do_not_depend_on_how_the_text_was_chunked() {
    let text = "Hello there, my 友人! How are you? I am well, thanks.";
    let whole = drive(&mut sentences(15, None), &[text]);
    let per_char: Vec<String> = text.chars().map(String::from).collect();
    let per_char: Vec<&str> = per_char.iter().map(String::as_str).collect();
    assert_eq!(drive(&mut sentences(15, None), &per_char), whole);
    assert_eq!(whole, vec!["Hello there, my 友人!", "How are you? I am well, thanks."]);
}
