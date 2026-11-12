use super::*;
use lk_agents_abi::SentenceSplitter as _;

fn drive(s: &mut SentenceBuffer, chunks: &[&str]) -> Vec<String> {
    let mut out: Vec<String> = chunks.iter().flat_map(|c| s.push(c)).collect();
    out.extend(s.flush());
    out
}

// Catches: losing text at a chunk boundary. The model writes tokens, not sentences, so a
// sentence usually arrives across several pushes, and every character has to come back out.
// `retain_format` is on, so the whitespace between sentences belongs to the one that follows.
#[test]
fn a_sentence_split_across_chunks_comes_back_whole() {
    assert_eq!(
        drive(
            &mut SentenceBuffer::default(),
            &["Hel", "lo the", "re. How ", "are you? I am", " well."]
        ),
        vec!["Hello there. How are you?", " I am well."]
    );
}

// Catches: the floor not applying on the agent-speech path, where this is the tokenizer. "Sure."
// is one word under the four-word floor, so it merges forward and the synthesiser does not
// breathe after it.
#[test]
fn a_short_sentence_is_merged_forward_not_spoken_alone() {
    assert_eq!(
        drive(&mut SentenceBuffer::default(), &["Sure. ", "Let me look."]),
        vec!["Sure. Let me look."]
    );
}

// Catches: a cut by bytes rather than characters. Two Cyrillic sentences (two bytes per
// letter): the first goes out once the second starts, and the rest is carried over intact.
#[test]
fn a_multibyte_buffer_is_cut_at_the_codepoint_and_carried_over_correctly() {
    let mut s = SentenceBuffer::default();
    assert_eq!(
        s.push("Это первое достаточно длинное предложение."),
        Vec::<String>::new(),
        "it could still go on"
    );
    assert_eq!(
        s.push(" А это второе достаточно длинное предложение."),
        vec!["Это первое достаточно длинное предложение."]
    );
    assert_eq!(s.flush(), vec![" А это второе достаточно длинное предложение."]);
}

// Catches: the last sentence going out before the text ends, or a flush that repeats it.
#[test]
fn the_final_sentence_is_held_until_flush() {
    let mut s = SentenceBuffer::default();
    assert_eq!(s.push("This is a sufficiently long first sentence."), Vec::<String>::new());
    let out = s.push(" And this is a sufficiently long second one.");
    assert_eq!(out, vec!["This is a sufficiently long first sentence."]);
    assert_eq!(s.flush(), vec![" And this is a sufficiently long second one."]);
    assert_eq!(s.flush(), Vec::<String>::new(), "flush is not repeatable");
}

// Catches: `retain_format` off not reaching the stream.
#[test]
fn retain_format_off_collapses_newlines_through_the_buffer() {
    let opts = SentenceOptions::default();
    let out = drive(
        &mut SentenceBuffer::with_options(opts),
        &["This is a sufficiently long first sentence.\n\n", "And this is a second one."],
    );
    assert_eq!(out, ["This is a sufficiently long first sentence.", "And this is a second one."]);
}

// `tests/test_tokenizer.py`'s text, fed in chunks of 1, 2, 4, ... characters.
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

/// The reference's `[1, 2, 4]` chunking.
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

// The public tokenizer's stream says what its `tokenize` says, chunk for chunk.
#[test]
fn streamed_sentence_tokenizer_matches_its_split() {
    use lk_agents_abi::tokenize::bindings::SpeechSentenceStream;
    let tokenizer = SpeechSentenceTokenizer::new(4, None, false, None);
    let mut s = tokenizer.stream();
    let chunks = reference_chunks(TEXT);
    let mut out: Vec<String> =
        chunks.iter().flat_map(|c| SpeechSentenceStream::push(&mut s, c.clone())).collect();
    out.extend(SpeechSentenceStream::flush(&mut s));
    assert_eq!(
        out,
        [
            "Hi! LiveKit is a platform for live audio and video applications and services.",
            "R.T.C stands for Real-Time Communication... again R.T.C. Mr. Theo is testing the \
             sentence tokenizer.",
            "This is a test.",
            "Another test. A short sentence.",
            "A longer sentence that is longer than the previous sentence. f(x) = x * 2.54 + 42.",
            "Hey! Hi! Hello! This is a sentence.",
            "这是一个中文句子。",
            "これは日本語の文章です。",
            "你好！LiveKit是一个直播音频和视频应用程序和服务的平台。",
            "This is a sentence contains   consecutive spaces.",
        ]
    );
    assert_eq!(out, tokenizer.tokenize(TEXT.to_owned(), None));
}

// Catches: a `<break time="1s"/>` arriving one character at a time being cut, or split off
// its sentence.
#[test]
fn a_break_tag_mid_sentence_never_splits() {
    let opts = SentenceOptions {
        min_words: 1,
        markup: Some(lk_agents_core::lexer::Vocabulary::All),
        ..SentenceOptions::default()
    };
    let text = "Hello <break time=\"1s\"/> there my friend. Second sentence here.";
    let by_char: Vec<String> = text.chars().map(String::from).collect();
    let pieces: Vec<&str> = by_char.iter().map(String::as_str).collect();
    assert_eq!(
        drive(&mut SentenceBuffer::with_options(opts), &pieces),
        ["Hello <break time=\"1s\"/> there my friend.", "Second sentence here."]
    );
}

// Catches: the provider limits not reaching the stream. Expressive batches up to the cap; plain
// speaks sentence by sentence and never batches past it.
#[test]
fn a_provider_cap_bounds_the_batches() {
    let text = "This is one more sentence. ".repeat(80);
    let expressive = drive(
        &mut sentence_splitter(Some("inworld"), markup_vocabulary("inworld", true)),
        &[&text],
    );
    assert!(expressive.len() > 1 && expressive.iter().all(|c| c.chars().count() <= 900));
    assert!(expressive[0].chars().count() > 880, "batched up to the cap: {}", expressive[0]);
    let plain = drive(&mut sentence_splitter(Some("inworld"), None), &[&text]);
    assert_eq!(plain.len(), 80);
}
