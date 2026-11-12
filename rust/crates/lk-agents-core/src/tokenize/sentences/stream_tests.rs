use std::path::Path;

use super::{
    SentenceOptions, SentenceSplitter, SentenceStream, SpeechSentenceTokenizer, sentence_splitter,
    spoken_words,
};
use crate::lexer::Vocabulary;

fn split(text: &str, opts: &SentenceOptions) -> Vec<String> {
    SentenceSplitter::default().sentences(text, opts).into_iter().map(|s| s.text).collect()
}

fn stream(pieces: &[&str], opts: &SentenceOptions) -> Vec<String> {
    feed(&mut SentenceStream::new(opts.clone()), pieces)
}

fn feed(stream: &mut SentenceStream, pieces: &[&str]) -> Vec<String> {
    let mut out: Vec<String> = pieces.iter().flat_map(|p| stream.push(p)).collect();
    out.extend(stream.flush());
    out
}

/// One splitter and one stream per option set, as their automata warm up.
struct Pair {
    opts: SentenceOptions,
    splitter: SentenceSplitter,
    stream: SentenceStream,
}

fn pairs() -> Vec<Pair> {
    options()
        .into_iter()
        .map(|opts| Pair {
            splitter: SentenceSplitter::default(),
            stream: SentenceStream::new(opts.clone()),
            opts,
        })
        .collect()
}

fn options() -> Vec<SentenceOptions> {
    let base = SentenceOptions::default();
    vec![
        base.clone(),
        SentenceOptions {
            min_words: 1,
            clause_words: Some(5),
            max_token_len: Some(60),
            ..base.clone()
        },
        SentenceOptions { markup: Some(Vocabulary::All), retain_format: true, ..base },
    ]
}

/// The stream, fed `text` cut at every character and one character at a time, emits the
/// chunks the whole text splits into. Returns what differs.
fn mismatches(pair: &mut Pair, text: &str) -> Option<String> {
    let want: Vec<String> =
        pair.splitter.sentences(text, &pair.opts).into_iter().map(|s| s.text).collect();
    let at: Vec<usize> = text.char_indices().map(|(i, _)| i).chain([text.len()]).collect();
    let by_char: Vec<&str> = at.windows(2).map(|w| &text[w[0]..w[1]]).collect();
    let halves = at.iter().map(|&i| vec![&text[..i], &text[i..]]);
    std::iter::once(by_char).chain(halves).find_map(|pieces| {
        let got = feed(&mut pair.stream, &pieces);
        (got != want).then(|| format!("{text:?} fed {pieces:?}\n  got  {got:?}\n  want {want:?}"))
    })
}

fn texts(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            serde_json::from_str::<serde_json::Value>(l).unwrap()["text"]
                .as_str()
                .unwrap()
                .to_owned()
        })
        .collect()
}

const STREAMED: &[&str] = &[
    concat!(
        "Hi! LiveKit is a platform for live audio and video applications and services. \n\n",
        "R.T.C stands for Real-Time Communication... again R.T.C. ",
        "Mr. Theo is testing the sentence tokenizer. \nThis is a test. Another test. ",
        "A short sentence.\nA longer sentence that is longer than the previous sentence. ",
        "f(x) = x * 2.54 + 42. Hey!\n Hi! Hello! \n\nThis is a sentence. ",
        "这是一个中文句子。これは日本語の文章です。你好！LiveKit是一个直播音频和视频应用程序和服务的平台。",
        "\nThis is a sentence contains   consecutive spaces."
    ),
    "Hello <break time=\"1s\"/> there my friend. Second sentence here.",
    "<emphasis>One. Two.</emphasis> Three <break/>. Four, five, six, seven, eight, nine: ten.",
    "Well, the total is 1,000 dollars, due at 12:30; after that, we close, sadly, for good.",
    "我们明天见，好吗？请把文件、照片和视频发给我；谢谢：再见。",
];

// The certainty the stream rests on: whatever the pieces, it says what the whole text splits
// into, chunk for chunk.
#[test]
fn the_stream_says_what_the_whole_text_splits_into() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/sentences/corpus.jsonl");
    let corpus = texts(&path);
    let texts = STREAMED.iter().copied().chain(corpus.iter().map(String::as_str));
    let mut pairs = pairs();
    let mut failures = Vec::new();
    for text in texts {
        failures.extend(pairs.iter_mut().filter_map(|pair| mismatches(pair, text)));
    }
    assert!(failures.is_empty(), "{} differ:\n{}", failures.len(), failures.join("\n"));
}

// `cargo test --release ... stream -- --ignored`: the same property over
// the experiment corpus (`corpus/*.jsonl`).
#[test]
#[ignore = "needs the experiment corpus"]
fn the_stream_says_what_the_whole_text_splits_into_on_the_experiment_corpus() {
    let corpus = super::sentences_tests::experiment_corpus();
    let mut pairs = pairs();
    let mut failures = Vec::new();
    let mut documents = 0;
    for entry in std::fs::read_dir(corpus.join("corpus")).unwrap() {
        for text in texts(&entry.unwrap().path()) {
            documents += 1;
            failures.extend(pairs.iter_mut().filter_map(|pair| mismatches(pair, &text)));
        }
    }
    println!("{documents} documents, {} differ", failures.len());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

// Catches: a sentence held back by the next one when nothing to come could change it.
#[test]
fn a_sentence_goes_out_once_its_end_is_certain() {
    let opts = SentenceOptions { min_words: 1, ..SentenceOptions::default() };
    let mut stream = SentenceStream::new(opts);
    assert_eq!(stream.push("This is the first sentence."), Vec::<String>::new());
    assert_eq!(stream.push(" And"), vec!["This is the first sentence."]);
    assert_eq!(stream.flush(), vec!["And"]);
    assert_eq!(stream.flush(), Vec::<String>::new(), "flush is not repeatable");
}

// Catches: short sentences spoken alone; the floor is in words, not characters.
#[test]
fn short_sentences_batch_to_four_words() {
    let opts = SentenceOptions::default();
    assert_eq!(
        split("Sure. Let me look. Okay then, done.", &opts),
        ["Sure. Let me look.", "Okay then, done."]
    );
    assert_eq!(split("好的。我看看。", &opts), ["好的。我看看。"]);
}

#[test]
fn spoken_words_count_what_is_said() {
    assert_eq!(spoken_words("Hello there, my friend."), 4);
    assert_eq!(spoken_words("don't stop... 2.54"), 3);
    assert_eq!(spoken_words("你好！LiveKit是一个平台。"), 8);
    assert_eq!(spoken_words("これは日本語です。"), 8);
    assert_eq!(spoken_words("สวัสดีครับ"), 4);
    assert_eq!(spoken_words("  "), 0);
    assert_eq!(spoken_words("well - ... ok"), 2);
}

fn clauses(text: &str) -> Vec<String> {
    let opts =
        SentenceOptions { min_words: 1, clause_words: Some(4), ..SentenceOptions::default() };
    split(text, &opts)
}

// Catches: a clause break in a sentence too short to need one, or inside a number or a time.
#[test]
fn a_long_sentence_breaks_at_a_clause_mark_and_a_short_one_never() {
    assert_eq!(clauses("Yes, of course, sure."), ["Yes, of course, sure."]);
    assert_eq!(
        clauses("The total comes to 1,000 dollars at 12:30, and then we close."),
        ["The total comes to 1,000 dollars at 12:30,", "and then we close."]
    );
    assert_eq!(
        clauses("First we open the door; then we walk in."),
        ["First we open the door;", "then we walk in."]
    );
    assert_eq!(
        clauses("Here is what we need: bread and milk."),
        ["Here is what we need:", "bread and milk."]
    );
    assert_eq!(
        clauses("我们明天早上见面，然后一起去公司。"),
        ["我们明天早上见面，", "然后一起去公司。"]
    );
    assert_eq!(clauses("请带上护照、机票、钱包和手机。"), ["请带上护照、", "机票、钱包和手机。"]);
    assert_eq!(clauses("合計は1，000円です。"), ["合計は1，000円です。"]);
    let off = SentenceOptions { min_words: 1, ..SentenceOptions::default() };
    assert_eq!(split("First we open the door; then we walk in.", &off).len(), 1);
}

// Catches: batching past the provider's limit, or cutting a sentence that is longer than it.
#[test]
fn batching_stops_at_the_length_cap_and_a_long_sentence_goes_whole() {
    let opts =
        SentenceOptions { min_words: 100, max_token_len: Some(30), ..SentenceOptions::default() };
    assert_eq!(
        split("One two three. Four five six. A sentence longer than thirty characters.", &opts),
        ["One two three. Four five six.", "A sentence longer than thirty characters."]
    );
}

// Catches: a chunk ending inside a tag or inside the span a wrapping tag opens.
#[test]
fn markup_is_never_split() {
    let opts = SentenceOptions {
        min_words: 1,
        markup: Some(Vocabulary::All),
        ..SentenceOptions::default()
    };
    let text = "Hello <break time=\"1s\"/> there. <emphasis>One. Two.</emphasis> Three.";
    let by_char: Vec<String> = text.chars().map(String::from).collect();
    let pieces: Vec<&str> = by_char.iter().map(String::as_str).collect();
    // a tag at a cut goes with what follows, so the closing tag holds `Two.` with `Three.`
    let want = ["Hello <break time=\"1s\"/> there.", "<emphasis>One. Two.</emphasis> Three."];
    assert_eq!(stream(&pieces, &opts), want);
}

// Catches: a wrapper the model never closes holding the rest of the reply until it ends.
#[test]
fn an_unclosed_wrapper_stops_holding_sentences_once_far_behind() {
    let opts = SentenceOptions {
        markup: Some(crate::lexer::Vocabulary::LiveKit),
        ..SentenceOptions::default()
    };
    let mut stream = SentenceStream::new(opts);
    let mut early = stream.push("<expr type=\"prosody\" label=\"slow\">");
    for n in 0..40 {
        early.extend(stream.push(&format!("This is sentence number {n} of the reply. ")));
    }
    assert!(!early.is_empty(), "nothing reached the voice before the reply ended");
}

/// `tests/test_tokenizer.py`'s `[1, 2, 4]` chunking.
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
    use lk_agents_abi::tokenize::bindings::{
        SpeechSentenceStream, SpeechSentenceTokenizer as Tokenizer,
    };
    let tokenizer = SpeechSentenceTokenizer::new(4, None, false, None);
    let mut s = tokenizer.stream();
    let mut out: Vec<String> = reference_chunks(STREAMED[0])
        .into_iter()
        .flat_map(|c| SpeechSentenceStream::push(&mut s, c))
        .collect();
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
    assert_eq!(out, tokenizer.tokenize(STREAMED[0].to_owned(), None));
}

// Catches: the provider limits not reaching the stream. Expressive batches up to the cap; plain
// speaks sentence by sentence and never batches past it.
#[test]
fn a_provider_cap_bounds_the_batches() {
    let text = "This is one more sentence. ".repeat(80);
    let markup = crate::tts::markup::markup_vocabulary("inworld", true);
    let expressive = feed(&mut sentence_splitter(Some("inworld"), markup), &[&text]);
    assert!(expressive.len() > 1 && expressive.iter().all(|c| c.chars().count() <= 900));
    assert!(expressive[0].chars().count() > 880, "batched up to the cap: {}", expressive[0]);
    let plain = feed(&mut sentence_splitter(Some("inworld"), None), &[&text]);
    assert_eq!(plain.len(), 80);
}
