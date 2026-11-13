//! The sentence splitter's speed, as an LLM streams a reply in pieces of one to seven characters:
//! each push's latency over the experiment corpus and over adversarial replies, and the cost of a
//! process's first stream. A synchronous call may take 10% of a 10ms frame
//! (`rust/bench-budgets.json`): 1ms at the 99.9th percentile.
//!
//! `make bench-sentences`, or:
//!
//! `cargo test --release -p lk-agents-runtime --test sentence_bench -- --ignored --nocapture
//! --test-threads=1`
#![allow(clippy::disallowed_methods, reason = "a benchmark measures real time")]

use std::path::Path;
use std::time::{Duration, Instant};

use lk_agents_core::lexer::Vocabulary;
use lk_agents_core::tokenize::sentences::{SentenceOptions, SentenceSplitter, SentenceStream};

/// The slowest push allowed, at the 99.9th percentile: a lone slower one is the OS pausing the
/// thread, not the splitter.
const BUDGET: Duration = Duration::from_millis(1);

/// A wrapper the model opens and never closes.
const OPEN: &str = "<expr type=\"prosody\" label=\"slow\">";

/// A reply shape, and how to write it at a length.
type Case = (&'static str, fn(usize) -> String);

/// The text cut as an LLM streams it: pieces of one to seven characters.
fn pieces(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut at = 0;
    let mut size = 1;
    let bounds: Vec<usize> = text.char_indices().map(|(i, _)| i).chain([text.len()]).collect();
    let mut k = 0;
    while k + 1 < bounds.len() {
        let next = (k + size).min(bounds.len() - 1);
        out.push(&text[at..bounds[next]]);
        at = bounds[next];
        k = next;
        size = size % 7 + 1;
    }
    out
}

/// Per-push durations and the whole text's duration.
fn time(stream: &mut SentenceStream, text: &str, pushes: &mut Vec<Duration>) -> Duration {
    let start = Instant::now();
    for piece in pieces(text) {
        let t = Instant::now();
        std::hint::black_box(stream.push(piece));
        pushes.push(t.elapsed());
    }
    let t = Instant::now();
    std::hint::black_box(stream.flush());
    pushes.push(t.elapsed());
    start.elapsed()
}

fn pct(sorted: &[Duration], p: f64) -> Duration {
    sorted[((sorted.len() - 1) as f64 * p) as usize]
}

fn options() -> Vec<(&'static str, SentenceOptions)> {
    let base = SentenceOptions::default();
    vec![
        ("default", base.clone()),
        ("markup", SentenceOptions { markup: Some(Vocabulary::LiveKit), ..base.clone() }),
        ("clauses", SentenceOptions { clause_words: Some(5), max_token_len: Some(120), ..base }),
    ]
}

fn texts(name: &str) -> Vec<String> {
    // the experiment corpus beside the checkout, as `lk-agents-core`'s ignored tests read it
    let corpus = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../sentence-corpus");
    let path = corpus.join("corpus").join(name);
    let lines =
        std::fs::read_to_string(&path).expect("the experiment corpus is beside the checkout");
    lines
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let doc: serde_json::Value = serde_json::from_str(l).expect("a corpus line is JSON");
            doc["text"].as_str().expect("a corpus document has a text").to_owned()
        })
        .collect()
}

#[test]
#[ignore = "a measurement, in release"]
fn speed_on_real_replies() {
    for file in [
        "llm.jsonl",
        "tau2.jsonl",
        "generated.jsonl",
        "zh.jsonl",
        "ja.jsonl",
        "hi.jsonl",
        "lists.jsonl",
    ] {
        let docs = texts(file);
        for (name, opts) in options() {
            let mut stream = SentenceStream::new(opts);
            let mut pushes = Vec::new();
            let mut whole = Vec::new();
            let mut chars = 0;
            for doc in &docs {
                chars += doc.chars().count();
                whole.push(time(&mut stream, doc, &mut pushes));
            }
            pushes.sort();
            assert!(pct(&pushes, 0.999) < BUDGET, "p99.9 over {BUDGET:?}");
            whole.sort();
            let total: Duration = whole.iter().sum();
            println!(
                "{file:16} {name:8} docs {:6} pushes {:8} | push p50 {:>8.2?} p99 {:>8.2?} p99.9 {:>8.2?} max {:>8.2?} | reply max {:>8.2?} | {:.0} chars/ms",
                docs.len(),
                pushes.len(),
                pct(&pushes, 0.5),
                pct(&pushes, 0.99),
                pct(&pushes, 0.999),
                pushes[pushes.len() - 1],
                whole[whole.len() - 1],
                chars as f64 / total.as_secs_f64() / 1000.0
            );
        }
    }
}

#[test]
#[ignore = "a measurement, in release"]
fn speed_on_adversarial_replies() {
    let cases: Vec<Case> = vec![
        ("no sentence end", |n| "word ".repeat(n / 5)),
        ("short sentences", |n| "Ok. ".repeat(n / 4)),
        ("normal sentences", |n| "The flight leaves at nine from gate twelve. ".repeat(n / 44)),
        ("open wrapper", |n| format!("{OPEN}{}", "Short one here, really. ".repeat(n / 24))),
        ("many tags", |n| "Hello <break time=\"1s\"/> there my friend. ".repeat(n / 42)),
        ("cjk no stop", |n| "我们今天去公园玩".repeat(n / 8)),
        ("cjk sentences", |n| "我们今天去公园玩。".repeat(n / 9)),
        ("digit readout", |n| "4 1 5 ".repeat(n / 6)),
        ("abbreviations", |n| "e.g. ".repeat(n / 5)),
        ("list", |n| "1. item\n".repeat(n / 8)),
        ("no spaces", |n| "a".repeat(n)),
    ];
    for (name, make) in &cases {
        for n in [1_000, 5_000, 20_000] {
            let text = make(n);
            for (opt, opts) in options() {
                let mut stream = SentenceStream::new(opts);
                let mut pushes = Vec::new();
                let whole = time(&mut stream, &text, &mut pushes);
                pushes.sort();
                assert!(pct(&pushes, 0.999) < BUDGET, "p99.9 over {BUDGET:?}");
                println!(
                    "{name:17} {n:6} {opt:8} | push p50 {:>8.2?} p99 {:>8.2?} max {:>8.2?} | whole {:>9.2?}",
                    pct(&pushes, 0.5),
                    pct(&pushes, 0.99),
                    pushes[pushes.len() - 1],
                    whole
                );
            }
        }
    }
}

#[test]
#[ignore = "a measurement, in release"]
fn speed_from_cold() {
    let reply = "Sure, I can help with that. Your flight AA 1234 leaves at 9:30 a.m. from gate B12. \
                 Would you like me to text you the confirmation? It's 4 1 5, 5 5 5, 0 1 2 3.";
    let t = Instant::now();
    drop(SentenceSplitter::default());
    println!("the rules compile, as process init does: {:?}", t.elapsed());
    let mut stream = SentenceStream::new(SentenceOptions::default());
    let mut pushes = Vec::new();
    let whole = time(&mut stream, reply, &mut pushes);
    println!("first reply: {whole:?}, slowest push {:?}", pushes.iter().max().unwrap());
    for i in 0..3 {
        let t = Instant::now();
        let mut stream = SentenceStream::new(SentenceOptions::default());
        let made = t.elapsed();
        let mut pushes = Vec::new();
        let whole = time(&mut stream, reply, &mut pushes);
        println!(
            "new stream {i}: made in {made:?}, reply {whole:?}, slowest push {:?}",
            pushes.iter().max().unwrap()
        );
    }
}
