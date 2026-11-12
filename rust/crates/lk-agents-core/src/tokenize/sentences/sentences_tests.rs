use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use super::{SentenceOptions, SentenceSplitter};

/// The full corpus the CI windows were chosen from: `sentence-corpus/` beside the checkout.
pub(super) fn experiment_corpus() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../sentence-corpus")
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/sentences")
}

fn jsonl(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn str_of(v: &Value) -> &str {
    v.as_str().unwrap()
}

/// Where each sentence starts, in characters.
fn starts(v: &Value) -> Vec<usize> {
    v.as_array().unwrap().iter().map(|n| n.as_u64().unwrap() as usize).collect()
}

// Every corpus window splits as recorded.
#[test]
fn the_corpus_splits_as_recorded() {
    check_corpus(false);
}

// No group's F1 falls below its floor in `f1_floors.tsv`.
#[test]
fn no_group_scores_below_its_f1_floor() {
    check_floors(false);
}

// After a change to the rules, rewrites `corpus.jsonl` and `f1_floors.tsv`; review their diff.
#[test]
#[ignore = "rewrites the recorded fixtures"]
fn record_sentence_fixtures() {
    check_corpus(true);
    check_floors(true);
}

fn check_corpus(record: bool) {
    let path = fixtures().join("corpus.jsonl");
    let mut splitter = SentenceSplitter::default();
    let docs = jsonl(&path);
    let mut changed = Vec::new();
    let mut lines = Vec::new();
    for doc in &docs {
        let text = str_of(&doc["text"]);
        let got: Vec<usize> = splitter.spans(text).iter().map(|s| s.0).collect();
        let want = starts(&doc["starts"]);
        if got != want {
            let chars: Vec<char> = text.chars().collect();
            let cut = |at: &[usize]| -> Vec<String> {
                at.iter()
                    .zip(at.iter().skip(1).chain([&chars.len()]))
                    .map(|(&a, &b)| chars[a..b].iter().collect::<String>().trim().to_owned())
                    .collect()
            };
            let id = str_of(&doc["id"]);
            changed.push(format!("{id}\n   got  {:?}\n   want {:?}", cut(&got), cut(&want)));
        }
        let mut line = json!({ "id": doc["id"], "text": doc["text"], "starts": got });
        if let Some(expect) = doc.get("expect") {
            line["expect"] = expect.clone();
        }
        lines.push(line.to_string());
    }
    if record {
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();
        return;
    }
    assert!(changed.is_empty(), "{} changed:\n{}", changed.len(), changed.join("\n"));
}

/// The group a labelled case is scored in: its source and language.
fn group(id: &str) -> String {
    let mut parts = id.split('#').next().unwrap().split('-');
    let source = parts.next().unwrap();
    if source == "issue" {
        source.to_owned()
    } else {
        format!("{source}/{}", parts.next().unwrap())
    }
}

/// True positives, predicted and expected sentence starts, per group. A predicted start where
/// none is expected is a bad break; a missed one holds speech back.
fn counts(splitter: &mut SentenceSplitter, docs: &[Value]) -> BTreeMap<String, [usize; 3]> {
    let mut out: BTreeMap<String, [usize; 3]> = BTreeMap::new();
    for doc in docs {
        let id = str_of(&doc["id"]);
        let text = str_of(&doc["text"]);
        let Some(expect) = doc.get("expect") else { continue };
        let expected: BTreeSet<usize> = starts(expect).into_iter().skip(1).collect();
        let predicted: BTreeSet<usize> = splitter.spans(text).iter().skip(1).map(|s| s.0).collect();
        let tp = predicted.intersection(&expected).count();
        let c = out.entry(group(id)).or_default();
        c[0] += tp;
        // A Tatoeba entry may hold several sentences: only its own boundaries are known.
        c[1] += if id.starts_with("tatoeba") { tp } else { predicted.len() };
        c[2] += expected.len();
    }
    out
}

fn f1([tp, predicted, expected]: [usize; 3]) -> f64 {
    if predicted == 0 && expected == 0 {
        return 1.0;
    }
    let p = tp as f64 / predicted.max(1) as f64;
    let r = tp as f64 / expected.max(1) as f64;
    if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 }
}

fn check_floors(record: bool) {
    let mut splitter = SentenceSplitter::default();
    let scores = counts(&mut splitter, &jsonl(&fixtures().join("corpus.jsonl")));
    let path = fixtures().join("f1_floors.tsv");
    let floors: BTreeMap<String, f64> = std::fs::read_to_string(&path)
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| {
            let mut cols = l.split('\t');
            (cols.next().unwrap().to_owned(), cols.next().unwrap().parse().unwrap())
        })
        .collect();
    let mut report = String::from("# group\tF1\tprecision\trecall\n");
    let mut below = Vec::new();
    for (group, c) in &scores {
        let score = (f1(*c) * 1000.0).floor() / 1000.0;
        let p = c[0] as f64 / c[1].max(1) as f64;
        let r = c[0] as f64 / c[2].max(1) as f64;
        report.push_str(&format!("{group}\t{score:.3}\t{p:.3}\t{r:.3}\n"));
        let floor = floors.get(group).copied().unwrap_or(0.0);
        if score < floor {
            below.push(format!("{group}: {score:.3} < {floor:.3}"));
        }
    }
    println!("{report}");
    if record {
        std::fs::write(&path, report).unwrap();
        return;
    }
    assert!(below.is_empty(), "{}", below.join("\n"));
}

// The experiment corpus holds `corpus/*.jsonl` and BlingFire's recorded `expected/*.jsonl`.
// This writes the current splits, with the rules that fired, to its `out/` and counts the
// documents split unlike BlingFire.
#[test]
#[ignore = "needs the experiment corpus"]
fn compare_with_blingfire_on_the_experiment_corpus() {
    let corpus = experiment_corpus();
    let out = Some(corpus.join("out"));
    std::fs::create_dir_all(corpus.join("out")).unwrap();
    let mut splitter = SentenceSplitter::default();
    let mut differ: BTreeMap<String, usize> = BTreeMap::new();
    for entry in std::fs::read_dir(corpus.join("corpus")).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let texts: BTreeMap<String, String> = jsonl(&path)
            .into_iter()
            .map(|d| (str_of(&d["id"]).to_owned(), str_of(&d["text"]).to_owned()))
            .collect();
        let mut lines = Vec::new();
        for golden in jsonl(&corpus.join("expected").join(&name)) {
            let id = str_of(&golden["id"]);
            let text = &texts[id];
            let got = splitter.spans(text);
            let matches = std::mem::take(&mut splitter.caches.matches);
            let mut fired = std::mem::take(&mut splitter.caches.fired);
            fired.sort();
            fired.dedup();
            let blingfire: Vec<(usize, usize)> =
                serde_json::from_value(golden["spans"].clone()).unwrap();
            if got != blingfire {
                *differ.entry(name.clone()).or_default() += 1;
            }
            let chars: Vec<char> = text.chars().collect();
            let sentences: Vec<String> =
                got.iter().map(|&(s, e)| chars[s..e].iter().collect()).collect();
            let line = json!({ "id": id, "spans": got, "sentences": sentences, "fired": fired, "matches": matches });
            lines.push(line.to_string());
        }
        if let Some(out) = &out {
            std::fs::write(out.join(&name), lines.join("\n") + "\n").unwrap();
        }
    }
    println!("documents that split unlike BlingFire: {differ:?}");
}

fn chunks(text: &str, min_words: usize, retain_format: bool) -> Vec<(String, usize, usize)> {
    let opts = SentenceOptions { min_words, retain_format, ..SentenceOptions::default() };
    SentenceSplitter::default().sentences(text, &opts)
}

// A short sentence merges forward without advancing its start; whitespace belongs to the next
// sentence; `retain_format` keeps it, otherwise newlines collapse and the text is trimmed.
#[test]
fn sentences_merge_short_ones_and_keep_every_character() {
    type Case<'a> = (&'a str, usize, bool, Vec<(&'a str, usize, usize)>);
    let cases: Vec<Case<'_>> = vec![
        (
            "Hello there. How are you? I am well.",
            1,
            false,
            vec![("Hello there.", 0, 12), ("How are you?", 12, 25), ("I am well.", 25, 36)],
        ),
        (
            "Hello there. How are you? I am well.",
            4,
            false,
            vec![("Hello there. How are you?", 0, 25), ("I am well.", 25, 36)],
        ),
        (
            "First line.\n\n   Second line.",
            1,
            true,
            vec![("First line.", 0, 11), ("\n\n   Second line.", 11, 28)],
        ),
        ("One.   Two.", 1, false, vec![("One.", 0, 4), ("Two.", 4, 11)]),
        (
            "一つ目です。二つ目です。",
            1,
            false,
            vec![("一つ目です。", 0, 6), ("二つ目です。", 6, 12)],
        ),
        ("no stop at all", 1, false, vec![("no stop at all", 0, 14)]),
        ("   \n\t ", 1, false, vec![]),
    ];
    for (text, min_words, retain, expected) in cases {
        let got = chunks(text, min_words, retain);
        let got: Vec<(&str, usize, usize)> =
            got.iter().map(|(s, a, b)| (s.as_str(), *a, *b)).collect();
        assert_eq!(got, expected, "{text:?} min_words={min_words} retain={retain}");
    }
}

// Trailing whitespace is not a sentence of its own; it belongs to none, as a stream has said the
// sentence before it arrives. https://github.com/livekit/agents/issues/6296
#[test]
fn trailing_whitespace_is_not_a_separate_sentence() {
    for tail in [" ", "\t  ", "\n\n  "] {
        let text = format!("Hello world.{tail}");
        assert_eq!(chunks(&text, 1, true), vec![("Hello world.".to_owned(), 0, 12)]);
    }
}
