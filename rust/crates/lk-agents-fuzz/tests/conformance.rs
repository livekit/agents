//! The recogniser fake against the truth. Every STT recording in the corpus (`corpus/recordings`
//! in lk-agents-test-support, `corpus/record.py`: real providers through LiveKit Inference) is
//! replayed through the fake that models its provider, and what the fake does is compared with
//! what the provider did: the fake's utterances are shaped as the recorded ones were, against the
//! recorded speech spans, and every delay lies within what the provider was recorded taking.
//! The realtime server is held to its recordings by inference's own tests.
use std::collections::{BTreeMap, BTreeSet};

use lk_agents_fuzz::search::stt_kinds;
use lk_agents_test_support::corpus;
use serde_json::Value;

fn recordings(prefix: &str) -> Vec<(String, Value)> {
    let dir = corpus::dir().join("recordings");
    let mut out: Vec<(String, Value)> = std::fs::read_dir(&dir)
        .expect("corpus/recordings")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with(prefix)))
        .map(|p| {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            let value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
            (name, value)
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    assert!(!out.is_empty(), "no {prefix} recordings");
    out
}

/// Each utterance's event kinds, from its start of speech to its end, runs collapsed, and the
/// interim-preflight alternation Flux repeats as long as the person goes on folded into one
/// `(interim > preflight)+`: how many times is the length of the turn, not its shape.
fn utterance_shapes(kinds: &[(f64, String)]) -> BTreeSet<String> {
    let mut shapes = BTreeSet::new();
    let mut cur: Vec<String> = Vec::new();
    for (_, k) in kinds {
        if k == "start_of_speech" {
            cur.clear();
        }
        if cur.last() != Some(k) {
            cur.push(k.clone());
        }
        if k == "end_of_speech" {
            let mut folded: Vec<String> = Vec::new();
            let mut i = 0;
            while i < cur.len() {
                let pair = cur.get(i).map(String::as_str) == Some("interim_transcript")
                    && cur.get(i + 1).map(String::as_str) == Some("preflight_transcript");
                if pair {
                    while cur.get(i).map(String::as_str) == Some("interim_transcript")
                        && cur.get(i + 1).map(String::as_str) == Some("preflight_transcript")
                    {
                        i += 2;
                    }
                    folded.push("(interim_transcript > preflight_transcript)+".into());
                } else {
                    folded.push(cur[i].clone());
                    i += 1;
                }
            }
            shapes.insert(folded.join(" > "));
            cur.clear();
        }
    }
    shapes
}

// Catches: a recogniser model whose utterances are shaped as the recorded provider's never were
// (a start of speech before its first words, a final with no usage before it, an end of speech
// apart from its final, a preflight from a model that sends none) or whose delays lie outside what
// it was recorded taking.
#[test]
fn the_recogniser_behaves_as_recorded() {
    let mut by_model: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for (_, rec) in recordings("stt-") {
        by_model.entry(rec["model"].as_str().unwrap().to_owned()).or_default().push(rec);
    }
    for model in corpus::STT_MODELS {
        let recs = by_model.get(*model).unwrap_or_else(|| panic!("no recording of {model}"));
        let mut recorded = BTreeSet::new();
        for rec in recs {
            let kinds: Vec<(f64, String)> = rec["events"]
                .as_array()
                .unwrap()
                .iter()
                .map(|e| (e["t"].as_f64().unwrap(), e["type"].as_str().unwrap().to_owned()))
                .collect();
            recorded.extend(utterance_shapes(&kinds));
        }
        let (first_lo, first_hi) = corpus::range(&["stt", model, "first_after_onset"]);
        let (final_lo, final_hi) = corpus::range(&["stt", model, "final_after_end"]);
        let mut fake = BTreeSet::new();
        for rec in recs {
            for (i, span) in rec["speech"].as_array().unwrap().iter().enumerate() {
                let dur =
                    ((span["t1"].as_f64().unwrap() - span["t0"].as_f64().unwrap()) * 1000.0) as u64;
                let words = "one two three four five six seven eight nine ten";
                for seed in 0..20 {
                    let kinds = stt_kinds(model, words, dur, seed * 31 + i as u64);
                    let as_owned: Vec<(f64, String)> =
                        kinds.iter().map(|(t, k)| (*t as f64, (*k).to_owned())).collect();
                    fake.extend(utterance_shapes(&as_owned));
                    // The first words of the first piece, and the last final, within the recorded
                    // delays (the cuts of a long turn place later pieces past the first).
                    if let Some((t, _)) = kinds
                        .iter()
                        .find(|(_, k)| *k == "interim_transcript" || *k == "final_transcript")
                    {
                        let lag = *t as f64;
                        assert!(
                            lag >= first_lo.min(dur as f64) - 1.0
                                && lag <= first_hi.max(dur as f64 + final_hi) + 1.0,
                            "{model}: first words {lag} ms after onset, recorded {first_lo}..{first_hi}"
                        );
                    }
                    if let Some((t, _)) = kinds.iter().rev().find(|(_, k)| *k == "final_transcript")
                    {
                        let tail = *t as f64 - dur as f64;
                        assert!(
                            tail <= final_hi + first_hi + 1.0 && tail >= final_lo.min(0.0) - 1.0,
                            "{model}: last final {tail} ms after the end, recorded {final_lo}..{final_hi}"
                        );
                    }
                }
            }
            // Noise, for a model that reports it.
            fake.extend(utterance_shapes(
                &stt_kinds(model, "", 400, 1)
                    .iter()
                    .map(|(t, k)| (*t as f64, (*k).to_owned()))
                    .collect::<Vec<_>>(),
            ));
        }
        for shape in &fake {
            assert!(
                recorded.contains(shape),
                "{model}: an utterance shaped as no recorded one was:\n  {shape}\nrecorded:\n  {}",
                recorded.iter().cloned().collect::<Vec<_>>().join("\n  ")
            );
        }
        // And every recorded shape is one the model makes.
        for shape in &recorded {
            assert!(
                fake.contains(shape),
                "{model}: the provider shaped an utterance as\n  {shape}\nand the model never does; it makes:\n  {}",
                fake.iter().cloned().collect::<Vec<_>>().join("\n  ")
            );
        }
    }
}
