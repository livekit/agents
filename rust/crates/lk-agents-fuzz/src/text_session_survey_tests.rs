//! The TextSession simulation (`lk_agents_fuzz::text_session`) over many seeds: every
//! conversation must keep every rule under the arrival order and BasicPOS, and the campaign must
//! reach every scenario. A finding of a (contract, subject kind) in `known_findings.tsv` is
//! counted, not failed.
//!
//! The campaign is `survey`, behind `--ignored` (the launch gate runs it): `TEXT_SESSION_SEEDS`
//! sizes it, `TEXT_SESSION_SEED=<n>` reruns one seed with its findings in full.
use std::collections::{BTreeMap, BTreeSet};

use lk_agents_fuzz::{boundary, sched::Sampling, text_session};

/// Every scenario the campaign must reach.
const REACH: [&str; 10] = [
    "TextSessionReconstructed",
    "TextSessionChildStarted",
    "TextSessionChildCompleted",
    "TextSessionParentParked",
    "TextSessionApprovalAsked",
    "TextSessionDepthRefused",
    "TextSessionInputRequired",
    "TextSessionCompleted",
    "TextSessionFailed",
    "TextSessionCancelled",
];

#[test]
fn a_seed_replays_its_run() {
    let (a, b) = (text_session::run(9, Sampling::Arrival), text_session::run(9, Sampling::Arrival));
    assert_eq!(a.reached, b.reached);
    assert_eq!(a.trace.turns.len(), b.trace.turns.len());
}

#[test]
#[ignore = "the campaign: the launch gate runs it"]
fn survey() {
    let seeds: Vec<u64> = match std::env::var("TEXT_SESSION_SEED") {
        Ok(one) => vec![one.parse().expect("a seed")],
        Err(_) => {
            let n: u64 =
                std::env::var("TEXT_SESSION_SEEDS").map_or(4096, |n| n.parse().expect("a count"));
            assert!(n > 0, "a campaign needs a positive number of seeds");
            (0..n).collect()
        }
    };
    let mut failures = Vec::new();
    let mut known: BTreeMap<String, u64> = BTreeMap::new();
    let mut reached: BTreeSet<&'static str> = BTreeSet::new();
    for &seed in &seeds {
        for sampling in [Sampling::Arrival, Sampling::BasicPos] {
            let outcome = text_session::run(seed, sampling);
            reached.extend(outcome.reached);
            for f in outcome.violations {
                if f.known() {
                    *known.entry(format!("{} ({})", f.contract, f.subject)).or_default() += 1;
                } else {
                    failures.push(format!(
                        "TEXT_SESSION_SEED={seed} ({sampling:?}): {} ({}): {}",
                        f.contract, f.subject, f.text
                    ));
                }
            }
        }
    }
    boundary::TEXT_SESSION.report(&reached, &[]);
    println!("{} seeds x 2 orders, reached {reached:?}, known {known:?}", seeds.len());
    if std::env::var("TEXT_SESSION_SEED").is_err() {
        let missing: Vec<&&str> = REACH.iter().filter(|r| !reached.contains(**r)).collect();
        assert!(missing.is_empty(), "the campaign never reached {missing:?}");
    }
    assert!(failures.is_empty(), "{} findings:\n{}", failures.len(), failures.join("\n"));
}
