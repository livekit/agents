//! The A2A simulation (`lk_agents_fuzz::a2a`) over many seeds: every run must keep every rule.
//!
//! A finding of a (contract, subject kind) in `known_findings.tsv` is counted, not failed.
//!
//! `cargo test` runs a small window and the replay check; the campaign is `survey`, behind
//! `--ignored` (the launch gate runs it): `A2A_SEEDS` sizes it, `A2A_SEED=<n>` reruns one seed
//! with its violations in full.
use std::collections::{BTreeMap, BTreeSet};

use lk_agents_core::a2a::Policy;
use lk_agents_fuzz::{a2a, boundary};

const MESSAGES: usize = 8;

/// What a campaign reached, by name, and how often.
fn reach(record: &a2a::Record, into: &mut BTreeMap<&'static str, u64>) {
    let answered = |p: Policy| {
        record.requests.iter().filter(|r| r.policy == p && r.answer.is_some()).count() as u64
    };
    let ledger = &record.ledger;
    for (name, n) in [
        ("Answer", record.requests.iter().filter(|r| r.answer.is_some()).count() as u64),
        ("Batch", answered(Policy::Batch)),
        ("Interrupt", answered(Policy::Interrupt)),
        ("Queue", answered(Policy::Queue)),
        ("Concurrent", answered(Policy::Concurrent)),
        ("Reject", answered(Policy::Reject)),
        ("WorkerLost", record.requests.iter().filter(|r| r.killed).count() as u64),
        ("LostAck", ledger.lost_acks),
    ] {
        *into.entry(name).or_default() += n;
    }
    for (kind, n) in &ledger.injected {
        *into.entry(kind).or_default() += n;
    }
}

/// Runs `seeds`; the failures, one line each with the command that reruns it, and the reach.
fn campaign(seeds: impl Iterator<Item = u64>) -> (Vec<String>, BTreeMap<&'static str, u64>) {
    let mut failures = Vec::new();
    let mut reached = BTreeMap::new();
    for seed in seeds {
        let record = a2a::run(seed, MESSAGES);
        reach(&record, &mut reached);
        let (known, new): (Vec<_>, Vec<_>) =
            a2a::rules::check(&record).into_iter().partition(|f| f.known());
        *reached.entry("KnownFinding").or_default() += known.len() as u64;
        if !new.is_empty() {
            let new: Vec<String> = new.iter().map(ToString::to_string).collect();
            failures.push(format!("A2A_SEED={seed} ({:?}): {new:?}", record.now));
        }
    }
    (failures, reached)
}

#[test]
fn a_small_window_keeps_every_a2a_rule() {
    let (failures, reached) = campaign(0..32);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(reached["Answer"] > 0, "{reached:?}");
}

#[test]
fn a_seed_replays_its_run() {
    let a = a2a::run(7, MESSAGES);
    let b = a2a::run(7, MESSAGES);
    let outcome = |r: &a2a::Record| {
        let statuses: Vec<_> = r.tasks.values().map(|t| t.status).collect();
        let answered = r.requests.iter().map(|q| (q.killed, q.answer.as_ref().map(Result::is_ok)));
        (r.now, statuses, answered.collect::<Vec<_>>())
    };
    assert_eq!(outcome(&a), outcome(&b));
}

#[test]
#[ignore = "the campaign: the launch gate runs it"]
fn survey() {
    let seeds: Vec<u64> = match std::env::var("A2A_SEED") {
        Ok(one) => vec![one.parse().expect("a seed")],
        Err(_) => {
            let n: u64 = std::env::var("A2A_SEEDS").map_or(1024, |n| n.parse().expect("a count"));
            assert!(n > 0, "a campaign needs a positive number of seeds");
            (0..n).collect()
        }
    };
    let (failures, reached) = campaign(seeds.iter().copied());
    let inputs: BTreeSet<&str> =
        reached.iter().filter(|(_, n)| **n > 0).map(|(name, _)| *name).collect();
    boundary::A2A.report(&inputs, &[]);
    println!("{} seeds: {reached:?}", seeds.len());
    if std::env::var("A2A_SEED").is_err() {
        let missing: Vec<&&str> =
            reached.keys().filter(|k| **k != "KnownFinding" && !inputs.contains(*k)).collect();
        assert!(missing.is_empty(), "the campaign never reached {missing:?}");
    }
    assert!(failures.is_empty(), "{} failing runs:\n{}", failures.len(), failures.join("\n"));
}
