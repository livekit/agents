//! The A2A simulation (`lk_agents_fuzz::a2a`) over many seeds: every run must keep every rule,
//! under the arrival order and BasicPOS over the due processes.
//!
//! A finding of a (contract, subject kind) in `known_findings.tsv` is counted, not failed.
//!
//! `cargo test` runs a small window and the replay check; the campaign is `survey`, behind
//! `--ignored` (the launch gate runs it): `A2A_SEEDS` sizes it, `A2A_SEED=<n>` reruns one seed
//! with its violations in full.
use std::collections::BTreeSet;

use lk_agents_fuzz::{a2a, boundary, sched::Sampling};

const MESSAGES: usize = 8;

/// What a campaign reached, by name, and how often.
fn reach(record: &a2a::Record, into: &mut std::collections::BTreeMap<&'static str, u64>) {
    let s = &record.seen;
    let cancels_live = s.cancels.iter().filter(|c| !c.before.is_terminal()).count() as u64;
    for (name, n) in [
        ("Answer", s.responses.len() as u64),
        ("Deduplicated", s.deduplicated),
        ("Continuation", s.continuations),
        ("Interrupted", s.interrupted),
        ("ClientCanceledTurn", s.client_canceled),
        ("CancelLive", cancels_live),
        ("CancelTerminal", s.cancels.len() as u64 - cancels_live),
        ("LostAck", s.lost_acks),
        ("UnresolvedAnswer", s.unresolved),
        ("RejectedSave", s.rejected),
        ("LeaseTimeout", s.lease_timeouts),
        ("WorkerLost", s.lost.len() as u64),
        ("Listing", s.listings.len() as u64),
        (
            "ForeignPageToken",
            s.listings.iter().filter(|l| l.foreign_token.is_some()).count() as u64,
        ),
    ] {
        *into.entry(name).or_default() += n;
    }
}

/// Runs `seeds`; the failures, one line each with the command that reruns it, and the reach.
fn campaign(
    seeds: impl Iterator<Item = u64>,
) -> (Vec<String>, std::collections::BTreeMap<&'static str, u64>) {
    let mut failures = Vec::new();
    let mut reached = std::collections::BTreeMap::new();
    for seed in seeds {
        for sampling in [Sampling::Arrival, Sampling::BasicPos] {
            let record = a2a::run(seed, MESSAGES, sampling);
            reach(&record, &mut reached);
            let (known, new): (Vec<_>, Vec<_>) =
                a2a::rules::check(&record).into_iter().partition(|f| f.known());
            *reached.entry("KnownFinding").or_default() += known.len() as u64;
            if !new.is_empty() {
                let new: Vec<String> = new.iter().map(ToString::to_string).collect();
                failures.push(format!(
                    "A2A_SEED={seed} ({sampling:?}, {} steps): {new:?}",
                    record.steps
                ));
            }
        }
    }
    (failures, reached)
}

#[test]
fn a_small_window_keeps_every_a2a_rule() {
    let (failures, reached) = campaign(0..16);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(reached["Answer"] > 0 && reached["Deduplicated"] > 0, "{reached:?}");
}

#[test]
fn a_seed_replays_its_run() {
    let a = a2a::run(7, MESSAGES, Sampling::BasicPos);
    let b = a2a::run(7, MESSAGES, Sampling::BasicPos);
    assert_eq!((a.steps, a.tasks, a.events), (b.steps, b.tasks, b.events));
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
