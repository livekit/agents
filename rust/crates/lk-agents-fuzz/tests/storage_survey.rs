//! The storage simulation (`lk_agents_fuzz::storage`) over many seeds: every run must keep every
//! invariant, under the arrival order and BasicPOS over the due processes.
//!
//! `cargo test` runs a small window and the replay check; the campaign is `survey`, behind
//! `--ignored` (the launch gate runs it): `STORAGE_SEEDS` sizes it, `STORAGE_SEED=<n>` reruns one
//! seed with its violations in full.
use std::collections::BTreeSet;

use lk_agents_fuzz::{boundary, sched::Sampling, storage};

const WORKERS: usize = 3;
const MESSAGES: usize = 8;

/// Runs `seeds`; the failures, one line each with the command that reruns it, and what was
/// reached.
fn campaign(seeds: impl Iterator<Item = u64>) -> (Vec<String>, [usize; 3]) {
    let mut failures = Vec::new();
    let mut reached = [0usize; 3];
    for seed in seeds {
        for sampling in [Sampling::Arrival, Sampling::BasicPos] {
            let outcome = storage::run(seed, WORKERS, MESSAGES, sampling);
            reached[0] += outcome.committed;
            reached[1] += outcome.resolved as usize;
            reached[2] += outcome.rejected;
            if !outcome.violations.is_empty() {
                failures.push(format!(
                    "STORAGE_SEED={seed} ({sampling:?}, {} steps): {:?}",
                    outcome.steps, outcome.violations
                ));
            }
        }
    }
    (failures, reached)
}

#[test]
fn a_small_window_keeps_every_storage_invariant() {
    let (failures, reached) = campaign(0..16);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(reached[0] > 0 && reached[1] > 0, "commits and lost acknowledgments were reached");
}

#[test]
fn a_seed_replays_its_run() {
    let a = storage::run(7, WORKERS, MESSAGES, Sampling::BasicPos);
    let b = storage::run(7, WORKERS, MESSAGES, Sampling::BasicPos);
    assert_eq!((a.steps, a.violations), (b.steps, b.violations));
}

#[test]
#[ignore = "the campaign: the launch gate runs it"]
fn survey() {
    let seeds: Vec<u64> = match std::env::var("STORAGE_SEED") {
        Ok(one) => vec![one.parse().expect("a seed")],
        Err(_) => {
            let n: u64 =
                std::env::var("STORAGE_SEEDS").map_or(1024, |n| n.parse().expect("a count"));
            assert!(n > 0, "a campaign needs a positive number of seeds");
            (0..n).collect()
        }
    };
    let (failures, [committed, resolved, rejected]) = campaign(seeds.iter().copied());
    let reached: BTreeSet<&str> = [
        (committed > 0).then_some("Commit"),
        (resolved > 0).then_some("LostAck"),
        (rejected > 0).then_some("RejectedSave"),
    ]
    .into_iter()
    .flatten()
    .collect();
    boundary::STORAGE.report(&reached, &[]);
    println!(
        "{} seeds: {committed} commits, {resolved} resolution lookups, {rejected} rejected saves",
        seeds.len()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
