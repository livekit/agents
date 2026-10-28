//! The worker's process pool simulation (`lk_agents_fuzz::worker`) over many seeds: every run must
//! keep every rule under the arrival order and BasicPOS.
//!
//! The campaign is `survey`, behind `--ignored` (the launch gate runs it): `WORKER_SEEDS` sizes
//! it; `WORKER_SEED=<n>` with the module's `replay` test prints a run in full.
use std::collections::{BTreeMap, BTreeSet};

use lk_agents_fuzz::{boundary, sched::Sampling, worker};

#[test]
#[ignore = "the campaign: the launch gate runs it"]
fn survey() {
    let n: u64 = std::env::var("WORKER_SEEDS").map_or(4096, |n| n.parse().expect("a count"));
    assert!(n > 0, "a campaign needs a positive number of seeds");
    let mut failed: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    let mut reached: BTreeSet<&'static str> = BTreeSet::new();
    for seed in 0..n {
        for sampling in [Sampling::Arrival, Sampling::BasicPos] {
            let outcome = worker::run(seed, sampling);
            for v in &outcome.violations {
                let rule = v.split(':').next().unwrap_or_default().to_owned();
                failed.entry(rule).or_insert((0, seed)).0 += 1;
            }
            reached.extend(outcome.reached);
        }
    }
    boundary::WORKER.report(&reached, &[]);
    for (rule, (runs, first)) in &failed {
        println!("NEW {rule}: {runs} runs, first seed {first}");
    }
    println!("{n} seeds x 2 orders, reached {reached:?}");
    assert!(failed.is_empty(), "worker findings: {failed:?}");
}
