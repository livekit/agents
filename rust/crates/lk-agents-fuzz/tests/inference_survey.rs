//! The inference simulations (`lk_agents_fuzz::inference`) over many seeds: the LLM request and
//! the avatar session machines and the VAD stream must keep every rule under the arrival order
//! and BasicPOS.
//!
//! The campaign is `survey`, behind `--ignored` (the launch gate runs it): `INFERENCE_SEEDS` sizes
//! it. A failure prints the seed; `LLM_SEED`/`AVATAR_SEED` replays print a run in full (the
//! modules' `replay` tests).
use std::collections::{BTreeMap, BTreeSet};

use lk_agents_fuzz::check::KNOWN_FINDINGS;
use lk_agents_fuzz::inference::{avatar, llm, vad};
use lk_agents_fuzz::{boundary, sched::Sampling};

#[test]
#[ignore = "the campaign: the launch gate runs it"]
fn survey() {
    let n: u64 = std::env::var("INFERENCE_SEEDS").map_or(4096, |n| n.parse().expect("a count"));
    assert!(n > 0, "a campaign needs a positive number of seeds");
    // (subject, rule) -> (runs failing it, the first seed)
    let mut failed: BTreeMap<(&str, String), (u64, u64)> = BTreeMap::new();
    let mut reached: BTreeSet<&'static str> = BTreeSet::new();
    let mut note = |subject: &'static str, seed: u64, violations: &[String]| {
        for v in violations {
            let rule = v.split(':').next().unwrap_or_default().to_owned();
            let entry = failed.entry((subject, rule)).or_insert((0, seed));
            entry.0 += 1;
        }
    };
    for seed in 0..n {
        for sampling in [Sampling::Arrival, Sampling::BasicPos] {
            let outcome = llm::run(seed, sampling);
            note("LlmRequest", seed, &outcome.violations);
            reached.extend(outcome.reached);
            let outcome = avatar::run(seed, sampling);
            note(avatar::subject(&outcome.trace), seed, &outcome.violations);
            reached.extend(outcome.reached);
            let outcome = vad::run(seed, sampling);
            note("VadStream", seed, &outcome.violations);
            reached.extend(outcome.reached);
        }
    }
    boundary::INFERENCE.report(&reached, &[]);
    let mut unknown = Vec::new();
    for ((subject, rule), (runs, first)) in &failed {
        let known = KNOWN_FINDINGS.iter().any(|k| k.contract == rule && k.subject == *subject);
        println!(
            "{} {subject} {rule}: {runs} runs, first seed {first}",
            if known { "known" } else { "NEW" }
        );
        if !known {
            unknown.push(format!("{subject} {rule} (seed {first})"));
        }
    }
    println!("{n} seeds x 2 orders, reached {reached:?}");
    assert!(unknown.is_empty(), "findings outside the known ones: {unknown:?}");
}
