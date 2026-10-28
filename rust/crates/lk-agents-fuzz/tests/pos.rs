//! Partial-order sampling, measured. Ignored by default: this is a measurement, not a check (the
//! scheduler's own tests are the small programs in `src/drive/queue.rs`).
//!
//! `cargo test --release -p lk-agents-fuzz --test pos -- --ignored --nocapture bench`
//!
//! `bench`: every queue policy on the same seeds, the way Yuan, Yang and Gu (CAV 2018) compare
//! samplers. Over `VOICE_SEEDS` (4096) campaign seeds per policy: what each reached, its distinct
//! findings (contract and subject kind), its cost, and per contract the seeds that failed it and
//! the seeds to its first failure in each of `VOICE_WINDOWS` (8) windows. Then the paper's method
//! proper, per contract that failed: its first failing case is the program, run under
//! `VOICE_SCHEDULES` (512) schedules per policy, the hit ratio and the schedules to the first hit
//! in each window; and between the two, that case's setup under `VOICE_SETUPS` (1024) fresh
//! conversations. `rust/crates/lk-agents-fuzz/bench/known_bugs.sh` runs it over the known bugs,
//! each reintroduced from its fix commit, and tabulates it.
use std::collections::BTreeSet;

use fastrand::Rng;
use lk_agents_fuzz::{
    drive::{Case, Policy, run},
    search,
};

fn env(key: &str) -> Option<u64> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}

/// `work` over `items` on every core, results in item order.
fn parallel<T: Sync, R: Send>(items: &[T], work: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(items.len());
    let mut done: Vec<(usize, R)> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(item) = items.get(i) else { break mine };
                        mine.push((i, work(item)));
                    }
                })
            })
            .collect();
        workers.into_iter().flat_map(|w| w.join().expect("a worker panicked")).collect()
    });
    done.sort_by_key(|(i, _)| *i);
    done.into_iter().map(|(_, r)| r).collect()
}

/// What one campaign seed left, for the benchmark.
struct Seeded {
    failed: BTreeSet<String>,
    findings: BTreeSet<(String, String)>,
    reached: Vec<String>,
    edges: BTreeSet<(String, String)>,
    /// The case, when it failed something: a program for the second half.
    case: Option<Case>,
}

/// Where in each window of `per` tries the first of `hits` (sorted indices) falls; `None` when
/// the window has none.
fn first_hits(hits: &[usize], tries: usize, windows: usize) -> Vec<Option<usize>> {
    let per = tries / windows;
    (0..windows).map(|w| hits.iter().find(|&&i| i / per == w).map(|i| i - w * per + 1)).collect()
}

/// The median of the windows' first hits, a window without one counted as past its end.
fn median(firsts: &[Option<usize>], per: usize) -> String {
    let mut all: Vec<usize> = firsts.iter().map(|f| f.unwrap_or(usize::MAX)).collect();
    all.sort();
    match all[all.len() / 2] {
        usize::MAX => format!(">{per}"),
        n => n.to_string(),
    }
}

#[test]
#[ignore]
fn bench() {
    let seeds = env("VOICE_SEEDS").unwrap_or(4096) as usize;
    let windows = env("VOICE_WINDOWS").unwrap_or(8) as usize;
    let schedules = env("VOICE_SCHEDULES").unwrap_or(512) as usize;
    let setups = env("VOICE_SETUPS").unwrap_or(1024) as usize;
    let budget = env("VOICE_BUDGET").unwrap_or(1500) as usize;
    let first = env("VOICE_FIRST_SEED").unwrap_or(0);
    assert!(seeds > 0 && schedules > 0 && setups > 0, "a benchmark of no seeds measures nothing");
    assert!(seeds >= windows && schedules >= windows && setups >= windows, "fewer tries than windows");
    let range: Vec<u64> = (first..first + seeds as u64).collect();

    let mut campaigns: Vec<(Policy, Vec<Seeded>, f64)> = Vec::new();
    for policy in Policy::ALL {
        let began = std::time::Instant::now();
        let seeded = parallel(&range, |&seed| {
            let setup = search::sample_setup(&mut Rng::with_seed(seed));
            let dials = search::Dials::of(seed);
            let outcome = search::explore_with(setup, seed, budget, dials, policy);
            Seeded {
                failed: outcome.failures.iter().map(|f| f.contract.clone()).collect(),
                findings: outcome
                    .failures
                    .iter()
                    .map(|f| (f.contract.clone(), f.subject.kind()))
                    .collect(),
                reached: outcome.reached.clone(),
                edges: outcome.edges.clone(),
                case: (!outcome.failures.is_empty()).then(|| outcome.case.clone()),
            }
        });
        campaigns.push((policy, seeded, began.elapsed().as_secs_f64()));
    }

    println!("\ncampaign, {seeds} seeds from {first} per policy:");
    println!(
        "  {:<9} {:>9} {:>8} {:>6} {:>8}",
        "policy", "seeds/s", "variants", "edges", "findings"
    );
    for (policy, seeded, secs) in &campaigns {
        let reached: BTreeSet<&String> = seeded.iter().flat_map(|s| &s.reached).collect();
        let edges: BTreeSet<&(String, String)> = seeded.iter().flat_map(|s| &s.edges).collect();
        let findings: BTreeSet<&(String, String)> =
            seeded.iter().flat_map(|s| &s.findings).collect();
        println!(
            "  {:<9} {:>9.0} {:>8} {:>6} {:>8}",
            format!("{policy:?}"),
            seeds as f64 / secs,
            reached.len(),
            edges.len(),
            findings.len()
        );
        println!(
            "TSV\tcampaign\t{policy:?}\t{:.1}\t{}\t{}\t{}",
            seeds as f64 / secs,
            reached.len(),
            edges.len(),
            findings.len()
        );
        for (contract, kind) in findings {
            println!("TSV\tfinding\t{policy:?}\t{contract}\t{kind}");
        }
    }

    let contracts: BTreeSet<&String> =
        campaigns.iter().flat_map(|(_, s, _)| s.iter().flat_map(|s| &s.failed)).collect();
    let per = seeds / windows;
    println!("\nper contract: failing seeds, and the median over {windows} windows of {per} seeds");
    println!("of the seeds to the first failure:");
    for contract in &contracts {
        let row: Vec<String> = campaigns
            .iter()
            .map(|(policy, seeded, _)| {
                let hits: Vec<usize> =
                    (0..seeded.len()).filter(|&i| seeded[i].failed.contains(*contract)).collect();
                let firsts = first_hits(&hits, seeds, windows);
                let found = firsts.iter().filter(|f| f.is_some()).count();
                println!(
                    "TSV\tseeds\t{policy:?}\t{contract}\t{}\t{seeds}\t{}\t{found}",
                    hits.len(),
                    median(&firsts, per)
                );
                format!("{policy:?} {:>4} {:>5}", hits.len(), median(&firsts, per))
            })
            .collect();
        println!("  {contract}: {}", row.join("  "));
    }

    // The paper's method: one program, the schedule the only thing that changes.
    let per = schedules / windows;
    println!(
        "\nper contract, its first failing case as the program: hits in {schedules} schedules, \
         and the median over {windows} windows of {per} of the schedules to the first hit:"
    );
    let programs: Vec<(&String, Case)> = contracts
        .iter()
        .filter_map(|contract| {
            let found = campaigns
                .iter()
                .flat_map(|(_, seeded, _)| seeded.iter())
                .filter(|s| s.failed.contains(*contract))
                .filter_map(|s| s.case.clone())
                .min_by_key(|case| case.schedule)?;
            Some((*contract, found))
        })
        .collect();
    for (contract, program) in &programs {
        let row: Vec<String> = Policy::ALL
            .iter()
            .map(|&policy| {
                let tries: Vec<u64> = (1..=schedules as u64).collect();
                let hit = parallel(&tries, |&s| {
                    let schedule = s.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ program.schedule;
                    let case = Case { schedule, policy, ..program.clone() };
                    run(case).failures.iter().any(|f| &f.contract == *contract)
                });
                let hits: Vec<usize> = (0..hit.len()).filter(|&i| hit[i]).collect();
                let firsts = first_hits(&hits, schedules, windows);
                println!(
                    "TSV\tprogram\t{policy:?}\t{contract}\t{}\t{schedules}\t{}\t{}",
                    hits.len(),
                    median(&firsts, per),
                    program.schedule
                );
                format!("{policy:?} {:>4} {:>5}", hits.len(), median(&firsts, per))
            })
            .collect();
        println!("  {contract} (seed {}): {}", program.schedule, row.join("  "));
    }
    // Between the two: the program's setup, every seed a fresh conversation under each policy,
    // so the inputs are not the ones one policy's run recorded.
    let per = setups / windows;
    println!(
        "\nper contract, its first failing case's setup: failing seeds of {setups}, and the median \
         over {windows} windows of {per} of the seeds to the first failure:"
    );
    for (contract, program) in &programs {
        let row: Vec<String> = Policy::ALL
            .iter()
            .map(|&policy| {
                let tries: Vec<u64> = (1..=setups as u64).collect();
                let hit = parallel(&tries, |&s| {
                    let seed = s.wrapping_mul(0x2545_f491_4f6c_dd1d) ^ program.schedule;
                    let dials = search::Dials::of(seed);
                    let setup = program.setup.clone();
                    let outcome = search::explore_with(setup, seed, budget, dials, policy);
                    outcome.failures.iter().any(|f| &f.contract == *contract)
                });
                let hits: Vec<usize> = (0..hit.len()).filter(|&i| hit[i]).collect();
                let firsts = first_hits(&hits, setups, windows);
                println!(
                    "TSV\tsetup\t{policy:?}\t{contract}\t{}\t{setups}\t{}",
                    hits.len(),
                    median(&firsts, per)
                );
                format!("{policy:?} {:>4} {:>5}", hits.len(), median(&firsts, per))
            })
            .collect();
        println!("  {contract}: {}", row.join("  "));
    }
}
