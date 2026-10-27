//! Partial-order sampling, measured. Ignored by default: these are measurements, not checks.
//!
//! `cargo test --release -p lk-agents-fuzz --test pos -- --ignored --nocapture commutation`
//!
//! - `commutation`: the conflict relation (`Event::accesses`) against the session itself. A
//!   campaign seed's case is replayed twice with a pair of pending events delivered back to back,
//!   once in each order: a pair declared independent that leaves a different session (the
//!   journal its two steps wrote, or the whole run's journal and findings) is a false
//!   independence. Pairs declared racing are swapped the same way as the control. `VOICE_SEEDS`
//!   (256) cases, `VOICE_SWAPS` (4) pairs each.
//! - `bench`: every queue policy on the same seeds, the way Yuan, Yang and Gu (CAV 2018) compare
//!   samplers. Over `VOICE_SEEDS` (4096) campaign seeds per policy: what each reached, its distinct
//!   findings (contract and subject kind), its cost, and per contract the seeds that failed it and
//!   the seeds to its first failure in each of `VOICE_WINDOWS` (8) windows. Then the paper's
//!   method proper, per contract that failed: its first failing case is the program, run under
//!   `VOICE_SCHEDULES` (512) schedules per policy, the hit ratio and the schedules to the first
//!   hit in each window; and between the two, that case's setup under `VOICE_SETUPS` (1024)
//!   fresh conversations. `rust/crates/lk-agents-fuzz/bench/known_bugs.sh` runs it over the known
//!   bugs, each reintroduced from its fix commit, and tabulates it.
use std::collections::{BTreeMap, BTreeSet};

use fastrand::Rng;
use lk_agents_fuzz::{
    drive::{
        Case, Policy,
        queue::{self, Swap},
        run,
    },
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

/// Wall-clock stamps cut to the millisecond: the session clock moves a microsecond per read, so
/// whichever of two steps in one instant goes second stamps a later microsecond. That is the
/// clock's tie-break, not state the two share.
fn mask(step: &str) -> String {
    let mut out = String::with_capacity(step.len());
    let mut rest = step;
    while let Some(at) = rest.find(|c: char| c.is_ascii_digit()) {
        let (head, tail) = rest.split_at(at);
        out.push_str(head);
        let len = tail.find(|c: char| !c.is_ascii_digit() && c != '.').unwrap_or(tail.len());
        let (number, tail) = tail.split_at(len);
        if head.ends_with("tv_nsec: ")
            && let Ok(nanos) = number.parse::<u64>()
        {
            out.push_str(&(nanos / 1_000_000).to_string());
        } else if head.ends_with("timestamp: ")
            && let Some(dot) = number.find('.')
        {
            out.push_str(&number[..(dot + 4).min(number.len())]);
        } else {
            out.push_str(number);
        }
        rest = tail;
    }
    out.push_str(rest);
    out
}

fn masked(steps: &[String]) -> Vec<String> {
    let mut steps: Vec<String> = steps.iter().map(|s| mask(s)).collect();
    steps.sort();
    steps
}

/// One swapped pair, both orders.
struct Swapped {
    racing: bool,
    kinds: (String, String),
    local_differs: bool,
    run_differs: bool,
    /// Unmasked: the clock's microsecond counted.
    raw_differs: bool,
    heads: usize,
    independent: usize,
}

fn swap(case: &Case, at: u64, racing: bool) -> Option<Swapped> {
    let mut sides = [false, true].map(|flip| {
        queue::arm(Some(Swap { at, flip, racing }));
        run(case.clone());
        queue::take()
    });
    let [a, b] = &mut sides;
    if a.pair.len() != 2 || b.pair.len() != 2 {
        return None;
    }
    a.pair_steps.sort();
    b.pair_steps.sort();
    if std::env::var_os("VOICE_SHOW").is_some()
        && !racing
        && masked(&a.pair_steps) != masked(&b.pair_steps)
    {
        let only = |x: &Vec<String>, y: &Vec<String>| -> Vec<String> {
            x.iter().filter(|s| !y.contains(s)).cloned().collect()
        };
        println!(
            "\n{:?} then {:?}: only one way {:#?}\nonly the other {:#?}",
            a.pair,
            b.pair,
            only(&a.pair_steps, &b.pair_steps),
            only(&b.pair_steps, &a.pair_steps)
        );
    }
    let mut kinds = (a.pair[0].clone(), a.pair[1].clone());
    if kinds.0 > kinds.1 {
        kinds = (kinds.1, kinds.0);
    }
    Some(Swapped {
        racing,
        kinds,
        local_differs: masked(&a.pair_steps) != masked(&b.pair_steps),
        raw_differs: a.pair_steps != b.pair_steps,
        run_differs: masked(&a.journal) != masked(&b.journal),
        heads: a.heads,
        independent: a.independent,
    })
}

#[test]
#[ignore]
fn commutation() {
    let seeds: Vec<u64> = (0..env("VOICE_SEEDS").unwrap_or(256)).collect();
    let swaps = env("VOICE_SWAPS").unwrap_or(4);
    let budget = env("VOICE_BUDGET").unwrap_or(600) as usize;
    let results: Vec<Vec<Swapped>> = parallel(&seeds, |&seed| {
        let setup = search::sample_setup(&mut Rng::with_seed(seed));
        let case = search::explore(setup, seed, budget).case;
        queue::arm(None);
        run(case.clone());
        let delivered = queue::take().delivered;
        let mut rng = Rng::with_seed(seed ^ 0x7377_6170);
        let mut found = Vec::new();
        for racing in [false, true] {
            let mut tries = 0;
            let mut got = 0;
            while got < swaps && tries < swaps * 8 && delivered > 1 {
                tries += 1;
                if let Some(swapped) = swap(&case, rng.u64(..delivered - 1), racing) {
                    found.push(swapped);
                    got += 1;
                }
            }
        }
        found
    });
    let all: Vec<&Swapped> = results.iter().flatten().collect();
    for racing in [false, true] {
        let pairs: Vec<&&Swapped> = all.iter().filter(|s| s.racing == racing).collect();
        let n = pairs.len().max(1) as f64;
        let local = pairs.iter().filter(|s| s.local_differs).count();
        let whole = pairs.iter().filter(|s| s.run_differs).count();
        let raw = pairs.iter().filter(|s| s.raw_differs).count();
        println!(
            "\n{} pairs: {}; their own steps differ when swapped: {local} ({:.1}%; {raw}, \
             {:.1}%, counting the clock's microsecond); the whole run differs: {whole} ({:.1}%)",
            if racing { "racing (control)" } else { "independent" },
            pairs.len(),
            100.0 * local as f64 / n,
            100.0 * raw as f64 / n,
            100.0 * whole as f64 / n,
        );
        let mut by_kind: BTreeMap<&(String, String), (usize, usize, usize)> = BTreeMap::new();
        for s in &pairs {
            let row = by_kind.entry(&s.kinds).or_default();
            row.0 += 1;
            row.1 += usize::from(s.local_differs);
            row.2 += usize::from(s.run_differs);
        }
        let mut rows: Vec<_> = by_kind.into_iter().collect();
        rows.sort_by_key(|(_, (n, local, whole))| std::cmp::Reverse((*local, *whole, *n)));
        println!("  {:>5} {:>6} {:>6}  pair", "pairs", "local", "run");
        for ((a, b), (n, local, whole)) in rows.iter().take(if racing { 12 } else { 40 }) {
            println!("  {n:>5} {local:>6} {whole:>6}  {a} | {b}");
        }
    }
    let heads: usize = all.iter().map(|s| s.heads).sum();
    let independent: usize = all.iter().map(|s| s.independent).sum();
    println!(
        "\nat the sampled deliveries: {heads} other pending heads, {independent} ({:.1}%) \
         declared independent of the policy's pick",
        100.0 * independent as f64 / heads.max(1) as f64
    );
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
