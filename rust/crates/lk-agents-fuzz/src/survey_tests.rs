//! A campaign over many seeds, reporting what it reached and what it found.
//!
//! Ignored by default: this is the search, not part of `cargo test`.
//! `cargo test --release -p lk-agents-fuzz --test survey -- --ignored --nocapture`
//!
//! Release, because a conversation is thousands of clock advances each carrying ten frames of
//! audio through the driver, and that path is sixteen times slower unoptimised. The numbers are
//! identical either way.
//!
//! One search, `search::explore`: a person on a phone, the developer's program, and models,
//! outputs and a room that behave and fail as real ones do, at a randomness drawn per seed from a
//! pure conversation to a pure uniform sampler. Its three numbers -- variants reached, distinct
//! edges, distinct findings -- are the yardstick every change to the search is measured by, at 64
//! seeds on three windows and at 256. Everything is also broken down by decade of randomness: what
//! each decade reaches, what only it reaches, and what it costs, so the distribution is
//! measured rather than chosen.
//!
//! - `VOICE_SEEDS` (32) and `VOICE_BUDGET` (3000 inputs) size a campaign. A beat of conversation
//!   is three inputs and a reply takes seconds of beats; at 450 most runs ended before the agent
//!   finished speaking, and 1500 took findings from 37 to 101 for thirteen seconds (2026-10-22).
//!   At 1500 half the runs still ended on the budget, about forty seconds into a call whose median
//!   is five minutes; 3000 found seven findings 1500 never reached, at twice the time (2026-10-27).
//! - `VOICE_FIRST_SEED` moves the seed window, which is how the noise floor is measured.
//! - `VOICE_SEED=4372` runs that one seed and prints its setup, every failure in full and the
//!   journal variants it reached. A seed is the whole of a run: nothing carries over from the
//!   seeds before it, so every finding is reported with the one-line command that reruns it.
//! - `VOICE_FAMILY=Decided` opens one journal family variant by variant. A finding outside the
//!   known ones fails the survey.
//! - `VOICE_TRACE=1` (any run) narrates every input as it is applied.
use std::collections::{BTreeMap, BTreeSet};

#[allow(unused_imports)]
use lk_agents_fuzz::{
    boundary,
    check::KNOWN_FINDINGS,
    drive::{Outcome, PHASES, Policy, Setup},
    input::Input,
    record::Kind,
    rules::RULES,
    search,
};

#[test]
#[ignore]
fn survey() {
    let env = |key: &str| std::env::var(key).ok().and_then(|v| v.parse::<u64>().ok());
    let single = env("VOICE_SEED");
    let seeds = if single.is_some() { 1 } else { env("VOICE_SEEDS").unwrap_or(32) };
    assert!(seeds > 0, "VOICE_SEEDS=0: a survey of no seeds proves nothing");
    // `VOICE_FIRST_SEED` moves the window: the same campaign over another set of seeds is how the
    // noise floor is measured, and a change is only a change if it beats that floor.
    let first = single.or(env("VOICE_FIRST_SEED")).unwrap_or(0);
    let budget = env("VOICE_BUDGET").unwrap_or(BUDGET as u64) as usize;
    // `VOICE_POLICY=BasicPos` runs every seed under one queue policy instead of its own draw.
    let policy = std::env::var("VOICE_POLICY").ok().map(|name| {
        Policy::named(&name).unwrap_or_else(|| panic!("VOICE_POLICY: no policy {name}"))
    });
    let window = Window { first, seeds, budget, single: single.is_some(), policy };
    campaign(&window);
}

const BUDGET: usize = 3000;

/// The one line that reruns a seed alone.
fn rerun(seed: u64, budget: usize, policy: Policy) -> String {
    format!(
        "VOICE_SEED={seed} VOICE_BUDGET={budget} VOICE_POLICY={policy:?} \
         uv run --no-sync cargo test --manifest-path rust/Cargo.toml -q --release \
         -p lk-agents-fuzz --test survey -- --ignored --nocapture"
    )
}

/// One seed, in full: what it was built with and everything it found and reached.
fn detail(seed: u64, budget: usize, outcome: &Outcome) {
    println!("\nseed {seed}, schedule {}: {:?}", outcome.case.schedule, outcome.case.setup);
    println!("rerun: {}", rerun(seed, budget, outcome.case.policy));
    println!(
        "randomness {:.4}, policy {:?}, {} inputs{}",
        search::randomness(seed),
        outcome.case.policy,
        outcome.case.inputs.len(),
        if outcome.stopped_early { ", stopped early" } else { "" }
    );
    println!("ending: {:?}", outcome.ending);
    println!("failures:");
    for f in &outcome.failures {
        println!("  {} on {:?}: {}", f.contract, f.subject, f.detail);
    }
    if outcome.failures.is_empty() {
        println!("  none");
    }
    println!("contracts (zero expectations means unexercised):");
    for rule in RULES {
        let made = outcome.expectations.get(rule.id()).copied().unwrap_or(0);
        let failures: Vec<_> =
            outcome.failures.iter().filter(|failure| failure.contract == rule.id()).collect();
        let status = if made == 0 {
            "UNEXERCISED"
        } else if failures.is_empty() {
            "passed"
        } else if failures.iter().all(|failure| failure.known()) {
            "KNOWN FINDING"
        } else {
            "NEW FINDING"
        };
        println!("  {status:<13} {:>6} expectations  {}", made, rule.id());
    }
    println!("journal variants reached:");
    let reached: BTreeSet<&String> = outcome.reached.iter().collect();
    for variant in reached {
        println!("  {variant}");
    }
}

/// Decades of randomness the report breaks down by, from 1e-4 to 1e3.
const BANDS: usize = 7;

/// What the seeds of one decade of randomness reached, and what they cost.
#[derive(Default)]
struct Band {
    seeds: usize,
    failing: usize,
    time: std::time::Duration,
    reached: BTreeSet<String>,
    edges: BTreeSet<(String, String)>,
    cells: BTreeSet<(&'static str, &'static str)>,
    /// Distinct `(contract, subject kind)` pairs found in the band.
    findings: BTreeSet<(String, String)>,
    /// Seeds failing each contract.
    hits: BTreeMap<String, usize>,
}

impl Band {
    fn absorb(&mut self, outcome: &Outcome, took: std::time::Duration) {
        self.seeds += 1;
        self.time += took;
        self.reached.extend(outcome.reached.iter().cloned());
        self.edges.extend(outcome.edges.iter().cloned());
        self.cells.extend(outcome.cells.iter().copied());
        self.failing += usize::from(!outcome.ok());
        self.findings
            .extend(outcome.failures.iter().map(|f| (f.contract.clone(), f.subject.kind())));
        let failed: BTreeSet<&String> = outcome.failures.iter().map(|f| &f.contract).collect();
        for contract in failed {
            *self.hits.entry(contract.clone()).or_default() += 1;
        }
    }
}

/// One row per band: what it reached and found, and what only it reached.
fn table(title: &str, rows: &[(String, &Band)]) {
    println!("\n{title}:");
    println!("  {:<14} seeds failing    time variants  edges  cells findings  only here", "");
    for (i, (label, band)) in rows.iter().enumerate() {
        let elsewhere: BTreeSet<&String> = rows
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .flat_map(|(_, (_, b))| &b.reached)
            .collect();
        let only: Vec<&str> =
            band.reached.iter().filter(|v| !elsewhere.contains(v)).map(String::as_str).collect();
        println!(
            "  {label:<14} {:>5} {:>7} {:>6.1}s {:>8} {:>6} {:>6} {:>8}  {}",
            band.seeds,
            band.failing,
            band.time.as_secs_f64(),
            band.reached.len(),
            band.edges.len(),
            band.cells.len(),
            band.findings.len(),
            only.join(", ")
        );
    }
}

/// One campaign's seeds, budget and search settings.
struct Window {
    first: u64,
    seeds: u64,
    budget: usize,
    /// `VOICE_SEED`: one seed, reported in full.
    single: bool,
    /// `VOICE_POLICY`.
    policy: Option<Policy>,
}

/// Every seed of the window, run on all cores and returned in seed order, so the report does not
/// depend on which thread finished first. A seed is the whole of its run, so runs share nothing.
fn run_all(window: &Window) -> Vec<(u64, Outcome, std::time::Duration)> {
    let seeds: Vec<u64> = (window.first..window.first + window.seeds).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(seeds.len());
    let mut done: Vec<(u64, Outcome, std::time::Duration)> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&seed) = seeds.get(i) else { break mine };
                        // A fresh setup per seed, from the seed alone: three journal families
                        // exist only under legs the default setup never builds. Fresh every
                        // time: mutating the campaign's most novel cases instead was measured on
                        // 2026-09-23 and lost a fifth of the edges and a quarter of the findings
                        // at every scale; the generator, not a mutator, is where novelty comes
                        // from.
                        let setup = search::sample_setup(&mut fastrand::Rng::with_seed(seed));
                        let began = std::time::Instant::now();
                        let outcome = match window.policy {
                            Some(policy) => {
                                let dials = search::Dials::of(seed);
                                search::explore_with(setup, seed, window.budget, dials, policy)
                            }
                            None => search::explore(setup, seed, window.budget),
                        };
                        mine.push((seed, outcome, began.elapsed()));
                    }
                })
            })
            .collect();
        workers.into_iter().flat_map(|w| w.join().expect("a survey worker panicked")).collect()
    });
    done.sort_by_key(|(seed, ..)| *seed);
    done
}

fn campaign(window: &Window) {
    println!(
        "configuration: first_seed={}, seeds={}, budget={}, policy={}",
        window.first,
        window.seeds,
        window.budget,
        window.policy.map_or_else(|| "per-seed".into(), |p| format!("{p:?}"))
    );
    println!(
        "build: {} {}; scope: voice/room over fake ports; seeds replay only on the same code/configuration",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION")
    );
    let (seeds, budget) = (window.seeds, window.budget);
    let mut reached: BTreeSet<String> = BTreeSet::new();
    let mut edges: BTreeSet<(String, String)> = BTreeSet::new();
    let mut findings: BTreeSet<(String, String)> = BTreeSet::new();
    let mut cells: BTreeMap<(&'static str, &'static str), u32> = BTreeMap::new();
    // Failures per contract, per decade of randomness.
    let mut found: BTreeMap<String, [usize; BANDS]> = BTreeMap::new();
    // One example per contract, so a finding can be read without saving the corpus.
    let mut example: BTreeMap<String, (u64, String)> = BTreeMap::new();
    // One example per finding outside the known ones, by (contract, subject kind).
    let mut new: BTreeMap<(String, String), (u64, String)> = BTreeMap::new();
    // Every failing seed, since one contract can fail for several causes.
    let mut failing: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    let mut exercised: BTreeMap<&'static str, u32> = BTreeMap::new();
    let mut bands: [Band; BANDS] = Default::default();
    let mut policies: BTreeMap<String, Band> = BTreeMap::new();
    // Runs in which both observers closed a speech: the precondition of a decided turn.
    let mut both_closed = 0;
    let mut heard = 0;
    // What each input was declared to open, and what it was seen to open.
    let mut declared: BTreeMap<&'static str, BTreeSet<Kind>> = BTreeMap::new();
    let mut observed: BTreeMap<&'static str, BTreeSet<Kind>> = BTreeMap::new();
    let started = std::time::Instant::now();
    // The busiest virtual instant, in steps of the run's loop, and where: NO_PROGRESS_STEPS's
    // derivation.
    let mut busiest = (0u32, 0u64);
    // How the runs ended (`drive::Ending`), by kind, with one seed each.
    let mut endings: BTreeMap<String, (usize, u64)> = BTreeMap::new();

    for (seed, outcome, took) in run_all(window) {
        busiest = busiest.max((outcome.busiest_instant, seed));
        let kind = format!("{:?}", outcome.ending);
        let kind = kind.split([' ', '{']).next().unwrap_or_default().to_owned();
        endings.entry(kind).or_insert((0, seed)).0 += 1;
        if window.single {
            println!("ending: {:?}", outcome.ending);
        }
        let band_of = ((search::randomness(seed).log10() + 4.0) as usize).min(BANDS - 1);
        bands[band_of].absorb(&outcome, took);
        policies.entry(format!("{:?}", outcome.case.policy)).or_default().absorb(&outcome, took);
        reached.extend(outcome.reached.iter().cloned());
        if window.single {
            detail(seed, budget, &outcome);
        }
        edges.extend(outcome.edges.iter().cloned());
        for cell in &outcome.cells {
            *cells.entry(*cell).or_default() += 1;
        }
        for (id, made) in &outcome.expectations {
            *exercised.entry(id).or_default() += made;
        }
        for input in &outcome.case.inputs {
            declared.entry(input.name()).or_default().extend(input.outcomes().iter().copied());
        }
        for (name, kinds) in &outcome.opened {
            observed.entry(name).or_default().extend(kinds.iter().copied());
        }
        let sent = |name: &str| outcome.case.inputs.iter().any(|i| i.name() == name);
        both_closed += usize::from(sent("VadEnd") && sent("SttFinal"));
        heard += usize::from(outcome.reached.iter().any(|v| v == "Playout::PlaybackStarted"));
        if !outcome.ok() {
            let mut here: BTreeSet<&str> = BTreeSet::new();
            for failure in &outcome.failures {
                let key = (failure.contract.clone(), failure.subject.kind());
                if !failure.known() {
                    new.entry(key.clone()).or_insert_with(|| (seed, failure.detail.clone()));
                }
                findings.insert(key);
                if here.insert(&failure.contract) {
                    found.entry(failure.contract.clone()).or_default()[band_of] += 1;
                    failing.entry(failure.contract.clone()).or_default().push(seed);
                }
                example
                    .entry(failure.contract.clone())
                    .or_insert_with(|| (seed, failure.detail.clone()));
            }
        }
    }

    let names = lk_agents_voice::trace::step_names();
    println!();
    println!("{seeds} seeds from {} x {budget} inputs in {:.1?}", window.first, started.elapsed());
    println!(
        "reached {} of {} journal variants, {} distinct edges, {} distinct findings",
        reached.len(),
        names.len(),
        edges.len(),
        findings.len()
    );
    let decades: Vec<(String, &Band)> = bands
        .iter()
        .enumerate()
        .map(|(i, b)| (format!("1e{:+}..1e{:+}", i as i32 - 4, i as i32 - 3), b))
        .collect();
    table("by randomness (uniform moves per conversation move)", &decades);
    let policies: Vec<(String, &Band)> = policies.iter().map(|(p, b)| (p.clone(), b)).collect();
    table("by queue policy", &policies);
    // gosim's yardstick: how often each policy hits each finding, per seed it ran.
    println!("\nhit rate per contract, by queue policy (failing seeds / seeds):");
    let contracts: BTreeSet<&String> = policies.iter().flat_map(|(_, b)| b.hits.keys()).collect();
    for contract in contracts {
        let rates: Vec<String> = policies
            .iter()
            .map(|(p, b)| {
                let hits = b.hits.get(contract).copied().unwrap_or(0);
                format!("{p} {:>5.1}%", 100.0 * hits as f64 / b.seeds.max(1) as f64)
            })
            .collect();
        println!("  {}  {contract}", rates.join("  "));
    }
    println!();
    println!("{both_closed} of {seeds} runs with both a VadEnd and an SttFinal");
    // A call where the agent was never heard reaches nothing that happens while it speaks.
    println!("{heard} of {seeds} runs where the person heard the agent");
    println!("busiest instant: {} steps of the run's loop (seed {})", busiest.0, busiest.1);
    let endings: Vec<String> =
        endings.iter().map(|(k, (n, seed))| format!("{k} {n} (seed {seed})")).collect();
    println!("endings: {}\n", endings.join(", "));

    // Per family, because the two ways to miss a variant have different fixes: a family at zero
    // is a leg no setup built, and a family part-reached is a search that has not got there yet.
    println!("by family:");
    let mut families: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for name in &names {
        let family = name.split("::").next().unwrap_or(name);
        let seen = &mut families.entry(family).or_default();
        seen.0 += usize::from(reached.contains(name));
        seen.1 += 1;
    }
    for (family, (seen, of)) in &families {
        println!("  {family:<12} {seen:>3} / {of}");
    }
    println!("missing:");
    for name in names.iter().filter(|n| !reached.contains(*n)) {
        println!("  {name}");
    }
    // `VOICE_FAMILY=Decided` opens one family up: which of its variants the search reached.
    if let Some(family) = std::env::var("VOICE_FAMILY").ok().filter(|f| !f.is_empty()) {
        println!("\n{family}, variant by variant:");
        for name in names.iter().filter(|n| n.split("::").next() == Some(family.as_str())) {
            let mark = if reached.contains(name) { "reached" } else { "missing" };
            println!("  {mark}  {name}");
        }
    }
    println!();
    println!("failing seeds per contract, by decade of randomness:");
    if found.is_empty() {
        println!("  none");
    }
    for (contract, by_band) in &found {
        let counts: Vec<String> = by_band.iter().map(|n| format!("{n:>6}")).collect();
        println!("  {}  {contract}", counts.join(""));
        let (seed, _) = example[contract];
        println!(
            "       seed {seed}: {}",
            rerun(seed, budget, window.policy.unwrap_or_else(|| Policy::of(seed)))
        );
        let all: Vec<String> = failing[contract].iter().map(u64::to_string).collect();
        println!("       all: {}", all.join(" "));
    }
    println!("\ncontracts never exercised:");
    let mut idle = 0;
    for rule in RULES {
        if exercised.get(rule.id()).copied().unwrap_or(0) == 0 {
            println!("       {}", rule.id());
            idle += 1;
        }
    }
    if idle == 0 {
        println!("       none");
    }

    // The gate: a (contract, subject kind) outside the known findings is new, whatever window
    // found it.
    println!("\nfindings outside the known ones (check::KNOWN_FINDINGS):");
    for ((contract, kind), (seed, what)) in &new {
        println!("  NEW {contract} on a {kind}: seed {seed}: {what}");
        println!(
            "    rerun: {}",
            rerun(*seed, budget, window.policy.unwrap_or_else(|| Policy::of(*seed)))
        );
    }
    if new.is_empty() {
        println!("  none");
    }
    // A known finding this window did not reach: reported, or, over the full window
    // (`VOICE_FULL_WINDOW=1`, the nightly gate) once its fixing regression exists, a failure
    // until it comes off the list.
    let full = std::env::var_os("VOICE_FULL_WINDOW").is_some();
    let mut stale = Vec::new();
    if !window.single {
        for known in KNOWN_FINDINGS.iter() {
            let key = (known.contract.to_owned(), known.subject.to_owned());
            if findings.contains(&key) {
                continue;
            }
            match known.fixed_by {
                Some(test) if full => {
                    println!("  FIXED {} on a {}: {test} fixes it; remove it", key.0, key.1);
                    stale.push(key);
                }
                Some(test) => println!("  not reached: {} on a {} (fixed by {test})", key.0, key.1),
                None => println!("  not reached: {} on a {} (no fix yet)", key.0, key.1),
            }
        }
    }

    // Only the sound direction: a row that opens something directly without declaring it.
    // The reverse cannot be checked, because a speech arrives from the driver later and no
    // timing rule can say which earlier input caused it.
    println!("\ninput rows that open something they do not declare:");
    let mut wrong = 0;
    for (name, claim) in &declared {
        let seen = observed.get(name).cloned().unwrap_or_default();
        let undeclared: Vec<_> = seen.difference(claim).collect();
        if !undeclared.is_empty() {
            println!("  {name}: opens {undeclared:?} undeclared");
            wrong += 1;
        }
    }
    if wrong == 0 {
        println!("  none");
    }
    // `VOICE_CELLS=1`: seeds in which each input landed in each phase.
    if std::env::var_os("VOICE_CELLS").is_some() {
        println!("\n=== input x phase, seeds ===");
        println!("input\t{}", PHASES.join("\t"));
        let inputs: BTreeSet<&str> = cells.keys().map(|(input, _)| *input).collect();
        for input in inputs {
            let row: Vec<String> = PHASES
                .iter()
                .map(|p| cells.get(&(input, *p)).copied().unwrap_or(0).to_string())
                .collect();
            println!("{input}\t{}", row.join("\t"));
        }
    }
    let delivered: BTreeSet<&str> = declared.keys().copied().collect();
    let opened: BTreeSet<String> =
        observed.values().flatten().map(|kind| format!("{kind:?}")).collect();
    boundary::VOICE.report(&delivered, &opened.into_iter().collect::<Vec<_>>());
    assert!(new.is_empty(), "new findings: {:?}", new.keys().collect::<Vec<_>>());
    assert!(stale.is_empty(), "fixed known findings still listed: {stale:?}");
}

#[test]
fn rerun_preserves_each_policy_and_the_input_budget() {
    for policy in Policy::ALL {
        let command = rerun(4372, 17, policy);
        assert!(command.contains("VOICE_SEED=4372"), "{command}");
        assert!(command.contains("VOICE_BUDGET=17"), "{command}");
        assert!(command.contains(&format!("VOICE_POLICY={policy:?}")), "{command}");
        assert!(command.contains("--manifest-path rust/Cargo.toml"), "{command}");
    }
}
