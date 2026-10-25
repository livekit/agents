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
//! - `VOICE_SEEDS` (32) and `VOICE_BUDGET` (1500 inputs) size a campaign. A beat of conversation
//!   is three inputs and a reply takes seconds of beats; at 450 most runs ended before the agent
//!   finished speaking, and 1500 took findings from 37 to 101 for thirteen seconds (2026-10-22).
//! - `VOICE_FIRST_SEED` moves the seed window, which is how the noise floor is measured.
//! - `VOICE_UNIFORM_SETUPS=1` draws every setup uniformly instead of from the arms that learn from
//!   seed to seed; the baseline a search change is measured against.
//! - `VOICE_FAMILY=Decided` opens one journal family variant by variant; `VOICE_DETAIL=1` adds the
//!   first seed and detail behind each finding; `VOICE_SAVE=1` writes every failing seed's case
//!   into `replay/`, where the replay test picks it up.
//! - `VOICE_TRACE=1` (any run) narrates every input as it is applied.
use std::collections::{BTreeMap, BTreeSet};

#[allow(unused_imports)]
use lk_agents_fuzz::{
    check::KNOWN_FINDINGS,
    drive::{Outcome, PHASES, Setup},
    input::Input,
    record::Kind,
    replay,
    rules::RULES,
    search,
};

#[test]
#[ignore]
fn survey() {
    let env = |key: &str| std::env::var(key).ok().and_then(|v| v.parse::<u64>().ok());
    let seeds = env("VOICE_SEEDS").unwrap_or(32);
    // `VOICE_FIRST_SEED` moves the window: the same campaign over another set of seeds is how the
    // noise floor is measured, and a change is only a change if it beats that floor.
    let first = env("VOICE_FIRST_SEED").unwrap_or(0);
    let budget = env("VOICE_BUDGET").unwrap_or(1500) as usize;
    // The setup arms learn from seed to seed unless `VOICE_UNIFORM_SETUPS=1`, under which every
    // seed's setup is a uniform draw: the baseline every search change is measured against.
    let learn = std::env::var_os("VOICE_UNIFORM_SETUPS").is_none();
    let window = Window { first, seeds, budget, learn };
    campaign(&window);
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
    learn: bool,
}

fn campaign(window: &Window) {
    let (seeds, budget) = (window.seeds, window.budget);
    let mut reached: BTreeSet<String> = BTreeSet::new();
    let mut edges: BTreeSet<(String, String)> = BTreeSet::new();
    let mut findings: BTreeSet<(String, String)> = BTreeSet::new();
    let mut cells: BTreeMap<(&'static str, &'static str), u32> = BTreeMap::new();
    // Failures per contract, per decade of randomness.
    let mut found: BTreeMap<String, [usize; BANDS]> = BTreeMap::new();
    // One example per contract, so a finding can be read without saving the corpus.
    let mut example: BTreeMap<String, (u64, String)> = BTreeMap::new();
    let mut exercised: BTreeMap<&'static str, u32> = BTreeMap::new();
    let mut bands: [Band; BANDS] = Default::default();
    let mut policies: BTreeMap<String, Band> = BTreeMap::new();
    // Runs in which both observers closed a speech: the precondition of a decided turn.
    let mut both_closed = 0;
    let save = std::env::var_os("VOICE_SAVE").is_some();
    let corpus = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("replay");
    // What each input was declared to open, and what it was seen to open.
    let mut declared: BTreeMap<&'static str, BTreeSet<Kind>> = BTreeMap::new();
    let mut observed: BTreeMap<&'static str, BTreeSet<Kind>> = BTreeMap::new();
    let started = std::time::Instant::now();

    // The setup's switches are arms: a seed that reached new ground rewards the values it drew.
    let mut arms = search::SetupArms::new();
    for seed in window.first..window.first + seeds {
        // A fresh setup per seed, from the seed and what the campaign has learned so far: three
        // journal families exist only under legs the default setup never builds, and no budget
        // reaches them without one. Fresh every time: mutating the campaign's most novel cases
        // instead was measured on 2026-09-23 and lost a fifth of the edges and a quarter of the
        // findings at every scale -- a near-copy of a run is a run's worth of novelty forgone,
        // and the generator, not a mutator, is where this search's novelty comes from.
        let (setup, drawn) = arms.sample(&mut fastrand::Rng::with_seed(seed));
        let began = std::time::Instant::now();
        let outcome = search::explore(setup, seed, budget);
        let band_of = ((search::randomness(seed).log10() + 4.0) as usize).min(BANDS - 1);
        let took = began.elapsed();
        bands[band_of].absorb(&outcome, took);
        policies.entry(format!("{:?}", outcome.case.policy)).or_default().absorb(&outcome, took);
        let known = reached.len();
        reached.extend(outcome.reached.iter().cloned());
        if window.learn {
            arms.learn(&drawn, reached.len() > known);
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
        if !outcome.ok() {
            let mut here: BTreeSet<&str> = BTreeSet::new();
            for failure in &outcome.failures {
                findings.insert((failure.contract.clone(), failure.subject.kind()));
                if here.insert(&failure.contract) {
                    found.entry(failure.contract.clone()).or_default()[band_of] += 1;
                }
                example
                    .entry(failure.contract.clone())
                    .or_insert_with(|| (seed, failure.detail.clone()));
            }
            if save {
                let contract = &outcome.failures[0].contract;
                let path = corpus.join(format!("{contract}-{seed}.json"));
                replay::save(&path, &outcome.case, &outcome).expect("the corpus is writable");
            }
        }
    }

    let names = lk_agents_voice::trace::step_names();
    println!();
    println!(
        "{seeds} seeds from {} x {budget} inputs in {:.1?}{}",
        window.first,
        started.elapsed(),
        if window.learn { "" } else { ", uniform setups" }
    );
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
    let leanings: Vec<String> = arms
        .leanings()
        .into_iter()
        .filter(|(_, _, mean)| (mean - 0.5).abs() >= 0.15)
        .map(|(switch, value, mean)| format!("{switch}={value} {mean:.2}"))
        .collect();
    println!(
        "switch leanings: {}",
        if leanings.is_empty() { "none".into() } else { leanings.join(", ") }
    );
    println!("{both_closed} of {seeds} runs with both a VadEnd and an SttFinal\n");

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
    // `VOICE_DETAIL=1` adds the first seed and detail behind each finding.
    let detail = std::env::var_os("VOICE_DETAIL").is_some();
    for (contract, by_band) in &found {
        let counts: Vec<String> = by_band.iter().map(|n| format!("{n:>6}")).collect();
        println!("  {}  {contract}", counts.join(""));
        if detail && let Some((seed, what)) = example.get(contract) {
            println!("       seed {seed}: {what}");
        }
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

    // The gate: a contract outside the known findings is new, whatever window found it.
    let new: Vec<&String> =
        found.keys().filter(|contract| !KNOWN_FINDINGS.contains(&contract.as_str())).collect();
    println!("\nfindings outside the known ones (check::KNOWN_FINDINGS):");
    for contract in &new {
        let (seed, what) = &example[*contract];
        println!("  NEW {contract}: seed {seed}: {what}");
    }
    if new.is_empty() {
        println!("  none");
    }
    let unseen: Vec<&&str> = KNOWN_FINDINGS.iter().filter(|k| !found.contains_key(**k)).collect();
    if !unseen.is_empty() {
        println!("known findings this window did not reach: {unseen:?}");
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
    assert!(new.is_empty(), "new findings: {new:?}");
}
