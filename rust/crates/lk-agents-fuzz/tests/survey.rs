//! A campaign over many seeds, reporting what it reached and what it found.
//!
//! Ignored by default: this is the search, not part of `cargo test`.
//! `cargo test --release -p lk-agents-fuzz --test survey -- --ignored --nocapture`
//!
//! Release, because a conversation is two thousand clock advances each carrying ten frames of
//! audio through the driver, and that path is sixteen times slower unoptimised: 256 seeds take
//! 50 s in a debug build and about 4 s in release. The numbers are identical either way.
//!
//! Every seed is run twice, once per mode. `conversation` is the search: a person on a phone,
//! drawn once per run, talking turn by turn to the agent while the providers answer and fail as
//! the case says. `chaos` is the uniform sampler over the whole table: shallow everywhere, and
//! kept as the check that every row applies and every declared open is real, not as a second
//! search. Each mode gets its own block; the `=== both modes ===` block is the union, and its
//! three numbers -- variants reached, distinct edges, distinct findings -- are the yardstick every
//! change to the search is measured by, at 64 seeds on three windows and at 256.
//!
//! - `VOICE_SEEDS` (32) and `VOICE_BUDGET` (150) size a campaign; a budget unit is one chaos row
//!   or one conversation beat, so `conversation` runs three rows per unit.
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
    drive::{Outcome, Setup},
    input::Input,
    record::Kind,
    replay,
    rules::RULES,
    search,
};

/// One way of exploring from a seed.
struct Mode {
    name: &'static str,
    explore: fn(Setup, u64, usize) -> Outcome,
    /// How many rows one unit of budget buys in this mode. A budget counts things that happen,
    /// not rows: a beat of conversation is three rows -- the frames, the detector's tick, the
    /// clock -- where a chaos step is one, and a budget in rows would give the conversation a
    /// third of the depth for the same number.
    cost: usize,
}

const MODES: [Mode; 2] = [
    Mode { name: "chaos", explore: search::explore, cost: 1 },
    Mode { name: "conversation", explore: search::explore_conversation, cost: 3 },
];

#[test]
#[ignore]
fn survey() {
    let env = |key: &str| std::env::var(key).ok().and_then(|v| v.parse::<u64>().ok());
    let seeds = env("VOICE_SEEDS").unwrap_or(32);
    // `VOICE_FIRST_SEED` moves the window: the same campaign over another set of seeds is how the
    // noise floor is measured, and a change is only a change if it beats that floor.
    let first = env("VOICE_FIRST_SEED").unwrap_or(0);
    let budget = env("VOICE_BUDGET").unwrap_or(150) as usize;
    // The setup arms learn from seed to seed unless `VOICE_UNIFORM_SETUPS=1`, under which every
    // seed's setup is a uniform draw: the baseline every search change is measured against.
    let learn = std::env::var_os("VOICE_UNIFORM_SETUPS").is_none();
    let window = Window { first, seeds, budget, learn };
    let mut union = Campaign::default();
    for mode in &MODES {
        union.absorb(&campaign(mode, &window));
    }
    // What the suite reaches is the union: the two modes are two ways in, and a variant one of
    // them reaches is covered. The three numbers a search change is judged by, in one place.
    let names = lk_agents_voice::trace::step_names();
    println!("\n=== both modes ===");
    println!(
        "reached {} of {} journal variants, {} distinct edges, {} distinct findings",
        union.reached.len(),
        names.len(),
        union.edges.len(),
        union.findings.len()
    );
    println!("missing:");
    for name in names.iter().filter(|n| !union.reached.contains(*n)) {
        println!("  {name}");
    }
}

/// One campaign's seeds, budget and search settings.
struct Window {
    first: u64,
    seeds: u64,
    budget: usize,
    learn: bool,
}

/// What a campaign reached: the yardstick every search change is measured by.
#[derive(Default)]
struct Campaign {
    reached: BTreeSet<String>,
    edges: BTreeSet<(String, String)>,
    /// Distinct `(contract, subject)` pairs: two seeds failing the same rule on the same kind of
    /// thing are one finding, not two.
    findings: BTreeSet<(String, String)>,
}

impl Campaign {
    fn absorb(&mut self, other: &Campaign) {
        self.reached.extend(other.reached.iter().cloned());
        self.edges.extend(other.edges.iter().cloned());
        self.findings.extend(other.findings.iter().cloned());
    }
}

fn campaign(mode: &Mode, window: &Window) -> Campaign {
    let (seeds, budget) = (window.seeds, window.budget * mode.cost);
    let mut reached: BTreeSet<String> = BTreeSet::new();
    let mut edges: BTreeSet<(String, String)> = BTreeSet::new();
    let mut findings: BTreeSet<(String, String)> = BTreeSet::new();
    let mut found: BTreeMap<String, usize> = BTreeMap::new();
    // One example per contract, so a finding can be read without saving the corpus.
    let mut example: BTreeMap<String, (u64, String)> = BTreeMap::new();
    let mut exercised: BTreeMap<&'static str, u32> = BTreeMap::new();
    let mut failing = 0;
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
        let outcome = (mode.explore)(setup, seed, budget);
        let known = reached.len();
        reached.extend(outcome.reached.iter().cloned());
        if window.learn {
            arms.learn(&drawn, reached.len() > known);
        }
        edges.extend(outcome.edges.iter().cloned());
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
            failing += 1;
            for failure in &outcome.failures {
                findings.insert((failure.contract.clone(), format!("{:?}", failure.subject)));
                *found.entry(failure.contract.clone()).or_default() += 1;
                example
                    .entry(failure.contract.clone())
                    .or_insert_with(|| (seed, failure.detail.clone()));
            }
            if save {
                let contract = &outcome.failures[0].contract;
                // The chaos corpus keeps its names; a conversation's case says which mode made it.
                let path = match mode.name {
                    "chaos" => corpus.join(format!("{contract}-{seed}.json")),
                    name => corpus.join(format!("{contract}-{name}-{seed}.json")),
                };
                replay::save(&path, &outcome.case, &outcome).expect("the corpus is writable");
            }
        }
    }

    let names = lk_agents_voice::trace::step_names();
    println!("\n=== mode: {} ===", mode.name);
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
    println!("{failing} of {seeds} seeds failing");
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
    // `VOICE_FAMILY=Decided` opens one family up: which of its variants this mode reached.
    if let Some(family) = std::env::var("VOICE_FAMILY").ok().filter(|f| !f.is_empty()) {
        println!("\n{family}, variant by variant:");
        for name in names.iter().filter(|n| n.split("::").next() == Some(family.as_str())) {
            let mark = if reached.contains(name) { "reached" } else { "missing" };
            println!("  {mark}  {name}");
        }
    }
    println!();
    println!("findings:");
    if found.is_empty() {
        println!("  none");
    }
    // `VOICE_DETAIL=1` adds the first seed and detail behind each finding.
    let detail = std::env::var_os("VOICE_DETAIL").is_some();
    for (contract, n) in &found {
        println!("  {n:>3}  {contract}");
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
    Campaign { reached, edges, findings }
}
