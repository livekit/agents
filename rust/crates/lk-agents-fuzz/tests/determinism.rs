//! The simulation must be reproducible: the same case twice must leave the same recording.
//!
//! This is a guard, not a demonstration. Tokio randomises the branch order of an unbiased
//! `select!`, so concurrent generations used to race differently on every run and a finding landed
//! on a different speech each time. `drive::simulate_with` pins that RNG and gives each case its own
//! runtime. Anything that reintroduces real nondeterminism -- a `HashMap` iterated for ordering,
//! wall-clock time read as data, a fresh `select!` outside a seeded runtime -- fails here first,
//! with a diff, instead of showing up months later as a corpus that stopped reproducing.
use lk_agents_fuzz::{
    drive::{Run, simulate_with},
    record::Recording,
    replay, search,
};

/// Wall-clock stamps the product puts on its own data, and the one hash that covers them.
///
/// Everything a rule can read, flattened. Two runs of one case must agree on all of it.
fn fingerprint(r: &Recording) -> String {
    let mut out = String::new();
    for (at, step) in r.journal() {
        out.push_str(&format!("J {} {:?}\n", at.seq, step));
    }
    for o in r.opened() {
        out.push_str(&format!("O {} {:?}#{} {:?} {:?}\n", o.at.seq, o.kind, o.nth, o.id, o.detail));
    }
    for (at, host) in r.host() {
        out.push_str(&format!("H {} {:?}\n", at.seq, host));
    }
    for (at, nth, answer) in r.results() {
        out.push_str(&format!("R {at:?} {nth} {answer:?}\n"));
    }
    for (at, sink) in r.sink() {
        out.push_str(&format!("S {} {:?}\n", at.seq, sink));
    }
    for (id, handle) in r.handles() {
        out.push_str(&format!("P {id} {handle:?}\n"));
    }
    out
}

fn replayed(case: &lk_agents_fuzz::drive::Case) -> String {
    simulate_with(case.schedule, async {
        let mut run = Run::start(&case.setup, case.schedule, case.policy).await;
        for input in &case.inputs {
            run.apply(input).await;
        }
        fingerprint(&run.record)
    })
}

/// Every saved case leaves the same recording twice over, down to the last journal step.
#[test]
fn a_saved_case_replays_identically() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("replay");
    let mut cases: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|entry| entry.expect("a readable directory entry").path())
        .filter(|path| path.extension().is_some_and(|e| e == "json"))
        .collect();
    cases.sort();
    assert!(!cases.is_empty(), "no saved cases to check determinism against");
    let mut reproducible: Vec<(String, bool)> = Vec::new();

    for path in cases {
        let case = replay::load(&path).case;
        let (first, second) = (replayed(&case), replayed(&case));
        if first != second {
            reproducible.push((path.display().to_string(), false));
            let at = first
                .lines()
                .zip(second.lines())
                .position(|(a, b)| a != b)
                .unwrap_or_else(|| first.lines().count().min(second.lines().count()));
            println!(
                "NOT REPRODUCIBLE {} line {at}:\n  {}\n  {}",
                path.display(),
                &first.lines().nth(at).unwrap_or("<end>")
                    [..140.min(first.lines().nth(at).unwrap_or("").len())],
                &second.lines().nth(at).unwrap_or("<end>")
                    [..140.min(second.lines().nth(at).unwrap_or("").len())],
            );
        } else {
            reproducible.push((path.display().to_string(), true));
        }
        // A fingerprint that is nearly empty would make this test pass without checking anything.
        assert!(
            first.lines().count() > 50,
            "{}: recording too small to be a real check",
            path.display()
        );
    }
    let bad = reproducible.iter().filter(|(_, ok)| !ok).count();
    println!("{} of {} cases reproduce", reproducible.len() - bad, reproducible.len());
    assert_eq!(bad, 0, "{bad} saved cases are not reproducible");
}

/// The same seed explores the same case, so a campaign is repeatable end to end. Setup included:
/// the survey samples one per seed, and a case is both halves.
#[test]
fn the_same_seed_explores_the_same_case() {
    let setup = |seed| search::sample_setup(&mut fastrand::Rng::with_seed(seed));
    assert_eq!(setup(7), setup(7), "one seed sampled two different setups");
    let first = search::explore(setup(7), 7, 300);
    let second = search::explore(setup(7), 7, 300);
    assert_eq!(first.case.inputs, second.case.inputs, "one seed sampled two different cases");
    assert_eq!(
        first.failures.iter().map(|f| (&f.contract, &f.subject)).collect::<Vec<_>>(),
        second.failures.iter().map(|f| (&f.contract, &f.subject)).collect::<Vec<_>>(),
        "one seed produced two different verdicts"
    );
}
