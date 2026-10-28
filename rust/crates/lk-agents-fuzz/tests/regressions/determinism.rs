//! The simulation must be reproducible: the same case twice must leave the same recording.
//!
//! This is a guard, not a demonstration. Tokio randomises the branch order of an unbiased
//! `select!`, so concurrent generations used to race differently on every run and a finding landed
//! on a different speech each time. `drive::simulate_with` pins that RNG and gives each case its own
//! runtime. Anything that reintroduces real nondeterminism -- a `HashMap` iterated for ordering,
//! wall-clock time read as data, a fresh `select!` outside a seeded runtime -- fails here first,
//! with a diff, instead of showing up months later as a seed that no longer reproduces its run.
use fastrand::Rng;
use lk_agents_fuzz::{
    drive::{Run, simulate_with},
    record::Recording,
    search,
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

fn rerun(case: &lk_agents_fuzz::drive::Case) -> String {
    simulate_with(case.schedule, async {
        let mut run = Run::start(&case.setup, case.schedule, case.policy).await;
        for input in &case.inputs {
            run.apply(input).await;
        }
        format!("{}D {}\n", fingerprint(&run.record), run.dispatch_order())
    })
}

/// A case the search explored, ending included, leaves the same recording twice over, down to
/// the last journal step.
#[test]
fn an_explored_case_reruns_identically() {
    let mut bad = 0;
    for seed in 0..6 {
        let case = search::explore(search::sample_setup(&mut Rng::with_seed(seed)), seed, 600).case;
        let (first, second) = (rerun(&case), rerun(&case));
        if first != second {
            bad += 1;
            let at = first
                .lines()
                .zip(second.lines())
                .position(|(a, b)| a != b)
                .unwrap_or_else(|| first.lines().count().min(second.lines().count()));
            let line = |text: &str| -> String {
                text.lines().nth(at).unwrap_or("<end>").chars().collect()
            };
            println!(
                "NOT REPRODUCIBLE seed {seed} line {at}:\n  {}\n  {}",
                line(&first),
                line(&second)
            );
        }
        // A fingerprint that is nearly empty would make this test pass without checking anything.
        assert!(first.lines().count() > 50, "seed {seed}: recording too small to be a real check");
    }
    assert_eq!(bad, 0, "{bad} explored cases are not reproducible");
}

/// The same seed explores the same case, so a campaign is repeatable end to end. Setup included:
/// the survey samples one per seed, and a case is both halves.
#[test]
fn the_same_seed_explores_the_same_case() {
    let setup = |seed| search::sample_setup(&mut Rng::with_seed(seed));
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
