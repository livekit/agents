//! The pool rules on traces written by hand: each rule rejects its counterexample and accepts the
//! clean run beside it; and a small campaign keeps every rule.

use super::*;

fn step(
    at: u64,
    input: In,
    effects: Vec<PoolEffect>,
    reservation: Option<ReservationState>,
    phase: Option<ProcessPhase>,
) -> Step {
    Step {
        at,
        input,
        effects,
        reservations: BTreeMap::from([(0, reservation)]),
        phases: phase.map(|ph| BTreeMap::from([(0, ph)])).unwrap_or_default(),
    }
}

/// One job: admitted, a process created, ready, reserved, launched; then the close releases it.
fn clean() -> Trace {
    use ReservationState::*;
    let p0 = ProcessId(0);
    let steps = vec![
        step(0, In::Start, vec![], None, None),
        step(
            10,
            In::Admit(0, "AJ_0".into()),
            vec![PoolEffect::Create(p0)],
            Some(Waiting),
            Some(ProcessPhase::Creating),
        ),
        step(40, In::Created(0), vec![], Some(Waiting), Some(ProcessPhase::Initializing)),
        step(
            900,
            In::Ready(0),
            vec![PoolEffect::Join(p0), PoolEffect::WatchExit(p0)],
            Some(Reserved(p0)),
            Some(ProcessPhase::Reserved),
        ),
        step(910, In::LaunchSucceeded(0, 0), vec![], Some(Completed), Some(ProcessPhase::Running)),
        step(911, In::Forget(0), vec![], None, Some(ProcessPhase::Running)),
        step(5_000, In::Close, vec![PoolEffect::Close(p0)], None, Some(ProcessPhase::Retiring)),
        step(5_300, In::Closed(0), vec![], None, Some(ProcessPhase::Retiring)),
        step(
            5_300,
            In::Joined(0, true),
            vec![PoolEffect::Released { process: p0, outcome: PoolJoinOutcome::Reaped }],
            None,
            Some(ProcessPhase::Cleaned),
        ),
        step(
            5_300,
            In::CleanupComplete,
            vec![PoolEffect::CloseComplete],
            None,
            Some(ProcessPhase::Cleaned),
        ),
    ];
    Trace { steps, jobs: BTreeMap::from([(0, "AJ_0".into())]) }
}

fn violations(rule: &str, trace: &Trace) -> Vec<String> {
    let (_, check) = RULES.iter().find(|(name, _)| *name == rule).expect("a rule");
    check(trace)
}

#[test]
fn the_clean_trace_keeps_every_rule() {
    assert_eq!(check(&clean()), Vec::<String>::new());
}

// Counterexample: an admitted request left waiting by the close.
#[test]
fn settle_rejects_a_request_the_close_left_waiting() {
    let mut trace = clean();
    trace.steps.truncate(3);
    trace.steps.push(step(
        5_000,
        In::Close,
        vec![PoolEffect::Close(ProcessId(0))],
        Some(ReservationState::Waiting),
        None,
    ));
    assert_eq!(violations("worker_admitted_requests_settle", &trace).len(), 2);
}

// Counterexample: a duplicate dispatch of the running job admitted, and run again.
#[test]
fn one_disposition_rejects_a_job_run_twice() {
    let mut trace = clean();
    trace.jobs.insert(1, "AJ_0".into());
    let mut dup =
        step(2_000, In::Admit(1, "AJ_0".into()), vec![], None, Some(ProcessPhase::Running));
    dup.reservations.insert(1, Some(ReservationState::Completed));
    trace.steps.insert(6, dup);
    assert_eq!(violations("worker_one_disposition_per_assignment", &trace).len(), 2);
}

// Counterexample: a stale `Ready` revives a retiring process, and a released one is closed again.
#[test]
fn no_revival_rejects_a_revived_process() {
    let mut trace = clean();
    trace.steps.insert(7, step(5_100, In::Ready(0), vec![], None, Some(ProcessPhase::Ready)));
    trace.steps.push(step(
        6_000,
        In::Wake,
        vec![PoolEffect::Close(ProcessId(0))],
        None,
        Some(ProcessPhase::Cleaned),
    ));
    assert_eq!(violations("worker_retired_processes_stay_retired", &trace).len(), 2);
}

// Counterexample: the close completes with a process never released nor reported.
#[test]
fn released_rejects_a_close_that_left_a_process() {
    let mut trace = clean();
    trace.steps[8].effects.clear();
    assert_eq!(violations("worker_processes_are_released", &trace).len(), 2);
}

#[test]
fn a_small_window_keeps_every_worker_rule() {
    let mut reached = BTreeSet::new();
    for seed in 0..256 {
        for sampling in [Sampling::Arrival, Sampling::BasicPos] {
            let outcome = run(seed, sampling);
            assert!(
                outcome.violations.is_empty(),
                "WORKER_SEED={seed} {sampling:?}: {:#?}",
                outcome.violations
            );
            reached.extend(outcome.reached);
        }
    }
    for label in [
        "SpawnFailed",
        "LaunchFailed",
        "Launched",
        "ExitFailed",
        "JoinFailed",
        "Cancel",
        "Refused",
        "RequestFailed",
        "CloseComplete",
    ] {
        assert!(reached.contains(label), "{label} not reached: {reached:?}");
    }
}

/// `WORKER_SEED=<n>` (`WORKER_SAMPLING=BasicPos`): one run's trace and violations in full.
#[test]
#[ignore = "a seed's replay"]
fn replay() {
    let seed = std::env::var("WORKER_SEED").expect("WORKER_SEED").parse().expect("a seed");
    let sampling = match std::env::var("WORKER_SAMPLING").as_deref() {
        Ok("BasicPos") => Sampling::BasicPos,
        _ => Sampling::Arrival,
    };
    let outcome = run(seed, sampling);
    for step in &outcome.trace.steps {
        println!(
            "{:>7} {:?} -> {:?}\n        {:?} {:?}",
            step.at, step.input, step.effects, step.reservations, step.phases
        );
    }
    println!("{:#?}", outcome.violations);
}
