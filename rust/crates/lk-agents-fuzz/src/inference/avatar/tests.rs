//! The avatar rules on traces written by hand: each rule rejects its counterexample and accepts
//! the clean run beside it; and a small campaign keeps every rule.

use super::*;

const R1: AvatarRequestId = AvatarRequestId(1);
const R2: AvatarRequestId = AvatarRequestId(2);

fn session(n: u64) -> ProviderSession {
    ProviderSession {
        session_id: Some(format!("s{n}")),
        provider_session_id: Some(format!("p{n}")),
        terminate_token: Some(format!("t{n}")),
        sample_rate: 24_000,
    }
}

fn step(at: u64, event: Ev, effects: Vec<Fx>) -> Step {
    Step { at, event, effects }
}

/// A clean run: created, adopted, closed and terminated, every attempt released.
fn clean() -> Trace {
    let s = session(1);
    let steps = vec![
        step(
            0,
            Ev::Create,
            vec![Fx::AvatarSessionCreate { request: R1, attempt: 0, avatar: AVATAR }],
        ),
        step(
            900,
            Ev::Created { request: R1, attempt: 0, session: s.clone() },
            vec![Fx::CreateFinished(Ok(s.clone()))],
        ),
        step(901, Ev::Adopted, vec![]),
        step(910, Ev::AttemptClosed { request: R1, attempt: 0 }, vec![]),
        step(
            5_000,
            Ev::Close,
            vec![Fx::AvatarSessionTerminate {
                request: R2,
                attempt: 0,
                avatar: AVATAR,
                session: s.clone(),
            }],
        ),
        step(
            5_200,
            Ev::Terminated { request: R2, attempt: 0 },
            vec![Fx::SessionTerminated(s.clone())],
        ),
        step(5_210, Ev::AttemptClosed { request: R2, attempt: 0 }, vec![]),
    ];
    Trace {
        steps,
        made: BTreeMap::from([(R1, s)]),
        running: Vec::new(),
        settled: true,
        read_limit: true,
    }
}

fn violations(rule: &str, trace: &Trace) -> Vec<String> {
    let (_, check) = RULES.iter().find(|(name, _)| *name == rule).expect("a rule");
    check(trace)
}

#[test]
fn the_clean_trace_keeps_every_rule() {
    assert_eq!(check(&clean()), Vec::<String>::new());
}

// Counterexample: the close never terminated the adopted session, and the gateway still runs it.
#[test]
fn never_leaked_rejects_a_session_left_running() {
    let mut trace = clean();
    trace.steps.truncate(4);
    trace.steps.push(step(5_000, Ev::Close, vec![]));
    trace.running = vec![session(1)];
    assert!(!violations("avatar_sessions_are_never_leaked", &trace).is_empty());
}

// Counterexample: a racing create accepted while the first is in flight, both adopted.
#[test]
fn one_session_rejects_a_second_creation() {
    let mut trace = clean();
    trace.steps.insert(
        1,
        step(
            100,
            Ev::Create,
            vec![Fx::AvatarSessionCreate { request: R2, attempt: 0, avatar: AVATAR }],
        ),
    );
    trace.steps.insert(4, step(902, Ev::Adopted, vec![]));
    assert_eq!(violations("avatar_creates_one_session", &trace).len(), 2);
}

// Counterexample: a close during creation that leaves the caller waiting, and an outcome after a
// cancel.
#[test]
fn one_outcome_rejects_an_unanswered_close_and_a_late_outcome() {
    let s = session(1);
    let mut closed = clean();
    closed.steps.truncate(1);
    closed.steps.push(step(300, Ev::Close, vec![]));
    assert!(!violations("avatar_caller_has_one_outcome", &closed).is_empty());
    let mut late = clean();
    late.steps.insert(1, step(300, Ev::Cancel, vec![]));
    late.steps[2] = step(
        900,
        Ev::Created { request: R1, attempt: 0, session: s.clone() },
        vec![Fx::CreateFinished(Ok(s))],
    );
    assert!(!violations("avatar_caller_has_one_outcome", &late).is_empty());
}

// Counterexample: a session retired as `CallerGone` with its caller still waiting, and a terminate
// of a session the gateway never made.
#[test]
fn retirements_reject_a_retirement_without_its_cause() {
    let mut trace = clean();
    trace.steps[1].effects =
        vec![Fx::Retiring { session: session(1), reason: Retirement::CallerGone }];
    trace.steps[4].effects = vec![Fx::AvatarSessionTerminate {
        request: R2,
        attempt: 0,
        avatar: AVATAR,
        session: session(9),
    }];
    assert_eq!(violations("avatar_retirements_have_their_cause", &trace).len(), 2);
}

// Counterexample: a terminate attempt whose release is never observed nor reported.
#[test]
fn released_rejects_an_unreleased_attempt() {
    let mut trace = clean();
    trace.steps.pop();
    assert!(!violations("avatar_attempts_are_released", &trace).is_empty());
}

// Counterexample: a create the close left reported unresolved early, and twice.
#[test]
fn retirement_bounded_rejects_an_early_and_a_repeated_report() {
    let mut trace = clean();
    trace.steps.truncate(1);
    trace.steps.push(step(1_000, Ev::Close, vec![Fx::CreateFinished(Err(CreateFailure::Closed))]));
    trace.steps.push(step(20_000, Ev::Wake, vec![Fx::RetirementIncomplete { request: R1 }]));
    trace.steps.push(step(31_000, Ev::Wake, vec![Fx::RetirementIncomplete { request: R1 }]));
    assert_eq!(violations("avatar_retirement_is_bounded", &trace).len(), 2);
}

/// A window of the campaign under both orders: every rule holds, and the run reaches refusals,
/// retries, every retirement, failed terminates and missing releases.
#[test]
fn a_small_window_keeps_every_avatar_rule() {
    let mut reached = BTreeSet::new();
    for seed in 0..512 {
        for sampling in [Sampling::Arrival, Sampling::BasicPos] {
            let outcome = run(seed, sampling);
            let unknown = unknown(&outcome.trace, &outcome.violations);
            assert!(unknown.is_empty(), "AVATAR_SEED={seed} {sampling:?}: {unknown:#?}");
            reached.extend(outcome.reached);
        }
    }
    for label in [
        "CreateRefused",
        "Retried",
        "Delivered",
        "ClosedWhileCreating",
        "CreateFailed",
        "RetiredCallerGone",
        "RetiredRefused",
        "RetiredClosed",
        "Unterminable",
        "Terminated",
        "TerminateFailed",
    ] {
        assert!(reached.contains(label), "{label} not reached: {reached:?}");
    }
}

/// `AVATAR_SEED=<n>` (and `AVATAR_SAMPLING=BasicPos`): one run's trace and violations in full.
#[test]
#[ignore = "a seed's replay"]
fn replay() {
    let seed = std::env::var("AVATAR_SEED").expect("AVATAR_SEED").parse().expect("a seed");
    let sampling = match std::env::var("AVATAR_SAMPLING").as_deref() {
        Ok("BasicPos") => Sampling::BasicPos,
        _ => Sampling::Arrival,
    };
    let outcome = run(seed, sampling);
    for step in &outcome.trace.steps {
        println!("{:>7} {:?}\n        -> {:?}", step.at, step.event, step.effects);
    }
    println!(
        "made {:?}\nrunning {:?}\nsettled {}",
        outcome.trace.made, outcome.trace.running, outcome.trace.settled
    );
    println!("{:#?}", outcome.violations);
}
