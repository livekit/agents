//! The LLM request rules on traces written by hand: each rule rejects its counterexample and
//! accepts the clean run beside it; and a small campaign keeps every rule.

use lk_agents_abi::errors::ApiError;

use super::*;

fn policy() -> RequestPolicy {
    RequestPolicy {
        connect_timeout: Some(dur(10_000)),
        write_timeout: Some(dur(10_000)),
        read_timeout: Some(dur(10_000)),
        total_timeout: None,
        max_retry: 3,
        retry_interval: dur(2_000),
    }
}

fn start(attempt: u32) -> LlmRequestEffect {
    LlmRequestEffect::LlmRequestStart { request_id: ID, attempt, llm: LLM }
}

fn delivered(content: &str) -> LlmRequestEffect {
    let delta = lk_agents_abi::ChoiceDelta { content: Some(content.into()), ..Default::default() };
    let chunk =
        lk_agents_abi::LlmChunk { id: "c".into(), delta: Some(delta), ..Default::default() };
    LlmRequestEffect::Deliver(chunk)
}

fn step(at: u64, event: Option<LlmRequestEvent>, effects: Vec<LlmRequestEffect>) -> Step {
    Step { at, event, effects }
}

fn chunk(attempt: u32) -> LlmRequestEvent {
    LlmRequestEvent::Chunk { attempt, chunk: text("hi") }
}

fn reset() -> AttemptFailure {
    AttemptFailure::Connection("reset".into())
}

/// A clean run: one failed attempt retried after 100 ms, then a streamed answer, both released.
fn clean() -> Trace {
    use LlmRequestEvent as E;
    let steps = vec![
        step(0, None, vec![start(0)]),
        step(0, Some(E::DemandStarted), vec![]),
        step(
            50,
            Some(E::Failed { attempt: 0, failure: reset() }),
            vec![LlmRequestEffect::AttemptFailed {
                attempt: 0,
                error: ApiError::connection(Some("reset".into())),
            }],
        ),
        step(60, Some(E::AttemptClosed { attempt: 0 }), vec![]),
        step(150, Some(E::Wake), vec![start(1)]),
        step(200, Some(E::Connected { attempt: 1 }), vec![]),
        step(210, Some(E::RequestWritten { attempt: 1 }), vec![]),
        step(600, Some(chunk(1)), vec![delivered("hi")]),
        step(700, Some(E::Ended { attempt: 1 }), vec![LlmRequestEffect::Finished(Ok(()))]),
        step(710, Some(E::AttemptClosed { attempt: 1 }), vec![]),
    ];
    Trace { policy: policy(), steps, settled: true }
}

fn violations(rule: &str, trace: &Trace) -> Vec<String> {
    let (_, check) = RULES.iter().find(|(name, _)| *name == rule).expect("a rule");
    check(trace)
}

#[test]
fn the_clean_trace_keeps_every_rule() {
    assert_eq!(check(&clean()), Vec::<String>::new());
}

// Counterexample: a second outcome, and a chunk after the first.
#[test]
fn one_outcome_rejects_a_second_outcome() {
    let mut trace = clean();
    trace.steps.push(step(
        800,
        Some(chunk(1)),
        vec![delivered("late"), LlmRequestEffect::Finished(Ok(()))],
    ));
    assert_eq!(violations("llm_request_has_one_outcome", &trace).len(), 2);
}

// Counterexample: an attempt launched after output reached the consumer.
#[test]
fn never_replayed_rejects_a_retry_after_output() {
    let mut trace = clean();
    trace.steps.insert(8, step(650, Some(LlmRequestEvent::Wake), vec![start(2)]));
    assert!(!violations("llm_output_is_never_replayed", &trace).is_empty());
}

// Counterexample: the first retry launched on `retry_interval` instead of 100 ms.
#[test]
fn backoff_rejects_a_first_retry_on_the_interval() {
    let mut trace = clean();
    trace.steps[4].at = 2_050;
    assert!(!violations("llm_retries_are_bounded_and_backed_off", &trace).is_empty());
}

// Counterexample: a read timeout while the consumer was not waiting.
#[test]
fn deadlines_reject_a_read_timeout_without_demand() {
    let mut trace = clean();
    trace.steps.insert(7, step(300, Some(LlmRequestEvent::DemandEnded), vec![]));
    trace.steps[8] = step(
        10_300,
        Some(LlmRequestEvent::Wake),
        vec![
            LlmRequestEffect::LlmRequestCancel { request_id: ID, attempt: 1 },
            LlmRequestEffect::AttemptFailed {
                attempt: 1,
                error: ApiError::timed_out(TimeoutPhase::Read),
            },
        ],
    );
    trace.steps.truncate(9);
    assert!(!violations("llm_deadlines_follow_the_policy", &trace).is_empty());
}

// Counterexample: a connection that hangs past its limit with nothing fired.
#[test]
fn deadlines_reject_a_missed_connect_limit() {
    let trace = Trace {
        policy: policy(),
        steps: vec![
            step(0, None, vec![start(0)]),
            step(30_000, Some(LlmRequestEvent::Cancel), vec![]),
        ],
        settled: true,
    };
    assert!(!violations("llm_deadlines_follow_the_policy", &trace).is_empty());
}

// Counterexample: a stale attempt's chunk delivered.
#[test]
fn current_only_rejects_a_stale_chunk() {
    let mut trace = clean();
    trace.steps.insert(5, step(160, Some(chunk(0)), vec![delivered("stale")]));
    assert!(!violations("llm_only_the_current_attempt_delivers", &trace).is_empty());
}

// Counterexample: the consumer cancels and the running attempt is never stopped, nor released.
#[test]
fn released_rejects_an_attempt_left_running() {
    let trace = Trace {
        policy: policy(),
        steps: vec![
            step(0, None, vec![start(0)]),
            step(100, Some(LlmRequestEvent::Cancel), vec![]),
        ],
        settled: false,
    };
    assert_eq!(violations("llm_attempts_are_released", &trace).len(), 3);
}

// Counterexample: a release reported missing before the cleanup deadline.
#[test]
fn released_rejects_an_early_cleanup_report() {
    let mut trace = clean();
    trace.steps.remove(9);
    trace.steps.push(step(
        800,
        Some(LlmRequestEvent::Wake),
        vec![LlmRequestEffect::CleanupIncomplete { request_id: ID, attempt: 1 }],
    ));
    assert!(!violations("llm_attempts_are_released", &trace).is_empty());
}

/// A window of the campaign under both orders: every rule holds, and the run reaches retries,
/// every give-up, the phase timeouts and stale chunks.
#[test]
fn a_small_window_keeps_every_llm_rule() {
    let mut reached = BTreeSet::new();
    for seed in 0..256 {
        for sampling in [Sampling::Arrival, Sampling::BasicPos] {
            let outcome = run(seed, sampling);
            assert!(
                outcome.violations.is_empty(),
                "LLM_SEED={seed} {sampling:?}: {:#?}",
                outcome.violations
            );
            reached.extend(outcome.reached);
        }
    }
    for label in [
        "Succeeded",
        "Retried",
        "GaveUpRetryable",
        "GaveUpFinal",
        "Cancelled",
        "ConnectTimeout",
        "ReadTimeout",
        "CleanupIncomplete",
    ] {
        assert!(reached.contains(label), "{label} not reached: {reached:?}");
    }
}
