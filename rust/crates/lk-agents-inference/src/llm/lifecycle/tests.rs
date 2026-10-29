//! `request_lifecycle_vectors`: the request machine stepped directly on supplied time.

use std::time::Duration;

use super::*;
use lk_agents_abi::HOST_TEARDOWN_TIMEOUT;
use lk_agents_abi::errors::{ApiErrorKind, TimeoutPhase};

const LLM: ResourceId = ResourceId(7);
const ID: LlmRequestId = LlmRequestId(1);

fn at(ms: u64) -> Ts {
    Ts::from_nanos(ms * 1_000_000)
}

fn policy() -> RequestPolicy {
    RequestPolicy {
        connect_timeout: Some(Duration::from_secs(10)),
        write_timeout: Some(Duration::from_secs(10)),
        read_timeout: Some(Duration::from_secs(10)),
        total_timeout: None,
        max_retry: 3,
        retry_interval: Duration::from_secs(2),
    }
}

fn start(policy: RequestPolicy) -> LlmRequest {
    let (request, effects) = LlmRequest::start(ID, LLM, policy, ("<think>", "</think>"), at(0));
    assert_eq!(effects, vec![started(0)]);
    request
}

fn started(attempt: u32) -> LlmRequestEffect {
    LlmRequestEffect::LlmRequestStart { request_id: ID, attempt, llm: LLM }
}

fn cancelled(attempt: u32) -> LlmRequestEffect {
    LlmRequestEffect::LlmRequestCancel { request_id: ID, attempt }
}

fn text(content: &str) -> WireChunk {
    serde_json::from_value(
        serde_json::json!({"id": "c", "choices": [{"delta": {"content": content}}]}),
    )
    .unwrap()
}

fn failure() -> AttemptFailure {
    AttemptFailure::Connection("reset".into())
}

/// Brings attempt `n` to the point where a response is expected, with the consumer waiting.
fn reading(request: &mut LlmRequest, attempt: u32, now: Ts) {
    request.handle(LlmRequestEvent::DemandStarted, now);
    request.handle(LlmRequestEvent::Connected { attempt }, now);
    request.handle(LlmRequestEvent::RequestWritten { attempt }, now);
}

fn finished(effects: &[LlmRequestEffect]) -> Option<&Result<(), ApiError>> {
    effects.iter().find_map(|e| match e {
        LlmRequestEffect::Finished(outcome) => Some(outcome),
        _ => None,
    })
}

// Catches: retrying without backoff, a first retry on `retry_interval` instead of 100 ms, or
// exceeding `max_retry` additional attempts.
#[test]
fn failures_before_output_are_retried_max_retry_times_with_backoff() {
    let mut request = start(policy());
    let mut now = at(0);
    for attempt in 0..=3u32 {
        let effects = request.handle(LlmRequestEvent::Failed { attempt, failure: failure() }, now);
        request.handle(LlmRequestEvent::AttemptClosed { attempt }, now);
        if attempt == 3 {
            let err = finished(&effects).unwrap().clone().unwrap_err();
            assert_eq!(err.kind, ApiErrorKind::Connection);
            assert!(err.retryable, "nothing was delivered: a fallback may still try");
            assert!(request.is_settled());
            return;
        }
        assert!(
            matches!(effects.as_slice(), [LlmRequestEffect::AttemptFailed { .. }]),
            "{effects:?}"
        );
        let backoff = if attempt == 0 { 100 } else { 2_000 };
        let due = now.saturating_add(Duration::from_millis(backoff));
        assert_eq!(
            request.handle(LlmRequestEvent::Wake, due.saturating_sub(Duration::from_millis(1))),
            vec![]
        );
        assert_eq!(request.handle(LlmRequestEvent::Wake, due), vec![started(attempt + 1)]);
        now = due;
    }
}

// Catches: replaying a request whose answer already reached the consumer.
#[test]
fn a_failure_after_visible_output_ends_the_request_unretryable() {
    let mut request = start(policy());
    reading(&mut request, 0, at(0));
    let effects = request.handle(LlmRequestEvent::Chunk { attempt: 0, chunk: text("Hel") }, at(5));
    assert!(
        matches!(effects.as_slice(), [LlmRequestEffect::Deliver(chunk)] if chunk.has_response())
    );
    let effects = request.handle(LlmRequestEvent::Failed { attempt: 0, failure: failure() }, at(6));
    let err = finished(&effects).unwrap().clone().unwrap_err();
    assert!(!err.retryable);
    assert!(!effects.iter().any(|e| matches!(e, LlmRequestEffect::LlmRequestStart { .. })));
}

// Catches: replay guarded on any chunk instead of on output the consumer can act on: the
// gateway's metadata-only leading delta or a tool call whose arguments are still streaming made a
// failure final, while a completed tool call must never be issued twice.
#[test]
fn only_actionable_output_blocks_a_retry() {
    let wire = |delta: serde_json::Value, finish: Option<&str>| -> WireChunk {
        serde_json::from_value(serde_json::json!({
            "id": "c",
            "choices": [{"delta": delta, "finish_reason": finish}],
        }))
        .unwrap()
    };
    let metadata = wire(
        serde_json::json!({"extra_content": {"livekit": {"inference_tier_billed": "s"}}}),
        None,
    );
    let call = wire(
        serde_json::json!({"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "lookup"}}]}),
        None,
    );
    let args = wire(
        serde_json::json!({"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}),
        None,
    );
    let done = wire(serde_json::json!({}), Some("tool_calls"));
    let attempts: [&[WireChunk]; 3] =
        [&[metadata.clone()], &[metadata, call.clone(), args.clone()], &[call, args, done]];
    let mut request = start(policy());
    let mut now = at(0);
    for (attempt, chunks) in (0u32..).zip(attempts) {
        reading(&mut request, attempt, now);
        for chunk in chunks {
            request.handle(LlmRequestEvent::Chunk { attempt, chunk: chunk.clone() }, now);
        }
        let effects = request.handle(LlmRequestEvent::Failed { attempt, failure: failure() }, now);
        if attempt < 2 {
            assert!(
                matches!(effects.as_slice(), [LlmRequestEffect::AttemptFailed { .. }]),
                "attempt {attempt}: {effects:?}"
            );
            request.handle(LlmRequestEvent::AttemptClosed { attempt }, now);
            now = request.wake_at().unwrap();
            assert_eq!(request.handle(LlmRequestEvent::Wake, now), vec![started(attempt + 1)]);
        } else {
            assert!(!finished(&effects).unwrap().clone().unwrap_err().retryable);
        }
    }
}

// Catches: status retry rules lost in the machine: a 401 retried, a 429 not.
#[test]
fn a_rejection_is_retried_only_when_its_status_is_transient() {
    let reject =
        |status| AttemptFailure::Rejected { status, message: "no".into(), request_id: None };
    let mut request = start(policy());
    let effects =
        request.handle(LlmRequestEvent::Failed { attempt: 0, failure: reject(429) }, at(0));
    assert!(matches!(effects.as_slice(), [LlmRequestEffect::AttemptFailed { .. }]));
    request.handle(LlmRequestEvent::Wake, at(100));
    let effects =
        request.handle(LlmRequestEvent::Failed { attempt: 1, failure: reject(401) }, at(100));
    let err = finished(&effects).unwrap().clone().unwrap_err();
    assert_eq!(
        (err.kind, err.status_code, err.retryable),
        (ApiErrorKind::Status, Some(401), false)
    );
}

// Catches: a read timer that counts the consumer's own pauses (backpressure) as remote
// silence, or one not restarted by progress.
#[test]
fn the_read_limit_counts_remote_silence_only_while_the_consumer_waits() {
    let mut request = start(RequestPolicy { max_retry: 0, ..policy() });
    reading(&mut request, 0, at(0));
    request.handle(LlmRequestEvent::Chunk { attempt: 0, chunk: text("a") }, at(9_000));
    assert_eq!(request.wake_at(), Some(at(19_000)), "progress restarts the limit");
    // The consumer stops pulling for a minute: no read timer runs meanwhile.
    request.handle(LlmRequestEvent::DemandEnded, at(9_001));
    assert_eq!(request.wake_at(), None);
    assert_eq!(request.handle(LlmRequestEvent::Wake, at(60_000)), vec![]);
    request.handle(LlmRequestEvent::DemandStarted, at(70_000));
    assert_eq!(request.wake_at(), Some(at(80_000)));
    let effects = request.handle(LlmRequestEvent::Wake, at(80_000));
    assert_eq!(effects[0], cancelled(0));
    let err = finished(&effects).unwrap().clone().unwrap_err();
    assert_eq!((err.kind, err.phase), (ApiErrorKind::Timeout, Some(TimeoutPhase::Read)));
}

// Catches: one timer for the whole exchange, or a timeout reported without its phase.
#[test]
fn connect_and_write_limits_report_their_own_phase() {
    let mut request =
        start(RequestPolicy { write_timeout: Some(Duration::from_secs(3)), ..policy() });
    let effects = request.handle(LlmRequestEvent::Wake, at(10_000));
    assert_eq!(effects[0], cancelled(0));
    assert!(
        matches!(&effects[1], LlmRequestEffect::AttemptFailed { error, .. } if error.phase == Some(TimeoutPhase::Connect))
    );
    request.handle(LlmRequestEvent::Wake, at(10_100));
    request.handle(LlmRequestEvent::Connected { attempt: 1 }, at(11_000));
    assert_eq!(request.wake_at(), Some(at(14_000)));
    let effects = request.handle(LlmRequestEvent::Wake, at(14_000));
    assert!(
        matches!(&effects[1], LlmRequestEffect::AttemptFailed { error, .. } if error.phase == Some(TimeoutPhase::Write))
    );
}

// Catches: a total deadline reset by retries or progress, or an attempt launched after it.
#[test]
fn the_total_limit_spans_every_attempt_and_stops_the_request() {
    let mut request =
        start(RequestPolicy { total_timeout: Some(Duration::from_secs(3)), ..policy() });
    request.handle(LlmRequestEvent::Failed { attempt: 0, failure: failure() }, at(0));
    request.handle(LlmRequestEvent::Wake, at(100));
    reading(&mut request, 1, at(100));
    request.handle(LlmRequestEvent::Progress { attempt: 1 }, at(2_900));
    assert_eq!(request.wake_at(), Some(at(3_000)));
    let effects = request.handle(LlmRequestEvent::Wake, at(3_000));
    assert_eq!(effects[0], cancelled(1));
    let err = finished(&effects).unwrap().clone().unwrap_err();
    assert_eq!((err.phase, err.retryable), (Some(TimeoutPhase::Total), false));
    request.handle(LlmRequestEvent::Failed { attempt: 1, failure: failure() }, at(3_001));
    assert!(request.handle(LlmRequestEvent::Wake, at(9_000)).iter().all(|e| !matches!(
        e,
        LlmRequestEffect::LlmRequestStart { .. } | LlmRequestEffect::Finished(_)
    )));
}

// Catches: a cancelled request that still retries, reports an outcome, or forgets the cleanup of
// the attempt it stopped.
#[test]
fn cancel_stops_the_attempt_and_waits_for_its_cleanup() {
    let mut request = start(policy());
    assert_eq!(request.handle(LlmRequestEvent::Cancel, at(10)), vec![cancelled(0)]);
    assert!(!request.is_settled(), "the attempt still owes its release");
    assert_eq!(
        request.handle(LlmRequestEvent::Failed { attempt: 0, failure: failure() }, at(11)),
        vec![]
    );
    assert_eq!(request.handle(LlmRequestEvent::AttemptClosed { attempt: 0 }, at(12)), vec![]);
    assert!(request.is_settled());
    assert_eq!(request.wake_at(), None);
}

// Catches: a late chunk from a superseded attempt delivered into the retry's answer.
#[test]
fn a_superseded_attempts_events_are_ignored() {
    let mut request = start(policy());
    request.handle(LlmRequestEvent::Failed { attempt: 0, failure: failure() }, at(0));
    request.handle(LlmRequestEvent::Wake, at(100));
    assert_eq!(
        request.handle(LlmRequestEvent::Chunk { attempt: 0, chunk: text("stale") }, at(101)),
        vec![]
    );
    assert_eq!(request.handle(LlmRequestEvent::Ended { attempt: 0 }, at(102)), vec![]);
    let effects =
        request.handle(LlmRequestEvent::Chunk { attempt: 1, chunk: text("fresh") }, at(103));
    assert_eq!(effects.len(), 1);
    assert_eq!(
        request.handle(LlmRequestEvent::Ended { attempt: 1 }, at(104)),
        vec![LlmRequestEffect::Finished(Ok(()))]
    );
}

// Catches: an attempt that never settles reported as clean termination, or reported forever
// (one typed observation, ownership stays with the runtime).
#[test]
fn a_missed_cleanup_deadline_is_reported_once() {
    let mut request = start(policy());
    request.handle(LlmRequestEvent::Cancel, at(0));
    let due = at(0).saturating_add(HOST_TEARDOWN_TIMEOUT);
    assert_eq!(request.wake_at(), Some(due));
    assert_eq!(
        request.handle(LlmRequestEvent::Wake, due),
        vec![LlmRequestEffect::CleanupIncomplete { request_id: ID, attempt: 0 }]
    );
    assert_eq!(request.wake_at(), None);
}

// Catches: a transport that closes an attempt without a terminal report leaving the consumer
// waiting forever.
#[test]
fn an_attempt_closed_without_an_outcome_fails() {
    let mut request = start(RequestPolicy { max_retry: 0, ..policy() });
    let effects = request.handle(LlmRequestEvent::AttemptClosed { attempt: 0 }, at(5));
    assert_eq!(finished(&effects).unwrap().clone().unwrap_err().kind, ApiErrorKind::Connection);
    assert!(request.is_settled());
}

#[test]
fn an_expired_acceptance_budget_never_authorizes_the_first_exchange() {
    let (_request, effects) = LlmRequest::start(
        ID,
        LLM,
        RequestPolicy { total_timeout: Some(Duration::ZERO), ..policy() },
        ("<think>", "</think>"),
        at(0),
    );
    assert!(
        !effects.iter().any(|effect| matches!(effect, LlmRequestEffect::LlmRequestStart { .. }))
    );
    let error = finished(&effects).expect("terminal outcome").as_ref().unwrap_err();
    assert_eq!(error.phase, Some(TimeoutPhase::Total));
}

// Catches: the time to first token latched on the failed attempt's metadata-only chunk, or
// measured from the retry instead of from the request: the caller waited through the failure and
// the backoff before the first token.
#[test]
fn time_to_first_token_spans_the_retry() {
    use lk_agents_core::telemetry::provider::LlmMetricState;

    let metadata: WireChunk = serde_json::from_value(serde_json::json!({
        "id": "c",
        "choices": [{"delta": {"extra_content": {"livekit": {"inference_tier_billed": "s"}}}}],
    }))
    .unwrap();
    let mut fold = LlmMetricState::new(0.0);
    let mut observe = |effects: Vec<LlmRequestEffect>, now: Ts| {
        let secs = now.saturating_duration_since(at(0)).as_secs_f64();
        for effect in effects {
            if let LlmRequestEffect::Deliver(chunk) = effect {
                fold.observe(&chunk, || secs);
            }
        }
    };
    let mut request = start(policy());
    reading(&mut request, 0, at(0));
    observe(request.handle(LlmRequestEvent::Chunk { attempt: 0, chunk: metadata }, at(10)), at(10));
    request.handle(LlmRequestEvent::Failed { attempt: 0, failure: failure() }, at(20));
    request.handle(LlmRequestEvent::AttemptClosed { attempt: 0 }, at(20));
    let retry = request.wake_at().unwrap();
    assert_eq!(request.handle(LlmRequestEvent::Wake, retry), vec![started(1)]);
    reading(&mut request, 1, retry);
    let first = retry.saturating_add(Duration::from_millis(30));
    observe(request.handle(LlmRequestEvent::Chunk { attempt: 1, chunk: text("hi") }, first), first);
    let ttft = fold.ttft().expect("the retry answered");
    assert!((ttft - 0.15).abs() < 1e-9, "ttft {ttft}");
}
