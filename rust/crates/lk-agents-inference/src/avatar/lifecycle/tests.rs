//! `avatar_creation_lifecycle` / `creation_adoption_vectors`: the avatar machine stepped
//! directly on supplied time.

use std::time::Duration;

use super::*;
use crate::attempts::ATTEMPT_CLEANUP_TIMEOUT;
use AvatarSessionEffect as Fx;
use AvatarSessionEvent as Ev;

const AVATAR: ResourceId = ResourceId(3);

fn at(ms: u64) -> Ts {
    Ts::from_nanos(ms * 1_000_000)
}

fn policy() -> RequestPolicy {
    RequestPolicy {
        connect_timeout: Some(Duration::from_secs(10)),
        write_timeout: Some(Duration::from_secs(10)),
        read_timeout: Some(Duration::from_secs(10)),
        total_timeout: None,
        max_retry: 2,
        retry_interval: Duration::from_secs(1),
    }
}

fn session(id: &str) -> ProviderSession {
    ProviderSession {
        session_id: Some(id.into()),
        provider_session_id: Some(format!("p-{id}")),
        terminate_token: Some("t".into()),
        sample_rate: 24_000,
    }
}

fn create(request: u64, attempt: u32) -> Fx {
    Fx::AvatarSessionCreate { request: AvatarRequestId(request), attempt, avatar: AVATAR }
}

fn terminate(request: u64, s: &ProviderSession) -> Fx {
    Fx::AvatarSessionTerminate { request: AvatarRequestId(request), attempt: 0, avatar: AVATAR, session: s.clone() }
}

fn created(request: u64, attempt: u32, s: &ProviderSession) -> Ev {
    Ev::Created { request: AvatarRequestId(request), attempt, session: s.clone() }
}

fn closed(request: u64, attempt: u32) -> Ev {
    Ev::AttemptClosed { request: AvatarRequestId(request), attempt }
}

fn written(avatar: &mut AvatarSessionLifecycle, request: u64, attempt: u32, ms: u64) {
    let (request, at_ms) = (AvatarRequestId(request), at(ms));
    avatar.handle(Ev::Connected { request, attempt }, at_ms);
    avatar.handle(Ev::RequestWritten { request, attempt }, at_ms);
}

fn reset() -> AttemptFailure {
    AttemptFailure::Connection("reset".into())
}

/// A lifecycle whose first create (request 1, attempt 0) is in flight.
fn creating() -> AvatarSessionLifecycle {
    let mut avatar = AvatarSessionLifecycle::new(AVATAR, policy());
    assert_eq!(avatar.handle(Ev::Create, at(0)), vec![create(1, 0)], "reserved and launched at once");
    avatar
}

// Catches: a second start launching a second billable create, the refusal disturbing the first,
// or a started instance starting again.
#[test]
fn a_second_caller_is_refused_and_the_first_creation_is_untouched() {
    let mut avatar = creating();
    assert_eq!(avatar.handle(Ev::Create, at(1)), vec![Fx::CreateRefused]);
    let s = session("a");
    assert_eq!(avatar.handle(created(1, 0, &s), at(2)), vec![Fx::CreateFinished(Ok(s.clone()))]);
    assert_eq!(avatar.handle(Ev::Create, at(3)), vec![Fx::CreateRefused], "a delivery is pending");
    assert_eq!(avatar.handle(Ev::Adopted, at(3)), vec![]);
    assert_eq!(avatar.handle(closed(1, 0), at(4)), vec![]);
    assert!(avatar.is_settled());
    assert_eq!(avatar.handle(Ev::Create, at(5)), vec![Fx::CreateRefused], "one instance is one avatar");
}

// Catches: a failed creation keeping its reservation, so the instance can never start.
#[test]
fn a_failed_creation_releases_its_reservation() {
    let mut avatar = creating();
    let rejected = AttemptFailure::Rejected { status: 400, message: "bad image".into(), request_id: None };
    let effects = avatar.handle(Ev::Failed { request: AvatarRequestId(1), attempt: 0, failure: rejected }, at(1));
    assert!(matches!(effects.as_slice(), [Fx::CreateFinished(Err(CreateFailure::Api(e)))] if e.status_code == Some(400)));
    assert_eq!(avatar.handle(Ev::Create, at(2)), vec![create(2, 0)]);
}

// Catches: a retry minting a new operation (a second provider session), or a stale attempt's
// answer of the same operation treated as a stray session and terminated under the caller.
#[test]
fn retries_keep_the_operation_and_any_attempts_answer_settles_it() {
    let mut avatar = creating();
    let effects = avatar.handle(Ev::Failed { request: AvatarRequestId(1), attempt: 0, failure: reset() }, at(0));
    assert!(matches!(effects.as_slice(), [Fx::AttemptFailed { request: AvatarRequestId(1), attempt: 0, .. }]));
    assert_eq!(avatar.handle(Ev::Wake, at(100)), vec![create(1, 1)], "same operation, next attempt");
    // Attempt 0's answer was already queued when it failed: the gateway replays one result per
    // operation, so it is the operation's answer and attempt 1 is stopped.
    let s = session("a");
    let effects = avatar.handle(created(1, 0, &s), at(101));
    assert_eq!(
        effects,
        vec![Fx::AvatarRequestCancel { request: AvatarRequestId(1), attempt: 1 }, Fx::CreateFinished(Ok(s.clone()))]
    );
    avatar.handle(Ev::Adopted, at(102));
    assert_eq!(avatar.handle(created(1, 1, &s), at(103)), vec![], "the replay is the held session");
}

// Catches: a creation whose caller left launching more attempts, or its late success adopted
// or leaked instead of terminated.
#[test]
fn a_late_success_after_the_caller_left_is_retired() {
    let mut avatar = creating();
    written(&mut avatar, 1, 0, 2);
    assert_eq!(avatar.handle(Ev::Cancel, at(5)), vec![], "the sent attempt is left to answer");
    let s = session("late");
    let effects = avatar.handle(created(1, 0, &s), at(900));
    assert_eq!(
        effects,
        vec![Fx::Retiring { session: s.clone(), reason: Retirement::CallerGone }, terminate(2, &s)]
    );
    assert!(!effects.iter().any(|e| matches!(e, Fx::CreateFinished(_))));
    let done = avatar.handle(Ev::Terminated { request: AvatarRequestId(2), attempt: 0 }, at(950));
    assert_eq!(done, vec![Fx::SessionTerminated(s)]);
    avatar.handle(closed(1, 0), at(951));
    avatar.handle(closed(2, 0), at(952));
    assert!(avatar.is_settled());
    assert_eq!(avatar.handle(Ev::Create, at(953)), vec![create(3, 0)], "nothing was adopted: it may start again");
}

// Catches: a caller cancelled during backoff still getting a retry launched.
#[test]
fn a_caller_leaving_during_backoff_ends_the_creation() {
    let mut avatar = creating();
    avatar.handle(Ev::Failed { request: AvatarRequestId(1), attempt: 0, failure: reset() }, at(0));
    assert_eq!(avatar.handle(Ev::Cancel, at(50)), vec![]);
    assert_eq!(avatar.handle(Ev::Wake, at(100)), vec![]);
    avatar.handle(closed(1, 0), at(101));
    assert!(avatar.is_settled());
}

// Catches: a session delivered to a caller that is gone left unowned.
#[test]
fn a_refused_delivery_keeps_cleanup_with_the_sender() {
    let mut avatar = creating();
    let s = session("a");
    avatar.handle(created(1, 0, &s), at(1));
    let effects = avatar.handle(Ev::DeliveryRefused, at(2));
    assert_eq!(effects, vec![Fx::Retiring { session: s.clone(), reason: Retirement::Refused }, terminate(2, &s)]);
}

// Catches: close during creation leaving the caller waiting, or the session created after it
// kept alive.
#[test]
fn close_during_creation_settles_the_caller_and_retires_the_result() {
    let mut avatar = creating();
    written(&mut avatar, 1, 0, 0);
    assert_eq!(avatar.handle(Ev::Close, at(1)), vec![Fx::CreateFinished(Err(CreateFailure::Closed))]);
    assert!(!avatar.is_settled(), "the running attempt still owes its answer");
    let s = session("a");
    let effects = avatar.handle(created(1, 0, &s), at(2));
    assert_eq!(effects, vec![Fx::Retiring { session: s.clone(), reason: Retirement::Closed }, terminate(2, &s)]);
    assert_eq!(avatar.handle(Ev::Create, at(3)), vec![Fx::CreateRefused]);
}

// Catches: an unsent create kept running after its caller left, or a stopped attempt that never
// releases reported as clean or waited on forever (decision 14: one typed observation, the
// runtime keeps the attempt).
#[test]
fn a_missed_cleanup_deadline_is_reported_once() {
    let mut avatar = creating();
    assert_eq!(
        avatar.handle(Ev::Cancel, at(0)),
        vec![Fx::AvatarRequestCancel { request: AvatarRequestId(1), attempt: 0 }],
        "the gateway cannot have received it: nothing to retire"
    );
    let due = at(0).saturating_add(ATTEMPT_CLEANUP_TIMEOUT);
    assert_eq!(avatar.wake_at(), Some(due));
    assert_eq!(avatar.handle(Ev::Wake, due), vec![Fx::CleanupIncomplete { request: AvatarRequestId(1), attempt: 0 }]);
    assert!(avatar.is_settled());
    assert_eq!(avatar.wake_at(), None);
    // A session the stopped attempt reports anyway is still terminated.
    let s = session("very-late");
    let effects = avatar.handle(created(1, 0, &s), at(9_000));
    assert_eq!(effects, vec![Fx::Retiring { session: s.clone(), reason: Retirement::Late }, terminate(2, &s)]);
}

// Catches: a failed terminate forgetting the session (it bills on), or a repeat close not
// retrying it.
#[test]
fn close_terminates_the_held_session_and_a_failure_keeps_it_held() {
    let mut avatar = creating();
    let s = session("a");
    avatar.handle(created(1, 0, &s), at(1));
    avatar.handle(Ev::Adopted, at(1));
    avatar.handle(closed(1, 0), at(1));
    assert_eq!(avatar.handle(Ev::Close, at(2)), vec![terminate(2, &s)]);
    let rejected = AttemptFailure::Rejected { status: 403, message: "no".into(), request_id: None };
    let effects = avatar.handle(Ev::Failed { request: AvatarRequestId(2), attempt: 0, failure: rejected }, at(3));
    assert!(matches!(effects.as_slice(), [Fx::TerminateFailed { session, .. }] if *session == s));
    avatar.handle(closed(2, 0), at(4));
    assert!(avatar.is_settled());
    assert_eq!(avatar.handle(Ev::Close, at(5)), vec![terminate(3, &s)]);
    assert_eq!(avatar.handle(Ev::Terminated { request: AvatarRequestId(3), attempt: 0 }, at(6)), vec![Fx::SessionTerminated(s)]);
}

// Catches: a terminate request sent without the identity it needs, or an unterminable session
// silently dropped.
#[test]
fn a_session_without_terminate_identity_is_released_with_a_report() {
    let mut avatar = creating();
    let s = ProviderSession { terminate_token: None, ..session("a") };
    avatar.handle(created(1, 0, &s), at(1));
    assert_eq!(
        avatar.handle(Ev::DeliveryRefused, at(2)),
        vec![Fx::Retiring { session: s.clone(), reason: Retirement::Refused }, Fx::Unterminable(s)]
    );
}

// Catches: a create the gateway received abandoned when its read timeout fires after close: the
// gateway still makes the session and nobody ends it (DST avatar_sessions_are_never_leaked, seed 0).
#[test]
fn a_received_create_cut_by_its_timeout_after_close_is_asked_again_and_retired() {
    let read = RequestPolicy { read_timeout: Some(Duration::from_secs(2)), max_retry: 0, ..policy() };
    let mut avatar = AvatarSessionLifecycle::new(AVATAR, read);
    avatar.handle(Ev::Create, at(0));
    written(&mut avatar, 1, 0, 10);
    assert_eq!(avatar.handle(Ev::Close, at(20)), vec![Fx::CreateFinished(Err(CreateFailure::Closed))]);
    let effects = avatar.handle(Ev::Wake, avatar.wake_at().unwrap());
    assert!(
        matches!(effects.as_slice(), [Fx::AvatarRequestCancel { attempt: 0, .. }, Fx::AttemptFailed { attempt: 0, .. }]),
        "the timed-out attempt is stopped and the operation retried: {effects:?}"
    );
    assert_eq!(avatar.handle(Ev::Wake, avatar.wake_at().unwrap()), vec![create(1, 1)], "same operation identity");
    let s = session("a");
    let effects = avatar.handle(created(1, 1, &s), at(3_000));
    assert_eq!(effects, vec![Fx::Retiring { session: s.clone(), reason: Retirement::Closed }, terminate(2, &s)]);
}

// Catches: a create whose last attempt lost its answer after the gateway received it failing its
// caller and forgetting the operation, so the session the gateway made bills on.
#[test]
fn a_create_failed_after_it_was_received_is_retired_once_its_caller_has_the_error() {
    let mut avatar = AvatarSessionLifecycle::new(AVATAR, RequestPolicy { max_retry: 0, ..policy() });
    avatar.handle(Ev::Create, at(0));
    written(&mut avatar, 1, 0, 10);
    let effects = avatar.handle(Ev::Failed { request: AvatarRequestId(1), attempt: 0, failure: reset() }, at(500));
    assert!(matches!(effects.as_slice(), [Fx::CreateFinished(Err(CreateFailure::Api(_)))]), "{effects:?}");
    assert_eq!(avatar.handle(Ev::Wake, avatar.wake_at().unwrap()), vec![create(1, 1)], "asked again under its identity");
    let s = session("a");
    let effects = avatar.handle(created(1, 1, &s), at(2_000));
    assert_eq!(effects, vec![Fx::Retiring { session: s.clone(), reason: Retirement::Late }, terminate(2, &s)]);
}

// Catches: a retirement that cannot learn the operation's outcome waited on forever, or ended
// silently (decision 14: one typed report; a later answer is still terminated).
#[test]
fn a_retirement_unanswered_by_its_deadline_is_reported_once() {
    let mut avatar = creating();
    written(&mut avatar, 1, 0, 10);
    avatar.handle(Ev::Close, at(20));
    let deadline = at(20).saturating_add(RETIREMENT_TIMEOUT);
    let mut effects = Vec::new();
    while let Some(due) = avatar.wake_at().filter(|due| *due <= deadline) {
        for effect in avatar.handle(Ev::Wake, due) {
            if let Fx::AvatarRequestCancel { request, attempt } = effect {
                avatar.handle(Ev::AttemptClosed { request, attempt }, due);
            }
            effects.push(effect);
        }
    }
    let reports: Vec<_> = effects.iter().filter(|e| matches!(e, Fx::RetirementIncomplete { .. })).collect();
    assert_eq!(reports, vec![&Fx::RetirementIncomplete { request: AvatarRequestId(1) }]);
    let s = session("very-late");
    let effects = avatar.handle(created(1, 0, &s), deadline.saturating_add(Duration::from_secs(1)));
    assert_eq!(effects, vec![Fx::Retiring { session: s.clone(), reason: Retirement::Late }, terminate(2, &s)]);
}
