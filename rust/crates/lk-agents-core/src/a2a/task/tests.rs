#![cfg(test)]

use super::*;
use crate::a2a::{DebounceSpec, PolicySpec};

fn machine(policy: Policy) -> TaskMachine {
    let mut wire = a2a::Message::new(a2a::Role::User, vec![a2a::Part::text("answer")]);
    wire.message_id = "m".into();
    TaskMachine::new(
        "lane".into(),
        Message::from_a2a(&wire, 0.0, None).unwrap(),
        "receipt".into(),
        None,
        &TurnOptions {
            policy: PolicySpec::All(policy),
            debounce: DebounceSpec::All(Duration::ZERO),
            wait_timeout: Some(Duration::from_secs(2)),
        },
    )
}

fn at(m: &mut TaskMachine, now: Duration, result: Result<Completion, StorageError>) -> Vec<Effect> {
    let id = m.pending.unwrap();
    m.handle(Event::Completed { id, result }, now)
}

fn operation(effects: Vec<Effect>) -> Operation {
    match effects.into_iter().next() {
        Some(Effect::Execute { operation, .. }) => operation,
        other => panic!("expected an operation, got {other:?}"),
    }
}

fn step(m: &mut TaskMachine, result: Completion) -> Operation {
    operation(at(m, Duration::ZERO, Ok(result)))
}

fn row(status: TaskState) -> TaskRow {
    TaskWrite { status, ..machine(Policy::Queue).submitted() }.row("lane")
}

#[test]
fn reject_resend_observes_the_original_task_after_a_lost_submit_race() {
    let mut m = machine(Policy::Reject);
    m.handle(Event::Start, Duration::ZERO);
    assert!(matches!(step(&mut m, Completion::Task(None)), Operation::Open { wait: false, .. }));
    assert!(matches!(
        step(&mut m, Completion::Opened { closed: false, picked_up: false }),
        Operation::Submit(_)
    ));
    assert!(matches!(step(&mut m, Completion::Submitted(false)), Operation::CloseRun));
    assert!(matches!(step(&mut m, Completion::Done), Operation::FindReceipt(id) if id == "m"));
}

#[test]
fn concurrent_handler_waits_for_a_conditional_task_claim() {
    let mut m = machine(Policy::Concurrent);
    m.handle(Event::Start, Duration::ZERO);
    step(&mut m, Completion::Task(None));
    step(&mut m, Completion::Submitted(true));
    assert!(matches!(
        step(&mut m, Completion::Done),
        Operation::Open { exclusive: false, pickup: true, .. }
    ));
    assert!(matches!(
        step(&mut m, Completion::Opened { closed: false, picked_up: false }),
        Operation::CloseRun
    ));
    assert!(matches!(step(&mut m, Completion::Done), Operation::ReadTask(_)));
}

#[test]
fn batch_preserves_the_callers_continuation_target() {
    let mut m = machine(Policy::Batch);
    m.phase = Phase::DrainPickup;
    m.pending = Some(1);
    let mut wire = m.message.to_a2a();
    wire.task_id = Some("waiting".into());
    m.rows = vec![InboxRow {
        id: 1,
        task_id: "receipt".into(),
        message: wire,
        received_at: 10,
        message_id: "m".into(),
        lane: "lane".into(),
    }];
    assert!(matches!(
        step(&mut m, Completion::Picked(Some(TaskState::Submitted))),
        Operation::RecoverOrphans { .. }
    ));
    let Operation::Run { batch, task_ids, .. } = step(&mut m, Completion::Done) else { panic!() };
    assert_eq!(batch[0].task_id.as_deref(), Some("waiting"));
    assert_eq!(task_ids, ["receipt"]);
}

#[test]
fn storage_failure_in_the_turn_abandons_the_run_and_fails_the_receipt() {
    let mut m = machine(Policy::Queue);
    m.phase = Phase::Run;
    m.pending = Some(4);
    let abandon =
        operation(at(&mut m, Duration::ZERO, Err(StorageError::Unavailable("offline".into()))));
    assert!(matches!(abandon, Operation::Abandon { outcome: RunOutcome::Failed, .. }));
    assert!(matches!(step(&mut m, Completion::Done), Operation::FailTask {
        task_id, reason: REASON_STORAGE_UNAVAILABLE } if task_id == "receipt"));
}

#[test]
fn a_lost_commit_acknowledgment_is_resolved_and_never_answered_failed() {
    let mut m = machine(Policy::Queue);
    m.phase = Phase::Run;
    m.pending = Some(1);
    let commit =
        step(&mut m, Completion::Ran { answer_task: "receipt".into(), output: Some("hi".into()) });
    assert!(matches!(commit, Operation::Commit { answer_task: Some(_), .. }));
    let resolve =
        operation(at(&mut m, Duration::ZERO, Err(StorageError::Uncertain("lost".into()))));
    assert!(matches!(resolve, Operation::ResolveLater { task_id } if task_id == "receipt"));
    assert!(matches!(step(&mut m, Completion::Done), Operation::ReadTask(_)));
    // the database is still unreachable: the task as this worker left it, still working
    let effects = at(&mut m, Duration::ZERO, Err(StorageError::Unavailable("down".into())));
    let [Effect::Finished(Ok(answer))] = effects.as_slice() else { panic!("{effects:?}") };
    assert_eq!((answer.task.as_ref(), answer.reason), (None, None), "never a failure");
}

#[test]
fn every_typed_storage_error_on_the_lane_claim_has_its_outcome() {
    // reject: the holder's task is named
    let mut m = machine(Policy::Reject);
    m.phase = Phase::Open;
    m.pending = Some(1);
    let rejected = StorageError::Rejected { running_task_id: None };
    assert!(matches!(
        operation(at(&mut m, Duration::ZERO, Err(rejected.clone()))),
        Operation::RunningTask
    ));
    let effects = at(&mut m, Duration::ZERO, Ok(Completion::RunningTask(Some("AST_r".into()))));
    assert!(matches!(&effects[..], [Effect::Finished(Err(StorageError::Rejected {
        running_task_id: Some(id) }))] if id == "AST_r"));
    // a waiter whose wait ran out fails its task lease-timeout
    let mut m = machine(Policy::Queue);
    m.phase = Phase::Open;
    m.pending = Some(1);
    assert!(matches!(
        operation(at(&mut m, Duration::ZERO, Err(rejected))),
        Operation::FailTask { reason: REASON_LEASE_TIMEOUT, .. }
    ));
    // anything else ends the request with its typed error
    for error in [
        StorageError::Unavailable("down".into()),
        StorageError::NotFound("AT_x".into()),
        StorageError::LeaseLost("taken".into()),
        StorageError::VersionConflict { lane: "lane".into(), expected: 1 },
        StorageError::Uncertain("lost".into()),
    ] {
        let mut m = machine(Policy::Queue);
        m.phase = Phase::Open;
        m.pending = Some(1);
        let effects = at(&mut m, Duration::ZERO, Err(error.clone()));
        assert!(matches!(&effects[..], [Effect::Finished(Err(e))] if *e == error), "{error}");
    }
}

/// Review of cf6a3c721: a resend that finds its task `submitted` must not watch it forever when
/// the waiter that wrote it died before taking the lane.
#[test]
fn a_submitted_task_whose_waiter_died_is_failed_once_the_wait_ran_out() {
    let mut m = machine(Policy::Queue);
    m.handle(Event::Start, Duration::ZERO);
    assert!(matches!(
        step(&mut m, Completion::Task(Some(row(TaskState::Submitted)))),
        Operation::ReadTask(_)
    ));
    let mut now = Duration::ZERO;
    let mut effects = at(&mut m, now, Ok(Completion::Task(Some(row(TaskState::Submitted)))));
    let mut opened = false;
    for _ in 0..64 {
        assert!(effects.is_empty(), "{effects:?}");
        now = m.wake_at().expect("polling");
        let next = operation(m.handle(Event::Wake, now));
        if let Operation::Open { wait: false, .. } = next {
            opened = true;
            break;
        }
        assert!(matches!(next, Operation::ReadTask(_)));
        effects = at(&mut m, now, Ok(Completion::Task(Some(row(TaskState::Submitted)))));
    }
    assert!(opened, "the stale submitted task was watched forever");
    assert!(now >= Duration::from_secs(2), "not before the wait ran out");
    assert!(matches!(
        step(&mut m, Completion::Opened { closed: false, picked_up: false }),
        Operation::FailTask { reason: REASON_WORKER_LOST, .. }
    ));
    assert!(matches!(step(&mut m, Completion::Done), Operation::RecoverOrphans { .. }));
    assert!(matches!(step(&mut m, Completion::Done), Operation::CloseRun));
    assert!(matches!(step(&mut m, Completion::Done), Operation::ReadTask(_)));
    let effects = at(&mut m, now, Ok(Completion::Task(Some(row(TaskState::Failed)))));
    assert!(matches!(&effects[..], [Effect::Finished(Ok(Answer { task: Some(_), .. }))]));
}

#[test]
fn batch_waits_past_the_polling_budget_without_answering_submitted() {
    let mut m = machine(Policy::Batch);
    m.poll(Duration::ZERO);
    m.poll(Duration::from_secs(120));
    assert_eq!(m.wake_at(), Some(Duration::from_secs(121)));
    assert!(matches!(m.phase, Phase::Poll { stale: false, .. }));
}

#[test]
fn stale_operation_completion_cannot_admit_replacement_work() {
    let mut m = machine(Policy::Queue);
    let issued = m.handle(Event::Start, Duration::ZERO);
    assert_eq!(issued.len(), 1);
    assert!(
        m.handle(Event::Completed { id: 0, result: Ok(Completion::Task(None)) }, Duration::ZERO)
            .is_empty()
    );
    assert!(matches!(step(&mut m, Completion::Task(None)), Operation::Submit(_)));
}
