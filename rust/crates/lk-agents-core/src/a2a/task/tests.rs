#![cfg(test)]

use super::*;
use crate::a2a::{DebounceSpec, PolicySpec};

fn machine(policy: Policy) -> TaskMachine {
    continuing(policy, None)
}

/// A machine for message `m`, continuing `target` when given.
fn continuing(policy: Policy, target: Option<&str>) -> TaskMachine {
    let mut wire = a2a::Message::new(a2a::Role::User, vec![a2a::Part::text("answer")]);
    wire.message_id = "m".into();
    wire.task_id = target.map(str::to_owned);
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
        0.0,
    )
}

fn opened(closed: bool, picked_up: bool) -> Completion {
    Completion::Opened { closed, picked_up, history_end: 0.0 }
}

/// Drives `m`, its orphans just recovered, through a turn whose pending question is `pending` and
/// whose handler answers `output` at once: the operation after the settlement.
fn turn(m: &mut TaskMachine, pending: Option<TaskRow>, output: HandlerOutput) -> Operation {
    m.phase = Phase::Run { path: Path::Main, step: RunStep::Recover };
    m.pending = Some(1);
    let mut op = step(m, Completion::Done);
    loop {
        op = match op {
            Operation::ReadPending(_) => step(m, Completion::Task(pending.clone())),
            Operation::Stage(staged) if staged.end.is_some() => return step(m, Completion::Done),
            Operation::Stage(_) | Operation::Handle(_) => step(m, Completion::Done),
            Operation::Next(_) => {
                step(m, Completion::Event(HandlerEvent::Output(Box::new(output.clone()))))
            }
            other => panic!("unexpected {other:?}"),
        }
    }
}

fn answered(text: &str) -> HandlerOutput {
    HandlerOutput { output: Some(text.into()), ..HandlerOutput::default() }
}

/// `waiting`, asking a question of a conversation tool's call.
fn question() -> TaskRow {
    let mut row =
        TaskWrite { task_id: "waiting".into(), ..row(TaskState::InputRequired).into() }.row("lane");
    row.message_id = Some("asked".into());
    row.internal = Some(TaskInternal {
        ask: Some(lk_agents_abi::storage::Ask {
            question: "which day?".into(),
            call_id: Some("call_1".into()),
            ..Default::default()
        }),
        streams: Vec::new(),
    });
    row
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
    assert!(matches!(
        step(&mut m, Completion::Task(None)),
        Operation::Open { access: Access::Exclusive { wait: false, .. }, .. }
    ));
    assert!(matches!(step(&mut m, opened(false, false)), Operation::Submit(_)));
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
        Operation::Open { access: Access::Concurrent, pickup: true, .. }
    ));
    assert!(matches!(step(&mut m, opened(false, false)), Operation::CloseRun));
    assert!(matches!(step(&mut m, Completion::Done), Operation::ReadTask(_)));
}

#[test]
fn batch_preserves_the_callers_continuation_target() {
    let mut m = machine(Policy::Batch);
    m.phase = Phase::Run { path: Path::Drain, step: RunStep::Pickup };
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
        Operation::ReadWorking
    ));
    let pending = step(&mut m, Completion::Tasks(Vec::new()));
    assert_eq!(pending, Operation::ReadPending(Some("waiting".into())));
}

#[test]
fn storage_failure_in_the_turn_abandons_the_run_and_fails_the_receipt() {
    let mut m = machine(Policy::Queue);
    m.phase = Phase::Run { path: Path::Main, step: RunStep::Recover };
    m.pending = Some(4);
    assert_eq!(step(&mut m, Completion::Done), Operation::ReadPending(None));
    let abandon =
        operation(at(&mut m, Duration::ZERO, Err(StorageError::Unavailable("offline".into()))));
    assert!(matches!(abandon, Operation::Abandon { outcome: RunOutcome::Failed, .. }));
    assert!(matches!(step(&mut m, Completion::Done), Operation::Pickup(t) if t == "receipt"));
    assert!(matches!(
        step(&mut m, Completion::Picked(Some(TaskState::Working))),
        Operation::FailOrphans { reason: REASON_STORAGE_UNAVAILABLE, failed }
            if failed.len() == 1 && failed[0].0 == "receipt"
    ));
}

/// A dead holder's `working` rows fail worker-lost, except the message's own task; a named task
/// that already ended is not failed.
#[test]
fn recovery_fails_exactly_the_orphans_and_never_an_ended_task() {
    let mut m = machine(Policy::Queue);
    m.phase = Phase::Run { path: Path::Main, step: RunStep::Open };
    m.pending = Some(1);
    assert!(matches!(step(&mut m, opened(false, true)), Operation::ReadWorking));
    let working = |id: &str| TaskRow { task_id: id.into(), ..row(TaskState::Working) };
    let fail = step(&mut m, Completion::Tasks(vec![working("receipt"), working("dead")]));
    assert!(
        matches!(&fail, Operation::FailOrphans { reason: REASON_WORKER_LOST, failed }
            if failed.iter().map(|f| f.0.as_str()).eq(["dead"])),
        "{fail:?}"
    );
    assert_eq!(step(&mut m, Completion::Done), Operation::ReadPending(None));
    // the lease ran out: the receipt fails lease-timeout, unless a client already ended it
    let mut m = machine(Policy::Queue);
    m.phase = Phase::Run { path: Path::Main, step: RunStep::Open };
    m.pending = Some(1);
    let rejected = StorageError::Rejected { running_task_id: None };
    assert!(matches!(operation(at(&mut m, Duration::ZERO, Err(rejected))), Operation::Pickup(_)));
    assert!(
        matches!(step(&mut m, Completion::Picked(None)), Operation::ReadTask(t) if t == "receipt")
    );
}

#[test]
fn a_lost_commit_acknowledgment_is_resolved_and_never_answered_failed() {
    let mut m = machine(Policy::Queue);
    let commit = turn(&mut m, None, answered("hi"));
    assert!(matches!(commit, Operation::Commit { answer_task: Some(_), .. }));
    let resolve =
        operation(at(&mut m, Duration::ZERO, Err(StorageError::Uncertain("lost".into()))));
    assert!(matches!(resolve, Operation::ResolveLater { task_id, .. } if task_id == "receipt"));
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
    m.phase = Phase::Run { path: Path::Main, step: RunStep::Open };
    m.pending = Some(1);
    let rejected = StorageError::Rejected { running_task_id: None };
    assert!(matches!(
        operation(at(&mut m, Duration::ZERO, Err(rejected.clone()))),
        Operation::ReadRuns(_)
    ));
    let runs = vec![
        LaneRun { task_id: Some("AST_ended".into()), interrupted: false },
        LaneRun { task_id: Some("AST_r".into()), interrupted: true },
    ];
    let effects = at(&mut m, Duration::ZERO, Ok(Completion::Runs(runs)));
    assert!(matches!(&effects[..], [Effect::Finished(Err(StorageError::Rejected {
        running_task_id: Some(id) }))] if id == "AST_r"));
    // a waiter whose wait ran out fails its task lease-timeout
    let mut m = machine(Policy::Queue);
    m.phase = Phase::Run { path: Path::Main, step: RunStep::Open };
    m.pending = Some(1);
    assert!(matches!(operation(at(&mut m, Duration::ZERO, Err(rejected))), Operation::Pickup(_)));
    assert!(matches!(
        step(&mut m, Completion::Picked(Some(TaskState::Submitted))),
        Operation::FailOrphans { reason: REASON_LEASE_TIMEOUT, .. }
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
        m.phase = Phase::Run { path: Path::Main, step: RunStep::Open };
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
    let mut reopened = false;
    for _ in 0..64 {
        assert!(effects.is_empty(), "{effects:?}");
        now = m.wake_at().expect("polling");
        let next = operation(m.handle(Event::Wake, now));
        if let Operation::Open { access: Access::Exclusive { wait: false, .. }, .. } = next {
            reopened = true;
            break;
        }
        assert!(matches!(next, Operation::ReadTask(_)));
        effects = at(&mut m, now, Ok(Completion::Task(Some(row(TaskState::Submitted)))));
    }
    assert!(reopened, "the stale submitted task was watched forever");
    assert!(now >= Duration::from_secs(2), "not before the wait ran out");
    assert!(matches!(step(&mut m, opened(false, false)), Operation::Pickup(_)));
    assert!(matches!(
        step(&mut m, Completion::Picked(Some(TaskState::Submitted))),
        Operation::FailOrphans { reason: REASON_WORKER_LOST, .. }
    ));
    assert!(matches!(step(&mut m, Completion::Done), Operation::ReadWorking));
    assert!(matches!(step(&mut m, Completion::Tasks(Vec::new())), Operation::CloseRun));
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

/// A2A_SEED=1337: a batch receipt whose every request gave up on a transient read failure while
/// watching was never drained.
#[test]
fn a_watcher_keeps_watching_through_a_transient_read_failure() {
    let mut m = machine(Policy::Batch);
    m.phase = Phase::Observe;
    m.pending = Some(1);
    let effects = at(&mut m, Duration::ZERO, Err(StorageError::Unavailable("down".into())));
    assert!(effects.is_empty(), "{effects:?}");
    let at_ = m.wake_at().expect("polls again");
    assert!(matches!(operation(m.handle(Event::Wake, at_)), Operation::ReadTask(_)));
}

/// The runtime door's known finding (answer_is_an_outcome/received_message_is_drained, Inbox): a
/// drain whose lane another holder took is not answered `Rejected` and does not give up its
/// pending receipt; it waits and drains it once the lane frees.
#[test]
fn a_refused_drain_waits_and_drains_later() {
    let mut m = machine(Policy::Batch);
    m.phase = Phase::Inbox;
    m.pending = Some(1);
    let row = InboxRow {
        id: 1,
        lane: "lane".into(),
        task_id: "receipt".into(),
        message_id: "m".into(),
        message: m.message.to_a2a(),
        received_at: 0,
    };
    assert!(matches!(step(&mut m, Completion::Inbox(vec![row.clone()])), Operation::Open { .. }));
    let refused = at(&mut m, Duration::ZERO, Err(StorageError::Rejected { running_task_id: None }));
    assert!(refused.is_empty(), "never answered Rejected: {refused:?}");
    let wake = m.wake_at().expect("waits");
    assert!(matches!(operation(m.handle(Event::Wake, wake)), Operation::ReadTask(_)));
    assert!(matches!(
        step(&mut m, Completion::Task(Some(row_of(TaskState::Submitted)))),
        Operation::ReadInbox
    ));
    assert!(matches!(step(&mut m, Completion::Inbox(vec![row])), Operation::Open { .. }));
}

/// The runtime door's known finding (message_executes_once, Concurrent): a concurrent message
/// whose task a client canceled before its turn does not run.
#[test]
fn a_concurrent_message_canceled_before_its_turn_does_not_run() {
    let mut m = machine(Policy::Concurrent);
    m.phase = Phase::Run { path: Path::Main, step: RunStep::Open };
    m.pending = Some(1);
    assert!(matches!(step(&mut m, opened(false, false)), Operation::CloseRun));
    assert!(matches!(step(&mut m, Completion::Done), Operation::ReadTask(_)));
}

fn row_of(status: TaskState) -> TaskRow {
    row(status)
}

/// A2A_SEED=154 239 (message_executes_once, Continuation): a continuation whose commit was
/// rejected was answered by the task it continued while its own receipt stayed `working`, later
/// failing worker-lost on its own. The rejected save fails the message's own task, the answer
/// names it, and the question it tried to answer stays pending.
#[test]
fn a_rejected_continuation_fails_its_own_task_and_leaves_the_question_pending() {
    for error in [
        StorageError::VersionConflict { lane: "lane".into(), expected: 1 },
        StorageError::Invalid("sqlite: disk I/O error".into()),
    ] {
        let mut m = continuing(Policy::Queue, Some("waiting"));
        let commit = turn(&mut m, Some(question()), answered("Tuesday"));
        assert!(matches!(commit, Operation::Commit { answer_task: Some(t), .. } if t == "waiting"));
        let pickup = operation(at(&mut m, Duration::ZERO, Err(error)));
        assert!(matches!(&pickup, Operation::Pickup(t) if t == "receipt"), "{pickup:?}");
        let fail = step(&mut m, Completion::Picked(Some(TaskState::Working)));
        assert!(
            matches!(&fail, Operation::FailOrphans { reason: REASON_STORAGE_UNAVAILABLE, .. }),
            "{fail:?}"
        );
        assert!(matches!(step(&mut m, Completion::Done), Operation::ReadTask(t) if t == "receipt"));
        let effects =
            at(&mut m, Duration::ZERO, Ok(Completion::Task(Some(row(TaskState::Failed)))));
        let [Effect::Finished(Ok(answer))] = effects.as_slice() else { panic!("{effects:?}") };
        assert_eq!(answer.task_id, "receipt");
    }
}

/// A2A_SEED=68 88 130 (one_question_pending, Lane): a reply naming a question that a turn ahead
/// already merged found no pending question and asked its own beside the newer one still waiting.
/// The reply answers the question waiting now.
#[test]
fn a_reply_to_a_merged_question_answers_the_one_waiting_now() {
    let mut m = continuing(Policy::Queue, Some("merged"));
    m.phase = Phase::Run { path: Path::Main, step: RunStep::Recover };
    m.pending = Some(1);
    assert_eq!(step(&mut m, Completion::Done), Operation::ReadPending(Some("merged".into())));
    assert_eq!(step(&mut m, Completion::Task(None)), Operation::ReadPending(None));
    let Operation::Stage(staged) = step(&mut m, Completion::Task(Some(question()))) else {
        panic!("the waiting question is settled before the handler")
    };
    let waiting = staged.transitions.iter().find(|t| t.task_id == "waiting").expect("settled");
    assert_eq!(waiting.status, TaskState::Completed);
    assert!(matches!(&staged.edits[..], [crate::a2a::HistoryEdit::Answer { .. }]));
}

/// A2A_SEED=181 350 616 (commit_is_what_the_run_staged, Concurrent): overlapping concurrent turns
/// both claimed one pending question with no lease to make the claim exclusive; the second's
/// terminal write was dropped and its client answered the first's outcome. A concurrent turn
/// answers on its own task and leaves the question to the lane's exclusive turns.
#[test]
fn a_concurrent_turn_leaves_a_pending_question_alone() {
    let mut m = continuing(Policy::Concurrent, Some("waiting"));
    m.phase = Phase::Run { path: Path::Main, step: RunStep::Recover };
    m.pending = Some(1);
    let Operation::Handle(handle) = step(&mut m, Completion::Done) else {
        panic!("a concurrent turn reads no pending question")
    };
    assert_eq!(handle.task_id, "receipt");
}

/// A2A_SEED=2 (request_answered_once, Inbox): a batch message whose drained turn asked a question
/// was watched forever: only a terminal row answered the request. A question is the turn's answer.
#[test]
fn a_watched_task_that_asks_a_question_is_answered() {
    let mut m = machine(Policy::Batch);
    m.phase = Phase::Observe;
    m.pending = Some(1);
    let effects =
        at(&mut m, Duration::ZERO, Ok(Completion::Task(Some(row(TaskState::InputRequired)))));
    let [Effect::Finished(Ok(answer))] = effects.as_slice() else { panic!("{effects:?}") };
    assert_eq!(answer.task.as_ref().map(|t| t.status), Some(TaskState::InputRequired));
}
