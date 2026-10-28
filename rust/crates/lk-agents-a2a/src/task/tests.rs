#![cfg(test)]

use super::*;
use crate::{DebounceSpec, PolicySpec};

fn machine(policy: Policy) -> TaskMachine {
    let mut wire = a2a::Message::new(a2a::Role::User, vec![a2a::Part::text("answer")]);
    wire.message_id = "m".into();
    TaskMachine::new(
        "thread".into(),
        "lane".into(),
        Message::from_a2a(&wire, 0.0, None).unwrap(),
        "receipt".into(),
        None,
        &TurnOptions {
            policy: PolicySpec::All(policy),
            debounce: DebounceSpec::All(Duration::ZERO),
            ..Default::default()
        },
    )
}

fn step(m: &mut TaskMachine, result: Completion) -> Operation {
    let id = m.pending.unwrap();
    let effects = m.handle(Event::Completed { id, result: Ok(result) }, Duration::ZERO);
    let Effect::Execute { operation, .. } = effects.into_iter().next().unwrap() else { panic!() };
    operation
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
fn storage_failure_after_admission_fails_the_receipt_as_an_orphan() {
    let mut m = machine(Policy::Queue);
    m.phase = Phase::Run;
    m.pending = Some(4);
    let effects = m.handle(
        Event::Completed { id: 4, result: Err(StorageError::Unavailable("offline".into())) },
        Duration::ZERO,
    );
    assert!(matches!(&effects[0], Effect::Execute { operation: Operation::FailOrphan {
        task_id, reason: REASON_STORAGE_UNAVAILABLE }, .. } if task_id == "receipt"));
}

#[test]
fn batch_waits_past_the_polling_budget_without_answering_submitted() {
    let mut m = machine(Policy::Batch);
    m.poll(Duration::ZERO);
    m.poll(Duration::from_secs(120));
    assert_eq!(m.wake_at(), Some(Duration::from_secs(121)));
    assert!(!matches!(m.phase, Phase::Finished));
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
