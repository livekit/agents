use super::*;
use crate::delegation::remote::{InteractionId, RemoteInteraction};
use serde_json::json;

fn target() -> RemoteTaskRef {
    RemoteTaskRef { task_id: "booking".into(), context_id: "trip".into() }
}

fn interaction() -> InteractionState {
    let mut interaction = RemoteInteraction::new(InteractionId {
        delegation_id: "delegate".into(),
        tool_id: "travel".into(),
        parent_call_id: "canonical-call".into(),
        endpoint: "https://travel.test".into(),
    })
    .unwrap();
    interaction.target_task(target()).unwrap();
    interaction.snapshot()
}

fn task(state: &str) -> a2a::Task {
    serde_json::from_value(json!({"id":"booking", "contextId":"trip",
        "status":{"state":state}}))
    .unwrap()
}

fn finished(step: CancellationStep) -> CancellationResult {
    let CancellationStep::Finished(result) = step else {
        panic!("the observation must settle this explicit control")
    };
    *result
}

#[test]
fn undispatched_cancellation_keeps_its_original_request_and_sdk_body() {
    let request: a2a::CancelTaskRequest = serde_json::from_value(json!({
        "id":"booking", "metadata":{"reason":"the person asked"}, "tenant":"tenant"
    }))
    .unwrap();
    let control = prepare(
        &interaction(),
        &target(),
        Some(RemoteRecovery::Dispatch {
            request_id: "original-request".into(),
            request: Box::new(A2ARequest::Cancel(request.clone())),
            streaming: false,
            input_request: None,
        }),
        "unused-new-request".into(),
    )
    .unwrap();
    let Control::Dispatch { request_id, task, request: resumed } = control else {
        panic!("a proved-undispatched control can retain its exact operation")
    };
    assert_eq!(request_id, "original-request");
    assert_eq!(resumed, request);
    assert_eq!(task, TaskRef { id: "booking".into(), context_id: "trip".into() });
    let mut changed = target();
    changed.context_id = "another-context".into();
    assert!(prepare(&interaction(), &changed, None, "fresh".into()).is_err());
}

#[test]
fn a_sent_control_only_observes_and_working_is_not_confirmed_cancellation() {
    let control = prepare(
        &interaction(),
        &target(),
        Some(RemoteRecovery::GetTask(target())),
        "unused-new-request".into(),
    )
    .unwrap();
    assert!(matches!(control, Control::Observe { .. }));
    let observed = task("TASK_STATE_WORKING");
    let result = finished(control.observed(Ok(observed.clone())));
    assert_eq!(result.observation.unwrap(), observed);
    assert!(matches!(result.confirmation, Err(ClientError::Protocol(_))));
    assert!(result.confirmed_task.is_none(), "unconfirmed work cannot settle queued cancels");
}

#[test]
fn a_lost_control_ack_can_succeed_only_after_reconciliation_observes_canceled() {
    let control = prepare(&interaction(), &target(), None, "cancel-request".into()).unwrap();
    let CancellationStep::Operation(control) = control
        .observed(Err(ClientError::Protocol("reply lost after applying cancellation".into())))
    else {
        panic!("an uncertain write must be observed, never replaced")
    };
    assert!(matches!(&control, Control::Reconcile { task, .. }
        if task.id == "booking" && task.context_id == "trip"));
    let observed = task("TASK_STATE_CANCELED");
    let result = finished(control.observed(Ok(observed.clone())));
    assert_eq!(result.observation.unwrap(), observed);
    assert_eq!(result.confirmation, Ok(()));
    let confirmed = result.confirmed_task.unwrap();
    assert_eq!(queued_cancel(&confirmed, &target()), Ok(()));
    let mut different = target();
    different.task_id = "another-task".into();
    assert!(queued_cancel(&confirmed, &different).is_err());
}

#[test]
fn a_refusal_keeps_its_protocol_data_and_a_failed_read_stays_with_the_original_call() {
    for read_fails in [false, true] {
        let error = ClientError::Rpc {
            code: a2a::error_code::TASK_NOT_CANCELABLE,
            message: "Booking is protected".into(),
            data: Some(json!({"stage":"booking"})),
        };
        let control = prepare(&interaction(), &target(), None, "cancel-request".into()).unwrap();
        let CancellationStep::Operation(control) = control.observed(Err(error.clone())) else {
            panic!("a refused or uncertain write is reconciled through the same task")
        };
        let observation = if read_fails {
            Err(ClientError::Protocol("the observation failed".into()))
        } else {
            Ok(task("TASK_STATE_INPUT_REQUIRED"))
        };
        let result = finished(control.observed(observation.clone()));
        assert_eq!(result.confirmation, Err(error));
        assert_eq!(
            result.observation, observation,
            "a control refusal cannot discard the original call's observed state or failure"
        );
    }
}
