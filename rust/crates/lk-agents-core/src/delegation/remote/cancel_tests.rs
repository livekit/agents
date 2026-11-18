use super::*;
use crate::delegation::remote::{InteractionId, RemoteInteraction};
use serde_json::json;

fn target() -> RemoteTaskRef {
    RemoteTaskRef { task_id: "booking".into(), context_id: "trip".into() }
}

fn interaction() -> InteractionState {
    let mut interaction = RemoteInteraction::new(InteractionId {
        remote_id: "remote".into(),
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

fn scope() -> RemoteControlScope {
    RemoteControlScope {
        project_origin: "https://travel.test".into(),
        project_issuer: "project-key-id".into(),
        caller_identity: "delegate".into(),
        thread_id: "trip".into(),
    }
}

fn retained(id: &str, proof: Option<RemoteControlScope>) -> RemoteAgentState {
    let mut remote = RemoteAgentState::new("https://travel.test".into());
    let send: a2a::SendMessageRequest = serde_json::from_value(json!({"message":{
        "messageId":format!("initial-{id}"),"role":"ROLE_USER","contextId":"trip",
        "parts":[{"text":"Book a seat"}]
    }}))
    .unwrap();
    super::super::journal::apply(
        "travel",
        &mut remote,
        lk_agents_abi::delegation::remote::RemoteInput::Prepare {
            request_id: "start".into(),
            request: A2ARequest::Send(Box::new(send)),
            streaming: false,
            origin: None,
            input_request: None,
            control_scope: proof.map(Box::new),
        },
    )
    .unwrap();
    super::super::journal::apply(
        "travel",
        &mut remote,
        lk_agents_abi::delegation::remote::RemoteInput::DispatchStarted {
            request_id: "start".into(),
        },
    )
    .unwrap();
    let observed: a2a::Task = serde_json::from_value(json!({"id":id,"contextId":"trip",
        "status":{"state":"TASK_STATE_WORKING"}}))
    .unwrap();
    super::super::journal::apply(
        "travel",
        &mut remote,
        lk_agents_abi::delegation::remote::RemoteInput::Response {
            request_id: "start".into(),
            response: Box::new(lk_agents_abi::delegation::remote::A2AResponse::Send(
                a2a::SendMessageResponse::Task(observed),
            )),
        },
    )
    .unwrap();
    remote
}

fn control_failure(remote: &mut RemoteAgentState, request_id: &str, failure: RemoteFailure) {
    use lk_agents_abi::delegation::remote::RemoteInput;
    for input in [
        RemoteInput::Prepare {
            request_id: request_id.into(),
            request: A2ARequest::Cancel(a2a::CancelTaskRequest {
                id: "booking".into(),
                metadata: None,
                tenant: None,
            }),
            streaming: false,
            origin: None,
            input_request: None,
            control_scope: None,
        },
        RemoteInput::DispatchStarted { request_id: request_id.into() },
        RemoteInput::Error { request_id: request_id.into(), error: failure },
    ] {
        super::super::journal::apply("travel", remote, input).unwrap();
    }
}

#[test]
fn fresh_retry_requires_a_later_intent_after_the_exact_typed_refusal() {
    use lk_agents_abi::session::SessionRecovery;
    let mut remote = retained("booking", Some(scope()));
    control_failure(
        &mut remote,
        "refused",
        RemoteFailure::Rpc {
            error: a2a::JsonRpcError {
                code: a2a::error_code::TASK_NOT_CANCELABLE,
                message: "protected".into(),
                data: Some(json!({"stage":"booking"})),
            },
        },
    );
    assert_eq!(retry_after(&remote, &target()).as_deref(), Some("refused"));
    let old = SessionRecovery::Cancellation { task: target(), retry_after: None };
    assert_eq!(
        crate::session::remote::recover(&remote, &old).unwrap(),
        Some(RemoteRecovery::GetTask(target())),
        "an overlapping intent cannot repeat the write"
    );
    let later =
        SessionRecovery::Cancellation { task: target(), retry_after: Some("refused".into()) };
    assert_eq!(crate::session::remote::recover(&remote, &later).unwrap(), None);
    control_failure(
        &mut remote,
        "new-intent",
        RemoteFailure::Rpc {
            error: a2a::JsonRpcError {
                code: a2a::error_code::INTERNAL_ERROR,
                message: "operation outcome is unknown".into(),
                data: None,
            },
        },
    );
    assert_eq!(retry_after(&remote, &target()), None);
    assert_eq!(
        crate::session::remote::recover(&remote, &later).unwrap(),
        Some(RemoteRecovery::GetTask(target())),
        "old refusal cannot authorize a third request"
    );
}

#[test]
fn transport_or_unknown_rpc_errors_never_become_definite_refusal_from_their_words() {
    for failure in [
        RemoteFailure::Protocol { message: "TASK_NOT_CANCELABLE protected refused".into() },
        RemoteFailure::Rpc {
            error: a2a::JsonRpcError {
                code: a2a::error_code::INTERNAL_ERROR,
                message: "TASK_NOT_CANCELABLE".into(),
                data: None,
            },
        },
        RemoteFailure::Rpc {
            error: a2a::JsonRpcError {
                code: -33077,
                message: "refused cancellation".into(),
                data: None,
            },
        },
    ] {
        let mut remote = retained("booking", Some(scope()));
        control_failure(&mut remote, "uncertain", failure);
        assert_eq!(retry_after(&remote, &target()), None);
    }
}

#[test]
fn idle_authority_is_exactly_the_original_task_project_principal_and_bound_thread() {
    let remote = retained("booking", Some(scope()));
    authorize(&remote, &target(), &scope()).unwrap();
    for changed in [
        RemoteControlScope { project_origin: "https://other-project.test".into(), ..scope() },
        RemoteControlScope { project_issuer: "another-project-key".into(), ..scope() },
        RemoteControlScope { caller_identity: "another-principal".into(), ..scope() },
        RemoteControlScope { thread_id: "foreign-thread".into(), ..scope() },
    ] {
        assert!(authorize(&remote, &target(), &changed).is_err());
    }
    let custom = retained("booking", None);
    assert!(
        authorize(&custom, &target(), &scope()).is_err(),
        "same origin and task context do not upgrade custom auth to worker authority"
    );
    let mut mixed = remote;
    let unrelated = retained("custom", None);
    mixed.tasks.extend(unrelated.tasks);
    assert!(
        authorize(
            &mixed,
            &RemoteTaskRef { task_id: "custom".into(), context_id: "trip".into() },
            &scope()
        )
        .is_err(),
        "proof belongs to one task, never the entire endpoint"
    );
}

#[test]
fn waiting_for_a_fence_does_not_turn_old_intent_into_a_retry_after_new_refusal() {
    let mut remote = retained("booking", Some(scope()));
    let earlier = remote.cancellation_evidence(&target());
    assert_eq!(retry_after_evidence(earlier.as_ref()), None);
    control_failure(
        &mut remote,
        "refused-while-waiting",
        RemoteFailure::Rpc {
            error: a2a::JsonRpcError {
                code: a2a::error_code::TASK_NOT_CANCELABLE,
                message: "protected".into(),
                data: None,
            },
        },
    );
    assert!(matches!(
        route(
            &remote,
            &target(),
            false,
            Some(&scope()),
            retry_after_evidence(earlier.as_ref()).as_deref()
        )
        .unwrap(),
        CancellationRoute::Retained { recovery: Some(RemoteRecovery::GetTask(_)) }
    ));
    let observed = remote.cancellation_evidence(&target());
    let retry = retry_after_evidence(observed.as_ref());
    assert_eq!(retry.as_deref(), Some("refused-while-waiting"));
    assert!(matches!(
        route(&remote, &target(), false, Some(&scope()), retry.as_deref()).unwrap(),
        CancellationRoute::Retained { recovery: None }
    ));
}
