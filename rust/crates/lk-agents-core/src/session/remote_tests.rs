use super::*;
use lk_agents_abi::delegation::remote::{A2AResponse, RemoteFailure};
use serde_json::json;

fn task() -> RemoteTaskRef {
    RemoteTaskRef { task_id: "booking".into(), context_id: "trip".into() }
}

fn apply(remote: &mut RemoteAgentState, input: RemoteInput) {
    crate::delegation::remote::journal::apply("booking", remote, input).unwrap();
}

fn observe(remote: &mut RemoteAgentState, state: &str) {
    apply(
        remote,
        RemoteInput::Response {
            request_id: "start".into(),
            response: Box::new(A2AResponse::Stream(
                serde_json::from_value(json!({"task":{"id":"booking", "contextId":"trip",
                "status":{"state":state}}}))
                .unwrap(),
            )),
        },
    );
}

fn working(state: &str) -> RemoteAgentState {
    let mut remote = RemoteAgentState::new("https://booking.test".into());
    apply(&mut remote, RemoteInput::Prepare {
        request_id: "start".into(), request: A2ARequest::Send(serde_json::from_value(json!({
            "message":{"messageId":"business", "role":"ROLE_USER", "parts":[{"text":"Book a room"}]}
        })).unwrap()), streaming: true, origin: None, input_request: None,
    });
    apply(&mut remote, RemoteInput::DispatchStarted { request_id: "start".into() });
    observe(&mut remote, state);
    remote
}

fn prepare_cancel(remote: &mut RemoteAgentState) -> A2ARequest {
    let request = A2ARequest::Cancel(
        serde_json::from_value(json!({
            "id":"booking", "metadata":{"reason":"the human requested cancellation"}
        }))
        .unwrap(),
    );
    apply(
        remote,
        RemoteInput::Prepare {
            request_id: "original-cancel".into(),
            request: request.clone(),
            streaming: false,
            origin: None,
            input_request: None,
        },
    );
    request
}

#[test]
fn cancellation_recovery_reuses_the_exact_prepared_control_until_dispatch_may_have_started() {
    let mut remote = working("TASK_STATE_WORKING");
    let target = SessionRecovery::Cancellation(task());
    assert_eq!(recover(&remote, &target).unwrap(), None);
    let request = prepare_cancel(&mut remote);
    apply(
        &mut remote,
        RemoteInput::Prepare {
            request_id: "later-read".into(),
            request: A2ARequest::Get(serde_json::from_value(json!({"id":"booking"})).unwrap()),
            streaming: false,
            origin: None,
            input_request: None,
        },
    );
    let saved = remote.clone();
    assert_eq!(
        recover(&remote, &target).unwrap(),
        Some(RemoteRecovery::Dispatch {
            request_id: "original-cancel".into(),
            request: Box::new(request),
            streaming: false,
            input_request: None,
        })
    );
    assert_eq!(remote, saved, "inspection neither sends nor replaces a recorded control");
    apply(&mut remote, RemoteInput::DispatchStarted { request_id: "original-cancel".into() });
    apply(
        &mut remote,
        RemoteInput::Error {
            request_id: "original-cancel".into(),
            error: RemoteFailure::Transport {
                kind: lk_agents_abi::errors::ErrorKind::Server,
                message: "receipt lost".into(),
                status_code: None,
            },
        },
    );
    assert_eq!(recover(&remote, &target).unwrap(), Some(RemoteRecovery::GetTask(task())));
}

#[test]
fn cancellation_recovery_checks_the_saved_context_and_never_starts_a_terminal_control() {
    let remote = working("TASK_STATE_WORKING");
    let mut foreign = task();
    foreign.context_id = "another-conversation".into();
    assert!(recover(&remote, &SessionRecovery::Cancellation(foreign)).is_err());
    let mut unknown = task();
    unknown.task_id = "another-task".into();
    assert!(recover(&remote, &SessionRecovery::Cancellation(unknown)).is_err());
    assert_eq!(
        recover(&working("TASK_STATE_CANCELED"), &SessionRecovery::Cancellation(task())).unwrap(),
        Some(RemoteRecovery::GetTask(task()))
    );
    assert!(
        recover(&working("TASK_STATE_COMPLETED"), &SessionRecovery::Cancellation(task())).is_err()
    );
    let mut completed_after_prepare = working("TASK_STATE_WORKING");
    prepare_cancel(&mut completed_after_prepare);
    observe(&mut completed_after_prepare, "TASK_STATE_COMPLETED");
    assert_eq!(
        recover(&completed_after_prepare, &SessionRecovery::Cancellation(task())).unwrap(),
        Some(RemoteRecovery::GetTask(task()))
    );
}

#[test]
fn an_observed_cancellation_needs_no_control_but_still_requires_exact_task_identity() {
    assert!(cancel_required(&working("TASK_STATE_WORKING"), &task()).unwrap());
    let canceled = working("TASK_STATE_CANCELED");
    assert!(!cancel_required(&canceled, &task()).unwrap());
    let mut foreign = task();
    foreign.context_id = "another-context".into();
    assert!(cancel_required(&canceled, &foreign).is_err());
    assert!(cancel_required(&working("TASK_STATE_COMPLETED"), &task()).is_err());
}

#[test]
fn terminal_observation_between_cancel_recovery_and_dispatch_prevents_a_control_send() {
    let mut remote = working("TASK_STATE_WORKING");
    assert_eq!(recover(&remote, &SessionRecovery::Cancellation(task())).unwrap(), None);
    observe(&mut remote, "TASK_STATE_COMPLETED");
    let mut state = SessionSnapshot::new("session".into(), "execution".into(), "desk".into());
    state.remote_agents.insert("booking".into(), remote);
    let before = state.clone();
    assert!(
        super::apply(
            &mut state,
            RemoteCommand {
                remote_id: "booking".into(),
                endpoint: "https://booking.test".into(),
                input: RemoteInput::Prepare {
                    request_id: "too-late".into(),
                    request: A2ARequest::Cancel(
                        serde_json::from_value(json!({"id":"booking"})).unwrap()
                    ),
                    streaming: false,
                    origin: None,
                    input_request: None,
                },
            }
        )
        .is_err()
    );
    assert_eq!(state, before, "fresh controls are rechecked at native Prepare admission");

    let mut remote = working("TASK_STATE_WORKING");
    prepare_cancel(&mut remote);
    observe(&mut remote, "TASK_STATE_CANCELED");
    state.remote_agents.insert("booking".into(), remote);
    let before = state.clone();
    assert!(
        super::apply(
            &mut state,
            RemoteCommand {
                remote_id: "booking".into(),
                endpoint: "https://booking.test".into(),
                input: RemoteInput::DispatchStarted { request_id: "original-cancel".into() },
            }
        )
        .is_err()
    );
    assert_eq!(state, before, "terminal work cannot cross an unsent cancellation's I/O boundary");
}
