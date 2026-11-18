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
            control_scope: None,
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
            control_scope: None,
        },
    );
    request
}

#[test]
fn cancellation_recovery_reuses_the_exact_prepared_control_until_dispatch_may_have_started() {
    let mut remote = working("TASK_STATE_WORKING");
    let target = SessionRecovery::Cancellation { task: task(), retry_after: None };
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
            control_scope: None,
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
    assert!(
        recover(&remote, &SessionRecovery::Cancellation { task: foreign, retry_after: None })
            .is_err()
    );
    let mut unknown = task();
    unknown.task_id = "another-task".into();
    assert!(
        recover(&remote, &SessionRecovery::Cancellation { task: unknown, retry_after: None })
            .is_err()
    );
    assert_eq!(
        recover(
            &working("TASK_STATE_CANCELED"),
            &SessionRecovery::Cancellation { task: task(), retry_after: None }
        )
        .unwrap(),
        Some(RemoteRecovery::GetTask(task()))
    );
    assert!(
        recover(
            &working("TASK_STATE_COMPLETED"),
            &SessionRecovery::Cancellation { task: task(), retry_after: None }
        )
        .is_err()
    );
    let mut completed_after_prepare = working("TASK_STATE_WORKING");
    prepare_cancel(&mut completed_after_prepare);
    observe(&mut completed_after_prepare, "TASK_STATE_COMPLETED");
    assert_eq!(
        recover(
            &completed_after_prepare,
            &SessionRecovery::Cancellation { task: task(), retry_after: None }
        )
        .unwrap(),
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
    assert_eq!(
        recover(&remote, &SessionRecovery::Cancellation { task: task(), retry_after: None })
            .unwrap(),
        None
    );
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
                    control_scope: None,
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

#[test]
fn observed_cancellation_blocks_a_prepared_task_message_before_its_first_dispatch() {
    let mut remote = working("TASK_STATE_WORKING");
    apply(
        &mut remote,
        RemoteInput::Prepare {
            request_id: "correction".into(),
            request: A2ARequest::Send(
                serde_json::from_value(json!({
                    "message":{"messageId":"correction", "role":"ROLE_USER", "taskId":"booking",
                        "contextId":"trip", "parts":[{"text":"Actually Saturday"}]}
                }))
                .unwrap(),
            ),
            streaming: true,
            origin: None,
            input_request: None,
            control_scope: None,
        },
    );
    observe(&mut remote, "TASK_STATE_CANCELED");
    let mut state = SessionSnapshot::new("session".into(), "execution".into(), "desk".into());
    state.remote_agents.insert("booking".into(), remote);
    let before = state.clone();
    assert!(
        super::apply(
            &mut state,
            RemoteCommand {
                remote_id: "booking".into(),
                endpoint: "https://booking.test".into(),
                input: RemoteInput::DispatchStarted { request_id: "correction".into() },
            }
        )
        .is_err()
    );
    assert_eq!(state, before, "terminal work cannot cross a prepared message's I/O boundary");
}

#[test]
fn settling_compacts_old_replies_but_keeps_the_send_of_an_unresolved_call() {
    use lk_agents_abi::llm::tool_executor::{REMOTE_REPORT_KEY, RemoteReport};
    use lk_agents_abi::{ChatItem, FunctionCall, FunctionCallOutput};
    let mut snapshot = SessionSnapshot::new("session".into(), "execution".into(), "front".into());
    let root = snapshot.root_frame_id.clone();
    let call = |id: &str, report: RemoteReport| FunctionCall {
        id: id.into(),
        call_id: id.into(),
        name: "remote_booking".into(),
        arguments: "{}".into(),
        created_at: std::time::UNIX_EPOCH,
        extra: [(REMOTE_REPORT_KEY.into(), serde_json::to_value(report).unwrap())]
            .into_iter()
            .collect(),
        group_id: None,
    };
    let resolved = call("resolved", RemoteReport::Completed);
    let history = &mut snapshot.frames.get_mut(&root).unwrap().history;
    history.push(ChatItem::FunctionCall(call("pending", RemoteReport::Pending)));
    history.push(ChatItem::FunctionCallOutput(FunctionCallOutput::new(
        resolved.output_id(),
        "resolved",
        "Booked",
        false,
        std::time::UNIX_EPOCH,
    )));
    history.push(ChatItem::FunctionCall(resolved));
    let command = |input| RemoteCommand {
        remote_id: "booking".into(),
        endpoint: "https://booking.test".into(),
        input,
    };
    let exchanges = crate::delegation::remote::journal::RETAINED_SETTLED_EVENTS;
    for n in 0..exchanges {
        let id = format!("send_{n}");
        let call_id = match n {
            0 => "pending",
            1 => "resolved",
            _ => "other",
        };
        let origin = ToolCallRef { agent_id: root.value.clone(), call_id: call_id.into() };
        super::apply(
            &mut snapshot,
            command(RemoteInput::Prepare {
                request_id: id.clone(),
                request: A2ARequest::Send(
                    serde_json::from_value(json!({
                        "message":{"messageId":id, "role":"ROLE_USER", "parts":[{"text":"Book"}]}
                    }))
                    .unwrap(),
                ),
                streaming: false,
                origin: Some(origin),
                input_request: None,
                control_scope: None,
            }),
        )
        .unwrap();
        super::apply(
            &mut snapshot,
            command(RemoteInput::DispatchStarted { request_id: id.clone() }),
        )
        .unwrap();
        super::apply(&mut snapshot, command(RemoteInput::Response {
            request_id: id.clone(),
            response: Box::new(A2AResponse::Send(serde_json::from_value(json!({"message":{
                "messageId":format!("reply_{n}"), "role":"ROLE_AGENT", "parts":[{"text":"Done"}]
            }})).unwrap())),
        }))
        .unwrap();
        super::apply(&mut snapshot, command(RemoteInput::Ended { request_id: id })).unwrap();
    }
    let remote = &snapshot.remote_agents["booking"];
    let settled = remote.events.iter().filter(|event| event.request_id != "send_0").count();
    assert!(settled <= exchanges, "settled history is bounded");
    assert!(!remote.requests.contains_key("send_1"), "a resolved call's send can go");
    let pending = ToolCallRef { agent_id: root.value.clone(), call_id: "pending".into() };
    assert!(
        matches!(recovery(remote, &pending).unwrap(), Some(RemoteRecovery::Observed(_))),
        "the unresolved call still recovers its original reply"
    );
}
