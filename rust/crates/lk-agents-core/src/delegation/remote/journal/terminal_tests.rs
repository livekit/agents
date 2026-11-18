use super::*;
use lk_agents_abi::delegation::remote::{A2ARequest, RemoteInput};
use serde_json::{Value, json};

fn target() -> RemoteTaskRef {
    RemoteTaskRef { task_id: "booking".into(), context_id: "trip".into() }
}
fn start(state: &mut RemoteAgentState, id: &str, get: bool) {
    let request = if get {
        A2ARequest::Get(a2a::GetTaskRequest {
            id: "booking".into(),
            history_length: None,
            tenant: None,
        })
    } else {
        A2ARequest::Send(
            serde_json::from_value(json!({
                "message":{"messageId":id,"role":"ROLE_USER","parts":[{"text":"Book"}]}
            }))
            .unwrap(),
        )
    };
    super::super::apply(
        "booking",
        state,
        RemoteInput::Prepare {
            request_id: id.into(),
            request,
            streaming: !get,
            origin: None,
            input_request: None,
            control_scope: None,
        },
    )
    .unwrap();
    super::super::apply("booking", state, RemoteInput::DispatchStarted { request_id: id.into() })
        .unwrap();
}
fn event(state: &mut RemoteAgentState, id: &str, value: Value) {
    let response = if state.requests[id].method == a2a::methods::GET_TASK {
        A2AResponse::Task(serde_json::from_value(value).unwrap())
    } else {
        A2AResponse::Stream(serde_json::from_value(value).unwrap())
    };
    super::super::apply(
        "booking",
        state,
        RemoteInput::Response { request_id: id.into(), response: Box::new(response) },
    )
    .unwrap();
}
fn accepted(state: &RemoteAgentState) -> a2a::Task {
    let Some(a2a::SendMessageResponse::Task(task)) = observed(state, &target()).unwrap() else {
        panic!("the task has accepted terminal evidence")
    };
    task
}
fn working() -> Value {
    json!({"task":{"id":"booking","contextId":"trip",
        "metadata":{"https://livekit.io/a2a/ext/agents/v1/progress":true,"https://livekit.io/a2a/ext/agents/v1/verbatim":"Old progress"},
        "status":{"state":"TASK_STATE_WORKING"}}})
}

#[test]
fn stale_working_and_input_required_remain_raw_without_reopening_an_accepted_result() {
    for stale in ["TASK_STATE_WORKING", "TASK_STATE_INPUT_REQUIRED"] {
        let mut state = RemoteAgentState::new("https://booking.test".into());
        start(&mut state, "original", false);
        event(&mut state, "original", working());
        event(
            &mut state,
            "original",
            json!({"task":{"id":"booking","contextId":"trip",
            "status":{"state":"TASK_STATE_COMPLETED"},
            "artifacts":[{"artifactId":"answer","parts":[{"text":"Booked"}]}]}}),
        );
        let sequence = state.tasks["booking"].terminal_event;
        start(&mut state, "earlier-poll", true);
        event(
            &mut state,
            "earlier-poll",
            json!({"id":"booking","contextId":"trip",
            "metadata":{"https://livekit.io/a2a/ext/agents/v1/verbatim":"Stale words"}, "status":{"state":stale}}),
        );
        let task = accepted(&state);
        assert_eq!(task.status.state, a2a::TaskState::Completed);
        assert!(task.metadata.is_none(), "terminal omission cannot inherit working flags");
        assert_eq!(task.artifacts.unwrap()[0].parts[0].as_text(), Some("Booked"));
        assert_eq!(state.tasks["booking"].terminal_event, sequence);
        assert!(state.tasks["booking"].input_requests.is_empty());
        assert!(
            matches!(&state.events.last().unwrap().payload, A2AJournalPayload::Response(response)
            if matches!(response.as_ref(), A2AResponse::Task(task) if !task.status.state.is_terminal()))
        );
        super::super::validate("booking", &state).unwrap();
    }
}

#[test]
fn explicit_terminal_flags_come_from_that_exact_status_event() {
    let mut state = RemoteAgentState::new("https://booking.test".into());
    start(&mut state, "original", false);
    event(&mut state, "original", working());
    event(
        &mut state,
        "original",
        json!({"statusUpdate":{"taskId":"booking","contextId":"trip",
        "metadata":{"https://livekit.io/a2a/ext/agents/v1/verbatim":"Booked"}, "status":{"state":"TASK_STATE_COMPLETED",
            "message":{"messageId":"final","role":"ROLE_AGENT","parts":[{"text":"Booked"}]}}}}),
    );
    let task = accepted(&state);
    assert_eq!(
        task.metadata.as_ref().unwrap()["https://livekit.io/a2a/ext/agents/v1/verbatim"],
        json!("Booked")
    );
    let normalized = crate::a2a::client::ReplyState::default()
        .accept(a2a::StreamResponse::Task(task))
        .unwrap()
        .unwrap();
    assert_eq!(normalized.metadata["livekit/verbatim"], json!("Booked"));
    assert_eq!(normalized.message.unwrap().parts[0].as_text(), Some("Booked"));
}

#[test]
fn a_terminal_snapshot_after_a_stream_gap_does_not_promote_cached_partial_artifacts() {
    let mut state = RemoteAgentState::new("https://booking.test".into());
    start(&mut state, "broken-stream", false);
    event(&mut state, "broken-stream", working());
    event(
        &mut state,
        "broken-stream",
        json!({"artifactUpdate":{"taskId":"booking","contextId":"trip",
        "artifact":{"artifactId":"answer","parts":[{"text":"Partial"}]}, "lastChunk":false}}),
    );
    super::super::apply(
        "booking",
        &mut state,
        RemoteInput::Ended { request_id: "broken-stream".into() },
    )
    .unwrap();
    start(&mut state, "poll", true);
    event(
        &mut state,
        "poll",
        json!({"id":"booking","contextId":"trip","status":{"state":"TASK_STATE_COMPLETED"}}),
    );
    assert_eq!(
        state.tasks["booking"].task.artifacts.as_ref().unwrap()[0].parts[0].as_text(),
        Some("Partial"),
        "raw accepted chunks remain available to inspection"
    );
    assert!(
        accepted(&state).artifacts.unwrap().is_empty(),
        "missing content cannot become a final result"
    );
    assert!(matches!(
        super::super::recovery(&state, "broken-stream").unwrap(),
        lk_agents_abi::delegation::remote::RemoteRecovery::Observed(_)
    ));
}

#[test]
fn an_uninterrupted_terminal_stream_keeps_two_intentionally_equal_appends() {
    let mut state = RemoteAgentState::new("https://booking.test".into());
    start(&mut state, "original", false);
    event(&mut state, "original", working());
    for append in [false, true] {
        event(
            &mut state,
            "original",
            json!({"artifactUpdate":{"taskId":"booking","contextId":"trip",
            "artifact":{"artifactId":"answer","parts":[{"text":"Equal"}]},"append":append}}),
        );
    }
    event(
        &mut state,
        "original",
        json!({"statusUpdate":{"taskId":"booking","contextId":"trip",
        "status":{"state":"TASK_STATE_COMPLETED"}}}),
    );
    let task = accepted(&state);
    assert_eq!(
        task.artifacts.unwrap()[0].parts.len(),
        2,
        "payload equality is not a replay cursor"
    );
}

#[test]
fn a_cancel_answered_by_completion_keeps_the_intent_and_accepts_the_real_result() {
    let mut state = RemoteAgentState::new("https://booking.test".into());
    start(&mut state, "start", false);
    event(&mut state, "start", working());
    super::super::apply(
        "booking",
        &mut state,
        RemoteInput::Prepare {
            request_id: "cancel".into(),
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
    )
    .unwrap();
    super::super::apply(
        "booking",
        &mut state,
        RemoteInput::DispatchStarted { request_id: "cancel".into() },
    )
    .unwrap();
    let completed: a2a::Task = serde_json::from_value(json!({"id":"booking","contextId":"trip",
        "status":{"state":"TASK_STATE_COMPLETED"},
        "artifacts":[{"artifactId":"receipt","parts":[{"text":"Booked Friday"}]}]}))
    .unwrap();
    super::super::apply(
        "booking",
        &mut state,
        RemoteInput::Response {
            request_id: "cancel".into(),
            response: Box::new(A2AResponse::Task(completed.clone())),
        },
    )
    .unwrap();
    let task = &state.tasks["booking"];
    assert!(task.cancel_requested, "the person's intent is kept beside the observed outcome");
    assert_eq!(task.task.status.state, a2a::TaskState::Completed);
    let accepted = accepted(&state);
    assert_eq!(accepted.status.state, a2a::TaskState::Completed);
    let receipt = &accepted.artifacts.as_ref().unwrap()[0];
    assert_eq!(receipt.parts[0].as_text(), Some("Booked Friday"));
    let control =
        crate::delegation::remote::cancel::operation(&target(), None, "cancel".into()).unwrap();
    let crate::delegation::remote::cancel::CancellationStep::Finished(result) =
        control.observed(Ok(completed.clone()))
    else {
        panic!("a task answer settles the control")
    };
    assert_eq!(result.observation.unwrap(), completed, "late success is delivered as observed");
    assert!(result.confirmation.is_err(), "completion is not a confirmed cancellation");
    assert!(result.confirmed_task.is_none());
}
