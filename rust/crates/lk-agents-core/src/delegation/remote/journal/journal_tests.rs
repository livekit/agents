use super::*;
use lk_agents_abi::a2a_protocol as a2a;
use serde_json::json;

fn send(id: &str, text: &str, task: Option<(&str, &str)>) -> RemoteInput {
    let (context, task) = task.map_or((None, None), |(context, task)| (Some(context), Some(task)));
    RemoteInput::Prepare {
        request_id: id.into(),
        request: A2ARequest::Send(
            serde_json::from_value(json!({
                "message":{"messageId":format!("message_{id}"), "role":"ROLE_USER",
                    "parts":[{"text":text}], "contextId":context,"taskId":task}
            }))
            .unwrap(),
        ),
        streaming: true,
        origin: None,
        input_request: None,
    }
}

fn dispatched(state: &mut RemoteAgentState, id: &str, input: RemoteInput) {
    apply(state, input).unwrap();
    apply(state, RemoteInput::DispatchStarted { request_id: id.into() }).unwrap();
}

fn observe(state: &mut RemoteAgentState, id: &str, value: serde_json::Value) {
    let response: a2a::StreamResponse = serde_json::from_value(value).unwrap();
    apply(
        state,
        RemoteInput::Response {
            request_id: id.into(),
            response: Box::new(A2AResponse::Stream(response)),
        },
    )
    .unwrap();
}

fn input_task() -> serde_json::Value {
    json!({"task":{"id":"booking","contextId":"trip","status":{
        "state":"TASK_STATE_INPUT_REQUIRED", "message":{
            "messageId":"question_1","role":"ROLE_AGENT","parts":[{"text":"Which day?"}]}
    }}})
}

#[test]
fn a_lost_initial_response_stays_uncertain_after_restart_without_resending_work() {
    let mut state = RemoteAgentState::new("https://booking.test".into());
    apply(&mut state, send("reserve", "Reserve a seat", None)).unwrap();
    assert_eq!(recovery(&state, "reserve").unwrap(), Recovery::Dispatch);
    apply(&mut state, RemoteInput::DispatchStarted { request_id: "reserve".into() }).unwrap();
    apply(
        &mut state,
        RemoteInput::Error {
            request_id: "reserve".into(),
            error: RemoteFailure::Transport {
                kind: lk_agents_abi::errors::ErrorKind::Connection,
                message: "the response connection ended".into(),
                status_code: None,
            },
        },
    )
    .unwrap();
    let mut restarted: RemoteAgentState =
        serde_json::from_value(serde_json::to_value(state).unwrap()).unwrap();
    assert_eq!(recovery(&restarted, "reserve").unwrap(), Recovery::Uncertain);
    assert!(
        apply(&mut restarted, RemoteInput::DispatchStarted { request_id: "reserve".into() })
            .is_err()
    );
    assert_eq!(restarted.events.len(), 2, "dispatch markers do not repeat message content");
    assert_eq!(restarted.messages.len(), 1);
    validate(&restarted).unwrap();
    restarted.requests.get_mut("reserve").unwrap().request_event = 1;
    assert!(validate(&restarted).is_err(), "a changed payload reference cannot be resumed");
}

#[test]
fn a_lost_input_response_is_reconciled_from_the_same_task_without_another_send() {
    let mut state = RemoteAgentState::new("https://booking.test".into());
    dispatched(&mut state, "start", send("start", "Book a seat", None));
    observe(&mut state, "start", input_task());
    let occurrence_id = state.tasks["booking"].input_requests.keys().next().unwrap().clone();
    let mut response = send("answer", "Friday", Some(("trip", "booking")));
    if let RemoteInput::Prepare { input_request, .. } = &mut response {
        *input_request = Some(InputRequestRef {
            task: RemoteTaskRef { context_id: "trip".into(), task_id: "booking".into() },
            occurrence_id: occurrence_id.clone(),
        });
    }
    dispatched(&mut state, "answer", response);
    apply(
        &mut state,
        RemoteInput::Error {
            request_id: "answer".into(),
            error: RemoteFailure::Transport {
                kind: lk_agents_abi::errors::ErrorKind::Connection,
                message: "accepted remotely, response lost".into(),
                status_code: None,
            },
        },
    )
    .unwrap();
    let mut restarted: RemoteAgentState =
        serde_json::from_value(serde_json::to_value(&state).unwrap()).unwrap();
    assert_eq!(
        recovery(&restarted, "answer").unwrap(),
        Recovery::GetTask(RemoteTaskRef { context_id: "trip".into(), task_id: "booking".into() })
    );
    let input = &restarted.tasks["booking"].input_requests[&occurrence_id];
    assert_eq!(input.state, InputRequestState::Responding);
    assert_eq!(input.response_message_id.as_deref(), Some("message_answer"));
    dispatched(
        &mut restarted,
        "reconcile",
        RemoteInput::Prepare {
            request_id: "reconcile".into(),
            request: A2ARequest::Get(a2a::GetTaskRequest {
                id: "booking".into(),
                history_length: None,
                tenant: None,
            }),
            streaming: false,
            origin: None,
            input_request: None,
        },
    );
    let task = serde_json::from_value(json!({"id":"booking","contextId":"trip",
        "status":{"state":"TASK_STATE_COMPLETED"},"history":[state.messages["message_answer"]]
    }))
    .unwrap();
    apply(
        &mut restarted,
        RemoteInput::Response {
            request_id: "reconcile".into(),
            response: Box::new(A2AResponse::Task(task)),
        },
    )
    .unwrap();
    assert_eq!(
        restarted.tasks["booking"].input_requests[&occurrence_id].state,
        InputRequestState::Responded
    );
    assert_eq!(
        restarted
            .events
            .iter()
            .filter(|entry| matches!(
                &entry.payload,
                A2AJournalPayload::Request(A2ARequest::Send(_))
            ))
            .count(),
        2
    );
    assert!(!restarted.tasks["booking"].cancel_requested);
    validate(&restarted).unwrap();
    restarted
        .tasks
        .get_mut("booking")
        .unwrap()
        .input_requests
        .get_mut(&occurrence_id)
        .unwrap()
        .response_message_id = Some("message_start".into());
    assert!(validate(&restarted).is_err(), "a response cannot be reassigned to another message");
}

#[test]
fn repeated_questions_keep_their_occurrence_but_identical_wording_is_not_identity() {
    let mut state = RemoteAgentState::new("https://booking.test".into());
    dispatched(&mut state, "start", send("start", "Book a seat", None));
    observe(&mut state, "start", input_task());
    observe(&mut state, "start", input_task());
    assert_eq!(state.tasks["booking"].input_requests.len(), 1);
    let mut next = input_task();
    next["task"]["status"]["message"]["messageId"] = json!("question_2");
    observe(&mut state, "start", next);
    let requests: Vec<_> = state.tasks["booking"].input_requests.values().collect();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].state, InputRequestState::Superseded);
    assert_eq!(requests[1].state, InputRequestState::Open);
}

#[test]
fn direct_contextless_messages_and_several_tasks_are_independent_exchanges() {
    let mut state = RemoteAgentState::new("https://booking.test".into());
    for id in ["hello", "first", "second"] {
        dispatched(&mut state, id, send(id, "Request", None));
    }
    observe(
        &mut state,
        "hello",
        json!({"message":{"messageId":"hello_reply","role":"ROLE_AGENT","parts":[{"data":{"ok":true}}]}}),
    );
    assert!(state.tasks.is_empty());
    assert_eq!(recovery(&state, "hello").unwrap(), Recovery::Observed);
    for id in ["first", "second"] {
        observe(
            &mut state,
            id,
            json!({"task":{"id":id,"contextId":"shared","status":{"state":"TASK_STATE_WORKING"}}}),
        );
    }
    assert_eq!(state.tasks.len(), 2);
    assert!(state.messages.contains_key("hello_reply"));
    assert_eq!(state.default_context_id.as_deref(), Some("shared"));
}

#[test]
fn conflicting_peer_content_is_journaled_without_overwriting_accepted_messages() {
    let mut state = RemoteAgentState::new("https://booking.test".into());
    dispatched(&mut state, "start", send("start", "Hello", None));
    let original =
        json!({"message":{"messageId":"reply","role":"ROLE_AGENT","parts":[{"text":"Original"}]}});
    observe(&mut state, "start", original.clone());
    let mut conflict = original;
    conflict["message"]["parts"][0]["text"] = json!("Changed");
    let result = apply(
        &mut state,
        RemoteInput::Response {
            request_id: "start".into(),
            response: Box::new(A2AResponse::Stream(serde_json::from_value(conflict).unwrap())),
        },
    );
    assert!(result.is_err());
    assert_eq!(state.messages["reply"].text(), Some("Original"));
    assert_eq!(state.events.len(), 3, "original evidence survives projection refusal");
}

#[test]
fn interleaved_artifacts_and_auth_required_survive_limited_task_snapshots() {
    let mut state = RemoteAgentState::new("https://booking.test".into());
    dispatched(&mut state, "start", send("start", "Prepare files", None));
    for (id, text, append) in [("a", "first", false), ("b", "other", false), ("a", "second", true)]
    {
        observe(
            &mut state,
            "start",
            json!({"artifactUpdate":{"taskId":"booking","contextId":"trip",
            "artifact":{"artifactId":id,"parts":[{"text":text}]},"append":append}}),
        );
    }
    observe(
        &mut state,
        "start",
        json!({"task":{"id":"booking","contextId":"trip","status":{"state":"TASK_STATE_AUTH_REQUIRED"}}}),
    );
    let task = &state.tasks["booking"].task;
    assert_eq!(task.status.state, a2a::TaskState::AuthRequired);
    assert_eq!(task.artifacts.as_ref().unwrap()[0].parts.len(), 2);
    assert_eq!(task.artifacts.as_ref().unwrap()[1].parts[0].as_text(), Some("other"));
}
