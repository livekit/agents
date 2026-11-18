use super::*;
use lk_agents_abi::session::ToolCallRef;
use serde_json::{Value, json};

fn send(state: &mut RemoteAgentState, id: &str, origin: Option<ToolCallRef>) {
    let request = A2ARequest::Send(
        serde_json::from_value(json!({
            "message":{"messageId":format!("message_{id}"),"role":"ROLE_USER","parts":[{"text":id}]}
        }))
        .unwrap(),
    );
    let prepare = RemoteInput::Prepare {
        request_id: id.into(),
        request,
        streaming: true,
        origin,
        input_request: None,
        control_scope: None,
    };
    apply("booking", state, prepare).unwrap();
    apply("booking", state, RemoteInput::DispatchStarted { request_id: id.into() }).unwrap();
}

fn poll(state: &mut RemoteAgentState, id: &str, task: &str) {
    let request = A2ARequest::Get(a2a::GetTaskRequest {
        id: task.into(),
        history_length: None,
        tenant: None,
    });
    let prepare = RemoteInput::Prepare {
        request_id: id.into(),
        request,
        streaming: false,
        origin: None,
        input_request: None,
        control_scope: None,
    };
    apply("booking", state, prepare).unwrap();
    apply("booking", state, RemoteInput::DispatchStarted { request_id: id.into() }).unwrap();
}

fn observe(state: &mut RemoteAgentState, id: &str, value: Value) {
    let response = if state.requests[id].method == a2a::methods::GET_TASK {
        A2AResponse::Task(serde_json::from_value(value["task"].clone()).unwrap())
    } else {
        A2AResponse::Stream(serde_json::from_value(value).unwrap())
    };
    apply(
        "booking",
        state,
        RemoteInput::Response { request_id: id.into(), response: Box::new(response) },
    )
    .unwrap();
}

fn end(state: &mut RemoteAgentState, id: &str) {
    apply("booking", state, RemoteInput::Ended { request_id: id.into() }).unwrap();
}

fn task(id: &str, state: &str) -> Value {
    json!({"task":{"id":id,"contextId":"trip","status":{"state":state},
        "artifacts":[{"artifactId":"answer","parts":[{"text":id}]}]}})
}

fn ids<T>(map: &std::collections::BTreeMap<String, T>) -> Vec<&str> {
    map.keys().map(String::as_str).collect()
}

#[test]
fn settled_history_is_bounded_and_keeps_whole_recent_operations() {
    let mut state = RemoteAgentState::new("https://booking.test".into());
    for n in 0..10 {
        let id = format!("send_{n}");
        send(&mut state, &id, None);
        observe(&mut state, &id, task(&format!("task_{n}"), "TASK_STATE_COMPLETED"));
        end(&mut state, &id);
    }
    compact(&mut state, 4, |_| false);

    assert_eq!(ids(&state.requests), ["send_8", "send_9"], "oldest settled work goes first");
    assert_eq!(ids(&state.tasks), ["task_8", "task_9"]);
    assert_eq!(ids(&state.messages), ["message_send_8", "message_send_9"]);
    let sequences: Vec<u64> = state.events.iter().map(|event| event.sequence).collect();
    assert_eq!(sequences, [16, 17, 18, 19], "retained entries keep their local order");
    validate("booking", &state).unwrap();
    let target = RemoteTaskRef { task_id: "task_9".into(), context_id: "trip".into() };
    assert!(observed(&state, &target).unwrap().is_some(), "the accepted outcome remains");

    send(&mut state, "send_10", None);
    assert_eq!(state.requests["send_10"].request_event, 20, "sequences never restart");
    validate("booking", &state).unwrap();
}

#[test]
fn unfinished_work_pins_its_tasks_inputs_and_unprovable_or_unresolved_operations() {
    let mut state = RemoteAgentState::new("https://booking.test".into());
    // Settled: a completed task and a later poll of it go together.
    send(&mut state, "done", None);
    observe(&mut state, "done", task("done_task", "TASK_STATE_COMPLETED"));
    end(&mut state, "done");
    poll(&mut state, "done_poll", "done_task");
    observe(&mut state, "done_poll", task("done_task", "TASK_STATE_COMPLETED"));
    end(&mut state, "done_poll");
    // Pinned: a working task and every poll of it.
    send(&mut state, "working", None);
    observe(&mut state, "working", task("working_task", "TASK_STATE_WORKING"));
    end(&mut state, "working");
    poll(&mut state, "working_poll", "working_task");
    observe(&mut state, "working_poll", task("working_task", "TASK_STATE_WORKING"));
    end(&mut state, "working_poll");
    // Pinned: an open question and its source message.
    send(&mut state, "question", None);
    observe(
        &mut state,
        "question",
        json!({"task":{"id":"question_task","contextId":"trip","status":{
            "state":"TASK_STATE_INPUT_REQUIRED","message":{
                "messageId":"which_day","role":"ROLE_AGENT","parts":[{"text":"Which day?"}]}}}}),
    );
    end(&mut state, "question");
    // Pinned: a send that may have reached the peer with no observed outcome.
    send(&mut state, "uncertain", None);
    apply(
        "booking",
        &mut state,
        RemoteInput::Error {
            request_id: "uncertain".into(),
            error: RemoteFailure::Protocol { message: "connection lost".into() },
        },
    )
    .unwrap();
    // Pinned: a settled reply whose originating call has not resolved yet.
    let origin = ToolCallRef { agent_id: "frame_0".into(), call_id: "call".into() };
    send(&mut state, "origin", Some(origin.clone()));
    observe(
        &mut state,
        "origin",
        json!({"message":{"messageId":"reply","role":"ROLE_AGENT","parts":[{"text":"Done"}]}}),
    );
    end(&mut state, "origin");

    compact(&mut state, 0, |call| call == &origin);

    assert_eq!(
        ids(&state.requests),
        ["origin", "question", "uncertain", "working", "working_poll"],
        "a settled task and every request addressing it are removed together"
    );
    assert_eq!(ids(&state.tasks), ["question_task", "working_task"]);
    assert!(state.messages.contains_key("which_day"), "an open question keeps its source");
    assert!(state.messages.contains_key("reply"));
    assert!(!state.messages.contains_key("message_done"));
    assert_eq!(recovery(&state, "uncertain").unwrap(), RemoteRecovery::Uncertain);
    assert!(matches!(recovery(&state, "origin").unwrap(), RemoteRecovery::Observed(_)));
    validate("booking", &state).unwrap();
}
