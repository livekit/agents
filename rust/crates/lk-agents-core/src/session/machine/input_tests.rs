use super::*;
use lk_agents_abi::delegation::remote::{A2ARequest, A2AResponse, RemoteCommand, RemoteInput};
use lk_agents_abi::session::{
    AgentInputRequest, InputRequestRef, InputRequestState, SessionInputCommand,
};

fn remote(session: &mut Session, input: RemoteInput) -> Option<InputRequestRef> {
    session
        .remote(RemoteCommand {
            remote_id: "flights".into(),
            endpoint: "https://flights.test".into(),
            input,
        })
        .unwrap()
}

fn send(id: &str, text: &str, selected: Option<InputRequestRef>) -> RemoteInput {
    let task = selected.as_ref().map(|reference| reference.remote().unwrap().1);
    RemoteInput::Prepare {
        request_id: id.into(),
        request: A2ARequest::Send(
            serde_json::from_value(json!({
                "message": {"messageId": format!("message_{id}"), "role": "ROLE_USER",
                    "parts": [{"text": text}], "taskId": task.map(|task| &task.task_id),
                    "contextId": task.map(|task| &task.context_id)}
            }))
            .unwrap(),
        ),
        streaming: true,
        origin: None,
        input_request: selected,
    }
}

fn question(session: &mut Session) -> InputRequestRef {
    remote(session, send("start", "Find flights", None));
    remote(session, RemoteInput::DispatchStarted { request_id: "start".into() });
    remote(
        session,
        RemoteInput::Response {
            request_id: "start".into(),
            response: Box::new(A2AResponse::Stream(
                serde_json::from_value(json!({
                    "task": {"id": "flight", "contextId": "trip", "status": {
                        "state": "TASK_STATE_INPUT_REQUIRED", "message": {
                            "messageId": "question", "role": "ROLE_AGENT",
                            "parts": [{"text": "Which day?"}]}
                    }}
                }))
                .unwrap(),
            )),
        },
    )
    .unwrap()
}

fn local_question(session: &mut Session) -> InputRequestRef {
    let agent_id = session.state.root_frame_id.clone();
    session.state.frames.get_mut(&agent_id).unwrap().input_requests.insert(
        "name".into(),
        AgentInputRequest {
            state: InputRequestState::Open,
            source_message: message("name", Role::Assistant, "What is your name?"),
            response_message: None,
            origin: None,
            presented_message_id: None,
        },
    );
    InputRequestRef::Agent { agent_id: agent_id.value, request_id: "name".into() }
}

fn present(driver: &mut Driver, requests: Vec<InputRequestRef>) {
    driver.send(SessionEvent::DialogueDelivered {
        invocation_id: driver.invocation(),
        message: message("heard", Role::Assistant, "What is your name, and which day?"),
    });
    driver
        .session
        .input(SessionInputCommand::Presented { requests, message_id: "heard".into() })
        .unwrap();
}

fn answer(session: &mut Session, requests: Vec<InputRequestRef>, text: &str) {
    session.record_dialogue(message("answer", Role::User, text));
    session.input(SessionInputCommand::Respond { requests, message_id: "answer".into() }).unwrap();
}

#[test]
fn recognition_requires_actual_delivery_and_excludes_private_and_queued_questions() {
    let mut driver = Driver::new(config(false, ChatHistory::Fresh, false), None, "Find flights");
    let reference = question(&mut driver.session);
    assert!(driver.session.input_candidates().is_empty());
    assert!(
        driver
            .session
            .input(SessionInputCommand::Presented {
                requests: vec![reference.clone()],
                message_id: "question".into(),
            })
            .is_err(),
        "a peer's question is not proof of human delivery"
    );
    assert!(
        driver
            .session
            .input(SessionInputCommand::Presented {
                requests: vec![reference.clone()],
                message_id: "human-0".into(),
            })
            .is_err(),
        "a human message cannot be a delivery receipt"
    );
    present(&mut driver, vec![reference.clone()]);
    let candidates = driver.session.input_candidates();
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].reference, reference);
    assert_eq!(candidates[0].question, "Which day?");
    assert_eq!(candidates[0].presentation.id, "heard");
}

#[test]
fn grouped_answers_reserve_all_occurrences_or_none_and_repeat_idempotently() {
    let mut driver = Driver::new(config(false, ChatHistory::Fresh, false), None, "Find flights");
    let flight = question(&mut driver.session);
    let name = local_question(&mut driver.session);
    present(&mut driver, vec![flight.clone(), name.clone()]);
    driver.session.record_dialogue(message("answer", Role::User, "Ada, next Friday"));
    let before = driver.session.snapshot();
    let mut wrong = flight.clone();
    let InputRequestRef::RemoteTask { task, .. } = &mut wrong else { unreachable!() };
    task.context_id = "another trip".into();
    assert!(
        driver
            .session
            .input(SessionInputCommand::Respond {
                requests: vec![name.clone(), wrong],
                message_id: "answer".into(),
            })
            .is_err()
    );
    assert_eq!(driver.session.snapshot(), before, "a stale member must not reserve other answers");
    answer(&mut driver.session, vec![name.clone(), flight.clone()], "Ada, next Friday");
    let selected = driver.session.snapshot();
    answer(&mut driver.session, vec![name, flight], "Ada, next Friday");
    assert_eq!(driver.session.snapshot(), selected);
    assert!(driver.session.input_candidates().is_empty());
}

#[test]
fn an_answer_is_held_through_a_child_without_its_remote_tool_and_a_native_restart() {
    let mut driver = Driver::new(config(false, ChatHistory::Fresh, false), None, "Find flights");
    let reference = question(&mut driver.session);
    present(&mut driver, vec![reference.clone()]);
    driver.reply("", &[("book", json!({"nights":2}))]);
    driver.prepare(json!({"nights":2}));
    let child_id = driver.session.state.active_frame_id.clone();
    assert_ne!(child_id, driver.session.state.root_frame_id);
    assert!(driver.session.config.agents["booking"].definition.tools.is_empty());
    assert_eq!(driver.session.input_candidates()[0].reference, reference);
    answer(&mut driver.session, vec![reference.clone()], "Friday");
    let snapshot = driver.session.snapshot();
    assert_eq!(snapshot.active_frame_id, child_id);
    assert_eq!(snapshot.remote_agents["flights"].requests.len(), 1, "recognition cannot dispatch");
    let restored = serde_json::from_value(serde_json::to_value(snapshot).unwrap()).unwrap();
    let restored = Session::new(driver.config, input(Some(restored))).unwrap().snapshot();
    let held = &restored.remote_agents["flights"].tasks["flight"].input_requests
        [reference.remote().unwrap().2];
    assert_eq!(held.state, InputRequestState::Responding);
    assert_eq!(held.human_response_id.as_deref(), Some("answer"));
    assert_eq!(held.response_message_id, None);
    assert_eq!(restored.chat_history.iter().filter(|message| message.id == "answer").count(), 1);
}

#[test]
fn a_correction_retires_unsent_intent_and_preserves_its_original_journal_body() {
    let mut driver = Driver::new(config(false, ChatHistory::Fresh, false), None, "Find flights");
    let reference = question(&mut driver.session);
    present(&mut driver, vec![reference.clone()]);
    answer(&mut driver.session, vec![reference.clone()], "Friday");
    let old = send("response", "Friday", Some(reference.clone()));
    remote(&mut driver.session, old.clone());
    let events = driver.session.snapshot().remote_agents["flights"].events.clone();
    driver.session.record_dialogue(message("correction", Role::User, "Actually Saturday"));
    driver
        .session
        .input(SessionInputCommand::Correct {
            request: reference.clone(),
            previous_message_id: "answer".into(),
            message_id: "correction".into(),
        })
        .unwrap();
    let snapshot = driver.session.snapshot();
    assert_eq!(snapshot.remote_agents["flights"].events, events);
    assert!(snapshot.remote_agents["flights"].requests["response"].ended);
    assert!(
        driver
            .session
            .remote(RemoteCommand {
                remote_id: "flights".into(),
                endpoint: "https://flights.test".into(),
                input: old,
            })
            .is_err(),
        "the old immutable send must never be revived after a correction"
    );
    assert!(
        driver
            .session
            .remote(RemoteCommand {
                remote_id: "flights".into(),
                endpoint: "https://flights.test".into(),
                input: RemoteInput::DispatchStarted { request_id: "response".into() },
            })
            .is_err()
    );
    let restored = serde_json::from_value(serde_json::to_value(snapshot).unwrap()).unwrap();
    let mut restored = Session::new(driver.config, input(Some(restored))).unwrap();
    remote(&mut restored, send("corrected", "Saturday", Some(reference)));
    remote(&mut restored, RemoteInput::DispatchStarted { request_id: "corrected".into() });
    assert!(restored.snapshot().remote_agents["flights"].requests["corrected"].dispatch_started);
}

#[test]
fn a_response_cannot_be_corrected_after_dispatch_may_have_started() {
    let mut driver = Driver::new(config(false, ChatHistory::Fresh, false), None, "Find flights");
    let reference = question(&mut driver.session);
    present(&mut driver, vec![reference.clone()]);
    answer(&mut driver.session, vec![reference.clone()], "Friday");
    remote(&mut driver.session, send("response", "Friday", Some(reference.clone())));
    remote(&mut driver.session, RemoteInput::DispatchStarted { request_id: "response".into() });
    driver.session.record_dialogue(message("correction", Role::User, "Actually Saturday"));
    let before = driver.session.snapshot();
    assert!(
        driver
            .session
            .input(SessionInputCommand::Correct {
                request: reference,
                previous_message_id: "answer".into(),
                message_id: "correction".into(),
            })
            .is_err()
    );
    assert_eq!(driver.session.snapshot(), before);
}

#[test]
fn a_forged_human_or_presentation_reference_cannot_restore_as_a_routable_question() {
    let mut driver = Driver::new(config(false, ChatHistory::Fresh, false), None, "Find flights");
    let reference = question(&mut driver.session);
    present(&mut driver, vec![reference.clone()]);
    answer(&mut driver.session, vec![reference.clone()], "Friday");
    for forged_presentation in [true, false] {
        let mut state = driver.session.snapshot();
        let input = state
            .remote_agents
            .get_mut("flights")
            .unwrap()
            .tasks
            .get_mut("flight")
            .unwrap()
            .input_requests
            .get_mut(reference.remote().unwrap().2)
            .unwrap();
        if forged_presentation {
            input.presented_message_id = Some("question".into());
        } else {
            input.human_response_id = Some("heard".into());
        }
        assert!(Session::new(driver.config.clone(), super::input(Some(state))).is_err());
    }
}
