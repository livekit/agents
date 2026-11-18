use super::*;
use lk_agents_abi::a2a_protocol as a2a;
use serde_json::json;

fn apply(
    state: &mut RemoteAgentState,
    input: RemoteInput,
) -> Result<Option<InputRequestRef>, RemoteError> {
    super::apply("booking", state, input)
}

fn validate(state: &RemoteAgentState) -> Result<(), RemoteError> {
    super::validate("booking", state)
}

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
        control_scope: None,
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
fn an_input_receipt_identifies_the_accepted_occurrence_without_following_later_questions() {
    let mut state = RemoteAgentState::new("https://booking.test".into());
    dispatched(&mut state, "start", send("start", "Book a seat", None));
    let observe = |state: &mut RemoteAgentState, value| {
        apply(
            state,
            RemoteInput::Response {
                request_id: "start".into(),
                response: Box::new(A2AResponse::Stream(serde_json::from_value(value).unwrap())),
            },
        )
        .unwrap()
    };
    let first = observe(&mut state, input_task()).unwrap();
    assert_eq!(
        *first.remote().unwrap().1,
        RemoteTaskRef { context_id: "trip".into(), task_id: "booking".into() }
    );
    assert_eq!(observe(&mut state, input_task()), Some(first.clone()));
    let mut limited = input_task();
    limited["task"]["status"].as_object_mut().unwrap().remove("message");
    assert_eq!(observe(&mut state, limited), Some(first.clone()));
    assert_eq!(observe(&mut state, input_task()), Some(first.clone()));
    let mut next = input_task();
    next["task"]["status"]["message"]["messageId"] = json!("question_2");
    let second = observe(&mut state, next).unwrap();
    assert_ne!(first, second);
    let mut response = send("answer", "Friday", Some(("trip", "booking")));
    let RemoteInput::Prepare { input_request, .. } = &mut response else { unreachable!() };
    *input_request = Some(first);
    assert!(apply(&mut state, response).is_err(), "a late response cannot select the new question");
    assert_eq!(
        state.tasks["booking"].input_requests[second.remote().unwrap().2].state,
        InputRequestState::Open
    );
    assert_eq!(
        observe(
            &mut state,
            json!({"task":{"id":"booking","contextId":"trip",
        "status":{"state":"TASK_STATE_WORKING"}}})
        ),
        None
    );
    validate(&state).unwrap();
}

#[test]
fn an_unconfirmed_response_does_not_make_the_same_question_available_for_another_answer() {
    for with_message in [true, false] {
        let mut state = RemoteAgentState::new("https://booking.test".into());
        dispatched(&mut state, "start", send("start", "Book a seat", None));
        let response = A2AResponse::Stream(serde_json::from_value(input_task()).unwrap());
        let reference = apply(
            &mut state,
            RemoteInput::Response {
                request_id: "start".into(),
                response: Box::new(response.clone()),
            },
        )
        .unwrap()
        .unwrap();
        let mut answer = send("answer", "Friday", Some(("trip", "booking")));
        let RemoteInput::Prepare { input_request, .. } = &mut answer else { unreachable!() };
        *input_request = Some(reference.clone());
        dispatched(&mut state, "answer", answer);
        apply(
            &mut state,
            RemoteInput::Error {
                request_id: "answer".into(),
                error: RemoteFailure::Protocol { message: "the reply was lost".into() },
            },
        )
        .unwrap();
        // Another observer sees the same input-required status, without history proving the
        // first response arrived. That is uncertainty, not permission to ask and send it twice.
        dispatched(
            &mut state,
            "recover",
            RemoteInput::Prepare {
                request_id: "recover".into(),
                request: A2ARequest::Get(a2a::GetTaskRequest {
                    id: "booking".into(),
                    history_length: None,
                    tenant: None,
                }),
                streaming: false,
                origin: None,
                input_request: None,
                control_scope: None,
            },
        );
        let mut observed = state.tasks["booking"].task.clone();
        if !with_message {
            observed.status.message = None;
        }
        let receipt = apply(
            &mut state,
            RemoteInput::Response {
                request_id: "recover".into(),
                response: Box::new(A2AResponse::Task(observed)),
            },
        )
        .unwrap();
        assert_eq!(receipt, None);
        let input = &state.tasks["booking"].input_requests[reference.remote().unwrap().2];
        assert_eq!(input.state, InputRequestState::Responding);
        assert_eq!(input.response_message_id.as_deref(), Some("message_answer"));
        assert_eq!(input.source_message_id.as_deref(), Some("question_1"));
        assert_eq!(state.tasks["booking"].input_requests.len(), 1);
        validate(&state).unwrap();
    }
}

#[test]
fn a_rejected_response_cannot_complete_recovery_just_because_it_is_journaled() {
    let mut state = RemoteAgentState::new("https://booking.test".into());
    dispatched(&mut state, "start", send("start", "Hello", None));
    let response = serde_json::from_value(json!({"message":{
        "messageId":"", "role":"ROLE_AGENT", "parts":[{"text":"Invalid identity"}]
    }}))
    .unwrap();
    assert!(
        apply(
            &mut state,
            RemoteInput::Response {
                request_id: "start".into(),
                response: Box::new(A2AResponse::Stream(response)),
            }
        )
        .is_err()
    );
    assert_eq!(state.events.len(), 2, "the refused wire payload remains available for diagnosis");
    assert_eq!(recovery(&state, "start").unwrap(), RemoteRecovery::Uncertain);
}

#[test]
fn new_work_cannot_be_retargeted_by_a_task_or_message_from_another_context() {
    for response in [
        json!({"task":{"id":"foreign-task","contextId":"foreign","status":{"state":"TASK_STATE_WORKING"}}}),
        json!({"message":{"messageId":"foreign-message","contextId":"foreign","role":"ROLE_AGENT","parts":[{"text":"Wrong conversation"}]}}),
    ] {
        let mut state = RemoteAgentState::new("https://booking.test".into());
        let mut input = send("start", "Hello", None);
        let RemoteInput::Prepare { request: A2ARequest::Send(body), .. } = &mut input else {
            unreachable!()
        };
        body.message.context_id = Some("trip".into());
        dispatched(&mut state, "start", input);
        assert!(
            apply(
                &mut state,
                RemoteInput::Response {
                    request_id: "start".into(),
                    response: Box::new(A2AResponse::Stream(
                        serde_json::from_value(response).unwrap()
                    )),
                }
            )
            .is_err()
        );
        assert_eq!(state.events.len(), 2);
        assert!(state.tasks.is_empty());
        assert!(!state.messages.contains_key("foreign-message"));
        assert_eq!(recovery(&state, "start").unwrap(), RemoteRecovery::Uncertain);
    }
}

#[test]
fn a_lost_initial_response_stays_uncertain_after_restart_without_resending_work() {
    let mut state = RemoteAgentState::new("https://booking.test".into());
    apply(&mut state, send("reserve", "Reserve a seat", None)).unwrap();
    assert!(matches!(recovery(&state, "reserve").unwrap(), RemoteRecovery::Dispatch { .. }));
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
    assert_eq!(recovery(&restarted, "reserve").unwrap(), RemoteRecovery::Uncertain);
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
fn an_undispatched_request_can_resume_after_its_local_observer_failed() {
    let mut state = RemoteAgentState::new("https://booking.test".into());
    let request = send("reserve", "Reserve a seat", None);
    apply(&mut state, request.clone()).unwrap();
    apply(
        &mut state,
        RemoteInput::Error {
            request_id: "reserve".into(),
            error: RemoteFailure::Protocol {
                message: "the local observer stopped before dispatch".into(),
            },
        },
    )
    .unwrap();
    let mut restored: RemoteAgentState =
        serde_json::from_value(serde_json::to_value(state).unwrap()).unwrap();
    validate(&restored).unwrap();
    assert!(matches!(recovery(&restored, "reserve").unwrap(), RemoteRecovery::Dispatch { .. }));
    apply(&mut restored, request.clone()).unwrap();
    validate(&restored).unwrap();
    assert!(!restored.requests["reserve"].ended);
    apply(&mut restored, RemoteInput::DispatchStarted { request_id: "reserve".into() }).unwrap();
    assert!(
        apply(&mut restored, request).is_err(),
        "a started dispatch cannot be retried this way"
    );
    assert_eq!(restored.events.len(), 2, "preparation keeps one immutable request and its error");
    assert_eq!(restored.messages.len(), 1, "the outgoing message keeps its original identity");
    validate(&restored).unwrap();
}

#[test]
fn prepared_response_recovery_retains_its_exact_question_occurrence() {
    let mut state = RemoteAgentState::new("https://booking.test".into());
    dispatched(&mut state, "start", send("start", "Book a seat", None));
    let reference = apply(
        &mut state,
        RemoteInput::Response {
            request_id: "start".into(),
            response: Box::new(A2AResponse::Stream(serde_json::from_value(input_task()).unwrap())),
        },
    )
    .unwrap()
    .unwrap();
    let mut prepared = send("answer", "Friday", Some(("trip", "booking")));
    let RemoteInput::Prepare { input_request, .. } = &mut prepared else { unreachable!() };
    *input_request = Some(reference.clone());
    apply(&mut state, prepared).unwrap();
    let restored: RemoteAgentState =
        serde_json::from_value(serde_json::to_value(state).unwrap()).unwrap();
    let RemoteRecovery::Dispatch { request_id, request, input_request, .. } =
        recovery(&restored, "answer").unwrap()
    else {
        panic!("an undispatched answer retains its original operation")
    };
    assert_eq!(request_id, "answer");
    assert_eq!(input_request, Some(reference));
    let A2ARequest::Send(send) = *request else { panic!("the prepared response is a send") };
    assert_eq!(send.message.message_id, "message_answer");
    assert_eq!(send.message.task_id.as_deref(), Some("booking"));
}

#[test]
fn a_lost_input_response_is_reconciled_from_the_same_task_without_another_send() {
    let mut state = RemoteAgentState::new("https://booking.test".into());
    dispatched(&mut state, "start", send("start", "Book a seat", None));
    observe(&mut state, "start", input_task());
    let occurrence_id = state.tasks["booking"].input_requests.keys().next().unwrap().clone();
    let mut response = send("answer", "Friday", Some(("trip", "booking")));
    if let RemoteInput::Prepare { input_request, .. } = &mut response {
        *input_request = Some(InputRequestRef::RemoteTask {
            remote_id: "booking".into(),
            task: RemoteTaskRef { context_id: "trip".into(), task_id: "booking".into() },
            request_id: occurrence_id.clone(),
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
        RemoteRecovery::GetTask(RemoteTaskRef {
            context_id: "trip".into(),
            task_id: "booking".into()
        })
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
            control_scope: None,
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
fn an_acknowledged_response_can_lead_to_another_input_occurrence_with_the_same_question() {
    for with_message in [false, true] {
        let mut state = RemoteAgentState::new("https://booking.test".into());
        let mut question = input_task();
        if !with_message {
            question["task"]["status"].as_object_mut().unwrap().remove("message");
        }
        dispatched(&mut state, "start", send("start", "Book a seat", None));
        observe(&mut state, "start", question.clone());
        // Continue past single-digit occurrence IDs: map string order is not question order.
        for round in 0..12 {
            let occurrence_id = state.tasks["booking"]
                .input_requests
                .iter()
                .find(|(_, input)| input.state == InputRequestState::Open)
                .map(|(id, _)| id.clone())
                .unwrap();
            let request_id = format!("answer_{round}");
            let mut response = send(&request_id, "A partial response", Some(("trip", "booking")));
            if let RemoteInput::Prepare { input_request, .. } = &mut response {
                *input_request = Some(InputRequestRef::RemoteTask {
                    remote_id: "booking".into(),
                    task: RemoteTaskRef { context_id: "trip".into(), task_id: "booking".into() },
                    request_id: occurrence_id.clone(),
                });
            }
            dispatched(&mut state, &request_id, response);
            // The peer admits the response and asks for input again, without a new message ID
            // or an intervening working status. Admission does not mean the content sufficed.
            observe(&mut state, &request_id, question.clone());
            let requests = &state.tasks["booking"].input_requests;
            assert_eq!(requests.len(), round + 2);
            assert_eq!(requests[&occurrence_id].state, InputRequestState::Responded);
            assert_eq!(
                requests.values().filter(|input| input.state == InputRequestState::Open).count(),
                1
            );
            observe(&mut state, &request_id, question.clone());
            assert_eq!(
                state.tasks["booking"].input_requests.len(),
                round + 2,
                "repeating that observation is not another question"
            );
            validate(&state).unwrap();
        }
    }
}

#[test]
fn a_replaced_input_request_retires_an_unsent_response_but_keeps_dispatch_uncertainty() {
    for started in [false, true] {
        let mut state = RemoteAgentState::new("https://booking.test".into());
        dispatched(&mut state, "start", send("start", "Book a seat", None));
        observe(&mut state, "start", input_task());
        let occurrence_id = state.tasks["booking"].input_requests.keys().next().unwrap().clone();
        let mut response = send("answer", "Friday", Some(("trip", "booking")));
        if let RemoteInput::Prepare { input_request, .. } = &mut response {
            *input_request = Some(InputRequestRef::RemoteTask {
                remote_id: "booking".into(),
                task: RemoteTaskRef { context_id: "trip".into(), task_id: "booking".into() },
                request_id: occurrence_id.clone(),
            });
        }
        apply(&mut state, response).unwrap();
        if started {
            apply(&mut state, RemoteInput::DispatchStarted { request_id: "answer".into() })
                .unwrap();
        }
        // A subscription observes the task asking something else before this send starts,
        // or while a sent response still has no acknowledgement.
        dispatched(
            &mut state,
            "observe",
            RemoteInput::Prepare {
                request_id: "observe".into(),
                request: A2ARequest::Subscribe(a2a::SubscribeToTaskRequest {
                    id: "booking".into(),
                    tenant: None,
                }),
                streaming: true,
                origin: None,
                input_request: None,
                control_scope: None,
            },
        );
        let mut next = input_task();
        next["task"]["status"]["message"]["messageId"] = json!("question_2");
        observe(&mut state, "observe", next);
        let input = &state.tasks["booking"].input_requests[&occurrence_id];
        assert_eq!(
            input.state,
            if started { InputRequestState::Responding } else { InputRequestState::Superseded }
        );
        assert_eq!(input.response_message_id.as_deref(), Some("message_answer"));
        assert!(
            apply(&mut state, RemoteInput::DispatchStarted { request_id: "answer".into() })
                .is_err()
        );
        assert_eq!(state.requests["answer"].dispatch_started, started);
        assert_eq!(
            recovery(&state, "answer").unwrap(),
            RemoteRecovery::GetTask(RemoteTaskRef {
                context_id: "trip".into(),
                task_id: "booking".into()
            })
        );
        validate(&state).unwrap();
    }
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
    assert!(
        matches!(recovery(&state, "hello").unwrap(), RemoteRecovery::Observed(response) if matches!(response.as_ref(), a2a::SendMessageResponse::Message(message) if message.message_id == "hello_reply"))
    );
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

#[test]
fn limited_task_snapshots_preserve_optional_content_and_explicit_empty_fields_replace_it() {
    let mut state = RemoteAgentState::new("https://booking.test".into());
    dispatched(&mut state, "start", send("start", "Prepare receipt", None));
    observe(
        &mut state,
        "start",
        json!({"task": {
            "id":"booking", "contextId":"trip", "status":{"state":"TASK_STATE_WORKING"},
            "history":[{"messageId":"receipt-note", "role":"ROLE_AGENT", "parts":[
                {"data":{"receipt":42}},
                {"url":"https://example.test/receipt.pdf", "mediaType":"application/pdf", "filename":"receipt.pdf"}
            ], "extensions":["https://example.test/receipt-extension"], "metadata":{"note":"kept"}}],
            "artifacts":[{"artifactId":"receipt", "parts":[{"text":"Confirmed"}],
                "extensions":["https://example.test/receipt-extension"]}],
            "metadata":{"custom":{"revision":1}}
        }}),
    );
    let prior = state.tasks["booking"].task.clone();
    observe(
        &mut state,
        "start",
        json!({"task": {
            "id":"booking", "contextId":"trip", "status":{"state":"TASK_STATE_COMPLETED"}
        }}),
    );
    let limited = &state.tasks["booking"].task;
    assert_eq!(limited.history, prior.history);
    assert_eq!(limited.artifacts, prior.artifacts);
    assert_eq!(limited.metadata, prior.metadata);
    let A2AJournalPayload::Response(raw) = &state.events.last().unwrap().payload else {
        panic!("the original response is retained")
    };
    let A2AResponse::Stream(a2a::StreamResponse::Task(raw)) = raw.as_ref() else {
        panic!("the original SDK Task is retained")
    };
    assert!(
        raw.history.is_none() && raw.artifacts.is_none() && raw.metadata.is_none(),
        "projection never rewrites the journal's original SDK appearance"
    );
    observe(
        &mut state,
        "start",
        json!({"task": {
            "id":"booking", "contextId":"trip", "status":{"state":"TASK_STATE_COMPLETED"},
            "history":[], "artifacts":[], "metadata":{}
        }}),
    );
    let replaced = &state.tasks["booking"].task;
    assert!(replaced.history.as_ref().unwrap().is_empty());
    assert!(replaced.artifacts.as_ref().unwrap().is_empty());
    assert!(replaced.metadata.as_ref().unwrap().is_empty());
    assert_eq!(
        state.messages["receipt-note"].extensions.as_ref().unwrap(),
        &["https://example.test/receipt-extension"]
    );
}

fn task_poll(state: &mut RemoteAgentState, id: &str) {
    dispatched(
        state,
        id,
        RemoteInput::Prepare {
            request_id: id.into(),
            request: A2ARequest::Get(serde_json::from_value(json!({"id":"booking"})).unwrap()),
            streaming: false,
            origin: None,
            input_request: None,
            control_scope: None,
        },
    );
}

#[test]
fn a_late_nonterminal_task_poll_cannot_reopen_a_completed_task_for_continuation() {
    let mut state = RemoteAgentState::new("https://booking.test".into());
    dispatched(&mut state, "start", send("start", "Book a room", None));
    observe(
        &mut state,
        "start",
        json!({"task": {
            "id":"booking", "contextId":"trip", "status":{
                "state":"TASK_STATE_COMPLETED", "timestamp":"2026-11-18T00:00:02Z"
            }, "artifacts":[{"artifactId":"receipt", "parts":[{"text":"Booked"}]}]
        }}),
    );
    let terminal = state.tasks["booking"].clone();
    task_poll(&mut state, "old-poll");
    let stale: a2a::Task = serde_json::from_value(json!({
        "id":"booking", "contextId":"trip", "status":{
            "state":"TASK_STATE_WORKING", "timestamp":"2026-11-18T00:00:01Z"
        }, "artifacts":[]
    }))
    .unwrap();
    apply(
        &mut state,
        RemoteInput::Response {
            request_id: "old-poll".into(),
            response: Box::new(A2AResponse::Task(stale.clone())),
        },
    )
    .unwrap();
    assert_eq!(
        state.tasks["booking"], terminal,
        "a stale poll cannot replace terminal status, input state or final artifacts"
    );
    assert!(
        crate::session::remote::task(&state, "booking").is_err(),
        "explicit task_id continuation stays refused after completion"
    );
    assert!(
        matches!(&state.events.last().unwrap().payload,
            A2AJournalPayload::Response(response) if response.as_ref() == &A2AResponse::Task(stale)
        ),
        "the exact raw stale evidence remains in communication history"
    );
    validate(&state).unwrap();
}

#[test]
fn conflicting_terminal_observations_keep_the_first_terminal_state_and_fail_protocol_projection() {
    for first in
        ["TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED", "TASK_STATE_REJECTED"]
    {
        let mut state = RemoteAgentState::new("https://booking.test".into());
        dispatched(&mut state, "start", send("start", "Request", None));
        observe(
            &mut state,
            "start",
            json!({"task": {
                "id":"booking", "contextId":"trip", "status":{"state":first}
            }}),
        );
        let terminal = state.tasks["booking"].clone();
        task_poll(&mut state, "conflict-poll");
        let second = if first == "TASK_STATE_COMPLETED" {
            "TASK_STATE_CANCELED"
        } else {
            "TASK_STATE_COMPLETED"
        };
        let observed: a2a::Task = serde_json::from_value(json!({
            "id":"booking", "contextId":"trip", "status":{"state":second}
        }))
        .unwrap();
        let result = apply(
            &mut state,
            RemoteInput::Response {
                request_id: "conflict-poll".into(),
                response: Box::new(A2AResponse::Task(observed.clone())),
            },
        );
        assert!(result.is_err(), "conflicting terminal evidence requires protocol reconciliation");
        assert_eq!(state.tasks["booking"], terminal);
        assert!(crate::session::remote::task(&state, "booking").is_err());
        assert!(matches!(&state.events.last().unwrap().payload,
            A2AJournalPayload::Response(response) if response.as_ref() == &A2AResponse::Task(observed)
        ));
        validate(&state).unwrap();
    }
}
