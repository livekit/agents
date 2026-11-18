use super::*;
use lk_agents_abi::delegation::remote::{A2ARequest, A2AResponse, RemoteCommand, RemoteInput};
use lk_agents_abi::session::{
    AgentInputRequest, InputRequestRef, InputRequestState, SessionInputCommand,
};

fn routing_id(session: &Session) -> String {
    let Phase::Routing { request_id, .. } = &session.phase else {
        panic!("the model has no input classification to answer")
    };
    request_id.clone()
}

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
        control_scope: None,
    }
}

fn question(session: &mut Session) -> InputRequestRef {
    question_for(session, "start", "flight", "question")
}

fn question_for(session: &mut Session, request: &str, task: &str, source: &str) -> InputRequestRef {
    remote(session, send(request, "Find flights", None));
    remote(session, RemoteInput::DispatchStarted { request_id: request.into() });
    remote(
        session,
        RemoteInput::Response {
            request_id: request.into(),
            response: Box::new(A2AResponse::Stream(
                serde_json::from_value(json!({
                    "task": {"id": task, "contextId": "trip", "status": {
                        "state": "TASK_STATE_INPUT_REQUIRED", "message": {
                            "messageId": source, "role": "ROLE_AGENT",
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
fn held_remote_responses_remain_distinct_until_a_proposed_call_selects_one() {
    use crate::delegation::remote::routing;
    use lk_agents_abi::delegation::routing::{CallAdmission, CallRoute};
    let mut driver = Driver::new(config(false, ChatHistory::Fresh, false), None, "Find flights");
    let first = question(&mut driver.session);
    let second = question_for(&mut driver.session, "another", "return-flight", "return-date");
    present(&mut driver, vec![first.clone(), second.clone()]);
    answer(&mut driver.session, vec![second.clone()], "Friday");
    let before = driver.session.snapshot();
    let offered = routing::offered(&before);
    let request = routing::request(
        &before,
        "flights",
        "https://flights.test",
        json!({"request":"Friday"}),
        None,
        &offered,
    )
    .unwrap();
    assert_eq!(
        request.candidates.iter().map(|candidate| candidate.responses.len()).sum::<usize>(),
        1
    );
    let admitted =
        routing::admit(&before, &request, Some(Ok(CallRoute::Respond(second.clone())))).unwrap();
    assert!(matches!(admitted, CallAdmission::Respond {request, ..} if request == second));
    assert!(
        routing::request(&before, "flights", "https://another.test", json!({}), None, &offered)
            .is_err()
    );

    answer(&mut driver.session, vec![first.clone()], "Friday");
    let before = driver.session.snapshot();
    let offered = routing::offered(&before);
    let request = routing::request(
        &before,
        "flights",
        "https://flights.test",
        json!({"request":"Friday"}),
        None,
        &offered,
    )
    .unwrap();
    assert!(routing::admit(&before, &request, Some(Ok(CallRoute::Unclear))).is_err());
    let explicit = routing::request(
        &before,
        "flights",
        "https://flights.test",
        json!({"request":"Why is the date needed?"}),
        Some("flight"),
        &offered,
    )
    .unwrap();
    let task = first.remote().unwrap().1.clone();
    assert_eq!(routing::without_classifier(&explicit), CallRoute::Continue(task.clone()));
    assert_eq!(
        routing::admit(&before, &explicit, Some(Ok(CallRoute::Continue(task.clone())))).unwrap(),
        CallAdmission::Continue(task)
    );
    assert_eq!(
        driver.session.snapshot(),
        before,
        "classification never reserves a response itself"
    );
    assert!(
        routing::request(
            &before,
            "flights",
            "https://flights.test",
            json!({}),
            Some("unknown"),
            &offered
        )
        .is_err()
    );
}

#[test]
fn task_continuation_uses_saved_context_without_consuming_a_question() {
    let mut driver = Driver::new(config(false, ChatHistory::Fresh, false), None, "Find flights");
    let reference = question(&mut driver.session);
    let before = driver.session.snapshot();
    let target = driver.session.remote_task("flights", "https://flights.test", "flight").unwrap();
    assert_eq!(target.context_id, "trip");
    assert_eq!(target.task_id, "flight");
    assert_eq!(driver.session.snapshot(), before, "addressing work does not answer its question");
    assert!(driver.session.remote_task("flights", "https://another.test", "flight").is_err());
    assert!(driver.session.remote_task("hotels", "https://hotels.test", "flight").is_err());
    assert!(driver.session.remote_task("flights", "https://flights.test", "missing").is_err());
    let InputRequestRef::RemoteTask { remote_id, task, .. } = reference else { unreachable!() };
    let saved = driver
        .session
        .state
        .remote_agents
        .get_mut(&remote_id)
        .unwrap()
        .tasks
        .get_mut(&task.task_id)
        .unwrap();
    saved.task.status.state = lk_agents_abi::a2a_protocol::TaskState::Working;
    saved.input_requests.clear();
    assert_eq!(
        driver.session.remote_task("flights", "https://flights.test", "flight").unwrap(),
        target,
        "corrections can address working tasks without any pending question"
    );
    driver
        .session
        .state
        .remote_agents
        .get_mut(&remote_id)
        .unwrap()
        .tasks
        .get_mut(&task.task_id)
        .unwrap()
        .task
        .status
        .state = lk_agents_abi::a2a_protocol::TaskState::Completed;
    let before = driver.session.snapshot();
    assert!(driver.session.remote_task("flights", "https://flights.test", "flight").is_err());
    let command = RemoteCommand {
        remote_id,
        endpoint: "https://flights.test".into(),
        input: RemoteInput::Prepare {
            request_id: "late-correction".into(),
            request: A2ARequest::Send(serde_json::from_value(json!({
                "message": {"messageId": "correction", "role": "ROLE_USER",
                    "parts": [{"text": "Actually Saturday"}], "taskId": "flight", "contextId": "trip"}
            })).unwrap()),
            streaming: true, origin: None, input_request: None,
                    control_scope: None,
        },
    };
    assert!(
        driver.session.remote(command).is_err(),
        "admission must revalidate before preparing I/O"
    );
    assert_eq!(driver.session.snapshot(), before);
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
    let candidates = driver.session.input_candidates();
    assert_eq!(candidates.len(), 2);
    assert!(candidates.iter().all(|candidate| {
        candidate.response.as_ref().is_some_and(|response| response.id == "answer")
    }));
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
    remote(&mut restored, send("corrected", "Actually Saturday", Some(reference.clone())));
    remote(&mut restored, RemoteInput::DispatchStarted { request_id: "corrected".into() });
    assert!(restored.snapshot().remote_agents["flights"].requests["corrected"].dispatch_started);
    let dispatched = restored.snapshot();
    restored
        .input(SessionInputCommand::Correct {
            request: reference,
            previous_message_id: "answer".into(),
            message_id: "correction".into(),
        })
        .unwrap();
    assert_eq!(
        restored.snapshot(),
        dispatched,
        "repeating admission cannot retire the unchanged send"
    );
}

#[test]
fn a_response_cannot_be_corrected_after_dispatch_may_have_started() {
    let mut driver = Driver::new(config(false, ChatHistory::Fresh, false), None, "Find flights");
    let reference = question(&mut driver.session);
    present(&mut driver, vec![reference.clone()]);
    answer(&mut driver.session, vec![reference.clone()], "Friday");
    remote(&mut driver.session, send("response", "Friday", Some(reference.clone())));
    remote(&mut driver.session, RemoteInput::DispatchStarted { request_id: "response".into() });
    assert!(driver.session.input_candidates().is_empty());
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
fn a_stale_correction_verdict_cannot_overwrite_a_newer_response_or_partly_reserve_a_group() {
    for dispatched in [false, true] {
        let mut previous =
            Driver::new(config(false, ChatHistory::Fresh, false), None, "Find flights");
        let flight = question(&mut previous.session);
        let name = local_question(&mut previous.session);
        present(&mut previous, vec![flight.clone(), name.clone()]);
        answer(&mut previous.session, vec![flight.clone()], "Friday");
        let mut session =
            Session::new(previous.config, input(Some(previous.session.snapshot()))).unwrap();
        let effects = session.handle(
            SessionEvent::Start(message("correction", Role::User, "Ada, actually Saturday")),
            Ts::ZERO,
        );
        let request = effects
            .iter()
            .find_map(|effect| match effect {
                SessionEffect::RouteInput(request) => Some(request),
                _ => None,
            })
            .expect("a held response is offered for correction");
        assert_eq!(
            request
                .candidates
                .iter()
                .find(|candidate| candidate.reference == flight)
                .unwrap()
                .response
                .as_ref()
                .unwrap()
                .id,
            "answer"
        );
        let expected = if dispatched {
            remote(&mut session, send("response", "Friday", Some(flight.clone())));
            remote(&mut session, RemoteInput::DispatchStarted { request_id: "response".into() });
            "answer"
        } else {
            session.record_dialogue(message("newer", Role::User, "Actually Sunday"));
            session
                .input(SessionInputCommand::Correct {
                    request: flight.clone(),
                    previous_message_id: "answer".into(),
                    message_id: "newer".into(),
                })
                .unwrap();
            "newer"
        };
        session.handle(
            SessionEvent::InputRouted {
                request_id: routing_id(&session),
                result: Some(Ok(InputRoute::Matched(vec![name, flight.clone()]))),
                usage: Default::default(),
            },
            Ts::ZERO,
        );
        let snapshot = session.snapshot();
        assert_eq!(
            snapshot.frames[&snapshot.root_frame_id].input_requests["name"].state,
            InputRequestState::Open,
            "a stale group cannot consume its otherwise valid member"
        );
        let request = &snapshot.remote_agents["flights"].tasks["flight"].input_requests
            [flight.remote().unwrap().2];
        assert_eq!(request.human_response_id.as_deref(), Some(expected));
    }
}

#[test]
fn a_correction_between_reading_the_response_and_preparation_refuses_the_stale_send() {
    let mut driver = Driver::new(config(false, ChatHistory::Fresh, false), None, "Find flights");
    let reference = question(&mut driver.session);
    present(&mut driver, vec![reference.clone()]);
    answer(&mut driver.session, vec![reference.clone()], "Friday");
    let human = crate::session::input::response(&driver.session.state, &reference).unwrap();
    let stale = send("stale", &human.raw_text_content().unwrap(), Some(reference.clone()));
    driver.session.record_dialogue(message("correction", Role::User, "Saturday"));
    driver
        .session
        .input(SessionInputCommand::Correct {
            request: reference.clone(),
            previous_message_id: "answer".into(),
            message_id: "correction".into(),
        })
        .unwrap();
    assert!(
        driver
            .session
            .remote(RemoteCommand {
                remote_id: "flights".into(),
                endpoint: "https://flights.test".into(),
                input: stale,
            })
            .is_err(),
        "a previously read answer must not claim the corrected reservation"
    );
    remote(&mut driver.session, send("current", "Saturday", Some(reference)));
    assert_eq!(driver.session.snapshot().remote_agents["flights"].requests.len(), 2);
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

#[test]
fn routing_precedes_the_current_agents_turn_and_checkpoints_the_selected_response() {
    let mut previous = Driver::new(config(false, ChatHistory::Fresh, false), None, "Find flights");
    let reference = question(&mut previous.session);
    present(&mut previous, vec![reference.clone()]);
    previous.reply("", &[("book", json!({"nights":2}))]);
    previous.prepare(json!({"nights":2}));
    let mut session = Session::new(previous.config, input(Some(previous.session.snapshot())))
        .unwrap()
        .with_persistence();
    let child_id = session.state.active_frame_id.clone();
    let effects = session.handle(
        SessionEvent::Start(message("reply", Role::User, "Friday, and what is checkout time?")),
        Ts::ZERO,
    );
    assert!(matches!(
        effects.as_slice(),
        [SessionEffect::Checkpoint(_), SessionEffect::RouteInput(_)]
    ));
    assert_eq!(session.snapshot().chat_history.last().unwrap().id, "reply");
    let effects = session.handle(
        SessionEvent::InputRouted {
            request_id: routing_id(&session),
            result: Some(Ok(InputRoute::Matched(vec![reference.clone()]))),
            usage: Default::default(),
        },
        Ts::ZERO,
    );
    assert!(matches!(effects.first(), Some(SessionEffect::Checkpoint(_))));
    assert_eq!(session.snapshot().active_frame_id, child_id);
    let input = &session.state.remote_agents["flights"].tasks["flight"].input_requests
        [reference.remote().unwrap().2];
    assert_eq!(input.human_response_id.as_deref(), Some("reply"));
    let history = effects
        .iter()
        .find_map(|effect| match effect {
            SessionEffect::Run {
                effect: agent_run::AgentRunEffect::LlmRequestStart { chat_ctx, .. },
                ..
            } => Some(chat_ctx),
            _ => None,
        })
        .unwrap();
    assert_eq!(history.items.iter().filter(|item| item.id() == "reply").count(), 1);
    assert!(history.items.iter().filter_map(ChatItem::as_message).any(|message| {
        message.role == Role::System
            && message
                .raw_text_content()
                .unwrap_or_default()
                .contains("not proof of remote delivery")
    }));
}

#[test]
fn routing_exposes_reservations_only_after_the_whole_group_is_accepted() {
    for stale in [false, true] {
        let mut driver =
            Driver::new(config(false, ChatHistory::Fresh, false), None, "Find flights");
        let flight = question(&mut driver.session);
        let name = local_question(&mut driver.session);
        present(&mut driver, vec![name.clone(), flight.clone()]);
        let offered = driver.session.input_candidates();
        let human = message("reply", Role::User, "Ada, next Friday");
        driver.session.record_dialogue(human.clone());
        if stale {
            driver
                .session
                .state
                .remote_agents
                .get_mut("flights")
                .unwrap()
                .tasks
                .get_mut("flight")
                .unwrap()
                .input_requests
                .get_mut(flight.remote().unwrap().2)
                .unwrap()
                .state = InputRequestState::Closed;
        }
        let before = driver.session.snapshot();
        let selected = vec![name, flight];
        let admission = crate::session::input::route(
            &mut driver.session.state,
            &human,
            &offered,
            Some(Ok(InputRoute::Matched(selected.clone()))),
        )
        .unwrap();
        if stale {
            assert!(admission.requests.is_empty(), "a partly stale group cannot be forwarded");
            assert_eq!(driver.session.snapshot(), before, "no member is partly reserved");
        } else {
            assert_eq!(admission.requests, selected);
            for reference in &admission.requests {
                assert_eq!(
                    crate::session::input::response(&driver.session.state, reference).unwrap().id,
                    human.id
                );
            }
        }
    }
}

#[test]
fn side_questions_ambiguous_verdicts_and_router_death_preserve_the_open_request() {
    for result in [Some(Ok(InputRoute::NoMatch)), Some(Ok(InputRoute::Unclear)), None] {
        let mut previous =
            Driver::new(config(false, ChatHistory::Fresh, false), None, "Find flights");
        let reference = question(&mut previous.session);
        present(&mut previous, vec![reference.clone()]);
        let mut session =
            Session::new(previous.config, input(Some(previous.session.snapshot()))).unwrap();
        session.handle(
            SessionEvent::Start(message("reply", Role::User, "Why do you need that?")),
            Ts::ZERO,
        );
        session.handle(
            SessionEvent::InputRouted {
                request_id: routing_id(&session),
                result,
                usage: Default::default(),
            },
            Ts::ZERO,
        );
        assert!(matches!(session.phase, Phase::Running(_)));
        assert_eq!(
            session.state.remote_agents["flights"].tasks["flight"].input_requests
                [reference.remote().unwrap().2]
                .state,
            InputRequestState::Open
        );
    }
}

#[test]
fn a_stale_or_late_detector_cannot_reserve_or_reopen_a_request() {
    for cancel in [false, true] {
        let mut previous =
            Driver::new(config(false, ChatHistory::Fresh, false), None, "Find flights");
        let reference = question(&mut previous.session);
        present(&mut previous, vec![reference.clone()]);
        let mut session =
            Session::new(previous.config, input(Some(previous.session.snapshot()))).unwrap();
        session.handle(SessionEvent::Start(message("reply", Role::User, "Friday")), Ts::ZERO);
        let route_id = routing_id(&session);
        if cancel {
            session.handle(SessionEvent::Cancel, Ts::ZERO);
        } else {
            remote(&mut session, RemoteInput::Response {
                request_id: "start".into(), response: Box::new(A2AResponse::Stream(serde_json::from_value(json!({
                    "task":{"id":"flight", "contextId":"trip", "status":{"state":"TASK_STATE_COMPLETED"}}
                })).unwrap())),
            });
        }
        session.handle(
            SessionEvent::InputRouted {
                request_id: route_id,
                result: Some(Ok(InputRoute::Matched(vec![reference.clone()]))),
                usage: Default::default(),
            },
            Ts::ZERO,
        );
        let observed = &session.state.remote_agents["flights"].tasks["flight"].input_requests
            [reference.remote().unwrap().2];
        assert_eq!(observed.human_response_id, None);
        assert_eq!(
            observed.state,
            if cancel { InputRequestState::Open } else { InputRequestState::Closed }
        );
        assert_eq!(matches!(session.phase, Phase::Done), cancel);
    }
}

#[test]
fn a_revised_transcript_is_classified_again_before_it_can_answer_a_question() {
    let mut previous = Driver::new(config(false, ChatHistory::Fresh, false), None, "Find flights");
    let reference = question(&mut previous.session);
    present(&mut previous, vec![reference.clone()]);
    let mut session =
        Session::new(previous.config, input(Some(previous.session.snapshot()))).unwrap();
    session.handle(SessionEvent::Start(message("reply", Role::User, "Friday")), Ts::ZERO);
    let original_request = routing_id(&session);
    session.record_dialogue(message("reply", Role::User, "Why do you need the date?"));
    let effects = session.handle(
        SessionEvent::InputRouted {
            request_id: original_request.clone(),
            result: Some(Ok(InputRoute::Matched(vec![reference.clone()]))),
            usage: Default::default(),
        },
        Ts::ZERO,
    );
    let request = effects
        .iter()
        .find_map(|effect| match effect {
            SessionEffect::RouteInput(request) => Some(request),
            _ => None,
        })
        .expect("a detector verdict cannot authorize different transcript words");
    assert_eq!(request.message.raw_text_content().as_deref(), Some("Why do you need the date?"));
    assert!(!effects.iter().any(|effect| matches!(effect, SessionEffect::Run { .. })));
    let revised_request = request.request_id.clone();
    assert_ne!(original_request, revised_request);
    let usage = session.usage();
    let late = session.handle(
        SessionEvent::InputRouted {
            request_id: original_request.clone(),
            result: Some(Ok(InputRoute::Matched(vec![reference.clone()]))),
            usage: lk_agents_abi::usage::Usage { requests: 99, ..Default::default() },
        },
        Ts::ZERO,
    );
    assert!(late.is_empty(), "the old verdict cannot be reinterpreted as the revised request");
    assert_eq!(session.usage(), usage, "a late receipt cannot charge its usage twice");
    assert!(
        session
            .handle(
                SessionEvent::InputRouted {
                    request_id: original_request,
                    result: None,
                    usage: Default::default()
                },
                Ts::ZERO,
            )
            .is_empty(),
        "an old routing task's death cannot end the revised classification"
    );

    let input = &session.state.remote_agents["flights"].tasks["flight"].input_requests
        [reference.remote().unwrap().2];
    assert_eq!(input.state, InputRequestState::Open);
    assert_eq!(input.human_response_id, None);
    let effects = session.handle(
        SessionEvent::InputRouted {
            request_id: revised_request,
            result: Some(Ok(InputRoute::NoMatch)),
            usage: Default::default(),
        },
        Ts::ZERO,
    );
    let history = effects
        .iter()
        .find_map(|effect| match effect {
            SessionEffect::Run {
                effect: agent_run::AgentRunEffect::LlmRequestStart { chat_ctx, .. },
                ..
            } => Some(chat_ctx),
            _ => None,
        })
        .expect("the current agent handles the revised side question");
    let human: Vec<_> = history
        .items
        .iter()
        .filter_map(ChatItem::as_message)
        .filter(|message| message.id == "reply")
        .collect();
    assert_eq!(human.len(), 1);
    assert_eq!(human[0].raw_text_content().as_deref(), Some("Why do you need the date?"));
}
