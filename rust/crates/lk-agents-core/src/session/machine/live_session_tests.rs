use super::*;
use lk_agents_abi::llm::tool_executor::ToolOutcome;
use lk_agents_abi::text_session::LiveTurnOrigin;

fn origin(id: &str) -> LiveTurnOrigin {
    LiveTurnOrigin {
        request_id: format!("request_{id}"),
        execution_id: format!("AST_{id}"),
        task_ids: vec![format!("AST_{id}")],
    }
}

fn last_turn(driver: &Driver) -> (u64, LiveTurnOrigin, bool) {
    driver
        .effects
        .iter()
        .rev()
        .find_map(|effect| match effect {
            SessionEffect::LiveTurnFinished { turn, origin, working, .. } => {
                Some((*turn, origin.clone(), *working))
            }
            _ => None,
        })
        .expect("a native turn awaits its fenced commit")
}

#[test]
fn a_live_tool_keeps_its_original_result_while_an_independent_turn_completes() {
    let mut config = config(false, ChatHistory::Fresh, false);
    config.agents.get_mut("desk").unwrap().agent_tools.clear();
    let mut driver = Driver {
        session: Session::new(config.clone(), input(None)).unwrap().with_live(origin("booking")),
        config,
        effects: Vec::new(),
        auto_release: true,
    };
    driver.send(SessionEvent::Start(message("human_A", Role::User, "Reserve a room.")));
    driver.reply("", &[("book", json!({"nights":2}))]);
    driver.prepare(json!({"nights":2}));
    let (owner, body, canonical) = driver
        .effects
        .iter()
        .rev()
        .find_map(|effect| match effect {
            SessionEffect::Run {
                invocation_id,
                effect: agent_run::AgentRunEffect::ToolExecute { call, function_call, .. },
            } => Some((invocation_id.clone(), *call, function_call.id.clone())),
            _ => None,
        })
        .unwrap();
    driver.run(agent_run::AgentRunEvent::ToolProgress {
        call: body,
        message: "Checking availability.".into(),
        silent: false,
    });
    driver.reply("I'm checking availability.", &[]);
    let (first, original, working) = last_turn(&driver);
    assert_eq!(original.task_ids, ["AST_booking"]);
    assert!(working);
    assert!(!driver.effects.iter().any(|effect| matches!(effect, SessionEffect::Finished { .. })));

    let queued = driver
        .session
        .admit_live(
            origin("question"),
            message("human_B", Role::User, "When is check-in?"),
            Ts::ZERO,
        )
        .unwrap();
    driver.effects.extend(queued);
    driver.send(SessionEvent::LiveTurnCommitted { turn: first, result: Ok(()) });
    driver.reply("Check-in is at 3pm.", &[]);
    let (side, source, working) = last_turn(&driver);
    assert_eq!(source.task_ids, ["AST_question"]);
    assert!(!working);
    driver.send(SessionEvent::LiveTurnCommitted { turn: side, result: Ok(()) });
    assert!(!driver.session.live_idle(), "the held body is still owned after the side reply");

    driver.send(SessionEvent::Run {
        invocation_id: owner,
        event: agent_run::AgentRunEvent::ToolSettled {
            call: body,
            report: ToolOutcome::Value {
                text: "Reservation R1".into(),
                reply_required: true,
                verbatim: false,
                more: None,
            }
            .into(),
        },
    });
    let (_, history) = driver.request();
    assert!(history.items.iter().any(|item| item.as_message().is_some_and(|message| {
        message.raw_text_content().as_deref() == Some("Check-in is at 3pm.")
    })));
    let outputs: Vec<_> = history
        .items
        .iter()
        .filter_map(|item| match item {
            ChatItem::FunctionCallOutput(output) if output.id == format!("{canonical}_out") => {
                Some(output)
            }
            _ => None,
        })
        .collect();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].output, "Reservation R1");
    driver.reply("Your reservation is R1.", &[]);
    let (final_turn, source, working) = last_turn(&driver);
    assert_eq!(source.task_ids, ["AST_booking"]);
    assert!(!working);
    assert_eq!(
        driver
            .effects
            .iter()
            .filter(|effect| matches!(
                effect,
                SessionEffect::Run { effect: agent_run::AgentRunEffect::ToolExecute { .. }, .. }
            ))
            .count(),
        1
    );
    driver.send(SessionEvent::LiveTurnCommitted { turn: final_turn, result: Ok(()) });
    assert!(driver.session.live_idle());
}

fn driver(config: SessionConfig) -> Driver {
    Driver {
        session: Session::new(config.clone(), input(None)).unwrap().with_live(origin("booking")),
        config,
        effects: Vec::new(),
        auto_release: true,
    }
}

fn body(driver: &Driver) -> (String, agent_run::CallId, FunctionCall) {
    driver
        .effects
        .iter()
        .rev()
        .find_map(|effect| match effect {
            SessionEffect::Run {
                invocation_id,
                effect: agent_run::AgentRunEffect::ToolExecute { call, function_call, .. },
            } => Some((invocation_id.clone(), *call, function_call.clone())),
            _ => None,
        })
        .unwrap()
}

fn value(text: &str) -> ToolOutcome {
    ToolOutcome::Value { text: text.into(), reply_required: true, verbatim: false, more: None }
}

fn root_body() -> (Driver, String, agent_run::CallId, FunctionCall) {
    let mut config = config(false, ChatHistory::Fresh, false);
    config.agents.get_mut("desk").unwrap().agent_tools.clear();
    let mut driver = driver(config);
    driver.send(SessionEvent::Start(message("human", Role::User, "Reserve a room.")));
    driver.reply("", &[("book", json!({"nights":2}))]);
    driver.prepare(json!({"nights":2}));
    let (invocation, body, call) = body(&driver);
    driver.run(agent_run::AgentRunEvent::ToolProgress {
        call: body,
        message: "Checking.".into(),
        silent: false,
    });
    driver.reply("I'm checking.", &[]);
    let (turn, _, working) = last_turn(&driver);
    assert!(working);
    driver.send(SessionEvent::LiveTurnCommitted { turn, result: Ok(()) });
    (driver, invocation, body, call)
}

#[test]
fn a_detached_final_question_uses_its_native_occurrence_and_original_canonical_call() {
    use lk_agents_abi::llm::tool_executor::ToolReport;
    use lk_agents_abi::session::{
        AgentInputRequest, InputRequestRef, InputRequestState, SessionInputCommand, ToolCallRef,
    };
    let (mut driver, original_invocation, body, call) = root_body();
    let agent = driver.session.state.active_frame_id.clone();
    let reference = InputRequestRef::Agent {
        agent_id: agent.value.clone(),
        request_id: "native-booking-question".into(),
    };
    driver
        .session
        .input(SessionInputCommand::Retain {
            agent_id: agent.value.clone(),
            request_id: "native-booking-question".into(),
            input: Box::new(AgentInputRequest {
                state: InputRequestState::Open,
                source_message: message("native-booking-question", Role::Assistant, "Which date?"),
                response_message: None,
                origin: Some(ToolCallRef {
                    agent_id: agent.value.clone(),
                    call_id: call.id.clone(),
                }),
                presented_message_id: None,
            }),
        })
        .unwrap();
    let before = driver.effects.len();
    driver.auto_release = false;
    driver.send(SessionEvent::Run {
        invocation_id: original_invocation.clone(),
        event: agent_run::AgentRunEvent::ToolSettled {
            call: body,
            report: ToolReport {
                input_request: Some(Box::new(reference.clone())),
                ..ToolOutcome::InputRequired { question: "Which date?".into(), verbatim: false }
                    .into()
            },
        },
    });
    let result = driver.effects[before..]
        .iter()
        .find_map(|effect| match effect {
            SessionEffect::InvocationSettled { result, .. } => Some(result),
            _ => None,
        })
        .expect("the existing terminal-question path presents the late body question");
    assert!(matches!(result.outcome, RunOutcome::InputRequired(_)));
    assert_eq!(result.input_requirements.len(), 1);
    let requirement = &result.input_requirements[0];
    assert_eq!(requirement.invocation_id, original_invocation);
    assert_eq!(requirement.call.id, call.id);
    assert_eq!(requirement.input_request, Some(reference));
    let presented = requirement.presentation_message_id.clone().unwrap();
    assert!(
        !driver.effects[before..].iter().any(|effect| matches!(
            effect,
            SessionEffect::Run { effect: agent_run::AgentRunEffect::LlmRequestStart { .. }, .. }
        )),
        "delegated questions do not become ordinary model continuations"
    );
    driver.present_text();
    let frame = &driver.session.state.frames[&agent];
    assert_eq!(
        frame.input_requests.len(),
        1,
        "the native occurrence is not copied into a fallback"
    );
    assert_eq!(
        frame.input_requests["native-booking-question"].presented_message_id,
        Some(presented)
    );
    assert!(frame.history.iter().any(|item| matches!(item,
        ChatItem::FunctionCallOutput(output) if output.id == call.output_id()
            && output.output == "waiting for the person's answer to: Which date?")));
    assert!(frame.history.iter().all(|item| !matches!(item,
        ChatItem::FunctionCall(call) if crate::tool_calls::provisional(call))));
}

#[test]
fn a_completed_child_keeps_its_accepted_output_until_all_owned_bodies_report() {
    let mut config = config(false, ChatHistory::Fresh, false);
    let mut lookup = config.agents["desk"].definition.tools[0].clone();
    lookup.spec.name = "lookup".into();
    config.agents.get_mut("booking").unwrap().definition.tools = vec![lookup];
    let mut driver = driver(config.clone());
    driver.send(SessionEvent::Start(message("human", Role::User, "Reserve a room.")));
    driver.reply("", &[("book", json!({"nights":2}))]);
    driver.prepare(json!({"nights":2}));
    let child = driver.session.state.active_frame_id.clone();
    let parent = driver.session.state.frames[&child].parent.clone().unwrap();
    let owner = driver.invocation();
    driver.reply("", &[("lookup", json!({"nights":1})), ("lookup", json!({"nights":2}))]);
    let calls: Vec<_> = driver
        .effects
        .iter()
        .filter_map(|effect| match effect {
            SessionEffect::Run {
                invocation_id,
                effect: agent_run::AgentRunEffect::ToolPrepare { call, arguments, .. },
            } if invocation_id == &owner => Some((*call, Value::Object(arguments.clone()))),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 2);
    for (call, arguments) in &calls {
        driver.run(agent_run::AgentRunEvent::ToolPrepared {
            call: *call,
            prepared: ResourceId(100 + call.0),
            identity: ArgumentIdentity::Json(arguments.clone()),
        });
    }
    for (call, _) in &calls {
        driver.run(agent_run::AgentRunEvent::ToolProgress {
            call: *call,
            message: "Checking.".into(),
            silent: false,
        });
    }
    let accepted = json!({"reference":"accepted-before-tools-finished"});
    driver.accepted(accepted.clone());
    let frame = &driver.session.state.frames[&child];
    assert!(frame.completed);
    assert_eq!(frame.output, Some(accepted.clone()));
    assert!(
        driver.session.state.frames[&parent.frame_id]
            .pending_calls
            .iter()
            .any(|pending| pending.parent == parent)
    );
    assert!(
        !driver.session.state.frames[&parent.frame_id].history.iter().any(|item| matches!(item,
        ChatItem::FunctionCallOutput(output) if output.id == format!("{}_out",parent.call_id)))
    );
    assert!(
        Session::new(config, input(Some(driver.session.snapshot()))).is_err(),
        "a saved progress result cannot recreate the local bodies or propagate accepted success"
    );
    let queued = driver
        .session
        .admit_live(origin("side"), message("side", Role::User, "When is check-in?"), Ts::ZERO)
        .unwrap();
    driver.effects.extend(queued);
    let (turn, _, working) = last_turn(&driver);
    assert!(working);
    let effects = driver.send(SessionEvent::LiveTurnCommitted { turn, result: Ok(()) });
    assert!(
        !effects.iter().any(|effect| matches!(effect, SessionEffect::InvocationStarted { .. })),
        "a queued human cannot replace an accepted child while its owned bodies remain"
    );

    let before = driver.effects.len();
    driver.send(SessionEvent::Run {
        invocation_id: owner.clone(),
        event: agent_run::AgentRunEvent::ToolSettled {
            call: calls[0].0,
            report: value("first settled").into(),
        },
    });
    assert!(driver.session.state.frames.contains_key(&child));
    assert_eq!(driver.session.state.frames[&child].output, Some(accepted.clone()));
    assert!(
        !driver.effects[before..]
            .iter()
            .any(|effect| matches!(effect, SessionEffect::InvocationStarted { .. }))
    );
    let (first, _, working) = last_turn(&driver);
    assert!(working);
    driver.send(SessionEvent::LiveTurnCommitted { turn: first, result: Ok(()) });
    driver.send(SessionEvent::Run {
        invocation_id: owner,
        event: agent_run::AgentRunEvent::ToolSettled {
            call: calls[1].0,
            report: value("second settled").into(),
        },
    });
    assert!(!driver.session.state.frames.contains_key(&child));
    let root = &driver.session.state.frames[&parent.frame_id];
    let results: Vec<_> = root
        .history
        .iter()
        .filter_map(|item| match item {
            ChatItem::FunctionCallOutput(output)
                if output.id == format!("{}_out", parent.call_id) =>
            {
                Some(output)
            }
            _ => None,
        })
        .collect();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].output, accepted.to_string());
    assert_eq!(
        driver
            .effects
            .iter()
            .filter(|effect| matches!(effect,
        SessionEffect::InvocationStarted {frame_id,..} if frame_id==&child))
            .count(),
        1,
        "the retained accepted child is propagated without running its model again"
    );
    driver.reply("Your room is reserved.", &[]);
    let (final_turn, source, working) = last_turn(&driver);
    assert_eq!(source.task_ids, ["AST_booking"]);
    assert!(!working);
    driver.send(SessionEvent::LiveTurnCommitted { turn: final_turn, result: Ok(()) });
    driver.reply("Check-in is at 3pm.", &[]);
    let (side, source, working) = last_turn(&driver);
    assert_eq!(source.task_ids, ["AST_side"]);
    assert!(!working);
    driver.send(SessionEvent::LiveTurnCommitted { turn: side, result: Ok(()) });
    assert!(driver.session.live_idle());
}

#[test]
fn losing_a_detached_local_body_retains_unknown_progress_and_refuses_resume() {
    let (mut driver, invocation, body, _) = root_body();
    let before = driver.effects.len();
    driver.send(SessionEvent::Run {
        invocation_id: invocation,
        event: agent_run::AgentRunEvent::CallLost { call: body, outcome: ToolOutcome::Cancelled },
    });
    assert!(driver.effects[before..].iter().any(|effect|matches!(effect,
        SessionEffect::LiveTurnFinished {result,..}
            if matches!(result.as_ref(),Err(TextSessionError::Routing(RoutingError::InvalidState(_)))))));
    assert!(Session::new(driver.config.clone(), input(Some(driver.session.snapshot()))).is_err());
    let (turn, _, _) = last_turn(&driver);
    driver.send(SessionEvent::LiveTurnCommitted { turn, result: Ok(()) });
    let effects = driver
        .session
        .admit_live(origin("retry"), message("retry", Role::User, "Any news?"), Ts::ZERO)
        .unwrap();
    assert!(effects.iter().any(|effect| matches!(effect,
        SessionEffect::LiveTurnFinished {result,..} if result.is_err())));
    assert!(
        !effects.iter().any(|effect| matches!(effect, SessionEffect::InvocationStarted { .. }))
    );
}

#[test]
fn detached_exact_output_is_minted_recorded_and_committed_under_its_original_source() {
    let mut config = config(false, ChatHistory::Fresh, false);
    config.agents.get_mut("desk").unwrap().agent_tools.clear();
    let mut driver = Driver {
        session: Session::new(config.clone(), input(None)).unwrap().with_live(origin("booking")),
        config,
        effects: Vec::new(),
        auto_release: true,
    };
    driver.send(SessionEvent::Start(message("human_A", Role::User, "Reserve a room.")));
    driver.reply("", &[("book", json!({"nights":2}))]);
    driver.prepare(json!({"nights":2}));
    let (owner, call) = driver
        .effects
        .iter()
        .rev()
        .find_map(|effect| match effect {
            SessionEffect::Run {
                invocation_id,
                effect: agent_run::AgentRunEffect::ToolExecute { call, .. },
            } => Some((invocation_id.clone(), *call)),
            _ => None,
        })
        .unwrap();
    driver.run(agent_run::AgentRunEvent::ToolProgress {
        call,
        message: "Checking availability.".into(),
        silent: false,
    });
    driver.reply("I'm checking availability.", &[]);
    let (first, _, _) = last_turn(&driver);
    let queued = driver
        .session
        .admit_live(
            origin("question"),
            message("human_B", Role::User, "When is check-in?"),
            Ts::ZERO,
        )
        .unwrap();
    driver.effects.extend(queued);
    driver.send(SessionEvent::LiveTurnCommitted { turn: first, result: Ok(()) });
    let foreground = driver.session.phase.invocation_mut().unwrap().id.clone();
    assert_ne!(foreground, owner);
    driver.session.recording = true;
    let output = agent_run::ToolOutputId(42);
    let effects = driver.send(SessionEvent::Run {
        invocation_id: owner.clone(),
        event: agent_run::AgentRunEvent::ToolOutputRequested {
            output,
            call: Some(call),
            text: "Reference ABC.".into(),
            stream: false,
        },
    });
    let exact = effects
        .iter()
        .find_map(|effect| match effect {
            SessionEffect::Run {
                invocation_id,
                effect: agent_run::AgentRunEffect::DeliverToolOutput { message, .. },
            } if invocation_id == &owner => Some(message.clone()),
            _ => None,
        })
        .unwrap();
    let effects = driver.send(SessionEvent::Run {
        invocation_id: owner.clone(),
        event: agent_run::AgentRunEvent::ToolOutputDelivered {
            output,
            outcome: lk_agents_abi::agent::run::SayOutcome::Delivered,
        },
    });
    assert!(effects.iter().any(|effect| matches!(effect, SessionEffect::ToolOutputCheckpoint { invocation_id, output: actual, .. } if invocation_id == &owner && actual == &output)));
    assert!(
        !effects.iter().any(|effect| matches!(
            effect,
            SessionEffect::Run { effect: agent_run::AgentRunEffect::ToolOutputSettled { .. }, .. }
        )),
        "the optional reader waits for its actual source checkpoint"
    );
    assert_eq!(
        driver
            .session
            .snapshot()
            .chat_history
            .iter()
            .filter(|message| message.id == exact.id)
            .count(),
        1
    );
    assert_eq!(driver.session.phase.invocation_mut().unwrap().id, foreground);
    assert!(
        !driver
            .session
            .phase
            .invocation_mut()
            .unwrap()
            .run
            .progress_new_items()
            .iter()
            .any(|item| item.id() == exact.id),
        "the current side turn cannot claim the original source's message"
    );
    let effects = driver.send(SessionEvent::ToolOutputCommitted {
        invocation_id: owner.clone(),
        output,
        result: Ok(()),
    });
    assert!(
        matches!(effects.as_slice(), [SessionEffect::Run { invocation_id, effect: agent_run::AgentRunEffect::ToolOutputSettled { outcome: lk_agents_abi::agent::run::SayOutcome::Delivered, .. } }] if invocation_id == &owner)
    );
    assert!(
        driver
            .send(SessionEvent::ToolOutputCommitted {
                invocation_id: owner,
                output,
                result: Ok(())
            })
            .is_empty()
    );
    assert!(!driver.session.live_idle(), "exact status did not finish the booking body");
}

#[test]
fn canceling_a_held_task_keeps_an_independent_foreground_reply_alive() {
    let (mut driver, owner, body, call) = root_body();
    let side = origin("independent");
    let effects = driver
        .session
        .admit_live(side.clone(), message("independent", Role::User, "When is check-in?"), Ts::ZERO)
        .unwrap();
    driver.effects.extend(effects);
    let foreground = driver.invocation();
    let tasks = vec!["AST_booking".to_owned()];
    assert!(driver.session.unowned_live_tasks(&tasks).is_empty());
    let effects = driver.session.cancel_live_tasks(&tasks, Ts::ZERO);
    assert_eq!(
        effects
            .iter()
            .filter(|effect| matches!(effect,
        SessionEffect::Run {invocation_id,effect:agent_run::AgentRunEffect::ToolCancel {call}}
            if invocation_id==&owner && *call==body))
            .count(),
        1
    );
    assert_eq!(driver.invocation(), foreground);
    driver.effects.extend(effects);
    driver.send(SessionEvent::Run {
        invocation_id: owner,
        event: agent_run::AgentRunEvent::ToolSettled {
            call: body,
            report: ToolOutcome::Cancelled.into(),
        },
    });
    assert_eq!(
        driver.invocation(),
        foreground,
        "the old body's cancellation cannot take the side reply"
    );
    driver.reply("Check-in is at 3pm.", &[]);
    let (turn, source, _) = last_turn(&driver);
    assert_eq!(source, side);
    driver.send(SessionEvent::LiveTurnCommitted { turn, result: Ok(()) });
    let (turn, source, working) = last_turn(&driver);
    assert_eq!(source.task_ids, ["AST_booking"]);
    assert!(!working);
    assert!(driver.effects.iter().rev().any(|effect| matches!(effect,
        SessionEffect::LiveTurnFinished {origin,result,..} if origin==&source
            && matches!(result.as_ref(),Err(TextSessionError::Run(RunError::Cancelled))))));
    assert_eq!(driver.session.state.frames[&driver.session.state.root_frame_id].history.iter()
        .filter(|item|matches!(item,ChatItem::FunctionCallOutput(output) if output.id==call.output_id())).count(),1);
    driver.send(SessionEvent::LiveTurnCommitted { turn, result: Ok(()) });
    assert!(driver.session.live_idle());
}
