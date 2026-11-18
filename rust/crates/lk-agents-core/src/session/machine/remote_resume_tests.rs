use super::*;
use lk_agents_abi::a2a_protocol as a2a;
use lk_agents_abi::delegation::remote::{A2ARequest, A2AResponse, RemoteCommand, RemoteInput};
use lk_agents_abi::llm::tool_executor::{REMOTE_REPORT_KEY, RemoteReport, ToolOutcome};

fn saved_child() -> (SessionConfig, SessionSnapshot, FunctionCall) {
    let mut config = config(false, ChatHistory::Fresh, false);
    let mut driver = Driver::new(config.clone(), None, "Book");
    driver.reply("", &[("book", json!({"nights":2}))]);
    driver.prepare(json!({"nights":2}));
    driver.auto_release = false;
    driver.accepted(json!({"reference":"accepted"}));
    let mut saved = driver.session.snapshot();
    let child = saved.active_frame_id.clone();
    let mut tool = config.agents["desk"].definition.tools[0].clone();
    tool.spec.name = "remote_booking".into();
    tool.remote = Some("booking-service".into());
    config.agents.get_mut("booking").unwrap().definition.tools.push(tool);
    let call = FunctionCall {
        id: "remote-original".into(),
        call_id: "provider-id".into(),
        name: "remote_booking".into(),
        arguments: r#"{"request":"Book a seat"}"#.into(),
        created_at: std::time::UNIX_EPOCH,
        extra: [(REMOTE_REPORT_KEY.into(), serde_json::to_value(RemoteReport::Pending).unwrap())]
            .into_iter()
            .collect(),
        group_id: None,
    };
    saved.frames.get_mut(&child).unwrap().history.extend([
        ChatItem::FunctionCall(call.clone()),
        lk_agents_abi::FunctionCallOutput::new(
            call.output_id(),
            &call.call_id,
            "Checking",
            false,
            std::time::UNIX_EPOCH,
        )
        .into(),
    ]);
    for input in [
        RemoteInput::Prepare { request_id: "original-request".into(), request: A2ARequest::Send(Box::new(serde_json::from_value(json!({"message":{"messageId":"original-message","role":"ROLE_USER","parts":[{"text":"Book a seat"}]}})).unwrap())), streaming: false, origin: Some(lk_agents_abi::session::ToolCallRef { agent_id: child.value, call_id: call.id.clone() }), input_request: None, control_scope: None },
        RemoteInput::DispatchStarted { request_id: "original-request".into() },
        RemoteInput::Response { request_id: "original-request".into(), response: Box::new(A2AResponse::Send(a2a::SendMessageResponse::Task(serde_json::from_value(json!({"id":"booking", "contextId":"trip", "status":{"state":"TASK_STATE_WORKING"}})).unwrap()))) },
        RemoteInput::Ended { request_id: "original-request".into() },
    ] {
        crate::session::remote::apply(&mut saved, RemoteCommand { remote_id: "booking-service".into(), endpoint: "https://booking.test".into(), input }).unwrap();
    }
    (config, saved, call)
}

fn finish_remote(driver: &mut Driver, call: &FunctionCall, state: &str) {
    let origin = lk_agents_abi::session::ToolCallRef {
        agent_id: driver.session.state.active_frame_id.value.clone(),
        call_id: call.id.clone(),
    };
    for input in [
        RemoteInput::Prepare {
            request_id: "reconcile".into(),
            request: A2ARequest::Get(a2a::GetTaskRequest {
                id: "booking".into(),
                history_length: None,
                tenant: None,
            }),
            streaming: false,
            origin: Some(origin),
            input_request: None,
            control_scope: None,
        },
        RemoteInput::DispatchStarted { request_id: "reconcile".into() },
        RemoteInput::Response {
            request_id: "reconcile".into(),
            response: Box::new(A2AResponse::Task(
                serde_json::from_value(
                    json!({"id":"booking","contextId":"trip","status":{"state":state}}),
                )
                .unwrap(),
            )),
        },
        RemoteInput::Ended { request_id: "reconcile".into() },
    ] {
        driver
            .session
            .remote(RemoteCommand {
                remote_id: "booking-service".into(),
                endpoint: "https://booking.test".into(),
                input,
            })
            .unwrap();
    }
}

#[test]
fn a_saved_accepted_child_waits_for_its_original_remote_report_before_parent_propagation() {
    let (config, saved, call) = saved_child();
    let child = saved.active_frame_id.clone();
    let parent_call = saved.frames[&child].parent.clone().unwrap();
    let mut driver = Driver::new(config, Some(saved), "Thanks");
    assert_eq!(driver.session.state.active_frame_id, child);
    assert!(!driver.effects.iter().any(|effect| matches!(
        effect,
        SessionEffect::Run { effect: agent_run::AgentRunEffect::LlmRequestStart { .. }, .. }
    )));
    finish_remote(&mut driver, &call, "TASK_STATE_COMPLETED");
    driver.prepare(json!({"request":"Book a seat"}));
    let internal = driver.effects.iter().find_map(|effect| match effect { SessionEffect::Run { effect:agent_run::AgentRunEffect::ToolExecute { call: internal, function_call, .. }, .. } => { assert_eq!(function_call.id, call.id); Some(*internal) }, _=>None }).unwrap();
    driver.run(agent_run::AgentRunEvent::ToolSettled {
        call: internal,
        report: ToolOutcome::Value {
            text: "Booked".into(),
            reply_required: true,
            verbatim: false,
            more: None,
        }
        .into(),
    });
    assert_eq!(driver.session.state.frames.len(), 1);
    let parent = &driver.session.state.frames[&parent_call.frame_id];
    assert_eq!(parent.history.iter().filter(|item| matches!(item, ChatItem::FunctionCallOutput(output) if output.id==format!("{}_out",parent_call.call_id) && output.output==r#"{"reference":"accepted"}"#)).count(),1);
}

#[test]
fn an_ambiguous_old_remote_report_refuses_without_propagating_the_accepted_child() {
    let (config, mut saved, call) = saved_child();
    let active = saved.active_frame_id.clone();
    for item in &mut saved.frames.get_mut(&active).unwrap().history {
        if let ChatItem::FunctionCall(item) = item
            && item.id == call.id
        {
            item.extra.remove(REMOTE_REPORT_KEY);
        }
    }
    let driver = Driver::new(config, Some(saved), "Thanks");
    assert!(driver.session.state.frames[&active].completed);
    assert!(driver.effects.iter().any(|effect| matches!(
        effect,
        SessionEffect::Finished {
            result: Err(TextSessionError::Routing(RoutingError::InvalidState(_))),
            ..
        }
    )));
    assert!(!driver.effects.iter().any(|effect| matches!(
        effect,
        SessionEffect::Run {
            effect: agent_run::AgentRunEffect::ToolPrepare { .. }
                | agent_run::AgentRunEffect::LlmRequestStart { .. },
            ..
        }
    )));
}
