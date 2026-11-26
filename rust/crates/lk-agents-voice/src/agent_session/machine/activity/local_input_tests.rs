use super::*;
use crate::agent_session::tests::{agent_update, driver};
use lk_agents_abi::session::{AgentInputRequest, InputRoute, ToolCallRef};
use lk_agents_abi::{ChatItem, FunctionCall, Ts};
use std::time::UNIX_EPOCH;

fn message(id: &str, role: Role, text: &str) -> ChatMessage {
    ChatMessage::text(id, role, text, UNIX_EPOCH)
}

fn question(
    state: &mut lk_agents_abi::session::snapshot::SessionSnapshot,
    id: &str,
) -> InputRequestRef {
    let agent_id = state.root_frame_id.value.clone();
    let reference = InputRequestRef::Agent { agent_id: agent_id.clone(), request_id: id.into() };
    let source = message(&format!("{id}/question"), Role::Assistant, "Please confirm the booking.");
    input::apply(
        state,
        vec![SessionInputCommand::Retain {
            agent_id: agent_id.clone(),
            request_id: id.into(),
            input: Box::new(AgentInputRequest {
                state: InputRequestState::Open,
                source_message: source.clone(),
                response_message: None,
                origin: Some(ToolCallRef { agent_id, call_id: id.into() }),
                presented_message_id: None,
            }),
        }],
    )
    .unwrap();
    state.chat_history.push(source.clone());
    input::apply(
        state,
        vec![SessionInputCommand::Presented {
            requests: vec![reference.clone()],
            message_id: source.id,
        }],
    )
    .unwrap();
    reference
}

fn waiting_call(id: &str) -> FunctionCall {
    let mut call = FunctionCall {
        id: id.into(),
        call_id: "reused-provider-id".into(),
        name: "book".into(),
        arguments: "{}".into(),
        created_at: UNIX_EPOCH,
        extra: Default::default(),
        group_id: None,
    };
    lk_agents_core::approval::set_mark(
        &mut call,
        lk_agents_abi::llm::tool_executor::Approval::Asked,
        "{}",
    );
    call
}

#[test]
fn a_side_question_does_not_consume_a_local_approval_occurrence() {
    let mut driver = driver();
    let mut state = lk_agents_abi::session::snapshot::SessionSnapshot::new(
        "session".into(),
        "task".into(),
        "agent".into(),
    );
    let request = question(&mut state, "call-item");
    driver.local_input_candidates(BTreeSet::from([request.clone()]));
    assert!(driver.needs_local_input_routing());
    let call = waiting_call("call-item");
    let output = crate::tool_executor::answer(
        &call,
        crate::tool_executor::ToolOutcome::InputRequired {
            question: "Please confirm the booking.".into(),
            verbatim: true,
        }
        .into(),
        UNIX_EPOCH,
    )
    .output;
    driver.commit_item(call.into(), Source::HostCtx, Visibility::Both);
    driver.commit_item(output.clone().into(), Source::HostCtx, Visibility::Both);
    let side = message("side", Role::User, "What does that cost?");
    state.chat_history.push(side.clone());
    let candidates = input::candidates(&state);
    input::route(&mut state, &side, &candidates, Some(Ok(InputRoute::NoMatch))).unwrap();
    assert!(driver.admit_local_responses(&mut state).unwrap().is_empty());
    assert!(driver.local_approval_answers("agent").is_empty());
    assert_eq!(
        state.frames[&state.root_frame_id].input_requests["call-item"].state,
        InputRequestState::Open
    );

    let answer = message("answer", Role::User, "Yes, book it.");
    driver.commit_item(answer.clone().into(), Source::HostCtx, Visibility::Both);
    state.chat_history.push(answer.clone());
    input::route(&mut state, &answer, &candidates, Some(Ok(InputRoute::Matched(vec![request]))))
        .unwrap();
    let delivery = driver.admit_local_responses(&mut state).unwrap().pop().unwrap();
    driver.local_input_delivered(delivery);
    assert!(!driver.needs_local_input_routing());
    assert_eq!(driver.local_approval_answers("agent"), BTreeSet::from(["call-item".into()]));
    assert_eq!(
        state.frames[&state.root_frame_id].input_requests["call-item"].state,
        InputRequestState::Responded
    );
    assert!(
        driver.session.chat_ctx.items().any(
            |item| matches!(item, ChatItem::FunctionCallOutput(recorded) if recorded == &output)
        ),
        "the source waiting outcome remains immutable"
    );
}

#[test]
fn a_checkpointed_match_keeps_its_agent_owner_when_the_floor_changes() {
    let mut driver = driver();
    let mut state = lk_agents_abi::session::snapshot::SessionSnapshot::new(
        "session".into(),
        "task".into(),
        "agent".into(),
    );
    let request = question(&mut state, "call-item");
    let answer = message("answer", Role::User, "Yes, book it.");
    driver.commit_item(answer.clone().into(), Source::HostCtx, Visibility::Both);
    state.chat_history.push(answer.clone());
    input::apply(
        &mut state,
        vec![SessionInputCommand::Respond { requests: vec![request], message_id: answer.id }],
    )
    .unwrap();
    let receipt = driver.admit_local_responses(&mut state).unwrap().pop().unwrap();
    driver.update_agent(Ts::ZERO, agent_update("another-agent", None));
    driver.local_input_delivered(receipt);
    assert!(driver.local_approval_answers("another-agent").is_empty());
    assert_eq!(driver.local_approval_answers("agent"), BTreeSet::from(["call-item".into()]));
}

#[test]
fn ending_a_live_body_closes_only_its_own_occurrences() {
    let mut driver = driver();
    let mut state = lk_agents_abi::session::snapshot::SessionSnapshot::new(
        "session".into(),
        "task".into(),
        "agent".into(),
    );
    question(&mut state, "saved-root");
    let body = question(&mut state, "body-request");
    driver.retain_body_question("body-output", "body-request", "body-call", "Which date?");
    driver.local_input_retained("body-request".into(), body.clone());
    driver.close_body_inputs("body-output");
    let closed: Vec<_> = driver
        .take_outbox()
        .into_iter()
        .filter_map(|effect| match effect {
            VoiceEffect::CloseLocalInputs { requests } => Some(requests),
            _ => None,
        })
        .flatten()
        .collect();
    assert_eq!(closed, vec![body]);
    input::apply(&mut state, vec![SessionInputCommand::Close { requests: closed }]).unwrap();
    assert_eq!(
        state.frames[&state.root_frame_id].input_requests["body-request"].state,
        InputRequestState::Closed
    );
    assert_eq!(
        state.frames[&state.root_frame_id].input_requests["saved-root"].state,
        InputRequestState::Open
    );
}

#[test]
fn retrying_a_local_checkpoint_cannot_reinterpret_a_revised_human_message() {
    let driver = driver();
    let mut state = lk_agents_abi::session::snapshot::SessionSnapshot::new(
        "session".into(),
        "task".into(),
        "agent".into(),
    );
    let request = question(&mut state, "call-item");
    let answer = message("answer", Role::User, "Yes, book it.");
    state.chat_history.push(answer.clone());
    input::apply(
        &mut state,
        vec![SessionInputCommand::Respond {
            requests: vec![request.clone()],
            message_id: answer.id.clone(),
        }],
    )
    .unwrap();
    let mut reloaded = state.clone();
    let receipt = driver.admit_local_responses(&mut state).unwrap().pop().unwrap();
    let revision = message("answer", Role::User, "No, do not book it.");
    *reloaded.chat_history.iter_mut().find(|message| message.id == answer.id).unwrap() = revision;
    input::apply(
        &mut reloaded,
        vec![SessionInputCommand::Correct {
            request,
            previous_message_id: answer.id.clone(),
            message_id: answer.id,
        }],
    )
    .unwrap();
    assert!(driver.readmit_local_responses(&mut reloaded, vec![receipt]).is_err());
    assert_eq!(
        reloaded.frames[&reloaded.root_frame_id].input_requests["call-item"].state,
        InputRequestState::Responding
    );
}

#[test]
fn a_successful_local_receipt_keeps_delivery_without_regranting_revised_human_words() {
    let mut driver = driver();
    driver.journalled = true;
    let mut state = lk_agents_abi::session::snapshot::SessionSnapshot::new(
        "session".into(),
        "task".into(),
        "agent".into(),
    );
    let reference = question(&mut state, "call-item");
    let answer = message("answer", Role::User, "Yes, book it.");
    driver.commit_item(answer.clone().into(), Source::HostCtx, Visibility::Both);
    state.chat_history.push(answer.clone());
    input::apply(
        &mut state,
        vec![SessionInputCommand::Respond {
            requests: vec![reference.clone()],
            message_id: answer.id.clone(),
        }],
    )
    .unwrap();
    let receipt = driver.admit_local_responses(&mut state).unwrap().pop().unwrap();
    let revised = message("answer", Role::User, "No, do not book it.");
    driver.commit_item(revised.clone().into(), Source::HostCtx, Visibility::Both);
    driver.local_input_delivered(receipt);
    assert!(driver.local_approval_answers("agent").is_empty());
    assert!(
        driver
            .session
            .chat_ctx
            .items()
            .any(|item| { matches!(item, ChatItem::Message(message) if message == &revised) })
    );
    let delivered = driver.take_outbox().into_iter().filter(|effect| matches!(effect,
        VoiceEffect::Journal(step) if matches!(step.as_ref(),
            crate::trace::Step::Tool(ToolStep::InputResponseDelivered { request, .. }) if request == &reference)
    )).count();
    assert_eq!(delivered, 1, "a successful frozen receipt still settles its exact delivery once");
    assert_eq!(
        state.frames[&state.root_frame_id].input_requests["call-item"].state,
        InputRequestState::Responded
    );
}

#[test]
fn metadata_normalization_keeps_approval_but_changed_human_content_or_identity_revokes_it() {
    use super::super::input::ReplyInputScope;
    use lk_agents_abi::llm::chat_context::protocol::{decode_chat_item, encode_chat_item};
    for changed in 0..4 {
        let mut driver = driver();
        let mut state = lk_agents_abi::session::snapshot::SessionSnapshot::new(
            "session".into(),
            "task".into(),
            "agent".into(),
        );
        let request = question(&mut state, "call-item");
        let mut answer = message("answer", Role::User, "Yes, book it.");
        answer.metrics.provider_request_ids = Some(vec!["stt-request".into()]);
        let scope = ReplyInputScope::Human(Box::new(answer.clone()));
        driver.commit_item(answer.clone().into(), Source::HostCtx, Visibility::Both);
        state.chat_history.push(answer.clone());
        input::apply(
            &mut state,
            vec![SessionInputCommand::Respond {
                requests: vec![request],
                message_id: answer.id.clone(),
            }],
        )
        .unwrap();
        let receipt = driver.admit_local_responses(&mut state).unwrap().pop().unwrap();
        let ChatItem::Message(mut normalized) =
            decode_chat_item(encode_chat_item(&answer.clone().into())).unwrap()
        else {
            panic!("a stored human remains a message")
        };
        assert_ne!(normalized, answer, "the real storage protocol normalizes metrics");
        driver.commit_item(normalized.clone().into(), Source::HostCtx, Visibility::Both);
        driver.local_input_delivered(receipt);
        let expected = BTreeSet::from(["call-item".into()]);
        assert_eq!(driver.scoped_local_approval_answers("agent", &scope), expected);

        normalized.transcript_confidence = Some(0.9);
        normalized.created_at += std::time::Duration::from_millis(20);
        normalized.extra.insert("timing-enriched".into(), serde_json::json!(true));
        normalized.metrics.transcription_delay = Some(0.02);
        driver.commit_item(normalized.clone().into(), Source::HostCtx, Visibility::Both);
        assert_eq!(
            driver.scoped_local_approval_answers("agent", &scope),
            expected,
            "metadata enrichment is the same human version"
        );
        match changed {
            0 => normalized.content = vec![lk_agents_abi::ChatContent::Text("No, stop.".into())],
            1 => normalized.content.push(lk_agents_abi::ChatContent::Image(
                lk_agents_abi::ImageContent {
                    id: "different-image".into(),
                    image: lk_agents_abi::ImageSource::Url("https://example.test/new.png".into()),
                    inference_width: None,
                    inference_height: None,
                    inference_detail: lk_agents_abi::InferenceDetail::Auto,
                    mime_type: None,
                },
            )),
            2 => normalized.id = "another-human".into(),
            _ => normalized.interrupted = true,
        }
        driver.commit_item(normalized.into(), Source::HostCtx, Visibility::Both);
        assert!(
            driver.scoped_local_approval_answers("agent", &scope).is_empty(),
            "changed words, images, human identity or interruption cannot keep approval"
        );
    }
}

#[test]
fn an_earlier_proposal_cannot_acquire_a_later_humans_approval() {
    scoped_approval_proposal(false, false);
}

#[test]
fn a_new_human_revokes_an_approval_before_a_deferred_proposal_is_admitted() {
    scoped_approval_proposal(false, true);
}

#[test]
fn a_realtime_proposal_accepts_delayed_matching_only_for_its_captured_human() {
    scoped_approval_proposal(true, false);
}

fn scoped_approval_proposal(realtime: bool, revoke: bool) {
    use super::super::input::{AfterCapture, ReplyInputGate};
    use super::super::reply_run::ReplyRun;
    use super::super::scheduling::SpeechKind;
    use super::super::speech::SpeechSpec;
    use crate::agent_session::{Answer, RequestId};
    use crate::generation::{PlanSpec, ReplySource};
    use crate::tool_executor::{AgentFrame, MadeCall, ToolGo};
    use lk_agents_abi::agent::{AgentDefinition, FunctionToolInfo, OutputSpec, ReadBack, Retries};
    use lk_agents_abi::llm::provider_format::tools::{DuplicateMode, DuplicateScope};
    use lk_agents_core::agent_run::{AgentRun, AgentRunEffect, AgentRunEvent, Opening, RunInput};

    let mut driver = driver();
    driver.journalled = true;
    let original = waiting_call("waiting-call");
    driver.commit_item(original.clone().into(), Source::HostCtx, Visibility::Both);
    let mut state = lk_agents_abi::session::snapshot::SessionSnapshot::new(
        "session".into(),
        "task".into(),
        "agent".into(),
    );
    let reference = question(&mut state, &original.id);
    let answer = message("answer", Role::User, "Yes, book it.");
    let old = message("earlier-human", Role::User, "Check the booking.");
    driver.commit_item(
        if revoke { answer.clone() } else { old.clone() }.into(),
        Source::HostCtx,
        Visibility::Both,
    );
    state.chat_history.push(answer.clone());
    input::apply(
        &mut state,
        vec![SessionInputCommand::Respond {
            requests: vec![reference.clone()],
            message_id: answer.id.clone(),
        }],
    )
    .unwrap();
    if revoke {
        let delivery = driver.admit_local_responses(&mut state).unwrap().pop().unwrap();
        driver.local_input_delivered(delivery);
    }

    let spec = lk_agents_abi::ToolSpec {
        name: "book".into(),
        description: "Book a room".into(),
        json_schema: serde_json::json!({"type":"object"}),
        flags: Default::default(),
        provider_tool: None,
    };
    struct Frame(AgentDefinition);
    impl AgentFrame for Frame {
        fn definition(&self) -> &AgentDefinition {
            &self.0
        }
        fn host(&self) -> &std::sync::Arc<dyn lk_agents_abi::agent::AgentCallbacks> {
            unreachable!("the test drives native admission without hosted callbacks")
        }
        fn host_context(&self) -> &lk_agents_abi::HostRef {
            unreachable!("the test drives native admission without hosted callbacks")
        }
    }
    let definition = AgentDefinition {
        name: "agent".into(),
        instructions: String::new(),
        dynamic_instructions: false,
        tools: vec![FunctionToolInfo {
            spec: spec.clone(),
            raw_schema: false,
            needs_approval: Some(ReadBack::Function),
            on_duplicate: DuplicateMode::Allow,
            duplicate_scope: DuplicateScope::Name,
            retries: None,
            remote: None,
            local_agent: None,
        }],
        output: OutputSpec::Text { validated: false },
        retries: Retries::default(),
    };
    let mut run = AgentRun::new(
        definition.clone(),
        RunInput {
            invocation_id: "scoped-proposal".into(),
            id_seed: 0,
            opening: Opening::Coordinated {
                approval_answers: driver.local_approval_answers("agent"),
            },
            history: driver.session.chat_ctx.items().cloned().collect(),
            delegated: false,
            mode: lk_agents_core::tool_outcome::Mode::Voice,
            llm: None,
            wall: lk_agents_abi::WallAnchor { wall: 0.0, at: Ts::ZERO },
            cleanup_timeout: lk_agents_abi::HOST_TEARDOWN_TIMEOUT,
        },
    )
    .unwrap();
    let request = run
        .handle(AgentRunEvent::Start, Ts::ZERO)
        .into_iter()
        .find_map(|effect| match effect {
            AgentRunEffect::PlanRequest { request, .. } => Some(request),
            _ => None,
        })
        .unwrap();
    let speech = driver.start_speech(
        Ts::ZERO,
        SpeechSpec::new(SpeechKind::Say, ReplySource::Text("pending reply".into()), true),
    );
    let mut reply = ReplyRun::new(
        run,
        Some(std::sync::Arc::new(Frame(definition))),
        PlanSpec {
            new_message: Some(if revoke { answer.clone() } else { old }),
            ..Default::default()
        },
        realtime,
        Some(request),
    );
    if realtime {
        reply.input_gate = ReplyInputGate::Capturing { request, after: AfterCapture::Human };
    }
    driver.activity.tools.replies.insert(speech, reply);
    if realtime {
        driver.expect_realtime_input(speech);
        driver.activity.speech_scheduler.speech_mut(speech).unwrap().authorize_generation();
    }
    driver.take_outbox();
    let proposal = FunctionCall {
        id: "proposal".into(),
        call_id: "proposal-wire".into(),
        name: "book".into(),
        arguments: "{}".into(),
        created_at: UNIX_EPOCH,
        extra: Default::default(),
        group_id: None,
    };
    driver.on_tool_made(
        Ts::ZERO,
        (speech, 1),
        MadeCall { call: proposal, spec: Some(spec), forbidden: false, agent: None },
        RequestId(700),
    );
    assert_eq!(driver.activity.tools.deferred.len(), 1);
    assert!(!driver.take_outbox().iter().any(|effect| matches!(effect,
        VoiceEffect::Journal(step) if matches!(step.as_ref(), crate::trace::Step::Tool(ToolStep::Calls { .. }))
    )), "authority is traced when this proposal is admitted");

    if revoke {
        driver.commit_item(
            message("new-human", Role::User, "No, stop.").into(),
            Source::HostCtx,
            Visibility::Both,
        );
    } else {
        driver.commit_item(answer.clone().into(), Source::HostCtx, Visibility::Both);
        let delivery = driver.admit_local_responses(&mut state).unwrap().pop().unwrap();
        if realtime {
            driver.route_realtime_input_for(speech, answer.clone());
            driver.input_offers(speech, request, vec![]);
            let routing = driver
                .take_outbox()
                .into_iter()
                .find_map(|effect| match effect {
                    VoiceEffect::RouteInput { request_id, .. } => Some(request_id),
                    _ => None,
                })
                .unwrap();
            driver.local_input_delivered(delivery);
            driver.input_routed(speech, routing, Some(Ok(Default::default())), Default::default());
        } else {
            driver.local_input_delivered(delivery);
        }
    }
    if !realtime {
        driver.activity.speech_scheduler.speech_mut(speech).unwrap().authorize_generation();
    }
    driver.settle_deferred_calls();
    assert!(driver.activity.tools.deferred.is_empty());
    let expected = if realtime { BTreeSet::from([original.id.clone()]) } else { BTreeSet::new() };
    assert_eq!(driver.activity.tools.replies[&speech].run.input_answers(), &expected);
    let effects = driver.take_outbox();
    let traced: Vec<_> = effects
        .iter()
        .filter_map(|effect| match effect {
            VoiceEffect::Journal(step) => match step.as_ref() {
                crate::trace::Step::Tool(ToolStep::Calls { item_id, input_answers, .. }) => {
                    Some((item_id.clone(), input_answers.clone()))
                }
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(traced, vec![("proposal".to_owned(), expected)]);
    let call = driver.activity.tools.replies[&speech].run.step_call_id("proposal-wire").unwrap();
    driver.on_tool_prepared(
        speech,
        call,
        Ok(lk_agents_abi::agent::ArgumentIdentity::Json(serde_json::json!({}))),
        RequestId(701),
    );
    let next = driver
        .take_outbox()
        .into_iter()
        .find_map(|effect| match effect {
            VoiceEffect::Answer(RequestId(701), Answer::ToolGo(go)) => Some(go),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        matches!(next, ToolGo::Execute { .. }),
        realtime,
        "only the proposal's own late-matched human can authorize its body"
    );
    if !realtime {
        assert!(matches!(next, ToolGo::ReadBack { .. }));
    }
}

fn body_question_driver() -> VoiceDriver {
    use crate::tool_executor::{AgentFrame, Toolset};
    use lk_agents_abi::agent::{
        AgentCallbacks, AgentDefinition, FunctionToolInfo, OutputSpec, Retries,
    };
    use lk_agents_abi::llm::provider_format::tools::{DuplicateMode, DuplicateScope};
    use std::sync::Arc;
    struct Frame(AgentDefinition);
    impl AgentFrame for Frame {
        fn definition(&self) -> &AgentDefinition {
            &self.0
        }
        fn host(&self) -> &Arc<dyn AgentCallbacks> {
            unreachable!("the prompt has no tool callback")
        }
        fn host_context(&self) -> &lk_agents_abi::HostRef {
            unreachable!("the prompt has no host callback")
        }
    }
    let mut driver = driver();
    let endpoint = "local://booking";
    let spec = lk_agents_abi::ToolSpec {
        name: "booking".into(),
        description: "Booking expert".into(),
        json_schema: serde_json::json!({"type":"object"}),
        flags: Default::default(),
        provider_tool: None,
    };
    let frame = Frame(AgentDefinition {
        name: "agent".into(),
        instructions: String::new(),
        dynamic_instructions: false,
        tools: vec![FunctionToolInfo {
            spec,
            raw_schema: true,
            needs_approval: None,
            on_duplicate: DuplicateMode::Allow,
            duplicate_scope: DuplicateScope::Name,
            retries: None,
            remote: None,
            local_agent: Some(endpoint.into()),
        }],
        output: OutputSpec::Text { validated: false },
        retries: Retries::default(),
    });
    driver.activity.toolset =
        Arc::new(Toolset { agent: Some(Arc::new(frame)), ..Default::default() });
    driver.retain_body_question("body-output", "body-question", "body-call", "Which date?");
    driver
        .session
        .experts
        .open("body-output", crate::experts::ExpertTool::local(endpoint, "booking"));
    driver.session.experts.ask("body-output", "body-question", "Which date?".into(), true, None);
    driver
}

#[test]
fn a_body_prompt_delivered_before_registration_retains_its_exact_presentation() {
    for retained_first in [false, true] {
        let mut driver = body_question_driver();
        let reference =
            InputRequestRef::Agent { agent_id: "root".into(), request_id: "body-question".into() };
        if retained_first {
            driver.local_input_retained("body-question".into(), reference.clone());
        }
        driver.try_present_experts(Ts::ZERO);
        let speech = driver
            .activity
            .speech_scheduler
            .speech_ids()
            .next()
            .expect("the expert prompt was scheduled");
        driver.activity.speech_scheduler.speech_mut(speech).unwrap().authorize_generation();
        driver.step_spoken(
            Ts::ZERO,
            speech,
            1,
            false,
            false,
            false,
            Some(Box::new(message("heard-body", Role::Assistant, "Which date?"))),
        );
        let mut effects = driver.take_outbox();
        if !retained_first {
            assert!(
                !effects.iter().any(|effect| matches!(effect, VoiceEffect::InputPresented { .. }))
            );
            driver.local_input_retained("body-question".into(), reference.clone());
            effects.extend(driver.take_outbox());
        }
        let presentations: Vec<_> = effects
            .into_iter()
            .filter_map(|effect| match effect {
                VoiceEffect::InputPresented { requests, message_id } => {
                    Some((requests, message_id))
                }
                _ => None,
            })
            .collect();
        assert_eq!(presentations, vec![(vec![reference], "heard-body".into())]);
    }
}

#[test]
fn an_interrupted_readback_stays_open_until_a_complete_prompt_is_delivered() {
    let mut driver = driver();
    let full = "Please confirm: Book two nights for Friday. Yes or no?";
    let mut state = lk_agents_abi::session::snapshot::SessionSnapshot::new(
        "session".into(),
        "task".into(),
        "agent".into(),
    );
    let reference = question(&mut state, "call-item");
    let input = state
        .frames
        .get_mut(&state.root_frame_id)
        .unwrap()
        .input_requests
        .get_mut("call-item")
        .unwrap();
    input.presented_message_id = None;
    input.source_message = message("call-item/question", Role::Assistant, full);
    let speech = driver
        .say_text(
            Ts::ZERO,
            full.into(),
            crate::agent_session::machine::activity::scheduling::VoiceRunParticipation::Outside,
        )
        .unwrap();
    driver.retain_local_question(speech, "call-item", full);
    driver.local_input_retained("call-item".into(), reference.clone());
    driver.presenting_local_input(speech, "call-item".into());
    driver.activity.speech_scheduler.speech_mut(speech).unwrap().authorize_generation();
    let mut partial = message("partial", Role::Assistant, "Please confirm:");
    partial.interrupted = true;
    driver.step_spoken(Ts::ZERO, speech, 1, true, false, false, Some(Box::new(partial)));
    assert!(
        !driver
            .take_outbox()
            .iter()
            .any(|effect| matches!(effect, VoiceEffect::InputPresented { .. }))
    );
    assert!(input::candidates(&state).is_empty(), "an unheard readback cannot authorize yes");
    assert_eq!(
        state.frames[&state.root_frame_id].input_requests["call-item"].state,
        InputRequestState::Open
    );

    let retry = driver
        .say_text(
            Ts::ZERO,
            full.into(),
            crate::agent_session::machine::activity::scheduling::VoiceRunParticipation::Outside,
        )
        .unwrap();
    driver.presenting_local_input(retry, "call-item".into());
    driver.activity.speech_scheduler.speech_mut(retry).unwrap().authorize_generation();
    let heard = message("heard-complete", Role::Assistant, full);
    driver.step_spoken(Ts::ZERO, retry, 1, false, false, false, Some(Box::new(heard.clone())));
    state.chat_history.push(heard);
    let presentations: Vec<_> = driver
        .take_outbox()
        .into_iter()
        .filter_map(|effect| match effect {
            VoiceEffect::InputPresented { requests, message_id } => {
                Some(SessionInputCommand::Presented { requests, message_id })
            }
            _ => None,
        })
        .collect();
    assert_eq!(presentations.len(), 1);
    input::apply(&mut state, presentations).unwrap();
    assert_eq!(input::candidates(&state).len(), 1, "the same request can receive a full retry");
}
