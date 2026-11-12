use super::offered_tool_infos;
use lk_agents_abi::ToolSpec;
use lk_agents_abi::agent::{FunctionToolInfo, ReadBack};
use lk_agents_abi::llm::provider_format::tools::{DuplicateMode, DuplicateScope};
use serde_json::json;

fn spec(name: &str) -> ToolSpec {
    ToolSpec {
        provider_tool: None,
        name: name.into(),
        description: "installed".into(),
        json_schema: json!({"type":"object"}),
        flags: Default::default(),
    }
}

#[test]
fn offered_schema_changes_preserve_installed_policies() {
    let installed = FunctionToolInfo {
        spec: spec("lookup"),
        raw_schema: true,
        needs_approval: Some(ReadBack::Default),
        on_duplicate: DuplicateMode::Reject,
        duplicate_scope: DuplicateScope::NameAndArgs,
        retries: Some(2),
        remote: Some("https://expert.invalid/lookup".into()),
    };
    let mut offered = installed.spec.clone();
    offered.description = "custom model-node description".into();
    let projected = offered_tool_infos(&[offered.clone()], &[installed]).unwrap();
    let [projected] = projected.as_slice() else { panic!("one offered tool") };
    assert_eq!(projected.spec, offered);
    assert!(projected.raw_schema);
    assert_eq!(projected.needs_approval, Some(ReadBack::Default));
    assert_eq!(projected.on_duplicate, DuplicateMode::Reject);
    assert_eq!(projected.duplicate_scope, DuplicateScope::NameAndArgs);
    assert_eq!(projected.retries, Some(2));
    assert_eq!(projected.remote.as_deref(), Some("https://expert.invalid/lookup"));
}

#[test]
fn unknown_offered_function_cannot_silently_disable_policy() {
    assert_eq!(
        offered_tool_infos(&[spec("unknown")], &[]).unwrap_err(),
        "an advertised function tool has no installed policy metadata"
    );
}

#[test]
fn cleanup_cancels_the_captured_call_even_after_its_acting_phase_is_gone() {
    use super::super::tools::tests::running;
    use super::*;

    let mut driver = crate::agent_session::tests::driver();
    let speech = SpeechId(7);
    let mut first = running("reply", 11);
    first.run_call = Some(CallId(1));
    first.executing = false; // Preparation/read-back already belong to the admitted call.
    let mut sibling = running("reply", 12);
    sibling.run_call = Some(CallId(2));
    driver.activity.tools.running.extend([first, sibling]);
    // Cleanup hides the run's acting calls; the live owner retains that link.
    assert!(!driver.activity.tools.replies.contains_key(&speech));
    driver.reply_effects(speech, vec![AgentRunEffect::ToolCancel { call: CallId(1) }]);
    let effects = driver.take_outbox();
    assert!(effects.iter().any(|effect| matches!(
        effect,
        VoiceEffect::CancelToolRefusal { speech: owner, call }
            if *owner == speech && *call == CallId(1)
    )));
    let cancelled: Vec<_> = effects
        .iter()
        .filter_map(|effect| match effect {
            VoiceEffect::CancelTool { hosted, .. } => Some(hosted.invocation.operation),
            _ => None,
        })
        .collect();
    assert_eq!(cancelled, [11], "a reused model ID cannot cancel its sibling");
}

#[test]
fn ending_an_admitted_call_does_not_remove_a_sibling_with_the_same_model_id() {
    use super::super::tools::tests::running;
    use super::*;

    let mut driver = crate::agent_session::tests::driver();
    let speech = SpeechId(7);
    let mut first = running("reply", 11);
    first.run_call = Some(CallId(1));
    let mut sibling = running("reply", 12);
    sibling.run_call = Some(CallId(2));
    driver.activity.tools.running.extend([first, sibling]);
    assert!(driver.activity.tools.end(speech, Some(CallId(2)), false));
    let remaining = &driver.activity.tools.running;
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].hosted.invocation.operation, 11);
    assert!(!driver.activity.tools.end(speech, None, false));
    assert_eq!(driver.activity.tools.running.len(), 1);
}

#[test]
fn shared_run_cleanup_deadline_wakes_voice_without_a_domain_timer() {
    use super::*;
    use lk_agents_abi::Ts;
    use std::time::{Duration, UNIX_EPOCH};

    let mut run = AgentRun::new(
        AgentDefinition {
            name: "agent".into(),
            instructions: String::new(),
            dynamic_instructions: false,
            tools: Vec::new(),
            output: OutputSpec::Structured { schema: json!({"type":"integer"}) },
            retries: Retries::default(),
        },
        RunInput {
            invocation_id: "cleanup-bound".into(),
            id_seed: 0,
            opening: Opening::Coordinated { approval_answers: Default::default() },
            history: Vec::new(),
            answer_tool: None,
            delegated: false,
            mode: lk_agents_core::tool_outcome::Mode::Voice,
            llm: None,
            wall: lk_agents_abi::WallAnchor { wall: 0.0, at: Ts::ZERO },
            cleanup_timeout: Duration::from_secs(5),
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
    let mut validations = Vec::new();
    for i in 0..2 {
        let call = FunctionCall {
            id: format!("candidate_{i}"),
            call_id: format!("call_{i}"),
            name: "final_result".into(),
            arguments: json!({"result":i}).to_string(),
            created_at: UNIX_EPOCH,
            extra: Default::default(),
            group_id: None,
        };
        validations.extend(
            run.handle(AgentRunEvent::CallMade { request, call }, Ts::ZERO).into_iter().filter_map(
                |effect| match effect {
                    AgentRunEffect::ValidateOutput { validation, .. } => Some(validation),
                    _ => None,
                },
            ),
        );
    }
    run.handle(AgentRunEvent::LlmEnded { request }, Ts::ZERO);
    assert_eq!(validations.len(), 2);
    run.handle(
        AgentRunEvent::OutputValidated {
            validation: validations[0],
            verdict: OutputVerdict::Failed("validator failed".into()),
        },
        Ts::ZERO,
    );
    let deadline = run.wake_at().expect("the other validation still owes cleanup");
    let mut driver = crate::agent_session::tests::driver();
    let speech = SpeechId(91);
    driver
        .activity
        .tools
        .replies
        .insert(speech, ReplyRun::new(run, None, PlanSpec::default(), false, None));
    assert_eq!(driver.wake_at(), Some(deadline));
    driver.handle(
        None,
        crate::agent_session::Now { at: deadline, wall: UNIX_EPOCH + Duration::from_secs(5) },
    );
    assert_eq!(driver.wake_at(), None, "cleanup bound cannot stay armed after terminal result");
    assert!(matches!(driver.activity.tools.replies[&speech].next, Some(StepNext::Done)));
    assert_eq!(
        driver.session.runs.outputs.get(&speech),
        Some(&VoiceRunOutcome::Failed(RunError::Host("validator failed".into())))
    );
}

#[test]
fn failed_reply_keeps_correlated_input_requirements_in_the_voice_run_failure() {
    use super::*;
    use lk_agents_abi::agent::run::RunInputRequirement;
    use std::time::Duration;
    let mut driver = crate::agent_session::tests::driver();
    driver.on_run(
        lk_agents_abi::Ts::ZERO,
        crate::run_result::RunSpec { user_input: None, host: None },
    );
    let speech = driver.start_speech(
        lk_agents_abi::Ts::ZERO,
        crate::agent_session::machine::activity::speech::SpeechSpec::new(
            crate::agent_session::machine::activity::scheduling::SpeechKind::Say,
            crate::generation::ReplySource::Text("owned".into()),
            true,
        ),
    );
    let run = AgentRun::new(
        AgentDefinition {
            name: "agent".into(),
            instructions: String::new(),
            dynamic_instructions: false,
            tools: Vec::new(),
            output: OutputSpec::Text { validated: false },
            retries: Retries::default(),
        },
        RunInput {
            invocation_id: "failed-reply".into(),
            id_seed: 0,
            opening: Opening::Coordinated { approval_answers: Default::default() },
            history: Vec::new(),
            answer_tool: None,
            delegated: false,
            mode: lk_agents_core::tool_outcome::Mode::Voice,
            llm: None,
            wall: lk_agents_abi::WallAnchor { wall: 0.0, at: lk_agents_abi::Ts::ZERO },
            cleanup_timeout: Duration::from_secs(5),
        },
    )
    .unwrap();
    driver
        .activity
        .tools
        .replies
        .insert(speech, ReplyRun::new(run, None, PlanSpec::default(), false, None));
    let requirement = RunInputRequirement {
        invocation_id: "failed-reply".into(),
        call: lk_agents_abi::a2a::turn::PendingCall {
            name: "ask".into(),
            call_id: "call-1".into(),
            answer_tool: Some("answer_city".into()),
        },
        question: "Which city?".into(),
        verbatim: true,
    };
    driver.reply_effects(
        speech,
        vec![AgentRunEffect::Finished {
            result: Box::new(Err(RunError::ToolRetriesExceeded {
                tool: "retry".into(),
                retries: 0,
            })),
            input_requirements: vec![requirement.clone()],
        }],
    );
    driver.release_speech(lk_agents_abi::Ts::ZERO, speech);
    driver.run_check();
    let settled = driver
        .settle_step()
        .into_iter()
        .find_map(|effect| match effect {
            VoiceEffect::Emit(crate::events::SessionEvent::RunSettled {
                result: Err(failure),
                ..
            }) => Some(failure),
            _ => None,
        })
        .unwrap();
    assert!(matches!(settled.error, RunError::ToolRetriesExceeded { .. }));
    assert_eq!(settled.input_requirements, [requirement]);
}

/// A reply with one `lookup` tool (read back first with `needs_approval`), its first step
/// requested and nothing else held.
fn tool_reply(
    driver: &mut crate::agent_session::VoiceDriver,
    speech: super::SpeechId,
    needs_approval: Option<ReadBack>,
) {
    use super::*;
    let info = FunctionToolInfo {
        spec: spec("lookup"),
        raw_schema: needs_approval.is_none(),
        needs_approval,
        on_duplicate: DuplicateMode::Allow,
        duplicate_scope: DuplicateScope::Name,
        retries: None,
        remote: None,
    };
    let mut run = AgentRun::new(
        AgentDefinition {
            name: "agent".into(),
            instructions: String::new(),
            dynamic_instructions: false,
            tools: vec![info],
            output: OutputSpec::Text { validated: false },
            retries: Retries::default(),
        },
        RunInput {
            invocation_id: "lost-call".into(),
            id_seed: 0,
            opening: Opening::Coordinated { approval_answers: Default::default() },
            history: Vec::new(),
            answer_tool: None,
            delegated: false,
            mode: lk_agents_core::tool_outcome::Mode::Voice,
            llm: None,
            wall: lk_agents_abi::WallAnchor { wall: 0.0, at: lk_agents_abi::Ts::ZERO },
            cleanup_timeout: std::time::Duration::from_secs(5),
        },
    )
    .unwrap();
    let request = run
        .handle(AgentRunEvent::Start, lk_agents_abi::Ts::ZERO)
        .into_iter()
        .find_map(|effect| match effect {
            AgentRunEffect::PlanRequest { request, .. } => Some(request),
            _ => None,
        })
        .unwrap();
    let reply = ReplyRun::new(run, None, PlanSpec::default(), false, Some(request));
    driver.activity.tools.replies.insert(speech, reply);
}

/// Every answer the driver owes, by request.
fn answers(
    driver: &mut crate::agent_session::VoiceDriver,
) -> Vec<(crate::agent_session::RequestId, crate::agent_session::Answer)> {
    driver
        .take_outbox()
        .into_iter()
        .filter_map(|effect| match effect {
            super::VoiceEffect::Answer(request, answer) => Some((request, answer)),
            _ => None,
        })
        .collect()
}

/// Where the call's task is when it dies.
#[derive(Clone, Copy, PartialEq)]
enum LostIn {
    Preparing,
    ReadingBack,
    Running,
}

#[test]
fn a_call_task_lost_mid_body_still_settles_its_step() {
    lost_call_settles_its_step(LostIn::Running);
}

#[test]
fn a_call_task_lost_while_preparing_still_settles_its_step() {
    lost_call_settles_its_step(LostIn::Preparing);
}

#[test]
fn a_call_task_lost_while_reading_back_still_settles_its_step() {
    lost_call_settles_its_step(LostIn::ReadingBack);
}

fn lost_call_settles_its_step(phase: LostIn) {
    use super::*;
    use crate::tool_executor::{MISSING_TOOL_OUTPUT, ToolOutcome};
    use lk_agents_abi::Ts;
    let mut driver = crate::agent_session::tests::driver();
    let speech = SpeechId(5);
    let approval = (phase == LostIn::ReadingBack).then_some(ReadBack::Function);
    tool_reply(&mut driver, speech, approval);
    let call = FunctionCall {
        id: "lookup-item".into(),
        call_id: "lookup-call".into(),
        name: "lookup".into(),
        arguments: "{}".into(),
        created_at: std::time::UNIX_EPOCH,
        extra: Default::default(),
        group_id: None,
    };
    let id = driver.reply_call_made(speech, call).expect("the run admits the call");
    assert!(matches!(driver.activity.reply_call_start(speech, id), ToolGo::Prepare { .. }));
    if phase != LostIn::Preparing {
        let prepared = RequestId(900);
        let identity = lk_agents_abi::agent::ArgumentIdentity::Json(json!({}));
        driver.on_tool_prepared(speech, id, Ok(identity), prepared);
        let go = answers(&mut driver).into_iter().find(|(r, _)| *r == prepared);
        let Some((_, Answer::ToolGo(go))) = go else { panic!("the prepared call moves on") };
        match phase {
            LostIn::ReadingBack => assert!(matches!(go, ToolGo::ReadBack { .. })),
            _ => assert!(matches!(go, ToolGo::Execute { .. })),
        }
    }
    // The call's task dies: the executor answers it for the step, without its run id.
    let lost = ToolOutcome::ToolError(MISSING_TOOL_OUTPUT.into());
    let status = Some(crate::events::ToolEndStatus::Error);
    driver.on_tool_ended(
        Ts::ZERO,
        (speech, None),
        "lookup-call".into(),
        status,
        None,
        lost.clone(),
    );
    let reply = &driver.activity.tools.replies[&speech];
    let (_, settled) = reply.run.step_call(id).expect("the step's call");
    let expected = if phase == LostIn::Running { lost } else { ToolOutcome::Internal };
    assert_eq!(settled.map(|s| &s.outcome), Some(&expected), "settled in the phase it waited");
    let waiter = RequestId(901);
    driver.on_tools_finished(Ts::ZERO, (speech, 1), step_end(), Some(waiter));
    assert!(
        answers(&mut driver).iter().any(|(r, a)| *r == waiter && matches!(a, Answer::ToolStep(_))),
        "the reply task waiting for its next step is answered"
    );
}

fn step_end() -> crate::tool_executor::StepEnd {
    crate::tool_executor::StepEnd {
        executions: Vec::new(),
        interrupted_before_tools: false,
        interrupted: false,
        forwarded_text_empty: true,
        max_tool_steps: 5,
        user_metrics: None,
    }
}

fn settled(step: u64) -> super::Settled {
    super::Settled { step: super::StepId(step), items: Vec::new(), terminal_call: None }
}

fn ended(step: u32) -> super::Ended {
    super::Ended { step, end: Box::new(step_end()) }
}

#[test]
fn step_results_hand_on_both_halves_in_either_order() {
    use super::StepResults;
    let mut results = StepResults::Open;
    assert!(results.settled(settled(1)).is_none());
    assert!(results.verdict().is_none(), "no verdict before the executor ends the step");
    let (s, e) = results.ended(ended(1)).expect("the second half hands both on");
    assert_eq!((s.step, e.step), (super::StepId(1), 1));
    assert_eq!(results.verdict(), Some(super::StepId(1)));
    assert!(matches!(results, StepResults::Open), "the verdict closes the step");

    assert!(results.ended(ended(2)).is_none());
    assert!(results.verdict().is_none(), "no verdict before the run settles the step");
    let (s, e) = results.settled(settled(2)).expect("the second half hands both on");
    assert_eq!((s.step, e.step), (super::StepId(2), 2));
    assert_eq!(results.verdict(), Some(super::StepId(2)));
    assert!(results.verdict().is_none(), "one verdict per step");
}

/// One reply's `instructions=` and `tool_choice=` stay with that reply and never reach the next.
/// https://github.com/livekit/agents/issues/4671 https://github.com/livekit/agents/issues/4242
/// https://github.com/livekit/agents/issues/1917 https://github.com/livekit/agents/issues/3834
#[test]
fn temporary_reply_configuration_does_not_leak() {
    use super::VoiceEffect;
    use crate::agent_session::ActivityCommand;
    use crate::generation::{GenEvent, Instructions, ReplyContext, TurnOutcome};
    use lk_agents_abi::{ToolChoice, Ts};
    let started = |effects: Vec<VoiceEffect>| -> Box<ReplyContext> {
        effects
            .into_iter()
            .find_map(|effect| match effect {
                VoiceEffect::StartSpeech { context, .. } => Some(context),
                _ => None,
            })
            .expect("the reply starts")
    };
    let texts = |context: &ReplyContext| -> String {
        let items = context.request.chat_ctx.items.iter();
        items.filter_map(|i| i.as_message()?.raw_text_content()).collect::<Vec<_>>().join("\n")
    };
    let mut driver = crate::agent_session::tests::driver();
    driver.on_frame(
        Ts::ZERO,
        ActivityCommand::UpdateInstructions(Instructions::plain("Speak English.")),
    );
    driver.take_outbox();

    let first = driver.generate_reply_speech(
        Ts::ZERO,
        None,
        Some("Only say goodbye.".into()),
        None,
        Some(ToolChoice::Forbidden),
        None,
        None,
        false,
        None,
    );
    let context = started(driver.settle_step());
    assert_eq!(context.tool_choice, Some(ToolChoice::Forbidden));
    assert!(texts(&context).contains("Only say goodbye."), "{}", texts(&context));

    // The persistent instructions change while that reply is out (as a tool of it would).
    driver.on_frame(
        Ts::ZERO,
        ActivityCommand::UpdateInstructions(Instructions::plain("Speak French.")),
    );
    let outcome =
        TurnOutcome { forwarded_text: "Goodbye.".into(), assistant_message: None, cut_at: None };
    driver.on_generation(
        Ts::ZERO,
        GenEvent::Done { speech: first, outcome: Box::new(outcome), interrupted: false },
    );
    driver.settle_step();

    driver.generate_reply_speech(Ts::ZERO, None, None, None, None, None, None, false, None);
    let context = started(driver.settle_step());
    let sent = texts(&context);
    assert_eq!(context.tool_choice, None, "the forced tool choice did not persist");
    assert!(!sent.contains("Only say goodbye."), "the one-off instructions leaked: {sent}");
    assert!(sent.contains("Speak French.") && !sent.contains("Speak English."), "{sent}");
}

/// Two sessions on one thread never mint the same assistant item id: each runs on the seed its
/// runtime drew, so its speeches, and the `{speech}/item_n` items of their replies, are its own.
#[test]
fn two_sessions_mint_distinct_reply_item_ids() {
    use super::VoiceEffect;
    use crate::agent_session::SessionParts;
    use crate::agent_session::tests::driver_with;
    use crate::agent_session::time::Now;
    use lk_agents_abi::Ts;
    let reply_item = |session_id: &str, id_seed: u64| -> String {
        let parts = SessionParts {
            agent: "agent".into(),
            session_id: session_id.into(),
            id_seed,
            max_unrecoverable_errors: 3,
            ..Default::default()
        };
        let mut driver =
            driver_with(parts, None, Now { at: Ts::ZERO, wall: std::time::UNIX_EPOCH });
        driver.generate_reply_speech(Ts::ZERO, None, None, None, None, None, None, false, None);
        let context = driver.settle_step().into_iter().find_map(|effect| match effect {
            VoiceEffect::StartSpeech { context, .. } => Some(context),
            _ => None,
        });
        context.expect("the reply starts").message_id
    };

    assert_ne!(reply_item("AS_first", 0x1111), reply_item("AS_second", 0x2222));
}
