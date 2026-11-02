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
    driver.activity.tools.replies.insert(
        speech,
        ReplyRun {
            run,
            agent: None,
            plan: PlanSpec {
                record: None,
                new_message: None,
                instructions: None,
                turn_instructions: None,
                tools: Vec::new(),
                on_enter: false,
                running_tool_calls: Vec::new(),
                audio_output_enabled: false,
                text_output_enabled: false,
                expressive: None,
            },
            realtime: false,
            request: None,
            calls: Default::default(),
            acting: false,
            cut: false,
            settling: None,
            settled_items: None,
            terminal_call: None,
            tool_end: None,
            verdict: None,
            reply_required: Default::default(),
            waiter: None,
            next: None,
            previous_user_metrics: None,
            accepted: Default::default(),
        },
    );
    assert_eq!(driver.wake_at(), Some(deadline));
    driver.handle(
        None,
        crate::agent_session::Now { at: deadline, wall: UNIX_EPOCH + Duration::from_secs(5) },
    );
    assert_eq!(driver.wake_at(), None, "cleanup bound cannot stay armed after terminal result");
    assert!(matches!(driver.activity.tools.replies[&speech].next, Some(StepNext::Done)));
    assert_eq!(
        driver.session.runs.outputs.get(&speech),
        Some(&RunOutput::Failed(RunError::Host("validator failed".into())))
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
        crate::agent_activity::speech::SpeechSpec::new(
            crate::agent_activity::scheduling::SpeechKind::Say,
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
    driver.activity.tools.replies.insert(
        speech,
        ReplyRun {
            run,
            agent: None,
            plan: PlanSpec {
                record: None,
                new_message: None,
                instructions: None,
                turn_instructions: None,
                tools: Vec::new(),
                on_enter: false,
                running_tool_calls: Vec::new(),
                audio_output_enabled: false,
                text_output_enabled: false,
                expressive: None,
            },
            realtime: false,
            request: None,
            calls: Default::default(),
            acting: false,
            cut: false,
            settling: None,
            settled_items: None,
            terminal_call: None,
            tool_end: None,
            verdict: None,
            reply_required: Default::default(),
            waiter: None,
            next: None,
            previous_user_metrics: None,
            accepted: Default::default(),
        },
    );
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
            VoiceEffect::Emit(SessionEvent::RunSettled { outcome, input_requirements, .. }) => {
                Some((outcome, input_requirements))
            }
            _ => None,
        })
        .unwrap();
    assert!(matches!(
        settled.0,
        crate::events::RunOutcome::Failed(crate::events::RunFailure::Run(
            RunError::ToolRetriesExceeded { .. }
        ))
    ));
    assert_eq!(settled.1, [requirement]);
}
