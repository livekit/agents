//! A queued update must still name the exact callback that owns the running call.

use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::ToolUpdate;
use crate::tool_executor::{HostedTool, Running, ToolOutcome};
use lk_agents_abi::agent::{AgentCallbacks, Invocation};
use lk_agents_abi::errors::HostError;
use lk_agents_abi::{BoxFuture, HostRef, Ts};
use std::sync::Arc;

struct UnusedHost;

impl AgentCallbacks for UnusedHost {
    fn render_instructions<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a Invocation,
    ) -> BoxFuture<'a, Result<String, HostError>> {
        unreachable!()
    }
    fn list_tools<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a Invocation,
    ) -> BoxFuture<'a, Result<Vec<lk_agents_abi::agent::FunctionToolInfo>, HostError>> {
        unreachable!()
    }
    fn prepare_tool<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a Invocation,
        _: serde_json::Map<String, serde_json::Value>,
    ) -> BoxFuture<
        'a,
        Result<
            Result<
                lk_agents_abi::agent::PreparedArguments,
                lk_agents_abi::agent::ToolArgumentError,
            >,
            HostError,
        >,
    > {
        unreachable!()
    }
    fn read_back<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a Invocation,
        _: &'a HostRef,
    ) -> BoxFuture<'a, Result<lk_agents_abi::agent::ReadBackAnswer, HostError>> {
        unreachable!()
    }
    fn run_tool<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a Invocation,
        _: &'a HostRef,
    ) -> BoxFuture<'a, Result<lk_agents_abi::agent::ToolCompletion, HostError>> {
        unreachable!()
    }
    fn validate_output<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a Invocation,
        _: lk_agents_abi::agent::OutputCandidate,
    ) -> BoxFuture<'a, Result<lk_agents_abi::agent::ValidatedOutput, HostError>> {
        unreachable!()
    }
    fn cancel(&self, _: &Invocation) {
        unreachable!()
    }
}

pub(in crate::agent_activity) fn running(invocation_id: &str, operation: u64) -> Running {
    let speech = SpeechId(7);
    let call = lk_agents_abi::FunctionCall {
        id: "new-call-item".into(),
        call_id: "reused".into(),
        name: "lookup".into(),
        arguments: "{}".into(),
        created_at: std::time::UNIX_EPOCH,
        extra: Default::default(),
        group_id: None,
    };
    let invocation = Invocation {
        operation,
        invocation_id: invocation_id.into(),
        agent_name: "agent".into(),
        execution_id: None,
        frame_id: None,
        context_scope: None,
        application_context: None,
        history: Vec::new(),
        function_call: Some(call.clone()),
        approval: None,
        speech: None,
        host_context: None,
    };
    Running {
        call,
        run_call: None,
        identity: None,
        executing: true,
        output_id: "new-output".into(),
        speech,
        step: 1,
        updates: 0,
        early: Vec::new(),
        hosted: HostedTool { host: Arc::new(UnusedHost), speech, invocation },
    }
}

#[test]
fn queued_progress_and_final_reports_cannot_target_a_reused_call_id() {
    let mut driver = crate::agent_session::tests::driver();
    driver.activity.tools.running.push(running("new-run", 2));
    let progress = || ToolUpdate::Progress { message: "working".into(), raw: "working".into() };
    driver.on_tool_update(Ts::ZERO, "old-run".into(), 2, "reused".into(), progress());
    driver.on_tool_update(
        Ts::ZERO,
        "new-run".into(),
        1,
        "reused".into(),
        ToolUpdate::Final { report: ToolOutcome::Nothing.into() },
    );
    assert!(driver.activity.tools.running[0].early.is_empty(), "stale receipt mutated replacement");
    driver.on_tool_update(Ts::ZERO, "new-run".into(), 2, "reused".into(), progress());
    assert_eq!(
        driver.activity.tools.running[0].early.len(),
        1,
        "the current callback still reports progress"
    );
}

#[test]
fn simultaneous_same_call_ids_use_the_exact_update_and_terminal_owner() {
    let mut driver = crate::agent_session::tests::driver();
    driver.activity.tools.running.push(running("old-run", 1));
    driver.activity.tools.running.push(running("new-run", 2));
    driver.on_tool_update(
        Ts::ZERO,
        "new-run".into(),
        2,
        "reused".into(),
        ToolUpdate::Progress { message: "new".into(), raw: "new".into() },
    );
    assert!(driver.activity.tools.running[0].early.is_empty());
    assert_eq!(driver.activity.tools.running[1].early.len(), 1);
    driver.activity.tools.running[1].updates = 1;
    driver.on_tool_update(
        Ts::ZERO,
        "new-run".into(),
        2,
        "reused".into(),
        ToolUpdate::Final { report: ToolOutcome::Nothing.into() },
    );
    assert_eq!(driver.activity.tools.running.len(), 1);
    assert_eq!(driver.activity.tools.running[0].hosted.invocation.invocation_id, "old-run");
    driver.activity.tools.running.push(running("replacement", 3));
    driver.on_tool_update(
        Ts::ZERO,
        "new-run".into(),
        2,
        "reused".into(),
        ToolUpdate::Final { report: ToolOutcome::Nothing.into() },
    );
    assert_eq!(
        driver.activity.tools.running.len(),
        2,
        "a stale final must not remove either live owner"
    );
}

#[test]
fn an_experts_question_keeps_the_exact_body_live_for_answer_progress_and_outcome() {
    use crate::experts::{ExpertTool, Floor, Presentation};
    let mut driver = crate::agent_session::tests::driver();
    let mut body = running("expert-run", 4);
    body.updates = 1;
    driver.activity.tools.running.push(body);
    let tools = [ExpertTool::new("https://expert.invalid/agent", "lookup")];
    driver.session.experts.open("reused", tools[0].clone());
    driver.session.experts.recorded("reused");
    driver.on_tool_update(
        Ts::ZERO,
        "expert-run".into(),
        4,
        "reused".into(),
        ToolUpdate::Final {
            report: ToolOutcome::InputRequired {
                question: "Which city?".into(),
                verbatim: false,
                tool: None,
            }
            .into(),
        },
    );
    assert_eq!(
        driver.activity.tools.running.len(),
        1,
        "an expert's input requirement is not body completion"
    );
    let floor = || Floor { tools: &tools, free: true, task: false };
    assert!(
        matches!(driver.session.experts.next(floor()), Some(Presentation::Question { question, .. }) if question == "Which city?")
    );
    let request = driver.session.experts.collect(floor()).expect("question owns its reply");
    driver.session.experts.answer(&request).expect("answer keeps original request identity");
    driver.on_tool_update(
        Ts::ZERO,
        "expert-run".into(),
        4,
        "reused".into(),
        ToolUpdate::Progress { message: "continued".into(), raw: "continued".into() },
    );
    assert!(
        matches!(driver.session.experts.next(floor()), Some(Presentation::Progress { text, .. }) if text == "continued")
    );
    driver.on_tool_update(
        Ts::ZERO,
        "expert-run".into(),
        4,
        "reused".into(),
        ToolUpdate::Final { report: ToolOutcome::Nothing.into() },
    );
    assert!(driver.activity.tools.running.is_empty());
    assert!(matches!(
        driver.session.experts.next(floor()),
        Some(Presentation::Outcome { outcome: ToolOutcome::Nothing, .. })
    ));
}

#[test]
fn foreground_holds_follow_the_captured_tool_owner_even_with_reused_call_ids() {
    use crate::agent_session::{SpeechCommand, VoiceCommandOrigin};
    let mut driver = crate::agent_session::tests::driver();
    let mut old = running("old-run", 1);
    old.speech = SpeechId(6);
    driver.activity.tools.running.push(old);
    driver.activity.tools.running.push(running("new-run", 2));
    let origin = |invocation_id: &str, operation| VoiceCommandOrigin::Tool {
        invocation_id: invocation_id.into(),
        operation,
        call_id: "reused".into(),
    };
    assert_eq!(driver.tool_speech(&origin("old-run", 1)), Some(SpeechId(6)));
    assert_eq!(driver.tool_speech(&origin("new-run", 2)), Some(SpeechId(7)));
    driver.activity.tools.running.remove(0);
    for (caller, expected) in [
        (origin("old-run", 1), None),
        (origin("new-run", 1), None),
        (origin("new-run", 2), Some(SpeechId(7))),
    ] {
        driver.command_origin = caller;
        driver.on_speech(Ts::ZERO, SpeechCommand::WaitForIdleAndHold { call: None });
        let asks = driver.session.idle.take_waiters();
        assert_eq!(asks.len(), 1);
        assert_eq!(asks[0].speech, expected, "stale tool must not own replacement foreground hold");
    }
    for caller in [
        VoiceCommandOrigin::Host,
        VoiceCommandOrigin::Hook { hook: "on_enter".into() },
        VoiceCommandOrigin::Amd,
        VoiceCommandOrigin::Ivr,
    ] {
        assert_eq!(driver.tool_speech(&caller), None);
    }
}

#[test]
fn forced_reply_cleanup_removes_exact_blocking_owners_and_preserves_other_lifetimes() {
    use crate::agent_session::{Now, effects::VoiceEffect};
    use crate::events::{SessionEvent, ToolEndStatus, ToolLifecycle};
    use crate::tool_executor::CallId;

    let mut driver = crate::agent_session::tests::driver();
    let speech = SpeechId(7);
    let mut first = running("reply", 11);
    first.run_call = Some(CallId(1));
    first.call.id = "first-item".into();
    let mut sibling = running("reply", 12);
    sibling.run_call = Some(CallId(2));
    sibling.call.id = "second-item".into();
    let mut detached = running("reply", 13);
    detached.run_call = Some(CallId(3));
    detached.updates = 1;
    let mut other = running("another-reply", 14);
    other.speech = SpeechId(8);
    other.hosted.speech = SpeechId(8);
    other.run_call = Some(CallId(1));
    driver.activity.tools.running.extend([first, sibling, detached, other]);

    driver.end_live_tools(Ts::ZERO, speech);
    let mut effects = driver.take_outbox();
    effects.extend(driver.handle(None, Now { at: Ts::ZERO, wall: std::time::UNIX_EPOCH }));
    let remaining: Vec<_> = driver
        .activity
        .tools
        .running
        .iter()
        .map(|running| running.hosted.invocation.operation)
        .collect();
    assert_eq!(remaining, [13, 14], "detached and unrelated speech work retain their owners");
    let cancelled: Vec<_> = effects
        .iter()
        .filter_map(|effect| match effect {
            VoiceEffect::CancelTool { hosted, .. } => Some(hosted.invocation.operation),
            _ => None,
        })
        .collect();
    assert_eq!(cancelled, [11, 12], "reused model IDs must not select another execution");
    let ended: Vec<_> = effects
        .iter()
        .enumerate()
        .filter_map(|(index, effect)| match effect {
            VoiceEffect::Emit(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Ended {
                status,
                ..
            })) => {
                assert_eq!(*status, ToolEndStatus::Cancelled);
                Some(index)
            }
            _ => None,
        })
        .collect();
    assert_eq!(ended.len(), 2, "each blocking owner publishes its end once");
    let first_item = effects
        .iter()
        .position(|effect| {
            matches!(effect,
                VoiceEffect::Emit(SessionEvent::ConversationItemAdded { item, .. })
                    if item.id() == "first-item" || item.id() == "second-item"
            )
        })
        .expect("force cleanup commits the ended calls' pairs");
    assert!(ended.iter().all(|index| *index < first_item), "ends precede committed pairs");

    driver.end_live_tools(Ts::ZERO, speech);
    assert!(driver.take_outbox().is_empty(), "repeated force cleanup must not end an owner twice");
}
