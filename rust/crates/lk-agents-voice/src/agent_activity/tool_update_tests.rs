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
        _: &'a Invocation,
    ) -> BoxFuture<'a, Result<String, HostError>> {
        unreachable!()
    }
    fn list_tools<'a>(
        &'a self,
        _: &'a Invocation,
    ) -> BoxFuture<'a, Result<Vec<lk_agents_abi::agent::FunctionToolInfo>, HostError>> {
        unreachable!()
    }
    fn prepare_tool<'a>(
        &'a self,
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
        _: &'a Invocation,
        _: &'a HostRef,
    ) -> BoxFuture<'a, Result<lk_agents_abi::agent::ReadBackAnswer, HostError>> {
        unreachable!()
    }
    fn run_tool<'a>(
        &'a self,
        _: &'a Invocation,
        _: &'a HostRef,
    ) -> BoxFuture<'a, Result<lk_agents_abi::agent::ToolCompletion, HostError>> {
        unreachable!()
    }
    fn validate_output<'a>(
        &'a self,
        _: &'a Invocation,
        _: lk_agents_abi::agent::OutputCandidate,
    ) -> BoxFuture<'a, Result<lk_agents_abi::agent::ValidatedOutput, HostError>> {
        unreachable!()
    }
    fn cancel(&self, _: &Invocation) {
        unreachable!()
    }
}

fn running(invocation_id: &str, operation: u64) -> Running {
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
        VoiceCommandOrigin::Hook,
        VoiceCommandOrigin::Amd,
        VoiceCommandOrigin::Ivr,
    ] {
        assert_eq!(driver.tool_speech(&caller), None);
    }
}
