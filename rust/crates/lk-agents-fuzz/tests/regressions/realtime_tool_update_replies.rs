//! A realtime model answers a background tool's final update on its own socket.
use lk_agents_abi::RealtimeCommandKind;
use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer};
use lk_agents_voice::trace::{RealtimeStep, Step, ToolStep};

// Catches: the deferred reply to a detached tool's update (`ctx.update()`, then its final
// result) built as a pipeline reply on a realtime session, which has no LLM: it failed at once
// with "cannot generate a reply: the session has no llm", the update was reported skipped, and
// the person never heard the tool's result (survey seed 769, once the fake realtime server called
// tools). The reference's deferred reply is `session.generate_reply()`, which a realtime model
// answers on its own socket.
#[test]
fn a_detached_realtime_tool_is_answered_on_the_socket() {
    let setup = Setup {
        realtime: Some(RealtimeSetup {
            turn_detection: false,
            tool_calls: Some(100),
            ..RealtimeSetup::default()
        }),
        tools: vec!["lookup".into()],
        // One round of tools: the follow-up speaks without calling another.
        max_tool_steps: Some(1),
        ..Setup::default()
    };
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(3_000) },
        Input::PlayoutStarted { segment: Ref::new(0) },
        Input::PlayoutFinished { segment: Ref::new(0) },
        Input::Advance { ms: Millis(100) },
        // The tool answers with a progress update and keeps running in the background.
        Input::ToolAnswered {
            tool: Ref::new(0),
            answer: ToolAnswer::Detached,
            text: AgentText("Checking.".into()),
        },
        // The model speaks the progress, and calls the tool again, which returns; the speaker
        // plays it.
        Input::Advance { ms: Millis(3_000) },
        Input::ToolAnswered {
            tool: Ref::new(1),
            answer: ToolAnswer::Return,
            text: AgentText(String::new()),
        },
        Input::PlayoutStarted { segment: Ref::new(1) },
        Input::PlayoutFinished { segment: Ref::new(1) },
        Input::Advance { ms: Millis(3_000) },
        Input::PlayoutStarted { segment: Ref::new(2) },
        Input::PlayoutFinished { segment: Ref::new(2) },
        Input::Advance { ms: Millis(100) },
        // News from the background, then its end, once the agent is idle.
        Input::ToolUpdate { tool: Ref::new(0), progress: Some(AgentText("It shipped.".into())) },
        Input::Advance { ms: Millis(100) },
        Input::ToolUpdate { tool: Ref::new(0), progress: None },
        Input::Advance { ms: Millis(1_000) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let journal = run.record.journal();
    let scheduled = journal
        .iter()
        .position(|(_, s)| matches!(s, Step::Tool(ToolStep::ReplyScheduled { .. })))
        .expect("the update's reply was scheduled");
    let asked = journal[scheduled..].iter().any(|(_, s)| {
        matches!(
            s,
            Step::Realtime(RealtimeStep::Sent { kind: RealtimeCommandKind::GenerateReply, .. })
        )
    });
    assert!(asked, "the update's reply never went to the realtime server");
    let failed = run.record.host_view().speeches().iter().any(|s| {
        s.error().is_some_and(|e| e.contains("has no llm"))
    });
    assert!(!failed, "the update's reply failed as a pipeline reply");
}
