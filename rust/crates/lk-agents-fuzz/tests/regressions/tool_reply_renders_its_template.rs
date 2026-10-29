//! A background tool's deferred reply is told what its `tool_handling` templates render.
use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
use lk_agents_fuzz::fakes::{TEMPLATED_REPLY, TEMPLATED_TOOL};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer};
use lk_agents_voice::trace::{Step, ToolStep};

// Catches (decision 109): the deferred reply scheduled with the framework's default instructions
// whatever the agent's or the tool's toolset's `reply_at_tail_template` /
// `reply_maybe_covered_template` said.
#[test]
fn a_deferred_reply_is_told_its_tools_template() {
    let setup = Setup {
        realtime: Some(RealtimeSetup {
            turn_detection: false,
            tool_calls: Some(100),
            ..RealtimeSetup::default()
        }),
        tools: vec![TEMPLATED_TOOL.into()],
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
        // News from the background, once the agent is idle.
        Input::ToolUpdate {
            tool: Ref::new(0),
            progress: Some(AgentText("It shipped.".into())),
            end: ToolAnswer::Return,
        },
        Input::Advance { ms: Millis(1_000) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let told: Vec<String> = run
        .record
        .journal()
        .iter()
        .filter_map(|(_, s)| match s {
            Step::Tool(ToolStep::ReplyScheduled { instructions, .. }) => Some(instructions.clone()),
            _ => None,
        })
        .collect();
    assert!(!told.is_empty(), "the update's reply was scheduled");
    assert!(told.iter().all(|t| t.starts_with(TEMPLATED_REPLY)), "told {told:?}");
}
