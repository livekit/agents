//! A live option change reaches the next reply.
use lk_agents_fuzz::drive::{Case, Setup, run};
use lk_agents_fuzz::input::{
    AgentText, Count, Input, Knob, Millis, Ref, ToolAnswer, ToolArgs, TurnMode,
};

/// One tool step under `max_tool_steps = 0`, with the limit raised to `raise` first when given.
fn tool_step(raise: Option<u32>) -> Case {
    let mut inputs = Vec::new();
    if let Some(steps) = raise {
        inputs.push(Input::Configure {
            knob: Knob::MaxToolSteps,
            mode: TurnMode::Default,
            ms: Millis(0),
            on: false,
            words: Count(steps),
        });
    }
    inputs.extend([
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20) },
        Input::LlmToolCall {
            chat: Ref::new(0),
            name: "lookup".into(),
            args: ToolArgs("{}".into()),
        },
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(20) },
        Input::ToolAnswered {
            tool: Ref::new(0),
            answer: ToolAnswer::Value,
            text: AgentText("ok".into()),
        },
        Input::Advance { ms: Millis(200) },
    ]);
    Case { setup: Setup { max_tool_steps: Some(0), ..Setup::default() }, inputs, ..Case::default() }
}

fn limit_reached(case: Case) -> bool {
    run(case).reached.iter().any(|v| v == "Core::PortFailure")
}

// Catches: replies reading the options as of the session's start, so `update_options(
// max_tool_steps=)` never reaches a reply.
#[test]
fn a_raised_max_tool_steps_reaches_the_next_reply() {
    assert!(
        limit_reached(tool_step(None)),
        "the step limit of 0 was never hit: the probe is blind"
    );
    assert!(!limit_reached(tool_step(Some(3))), "the raised limit did not reach the reply");
}
