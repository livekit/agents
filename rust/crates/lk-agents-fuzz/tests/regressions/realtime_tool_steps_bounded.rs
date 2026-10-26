//! A realtime model's tool rounds stop at `max_tool_steps`.
use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer};
use lk_agents_voice::trace::{Step, ToolStep};

// Catches: the realtime follow-up to a tool step sent with no `tool_choice`, so neither a spent
// step budget nor a draining agent stopped the model calling tools again: a model whose prompt
// has it call a tool every response looped, tool after tool, and the reply never ended (found
// once the fake realtime server called tools). The reference's `_realtime_generation_task` asks
// for the follow-up with `tool_choice="none"` when `max_tool_steps` is reached or the activity
// drains; the pipeline step already did.
#[test]
fn a_realtime_reply_stops_calling_tools_at_its_budget() {
    let setup = Setup {
        realtime: Some(RealtimeSetup {
            turn_detection: false,
            tool_calls: Some(100),
            ..RealtimeSetup::default()
        }),
        tools: vec!["lookup".into()],
        max_tool_steps: Some(1),
        ..Setup::default()
    };
    let mut inputs = vec![Input::GenerateReply { user_input: None, instructions: None, allow: None }];
    // Every call the model makes is answered and every segment played, for a few rounds.
    for round in 0..4 {
        inputs.push(Input::Advance { ms: Millis(3_000) });
        inputs.push(Input::PlayoutStarted { segment: Ref::new(round) });
        inputs.push(Input::PlayoutFinished { segment: Ref::new(round) });
        inputs.push(Input::ToolAnswered {
            tool: Ref::new(round),
            answer: ToolAnswer::Value,
            text: AgentText("ok".into()),
        });
    }
    inputs.push(Input::Advance { ms: Millis(3_000) });
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let calls = run
        .record
        .journal()
        .iter()
        .filter(|(_, s)| matches!(s, Step::Tool(ToolStep::CallStarted { .. })))
        .count();
    // The first reply's call, then one more step's (`max_tool_steps=1`), then the reply that may call
    // none.
    assert_eq!(calls, 2, "the model called tools past its budget of one step");
}
