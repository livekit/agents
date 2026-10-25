//! A filler that ends while the host is still sourcing its words finishes that hook too.
use lk_agents_fuzz::drive::{Case, Setup, run};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs};

// Catches: `end_filler` dropping the pending `FillerSource` hook without journalling
// `HookFinished`. The filler fires inside the `function_tools_executed` bound, the speech ends
// when that bound passes, and the host's answer then names a hook the driver no longer knows: the
// hook stayed open in the journal and the release phase never went quiet (soak seed 2).
#[test]
fn a_filler_ended_mid_source_finishes_its_hook() {
    let inputs = vec![
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20) },
        Input::LlmToolCall {
            chat: Ref::new(0),
            name: "lookup".into(),
            args: ToolArgs("{}".into()),
        },
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(20) },
        // No text: the host is asked for the filler's words when it fires.
        Input::FillerStart {
            tool: Ref::new(0),
            speech: Ref::new(0),
            delay: Millis(10),
            interval: None,
            text: None,
        },
        // `wait_for_idle()` in the body: the tool ends, and the speech with it, with no reply.
        Input::ToolAnswered {
            tool: Ref::new(0),
            answer: ToolAnswer::AfterIdle,
            text: AgentText(String::new()),
        },
        Input::Advance { ms: Millis(50) },
    ];
    let setup = Setup { hook_bound_ms: Some(20), ..Setup::default() };
    let outcome = run(Case { setup, inputs, ..Case::default() });
    assert!(outcome.ok(), "{:#?}", outcome.failures);
}
