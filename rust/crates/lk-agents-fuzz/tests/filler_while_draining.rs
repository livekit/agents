//! A filler whose say is refused while the session drains ends there.
use lk_agents_fuzz::drive::{Case, Setup, run};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolArgs};

// Catches: `fire_filler` counting a say the draining scheduler refused as spoken. The refused
// speech is interrupted at birth, the session stays idle, and with no delay and no interval the
// next fire comes in the same instant: the driver looped without virtual time moving (survey seed
// 1120 at 4096 seeds).
#[test]
fn a_filler_refused_while_draining_ends() {
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
        // The tool is still running when the close drains the scheduler.
        Input::Close { drain: true },
        Input::FillerStart {
            tool: Ref::new(0),
            speech: Ref::new(0),
            delay: Millis(0),
            interval: Some(Millis(0)),
            text: Some(AgentText("One moment.".into())),
        },
        Input::Advance { ms: Millis(50) },
    ];
    let outcome = run(Case { setup: Setup::default(), inputs, ..Case::default() });
    assert!(outcome.ok(), "{:#?}", outcome.failures);
}
