//! A filler whose say finishes at birth does not fire forever in one instant.
use lk_agents_fuzz::drive::{Case, RealtimeSetup, Setup, run};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};

// Catches: `with_filler(delay=0, interval=0)` on a realtime model's tool: the filler's say has no
// voice (a realtime session has no TTS), so it finished in the instant it was made, the session
// was idle again at once, and the next fire came in the same instant, forever: the session
// livelocked (`virtual_time_advances`, campaign seed 1090, once the fake realtime server called
// tools). A second fire in the instant of the last now ends the filler.
#[test]
fn a_voiceless_filler_with_no_interval_ends() {
    let setup = Setup {
        realtime: Some(RealtimeSetup {
            turn_detection: false,
            tool_calls: Some(100),
            ..RealtimeSetup::default()
        }),
        tools: vec!["lookup".into()],
        ..Setup::default()
    };
    let inputs = vec![
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(3_000) },
        Input::FillerStart {
            tool: Ref::new(0),
            speech: Ref::new(0),
            delay: Millis(0),
            interval: Some(Millis(0)),
            text: Some(AgentText("One moment.".into())),
        },
        // The reply's words play out; its tool runs on, and the agent is idle.
        Input::PlayoutStarted { segment: Ref::new(0) },
        Input::PlayoutFinished { segment: Ref::new(0) },
        Input::Advance { ms: Millis(1_000) },
    ];
    let outcome = run(Case { setup, inputs, ..Case::default() });
    let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"virtual_time_advances"), "{failed:?}");
}
