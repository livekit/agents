//! The release phase ends a person the detector last heard start, whatever ended before.
use lk_agents_fuzz::drive::{Case, Setup, run};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};

// Catches: the release phase counting the detector's starts against its ends, so an end with no
// start before it cancelled a later start and the person was left speaking. A reply held at
// launch behind them waited until the close (`speech_settles`, survey seed 4677).
#[test]
fn a_start_after_a_stray_end_is_ended() {
    let vad = || Ref::new(0);
    let inputs = vec![
        Input::VadEnd { vad: vad(), silence_ms: Millis(500) },
        Input::Advance { ms: Millis(20) },
        Input::VadStart { vad: vad(), speech_ms: Millis(300) },
        Input::Advance { ms: Millis(20) },
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20) },
        Input::LlmText { chat: Ref::new(0), text: AgentText("Hello there.".into()) },
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(20) },
    ];
    let setup = Setup { vad: true, ..Setup::default() };
    let outcome = run(Case { setup, inputs, ..Case::default() });
    let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"speech_settles"), "{failed:?}");
    assert!(outcome.expectations.get("speech_settles").is_some_and(|n| *n > 0));
}
