//! A recogniser that fails mid-utterance ends the speech it began.
use lk_agents_fuzz::drive::{Case, Setup, run};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs};

// Catches: the recogniser failing while the person spoke, so no end of speech ever came and the
// user was speaking for the rest of the call: never idle, a tool's `foreground()` waited on it,
// and the reply waiting on that tool ran until the close (`speech_settles`, survey seed 3036).
#[test]
fn a_failed_recogniser_does_not_leave_the_person_speaking() {
    let advance = || Input::Advance { ms: Millis(20) };
    let inputs = vec![
        Input::SttStart { stt: Ref::new(0) },
        advance(),
        Input::SttFail { stt: Ref::new(0) },
        advance(),
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        advance(),
        Input::LlmToolCall { chat: Ref::new(0), name: "lookup".into(), args: ToolArgs("{}".into()) },
        Input::LlmEnd { chat: Ref::new(0) },
        advance(),
        Input::ToolAnswered {
            tool: Ref::new(0),
            answer: ToolAnswer::AfterIdle,
            text: AgentText(String::new()),
        },
        advance(),
    ];
    let setup = Setup { stt: true, ..Setup::default() };
    let outcome = run(Case { setup, inputs, ..Case::default() });
    let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"speech_settles"), "{failed:?}");
    assert!(outcome.expectations.get("speech_settles").is_some_and(|n| *n > 0));
}
