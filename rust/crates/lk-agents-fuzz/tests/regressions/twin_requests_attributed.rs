//! Two requests planned on one context are each attributed to the speech that made them.
use lk_agents_fuzz::drive::{Case, Setup, run};
use lk_agents_fuzz::input::{Input, Millis, Ref, ToolArgs};

// Catches: requests with identical context ids left unattributed as ambiguous, so a failure
// injected into one was not found on its speech (`speech_error_is_injected`, survey seeds 331 and
// 531). The driver journals `ChatRequested` and opens the request in one poll, so plans and
// requests pair in order.
#[test]
fn a_failed_twin_request_is_its_speechs() {
    let reply = || Input::GenerateReply { user_input: None, instructions: None, allow: None };
    let inputs = vec![
        reply(),
        reply(),
        Input::Advance { ms: Millis(20) },
        // Output first, so the failure is not retried.
        Input::LlmToolCall {
            chat: Ref::new(1),
            name: "lookup".into(),
            args: ToolArgs("{}".into()),
        },
        Input::LlmFail { chat: Ref::new(1) },
        Input::Advance { ms: Millis(20) },
    ];
    let setup =
        Setup { tools: vec!["lookup".into()], max_unrecoverable_errors: 10, ..Setup::default() };
    let outcome = run(Case { setup, inputs, ..Case::default() });
    let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"speech_error_is_injected"), "{failed:?}");
    assert!(outcome.expectations.get("speech_error_is_injected").is_some_and(|n| *n > 0));
}
