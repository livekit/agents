//! A model failure is accounted by the session's error event, or by the interruption that
//! abandons its request.
use lk_agents_fuzz::drive::{Case, Setup, run};
use lk_agents_fuzz::input::{Input, Millis, Ref, ToolArgs};

fn judged(inputs: Vec<Input>) {
    let setup =
        Setup { tools: vec!["lookup".into()], max_unrecoverable_errors: 10, ..Setup::default() };
    let outcome = run(Case { setup, inputs, ..Case::default() });
    let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"provider_failure_is_accounted"), "{failed:?}");
    assert!(outcome.expectations.get("provider_failure_is_accounted").is_some_and(|n| *n > 0));
}

fn reply_with_a_call() -> Vec<Input> {
    vec![
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20) },
        Input::LlmToolCall {
            chat: Ref::new(0),
            name: "lookup".into(),
            args: ToolArgs("{}".into()),
        },
    ]
}

// Catches: a failure after the step's output, not retried and reported on the session's `error`
// event while the step's tool still ran, judged swallowed because the speech then ended
// interrupted rather than failed (survey seeds 4, 205, 265). The reference reports it through
// `_on_error` and never on the speech.
#[test]
fn a_failure_reported_as_an_error_event_is_accounted() {
    let mut inputs = reply_with_a_call();
    inputs.extend([
        Input::LlmFail { chat: Ref::new(0) },
        Input::Advance { ms: Millis(20) },
        // The tool still runs, so the interruption ends the speech only at its watchdog.
        Input::Interrupt { force: true, wait: false },
        Input::Advance { ms: Millis(6_000) },
    ]);
    judged(inputs);
}

// Catches: a failure delivered in the turn its speech was interrupted, judged swallowed though
// the interruption abandoned the request before it was read (survey seeds 207, 531, 583).
#[test]
fn a_failure_on_an_abandoned_request_is_accounted() {
    let mut inputs = reply_with_a_call();
    inputs.extend([
        Input::Interrupt { force: true, wait: false },
        Input::LlmFail { chat: Ref::new(0) },
        Input::Advance { ms: Millis(20) },
    ]);
    judged(inputs);
}
