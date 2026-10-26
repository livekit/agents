//! A speech waiting out a retry's backoff settles in the release phase.
use lk_agents_fuzz::drive::{Case, Setup, run};
use lk_agents_fuzz::input::{AgentText, Input, Millis, SayInput};

// Catches: the release phase reading only the driver's timers for what it still waits on, so a
// retry's backoff (journalled as `RetryArmed`, waited out on the clock) was not waited for and
// its speech was judged stuck (`speech_settles`, survey seed 1381).
#[test]
fn a_speech_behind_a_retry_settles() {
    let inputs = vec![
        // Past the away timer, so no deadline of the driver's own covers the backoff.
        Input::Advance { ms: Millis(16_000) },
        Input::TtsOutage { down: true },
        Input::Say {
            text: AgentText("Hello there.".into()),
            input: SayInput::Words,
            allow: None,
            to_ctx: true,
        },
        Input::Advance { ms: Millis(20) },
        Input::TtsOutage { down: false },
    ];
    let outcome = run(Case { setup: Setup::default(), inputs, ..Case::default() });
    let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"speech_settles"), "{failed:?}");
    assert!(outcome.expectations.get("speech_settles").is_some_and(|n| *n > 0));
}
