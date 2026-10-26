//! A second model failure on one speech is accounted by the retry armed after it.
use lk_agents_fuzz::drive::{Case, Setup, run};
use lk_agents_fuzz::input::{Input, Millis, Ref};

// Catches: `provider_failure_is_accounted` reading only a speech's first retry, so every failure
// after the first was judged unretried though the driver armed a retry for each (survey seeds
// 106, 139 and 196, among twelve of 38).
#[test]
fn each_failure_has_its_own_retry() {
    let inputs = vec![
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20) },
        Input::LlmFail { chat: Ref::new(0) },
        Input::Advance { ms: Millis(5_000) },
        Input::LlmFail { chat: Ref::new(1) },
        Input::Advance { ms: Millis(20) },
    ];
    let setup = Setup { max_unrecoverable_errors: 10, ..Setup::default() };
    let outcome = run(Case { setup, inputs, ..Case::default() });
    let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"provider_failure_is_accounted"), "{failed:?}");
    let made = outcome.expectations.get("provider_failure_is_accounted").copied().unwrap_or(0);
    assert_eq!(made, 2, "both failures were judged");
}
