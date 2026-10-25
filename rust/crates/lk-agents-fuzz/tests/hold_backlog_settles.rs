//! A long queue of `wait_for_idle_and_hold()` asks drains in the release phase.
use lk_agents_fuzz::drive::{Case, Setup, run};
use lk_agents_fuzz::input::{Input, Millis};

// Catches: the release phase counting each round that granted the next queued hold against its
// cap. Holds are granted one at a time, so a case that queued more asks than the cap was reported
// as a session that never went quiet (`session_settles`, survey seed 16 with 105 asks).
#[test]
fn more_queued_holds_than_release_rounds_still_settle() {
    let mut inputs = vec![Input::WaitForIdleAndHold {}; 150];
    inputs.push(Input::Advance { ms: Millis(20) });
    let outcome = run(Case { setup: Setup::default(), inputs, ..Case::default() });
    let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"session_settles"), "{failed:?}");
    assert!(!outcome.stopped_early);
}
