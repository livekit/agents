//! The case's finish releases a synthesis stream only when it held one back.
use lk_agents_fuzz::search;

// Catches: the finish answered every live synthesis stream with a release, though a stream the
// case never held produces on its own; one left open by a reply still waiting on authorization
// made every round "work to release", so the run hit its budget instead of going quiet and having
// its speeches judged (`session_settles`, survey seed 4743).
#[test]
fn a_synthesis_stream_the_case_never_held_lets_the_run_go_quiet() {
    let seed = 4743;
    let setup = search::sample_setup(&mut fastrand::Rng::with_seed(seed));
    let outcome = search::explore(setup, seed, 3_000);
    assert!(!outcome.stopped_early, "{:?}", outcome.ending);
    assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
}
