//! The case's finish releases a synthesis stream only when it held one back.
use lk_agents_fuzz::drive::{Outcome, Run, finish, simulate_with};
use lk_agents_fuzz::input::Input;
use lk_agents_fuzz::record::Kind;
use lk_agents_fuzz::search;

// Catches: the finish answered every live synthesis stream with a release, though a stream the
// case never held produces on its own; one left open by a reply still waiting on authorization
// made every round "work to release", so the run hit its budget instead of going quiet and having
// its speeches judged (`session_settles`, survey seed 4743; seed 82 draws the scenario since the
// fakes' delays are hand-set).
#[test]
fn a_synthesis_stream_the_case_never_held_lets_the_run_go_quiet() {
    let seed = 82;
    let setup = search::sample_setup(&mut fastrand::Rng::with_seed(seed));
    let case = search::explore(setup, seed, 3_000).case;
    let held = |nth: u32| {
        case.inputs.iter().any(|i| matches!(i, Input::TtsHold { tts } if tts.nth == nth))
    };
    simulate_with(case.schedule, async {
        let mut run = Run::start(&case.setup, case.schedule, case.policy).await;
        for input in &case.inputs {
            run.apply(input).await;
        }
        // The scenario: a synthesis stream the case never held, still open when the close began.
        let r = &run.record;
        let close = r.close_begun().expect("the case closed the session");
        let open_at_close =
            r.opened().iter().filter(|o| o.kind == Kind::Tts && o.at < close).any(|o| {
                !held(o.nth)
                    && !r
                        .closed()
                        .iter()
                        .any(|(at, k, n)| *k == Kind::Tts && *n == o.nth && *at < close)
            });
        assert!(open_at_close, "the case left no unheld synthesis stream open at its close");
        let outcome = finish(run, Outcome { case: case.clone(), ..Outcome::default() }).await;
        assert!(!outcome.stopped_early, "{:?}", outcome.ending);
        assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
    });
}
