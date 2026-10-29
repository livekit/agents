//! A realtime session swapped for a fresh one takes the person's turn along, at the finish too.
use lk_agents_fuzz::drive::{Outcome, Run, finish, simulate_with};
use lk_agents_fuzz::input::{Input, RtEvent};
use lk_agents_fuzz::search;

// Catches: the finish ended a person the realtime server had heard start, but read only a dropped
// connection as taking that turn along; after a fatal error the fallback swapped in a fresh
// session that never heard them, refused the stop every round, and the run hit its budget
// (`session_settles`, survey seeds 21875, 26055, 31823).
#[test]
fn a_swap_after_the_server_heard_the_person_lets_the_run_go_quiet() {
    let seed = 21875;
    let setup = search::sample_setup(&mut fastrand::Rng::with_seed(seed));
    let case = search::explore(setup, seed, 3_000).case;
    simulate_with(case.schedule, async {
        let mut run = Run::start(&case.setup, case.schedule, case.policy).await;
        for input in &case.inputs {
            run.apply(input).await;
        }
        // The scenario: the server's last word on the person was their start, then a swap.
        let heard: Vec<_> = run
            .record
            .applied()
            .iter()
            .filter(|a| a.accepted)
            .filter_map(|a| match &a.input {
                Input::RealtimeEvent { event, .. }
                    if matches!(
                        event,
                        RtEvent::SpeechStarted | RtEvent::SpeechStopped | RtEvent::FatalError
                    ) =>
                {
                    Some(event.clone())
                }
                _ => None,
            })
            .collect();
        assert!(
            matches!(heard.as_slice(), [.., RtEvent::SpeechStarted, RtEvent::FatalError]),
            "the scenario: a swap after the server heard the person start ({:?})",
            heard.iter().rev().take(3).collect::<Vec<_>>()
        );
        let outcome = finish(run, Outcome { case: case.clone(), ..Outcome::default() }).await;
        assert!(!outcome.stopped_early, "{:?}", outcome.ending);
        assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
    });
}
