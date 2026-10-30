//! A realtime session swapped for a fresh one takes the person's turn along, at the finish too.
use lk_agents_fuzz::drive::{Outcome, RealtimeSetup, Run, Setup, finish, simulate};
use lk_agents_fuzz::input::{Count, Input, Millis, Ref, RtEvent};

// Catches: the finish ended a person the realtime server had heard start, but read only a dropped
// connection as taking that turn along; after a fatal error the fallback swapped in a fresh
// session that never heard them, refused the stop every round, and the run hit its budget
// (`session_settles`, survey seeds 21875, 26055, 31823; 05ed23451).
#[test]
fn a_swap_after_the_server_heard_the_person_lets_the_run_go_quiet() {
    let setup = Setup {
        realtime: Some(RealtimeSetup { fallback: true, ..RealtimeSetup::default() }),
        ..Setup::default()
    };
    let server = |event| Input::RealtimeEvent { socket: Ref::new(0), event };
    simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            Input::Advance { ms: Millis(3_000) },
            Input::MicFrames { mic: Ref::new(0), chunks: Count(20) },
            Input::Advance { ms: Millis(200) },
            server(RtEvent::SpeechStarted),
            Input::Advance { ms: Millis(200) },
            // The server's last word on the person is their start: then it fails, and is swapped.
            server(RtEvent::FatalError),
            Input::Advance { ms: Millis(1_000) },
        ] {
            run.apply(&input).await;
        }
        assert!(
            run.record.applied().iter().all(|a| a.accepted),
            "the scenario: heard, then swapped"
        );
        let outcome = finish(run, Outcome::default()).await;
        assert!(!outcome.stopped_early, "{:?}", outcome.ending);
        assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
    });
}
