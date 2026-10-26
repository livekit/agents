//! A generation the realtime server opens on its own is a turn the case ends like any other.
use lk_agents_fuzz::drive::{Case, RealtimeSetup, Setup, run};
use lk_agents_fuzz::input::{Input, Millis, Ref, RtEvent};

// Catches: the fake server opening a second generation that it never counted as an open turn, so
// nothing ever ended it and its speech waited on the socket until the close (`speech_settles`,
// survey seed 179).
#[test]
fn a_server_generation_settles() {
    let setup =
        Setup { realtime: Some(RealtimeSetup { turn_detection: true }), ..Setup::default() };
    let inputs = vec![
        Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::SecondGeneration },
        Input::Advance { ms: Millis(20) },
    ];
    let outcome = run(Case { setup, inputs, ..Case::default() });
    let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"speech_settles"), "{failed:?}");
    assert!(outcome.expectations.get("speech_settles").is_some_and(|n| *n > 0));
}
