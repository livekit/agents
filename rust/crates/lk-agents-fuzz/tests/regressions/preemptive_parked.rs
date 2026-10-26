//! A preemptive reply parked for a turn that never ended is not a stuck speech.
use lk_agents_fuzz::drive::{Case, Setup, run};
use lk_agents_fuzz::input::{Input, Millis, Ref, UserText};

// Catches: `speech_settles` judging a preemptive reply, started on a preflight transcript, whose
// utterance ended without a final: the turn waits for words the case never sent, and the reply
// waits on the turn until the close discards it, as in the reference (survey seed 340).
#[test]
fn a_preemptive_reply_waits_on_its_turn() {
    let setup = Setup { stt: true, vad: true, preemptive: true, ..Setup::default() };
    let inputs = vec![
        Input::Advance { ms: Millis(100) },
        Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
        Input::SttStart { stt: Ref::new(0) },
        Input::Advance { ms: Millis(500) },
        Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(300) },
        Input::SttPreflight {
            utterance: Ref::new(0),
            text: UserText("yeah hi um I'm calling about my order".into()),
            speaker: None,
        },
        Input::Advance { ms: Millis(400) },
        Input::SttEnd { utterance: Ref::new(0) },
        Input::Advance { ms: Millis(5_000) },
    ];
    let outcome = run(Case { setup, inputs, ..Case::default() });
    let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(failed.is_empty(), "{:?}", outcome.failures);
    assert!(outcome.reached.iter().any(|v| v == "Emitted::PreemptiveGeneration"));
}
