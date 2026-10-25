//! A user turn that ended with no words to decide on does not hold `wait_for_idle()` off.
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{Input, Millis, Ref, TurnMode};
use lk_agents_fuzz::record::Answer;

// Catches: the recogniser's `Ended` phase (speech heard, no countdown started) read as a turn
// being decided. With no transcript coming, idle was never reached, and a tool waiting for idle
// inside its own speech waited forever while the speech waited on the tool (survey seed 1678,
// `session_settles`). The reference's `wait_for_idle` waits only on `_speaking` and a running
// `_end_of_turn_task`.
#[test]
fn idle_is_reached_after_a_turn_with_no_words() {
    let setup = Setup { vad: true, stt: true, turn: TurnMode::Stt, ..Setup::default() };
    let inputs = [
        Input::Advance { ms: Millis(20) },
        Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
        Input::Advance { ms: Millis(300) },
        Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
        Input::Advance { ms: Millis(500) },
        Input::WaitForIdle { hold: None },
        Input::Advance { ms: Millis(100) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let answers: Vec<_> = run
        .record
        .host_view()
        .api_calls()
        .iter()
        .filter(|c| matches!(c.input(), Input::WaitForIdle { .. }))
        .map(|c| c.returned().map(|(_, a)| a.clone()))
        .collect();
    assert_eq!(answers, [Some(Answer::IdleOk)]);
}
