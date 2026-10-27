//! A microphone attached as the session closes is let go with it.
use lk_agents_fuzz::drive::{Case, Policy, Setup, run};
use lk_agents_fuzz::input::{Input, Millis};

// Catches: a microphone attached in the step before the close left in the resource table. The
// core registers it and queues its source switch; the loop stopped before the runtime took the
// switch, and the runtime's finish applied it to the forwarder alone, so the id it held was never
// released (survey seed 2303, every_resource_released).
#[test]
fn a_mic_attached_at_the_close_is_released() {
    let inputs = vec![
        Input::Advance { ms: Millis(20) },
        Input::SetAudioInput { attach: true },
        Input::Close { drain: false },
    ];
    let leaked: Vec<u64> = (0..16)
        .filter(|&schedule| {
            let case = Case {
                setup: Setup { stt: true, vad: true, ..Setup::default() },
                inputs: inputs.clone(),
                schedule,
                random: Vec::new(),
                policy: Policy::BasicPos,
            };
            run(case).failures.iter().any(|f| f.contract == "every_resource_released")
        })
        .collect();
    assert!(leaked.is_empty(), "schedules leaving the microphone in the table: {leaked:?}");
}
