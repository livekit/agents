//! Frames left on a microphone replaced in the same turn may go with it.
use lk_agents_fuzz::drive::{Case, Setup, run};
use lk_agents_fuzz::input::{Count, Input, Millis, Ref};

// Catches: `forwarded_while_enabled` owing every listener the frames pushed to a microphone the
// case replaced before the driver's next turn, though the swap drops what the old one's reader
// had not read, as the reference's `_on_audio_input_changed` cancels it (survey seeds 64, 165,
// 257: 44 of the rule's 54 failures over seeds 0 to 5119).
#[test]
fn frames_on_a_replaced_microphone_are_not_owed() {
    let inputs = vec![
        Input::MicFrames { mic: Ref::new(0), chunks: Count(3) },
        Input::SetAudioInput { attach: true },
        Input::Advance { ms: Millis(100) },
    ];
    let setup = Setup { vad: true, stt: true, ..Setup::default() };
    let outcome = run(Case { setup, inputs, ..Case::default() });
    let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"forwarded_while_enabled"), "{failed:?}");
}
