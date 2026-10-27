//! Frames pushed between an audio input switch and the driver's turn that applies it may go
//! either way.
use lk_agents_fuzz::drive::{Case, Policy, Setup, run};
use lk_agents_fuzz::input::{Count, Input, Millis, Ref};
use lk_agents_voice::agent_session::IoKind;

/// Under every interleaving of the queue a few schedules sample, both orders included.
fn clean(inputs: Vec<Input>) {
    for schedule in 0..16 {
        let setup = Setup { vad: true, stt: true, ..Setup::default() };
        let inputs = inputs.clone();
        let outcome =
            run(Case { setup, inputs, schedule, policy: Policy::BasicPos, ..Case::default() });
        let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
        assert!(!failed.contains(&"forwarded_while_enabled"), "schedule {schedule}: {failed:?}");
    }
}

fn switch(enabled: bool) -> Input {
    Input::SetIoEnabled { leg: IoKind::AudioInput, enabled }
}

fn frames() -> Input {
    Input::MicFrames { mic: Ref::new(0), chunks: Count(3) }
}

// Catches: `forwarded_while_enabled` holding the driver to a switch it had not yet received.
// The switch rides the command queue to the driver's next turn while frames go straight to the
// pump, so a frame pushed just after a disable may still be forwarded (survey seeds 399, 2373)
// and one pushed just after an enable may still be dropped (seeds 4336, 4844).
#[test]
fn frames_after_a_disable_in_its_turn_may_be_heard() {
    clean(vec![switch(false), frames(), Input::Advance { ms: Millis(100) }]);
}

#[test]
fn frames_after_an_enable_in_its_turn_may_be_dropped() {
    let inputs = vec![
        switch(false),
        Input::Advance { ms: Millis(100) },
        switch(true),
        frames(),
        Input::Advance { ms: Millis(100) },
    ];
    clean(inputs);
}
