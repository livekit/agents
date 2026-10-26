//! An interruption by audio activity the voice-activity detector heard has a cause.
use lk_agents_fuzz::check::check_all;
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Confidence, Input, Millis, Ref, SayInput};
use lk_agents_voice::trace::{InterruptCause, InterruptStep, Step};

// Catches: `interruption_has_cause` taking only recognition events and turn commands as the
// person acting. The detector hearing speech over the agent is what `_interrupt_by_audio_activity`
// acts on, with no transcript yet (survey seed 65).
#[test]
fn a_vad_interruption_is_caused() {
    let setup = Setup { vad: true, ..Setup::default() };
    let text = AgentText("Your balance is due on the fourteenth.".into());
    let inputs = [
        Input::Say { text, input: SayInput::Words, allow: Some(true), to_ctx: true },
        Input::Advance { ms: Millis(20) },
        Input::PlayoutStarted { segment: Ref::new(0) },
        // Past the echo canceller's warm-up, which holds interruptions off.
        Input::Advance { ms: Millis(4_000) },
        Input::VadInference {
            vad: Ref::new(0),
            probability: Confidence(90),
            speaking: true,
            duration: Millis(900),
        },
        Input::MicFrames { mic: Ref::new(0), chunks: lk_agents_fuzz::input::Count(10) },
        Input::Advance { ms: Millis(100) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let cut = run.record.journal().iter().any(|(_, s)| {
        matches!(
            s,
            Step::Interrupt(InterruptStep::Interrupted {
                cause: InterruptCause::ByAudioActivity,
                ..
            })
        )
    });
    assert!(cut, "the detector's speech interrupted the agent");
    let (failures, _) = check_all(&run.record);
    let failed: Vec<_> = failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"interruption_has_cause"), "{failed:?}");
}
