//! The realtime server hearing the user start speaking interrupts the reply it is playing.
use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{Input, Millis, Ref, RtEvent};
use lk_agents_voice::trace::{GateOp, InterruptCause, InterruptStep, Step};

// Catches: the server's `input_speech_started` taking the guarded interruption by audio activity,
// which pauses a pausable sink instead of interrupting. On a real call the reply then sat paused
// after its generation had ended, no false-interruption timer ended the pause, and the next
// server turn interrupted it then, so its cancel landed on that new response and the agent went
// silent. The reference's `_on_input_speech_started` calls `interrupt(source="audio_activity")`.
#[test]
fn a_server_speech_onset_interrupts_the_playing_reply() {
    let setup = Setup {
        realtime: Some(RealtimeSetup { turn_detection: true, ..RealtimeSetup::default() }),
        pausable_output: true,
        ..Setup::default()
    };
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        // Past the echo canceller's warm-up, which holds interruptions back.
        Input::Advance { ms: Millis(4_000) },
        Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::SpeechStarted },
        Input::Advance { ms: Millis(20) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let journal = run.record.journal();
    let interrupted = journal.iter().any(|(_, step)| {
        matches!(
            step,
            Step::Interrupt(InterruptStep::Interrupted {
                cause: InterruptCause::ByAudioActivity,
                ..
            })
        )
    });
    let paused = journal.iter().any(|(_, step)| {
        matches!(step, Step::Interrupt(InterruptStep::PlayoutGate { op: GateOp::Pause, .. }))
    });
    assert!(interrupted, "the reply is interrupted");
    assert!(!paused, "not paused");
}
