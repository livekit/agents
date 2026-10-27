//! A filler's say on a realtime model that cannot say is refused, as the reference's `say`.
use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};
use lk_agents_voice::trace::{FillerEnd, GenerationStep, Step};

// Catches: the filler's text played as a pipeline speech with no voice on a realtime session
// whose model has no `say` (the DST agent's parity finding). The reference's `session.say` raises
// there ("without a TTS model or a RealtimeSession that supports say()"), which ends the filler
// with nothing spoken.
#[test]
fn a_filler_on_a_realtime_model_without_say_is_refused() {
    let setup = Setup {
        realtime: Some(RealtimeSetup {
            turn_detection: false,
            tool_calls: Some(100),
            ..RealtimeSetup::default()
        }),
        tools: vec!["lookup".into()],
        // No voice: with one, the TTS speaks the filler.
        tts: false,
        ..Setup::default()
    };
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(3_000) },
        Input::FillerStart {
            tool: Ref::new(0),
            speech: Ref::new(0),
            delay: Millis(0),
            interval: None,
            text: Some(AgentText("One moment.".into())),
        },
        Input::PlayoutStarted { segment: Ref::new(0) },
        Input::PlayoutFinished { segment: Ref::new(0) },
        Input::Advance { ms: Millis(1_000) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let journal = run.record.journal();
    let fired = journal
        .iter()
        .any(|(_, step)| matches!(step, Step::Generation(GenerationStep::FillerFired { .. })));
    let refused = journal.iter().any(|(_, step)| {
        matches!(
            step,
            Step::Generation(GenerationStep::FillerStopped { cause: FillerEnd::Refused, .. })
        )
    });
    assert!(!fired && refused, "fired {fired}, refused {refused}");
}
