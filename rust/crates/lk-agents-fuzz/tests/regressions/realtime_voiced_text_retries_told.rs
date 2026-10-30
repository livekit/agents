//! A text-only realtime reply spoken by a voice that does not stream tells each retry of a
//! sentence, as a pipeline reply does.
use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{Input, Millis};
use lk_agents_voice::events::{ErrorSource, SessionEvent};

// Catches: the realtime reply opened its TTS stream without the adapter's retry report, so a
// sentence retried while the voice was down was not the session's recoverable error and its wait
// was not journalled: the reply failed four seconds after the run looked quiet
// (`speech_settles`, survey seed 3409).
#[test]
fn each_retry_of_a_sentence_is_a_recoverable_error() {
    let setup = Setup {
        realtime: Some(RealtimeSetup {
            text_only: true,
            tool_calls: Some(0),
            ..RealtimeSetup::default()
        }),
        tts: true,
        chunked_tts: true,
        ..Setup::default()
    };
    let inputs = [
        Input::TtsOutage { down: true },
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(30_000) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let events = run.record.host_view().events();
    let retried = events
        .iter()
        .filter(|(_, ev)| {
            matches!(ev, SessionEvent::Error(e) if e.source == ErrorSource::Tts && e.recoverable)
        })
        .count();
    let failed = events.iter().any(|(_, ev)| {
        matches!(ev, SessionEvent::Error(e) if e.source == ErrorSource::Tts && !e.recoverable)
    });
    assert!(failed, "the scenario: the voice stays down, the reply's synthesis fails");
    assert_eq!(retried, 3, "each retry of the sentence is told");
}
