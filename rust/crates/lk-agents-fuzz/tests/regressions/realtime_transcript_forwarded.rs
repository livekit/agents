//! A realtime generation's words reach the transcription output.
use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, RtEvent, SayInput};
use lk_agents_fuzz::record::Sink;

// Catches: the realtime step recording the server's text as the transcript but never forwarding
// it, so a transcription output showed nothing of anything a realtime model said
// (`text_conservation`, survey seed 35). `_realtime_generation_task` forwards each message's text
// to `text_output` through `forward_generation`.
#[test]
fn a_realtime_say_is_shown() {
    let setup =
        Setup { realtime: Some(RealtimeSetup { turn_detection: false }), ..Setup::default() };
    let inputs = [
        Input::Say {
            text: AgentText("Hello there.".into()),
            input: SayInput::Words,
            allow: None,
            to_ctx: true,
        },
        Input::Advance { ms: Millis(20) },
        Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::GenerationDone },
        Input::Advance { ms: Millis(20) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let shown: String = run
        .record
        .sink()
        .iter()
        .filter_map(|(_, s)| match s {
            Sink::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(shown, "Hello there.");
    let flushed = run.record.sink().iter().any(|(_, s)| matches!(s, Sink::TextFlush { .. }));
    assert!(flushed, "the message's segment was closed");
}
