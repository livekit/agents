//! A realtime generation's words reach the transcription output.
use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{AGENT_TURNS, Input, Millis, Ref, RtEvent};
use lk_agents_fuzz::record::Sink;

// Catches: the realtime step recording the server's text as the transcript but never forwarding
// it, so a transcription output showed nothing of anything a realtime model said
// (`text_conservation`, survey seed 35). `_realtime_generation_task` forwards each message's text
// to `text_output` through `forward_generation`.
#[test]
fn a_realtime_reply_is_shown() {
    let setup =
        Setup { realtime: Some(RealtimeSetup { turn_detection: false, ..RealtimeSetup::default() }), ..Setup::default() };
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20) },
        Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::GenerationDone },
        Input::Advance { ms: Millis(20) },
        Input::PlayoutFinished { segment: Ref::new(0) },
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
    assert!(AGENT_TURNS.contains(&shown.as_str()) && !shown.is_empty(), "{shown:?}");
    let flushed = run.record.sink().iter().any(|(_, s)| matches!(s, Sink::TextFlush { .. }));
    assert!(flushed, "the message's segment was closed");
}
