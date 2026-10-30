//! A realtime reply whose audio finished playing before the server ended its generation is done,
//! and binds the output only for the message it sent.
use lk_agents_fuzz::drive::{NodeSetup, RealtimeSetup, Run, Setup, simulate};
use lk_agents_fuzz::fakes::RealtimeDelays;
use lk_agents_fuzz::input::{Input, Millis, Ref};
use lk_agents_fuzz::record::Sink;
use lk_agents_voice::events::SessionEvent;

// Catches: the reply held running until the session closed when the server's `response.done`
// came after the speaker reported its last message played: a binding that carried no message
// never flushed its transcription node, and its drain never ended (5574d1ccc, bc0b5de0c).
#[test]
fn the_reply_is_done_once_the_server_ends_it() {
    let setup = Setup {
        nodes: Some(NodeSetup {
            llm: false,
            tts: false,
            stt: false,
            llm_text: None,
            transcription: true,
            realtime_audio: false,
        }),
        // A server whose reply calls no tool and ends 30 s after its last audio.
        realtime: Some(RealtimeSetup {
            tool_calls: Some(0),
            delays: RealtimeDelays { done: Some(30_000), ..RealtimeDelays::default() },
            ..RealtimeSetup::default()
        }),
        sync_transcription: true,
        ..Setup::default()
    };
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        // The reply's last message is in the sink; the server has not yet said it is done.
        Input::Advance { ms: Millis(20_000) },
        Input::PlayoutFinished { segment: Ref::new(0) },
        Input::Advance { ms: Millis(20_000) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let done = run
        .record
        .host_view()
        .events()
        .into_iter()
        .filter(|(_, ev)| matches!(ev, SessionEvent::SpeechDone { .. }))
        .count();
    assert_eq!(done, 1);
}

// Catches: after the speaker reported the reply's one message played, the generation bound the
// output again for a next message before the server said whether one would come. None came, and
// that binding's transcript chain ended a segment the server never sent: the transcription output
// was flushed twice for one message (the reference binds the output once per message).
#[test]
fn the_transcript_ends_once_per_message() {
    let setup = Setup {
        nodes: Some(NodeSetup {
            llm: false,
            tts: false,
            stt: false,
            llm_text: None,
            transcription: true,
            realtime_audio: false,
        }),
        realtime: Some(RealtimeSetup {
            tool_calls: Some(0),
            delays: RealtimeDelays { done: Some(30_000), ..RealtimeDelays::default() },
            ..RealtimeSetup::default()
        }),
        ..Setup::default()
    };
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20_000) },
        Input::PlayoutFinished { segment: Ref::new(0) },
        Input::Advance { ms: Millis(40_000) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let done = run
        .record
        .host_view()
        .events()
        .into_iter()
        .any(|(_, ev)| matches!(ev, SessionEvent::SpeechDone { .. }));
    assert!(done, "the scenario: the server ended the reply");
    let flushes =
        run.record.sink().iter().filter(|(_, sink)| matches!(sink, Sink::TextFlush { .. })).count();
    assert_eq!(flushes, 1, "one message, one transcript segment");
}
