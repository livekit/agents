//! A realtime reply whose audio finished playing before the server ended its generation is done.
use lk_agents_fuzz::drive::{NodeSetup, RealtimeSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{Input, Millis, Ref};
use lk_agents_voice::events::SessionEvent;

// Catches: the generation task bound the output again for a next message after the speaker
// reported the last one played, and the server's `response.done` came after that report. No
// message came, so the transcription node of that binding was never flushed, and waiting for it
// to drain held the speech running until the session closed.
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
        // A server whose reply calls no tool and ends 12 ms after its last audio.
        realtime: Some(RealtimeSetup { seed: 1, ..RealtimeSetup::default() }),
        sync_transcription: true,
        ..Setup::default()
    };
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        // The reply's last message is in the sink; the server has not yet said it is done.
        Input::Advance { ms: Millis(1_112) },
        Input::PlayoutFinished { segment: Ref::new(0) },
        Input::Advance { ms: Millis(5_000) },
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
