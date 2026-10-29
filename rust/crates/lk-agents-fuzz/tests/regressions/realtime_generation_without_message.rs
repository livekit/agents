//! A realtime generation that ends before any message still ends its speech.
use lk_agents_fuzz::drive::{NodeSetup, Outcome, RealtimeSetup, Run, Setup, finish, simulate};
use lk_agents_fuzz::input::{Input, Millis, Ref, RtEvent};
use lk_agents_voice::events::SessionEvent;
use lk_agents_voice::trace::Step;

// Catches: the reply's binding opened the agent's `transcription_node` at activation, and a
// generation the socket's drop cancelled before its first message never flushed it, so the step
// waited on the node's drain until the session closed (`speech_settles`, survey seed 6784).
#[test]
fn a_generation_cancelled_before_its_first_message_ends_the_speech() {
    let setup = Setup {
        realtime: Some(RealtimeSetup {
            turn_detection: false,
            tool_calls: Some(0),
            ..RealtimeSetup::default()
        }),
        nodes: Some(NodeSetup {
            llm: false,
            tts: false,
            stt: false,
            llm_text: None,
            transcription: true,
            realtime_audio: false,
        }),
        ..Setup::default()
    };
    let rt = |event| Input::RealtimeEvent { socket: Ref::new(0), event };
    simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            // The server has opened the generation and sent no message yet.
            Input::Advance { ms: Millis(300) },
            rt(RtEvent::Disconnected),
            Input::Advance { ms: Millis(800) },
            rt(RtEvent::Reconnected),
            Input::Advance { ms: Millis(5_000) },
        ] {
            run.apply(&input).await;
        }
        let journal = run.record.journal();
        for (at, step) in journal.iter() {
            eprintln!("J {} {:?}", at.at_ms, step);
        }
        let done = journal
            .iter()
            .any(|(_, step)| matches!(step, Step::Emitted(SessionEvent::SpeechDone { .. })));
        assert!(done, "the speech whose generation was cancelled is done before the close");
        let outcome = finish(run, Outcome::default()).await;
        assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
    });
}
