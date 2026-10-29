//! A realtime reply is announced when asked for, and asks the server once its speech holds the
//! floor.
use lk_agents_abi::RealtimeCommandKind;
use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{Input, Millis, Ref, RtEvent};
use lk_agents_voice::trace::{RealtimeStep, SpeechStep, Step};

// Catches: the request going to the server at once and the speech created only when the server
// answered, so `speech_created` came after the model call and a reply queued behind a playing one
// made the server generate over it. The reference schedules the handle at `generate_reply()` and
// its `_realtime_reply_task` calls `rt_session.generate_reply()` once the speech is authorized.
#[test]
fn a_queued_realtime_reply_asks_the_server_once_authorized() {
    let setup = Setup {
        realtime: Some(RealtimeSetup { turn_detection: false, ..RealtimeSetup::default() }),
        ..Setup::default()
    };
    let reply = || Input::GenerateReply { user_input: None, instructions: None, allow: None };
    let inputs =
        [reply(), Input::Advance { ms: Millis(20) }, reply(), Input::Advance { ms: Millis(20) }];
    let done = [
        Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::GenerationDone },
        Input::Advance { ms: Millis(100) },
        Input::PlayoutFinished { segment: Ref::new(0) },
        Input::Advance { ms: Millis(2_000) },
    ];
    let count = |run: &Run, what: fn(&Step) -> bool| {
        run.record.journal().iter().filter(|(_, step)| what(step)).count()
    };
    let asked = |step: &Step| {
        matches!(
            step,
            Step::Realtime(RealtimeStep::Sent { kind: RealtimeCommandKind::GenerateReply, .. })
        )
    };
    let created = |step: &Step| matches!(step, Step::Speech(SpeechStep::ReplyCreated { .. }));
    let (queued, played) = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        let queued = (count(&run, created), count(&run, asked));
        for input in &done {
            run.apply(input).await;
        }
        (queued, count(&run, asked))
    });
    assert_eq!(queued, (2, 1), "both replies announced, only the playing one asked");
    assert_eq!(played, 2, "the second is asked once the first is done");
}
