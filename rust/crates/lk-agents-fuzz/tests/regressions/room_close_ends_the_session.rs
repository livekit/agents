//! The room closing ends the session, however the room IO's own close races it.
use lk_agents_abi::voice::events::CloseReason;
use lk_agents_fuzz::drive::{Outcome, RoomLegs, RoomSetup, Run, Setup, finish, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, RoomIn};
use lk_agents_voice::trace::{CloseStep, RoomEventKind, RoomStep, Step};

// Catches: the room IO's `Close` stopped its forwarder before the forwarder passed the same
// `Close` (and the link's end) to the session, so the session never closed and a later reply
// played into the closed room output, cut with no cause (`speech_is_heard`, campaign seed 19655).
#[test]
fn the_session_closes_with_its_room() {
    let setup = Setup {
        room: Some(RoomSetup { close_on_disconnect: false, delete_room_on_close: false }),
        room_legs: Some(RoomLegs { pre_connect: None }),
        ..Setup::default()
    };
    simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            Input::Room { event: RoomIn::Linked },
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            Input::LlmText { chat: Ref::new(0), text: AgentText("One moment please.".into()) },
            Input::Advance { ms: Millis(400) },
            Input::Room { event: RoomIn::Closed },
            Input::Advance { ms: Millis(5_000) },
        ] {
            run.apply(&input).await;
        }
        let journal: Vec<Step> = run.record.journal().iter().map(|(_, s)| s.clone()).collect();
        let closed = journal
            .iter()
            .position(|s| matches!(s, Step::Room(RoomStep::Event(RoomEventKind::Close))))
            .expect("the session hears the room close");
        assert!(
            journal[closed..].iter().any(|s| matches!(s,
                Step::Close(CloseStep::Requested { reason: CloseReason::JobShutdown, .. }))),
            "the room's close ends the session"
        );
        let outcome = finish(run, Outcome::default()).await;
        assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
    });
}
