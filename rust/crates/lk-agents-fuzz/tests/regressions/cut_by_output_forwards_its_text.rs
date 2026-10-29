//! A segment its output cuts, not the user, still forwards the rest of its words: only an
//! interruption stops the transcript.
use lk_agents_fuzz::drive::{Outcome, RoomLegs, RoomSetup, Run, Setup, finish, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, RoomIn};
use lk_agents_voice::trace::{PlayoutStep, Step};

// Catches: a synthesised segment whose room output closed mid-playout drained the words the model
// sent afterwards without forwarding them, so a completed reply's transcript stopped mid-word
// (`text_conservation`, survey seed 19205).
const FIRST: &str =
    "Your balance is one thousand two hundred and four dollars, and the next payment";

#[test]
fn a_reply_whose_room_closed_mid_segment_forwards_all_its_words() {
    let text = |t: &str| AgentText(t.into());
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
            // Still playing when the room closes: the next push fails and the synthesis stops.
            Input::LlmText { chat: Ref::new(0), text: text(FIRST) },
            Input::Advance { ms: Millis(400) },
            Input::Room { event: RoomIn::Closed },
            Input::Advance { ms: Millis(100) },
            Input::LlmText { chat: Ref::new(0), text: text(" due on the fourteenth.") },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(5_000) },
        ] {
            run.apply(&input).await;
        }
        let journal = run.record.journal();
        let cut = journal.iter().any(|(_, step)| {
            matches!(step, Step::Playout(PlayoutStep::PlaybackFinished { interrupted: true, .. }))
        });
        assert!(cut, "the room's closing cut the segment's playout");
        let forwarded: String = journal
            .iter()
            .filter_map(|(_, step)| match step {
                Step::Playout(PlayoutStep::TranscriptForwarded { text, .. }) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(forwarded, format!("{FIRST} due on the fourteenth."));
        let outcome = finish(run, Outcome::default()).await;
        assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
    });
}
