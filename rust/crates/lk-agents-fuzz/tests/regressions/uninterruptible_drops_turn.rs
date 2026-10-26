//! A user turn ending while an uninterruptible speech plays is dropped: no user item, no reply.
use lk_agents_abi::{ChatItem, Role};
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, SayInput, UserText};
use lk_agents_voice::{events::SessionEvent, trace::Step};

// Catches: the turn queued a reply behind the speech, which then answered words the agent had
// talked over. The reference's `_user_turn_completed_impl` returns with "skipping reply to user
// input, current speech generation cannot be interrupted" and keeps no user item; found by
// `scripts/parity.py say_uninterruptible`.
#[test]
fn a_turn_over_an_uninterruptible_speech_is_dropped() {
    let setup = Setup { stt: true, ..Setup::default() };
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        let inputs = [
            Input::Say {
                text: AgentText("A long answer that cannot be cut.".into()),
                input: SayInput::Words,
                allow: Some(false),
                to_ctx: true,
            },
            Input::Advance { ms: Millis(20) },
            Input::PlayoutStarted { segment: Ref::new(0) },
            Input::Advance { ms: Millis(200) },
            Input::SttStart { stt: Ref::new(0) },
            Input::SttFinal {
                utterance: Ref::new(0),
                text: UserText("stop please".into()),
                speaker: None,
            },
            Input::SttEnd { utterance: Ref::new(0) },
            Input::Advance { ms: Millis(3_000) },
        ];
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let mut speeches = 0;
    let mut user_items = 0;
    for (_, step) in run.record.journal().iter() {
        match step {
            Step::Emitted(SessionEvent::SpeechCreated { .. }) => speeches += 1,
            Step::Emitted(SessionEvent::ConversationItemAdded { item, .. }) => {
                if matches!(&**item, ChatItem::Message(m) if m.role == Role::User) {
                    user_items += 1;
                }
            }
            _ => {}
        }
    }
    assert_eq!((speeches, user_items), (1, 0), "only the say(), and no user item");
}
