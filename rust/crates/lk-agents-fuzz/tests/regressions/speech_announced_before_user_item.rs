//! `generate_reply(user_input=)` announces its speech, user-initiated, before the user item.
use lk_agents_abi::{ChatItem, Role};
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{Input, Millis, UserText};
use lk_agents_voice::{events::SessionEvent, trace::Step};

// Catches: the user item committed before `speech_created`, which a listener pairing the two saw
// in the wrong order. The reference emits `speech_created(user_initiated=True)` in
// `_generate_reply` and adds the user message once the reply task sees its speech scheduled
// (`_pipeline_reply_task_impl`); found by `scripts/parity.py generate_reply`.
#[test]
fn the_speech_is_announced_before_the_user_item() {
    let setup = Setup::default();
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        let reply = Input::GenerateReply {
            user_input: Some(UserText("hello there".into())),
            instructions: None,
            allow: None,
        };
        for input in [reply, Input::Advance { ms: Millis(20) }] {
            run.apply(&input).await;
        }
        run
    });
    let seen: Vec<&'static str> = run
        .record
        .journal()
        .iter()
        .filter_map(|(_, step)| match step {
            Step::Emitted(SessionEvent::SpeechCreated { user_initiated, .. }) => {
                assert!(user_initiated, "a generate_reply speech is user-initiated");
                Some("speech_created")
            }
            Step::Emitted(SessionEvent::ConversationItemAdded { item, .. }) => match &**item {
                ChatItem::Message(m) if m.role == Role::User => Some("user_item"),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(seen, ["speech_created", "user_item"]);
}
