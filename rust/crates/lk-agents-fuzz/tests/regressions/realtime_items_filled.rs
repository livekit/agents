//! A realtime server's empty records of its items stay out of history; the words that fill
//! them reach it, once, under the server's ids.
use lk_agents_abi::{ChatItem, Role};
use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{Count, Input, Millis, Ref, RtEvent};
use lk_agents_voice::events::SessionEvent;
use lk_agents_core::tts::markup::ChatMessageExt;

// Catches: `remote_item_added` committed into history as it came, empty, and the transcript and
// the reply that fill those items never told to the host: the user's final transcript upserted
// the empty record under its id (a replacement, never announced) and the reply was committed
// under an id of its own, beside its empty placeholder. On a real call history held an empty
// message per turn and no user words. The reference keeps the server's items as placeholders in
// the agent's context and commits the transcript and the message under the server's ids.
#[test]
fn server_items_reach_history_filled() {
    let setup = Setup {
        realtime: Some(RealtimeSetup {
            turn_detection: false,
            tool_calls: Some(0),
            ..RealtimeSetup::default()
        }),
        ..Setup::default()
    };
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20) },
        Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::GenerationDone },
        Input::Advance { ms: Millis(500) },
        Input::PlayoutFinished { segment: Ref::new(0) },
        Input::Advance { ms: Millis(20) },
        // The person speaks: a commit of an empty buffer is an error on a real server.
        Input::MicFrames { mic: Ref::new(0), chunks: Count(30) },
        Input::Advance { ms: Millis(300) },
        Input::CommitTurn { skip_reply: false },
        Input::Advance { ms: Millis(1_000) },
        Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::GenerationDone },
        Input::Advance { ms: Millis(500) },
        Input::PlayoutFinished { segment: Ref::new(1) },
        Input::Advance { ms: Millis(500) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let messages: Vec<(Role, String, String)> = run
        .record
        .host_view()
        .events()
        .into_iter()
        .filter_map(|(_, ev)| match ev {
            SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                ChatItem::Message(m) => {
                    Some((m.role, m.id.clone(), m.text_content().unwrap_or_default()))
                }
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert!(messages.iter().all(|(_, _, text)| !text.is_empty()), "{messages:#?}");
    let user: Vec<_> = messages.iter().filter(|(role, ..)| *role == Role::User).collect();
    assert_eq!(user.len(), 1, "{messages:#?}");
    assert!(user[0].1.starts_with("item_user_"), "{messages:#?}");
    let replies: Vec<_> = messages.iter().filter(|(role, ..)| *role == Role::Assistant).collect();
    assert_eq!(replies.len(), 2, "{messages:#?}");
    assert!(replies.iter().all(|(_, id, _)| id.starts_with("item_msg_")), "{messages:#?}");
}
