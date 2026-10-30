//! A reply interrupted before any of its audio was heard leaves no words behind.
use lk_agents_abi::{ChatItem, Role};
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};
use lk_agents_voice::events::SessionEvent;
use lk_agents_core::tts::markup::ChatMessageExt;

// Catches: a segment whose synthesis ended interrupted (never opened, or cut while its open was
// being retried) taking the lost-voice fallback, which forwards the segment's words as text and
// counts them heard. The reply was committed whole, `interrupted` set, though not a word of it
// played: on a real call the model then read its own unheard question as asked and answered past
// the caller. The reference commits only the synchronised transcript of what played: nothing.
#[test]
fn an_unheard_reply_is_not_committed() {
    let said = assistant_words(&[
        // Every synthesis fails at the open, so the segment sits in the retry until it is cut.
        Input::TtsOutage { down: true },
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20) },
        Input::LlmText {
            chat: Ref::new(0),
            text: AgentText("Your order shipped. Would you like a receipt?".into()),
        },
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(20) },
        Input::Interrupt { force: true, wait: false },
        Input::Advance { ms: Millis(5000) },
    ]);
    assert!(said.iter().all(|t| t.is_empty()), "committed unheard words: {said:?}");
}

fn assistant_words(inputs: &[Input]) -> Vec<String> {
    let run = simulate(async {
        let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
        for input in inputs {
            run.apply(input).await;
        }
        run
    });
    run.record
        .host_view()
        .events()
        .into_iter()
        .filter_map(|(_, ev)| match ev {
            SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                ChatItem::Message(m) if m.role == Role::Assistant => m.text_content(),
                _ => None,
            },
            _ => None,
        })
        .collect()
}
