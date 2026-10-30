//! A turn that takes the voice lane keeps the session's history whole, whatever its agent reads.
use lk_agents_abi::ChatItem;
use lk_agents_core::tts::markup::ChatMessageExt;
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, Role, UserText};
use lk_agents_voice::events::SessionEvent;

fn reply(chat: u32, said: &str) -> [Input; 9] {
    [
        Input::GenerateReply {
            user_input: Some(UserText("hello?".into())),
            instructions: None,
            allow: None,
        },
        Input::Advance { ms: Millis(20) },
        Input::LlmText { chat: Ref::new(chat), text: AgentText(said.into()) },
        Input::LlmEnd { chat: Ref::new(chat) },
        Input::Advance { ms: Millis(20) },
        Input::PlayoutStarted { segment: Ref::new(chat) },
        Input::PlayoutProgressed { segment: Ref::new(chat), ms: Millis(5_000) },
        Input::PlayoutFinished { segment: Ref::new(chat) },
        Input::Advance { ms: Millis(1_000) },
    ]
}

// Catches: the session's history was replaced by the lane storage gave back for a turn's
// request, the active agent's own view. Items that view does not hold (a question an earlier
// agent left waiting, what `update_chat_ctx` dropped from the view) left the history, so the
// person's answer never reached the waiting call and the next commit wrote the lane without them
// (`tool_output_follows_outcome`, survey seeds 13386, 15374).
#[test]
fn what_the_agent_does_not_read_stays_in_the_history() {
    // A store that keeps its lease for the whole call.
    let setup = Setup { storage_lost_at: Some(1_000), ..Setup::default() };
    // The agent's view becomes one note of the developer's; its first exchange stays in history.
    let dropped = Input::UpdateChatCtx {
        items: vec![Role::Assistant],
        text: AgentText("The office is closed on Monday.".into()),
    };
    let inputs: Vec<Input> = reply(0, "Hi, how can I help?")
        .into_iter()
        .chain([dropped])
        .chain(reply(1, "Sure."))
        .collect();
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    // The history as the host holds it: told item by item, replaced whole by a reload.
    let mut history: Vec<ChatItem> = Vec::new();
    for (_, event) in run.record.host_view().events() {
        match event {
            SessionEvent::HistoryReloaded { items } => history = items.clone(),
            SessionEvent::ConversationItemAdded { item, .. } => {
                match history.iter_mut().find(|i| i.id() == item.id()) {
                    Some(told) => *told = (**item).clone(),
                    None => history.push((**item).clone()),
                }
            }
            _ => {}
        }
    }
    let said = |text: &str| {
        history
            .iter()
            .any(|i| i.as_message().is_some_and(|m| m.text_content().as_deref() == Some(text)))
    };
    assert!(said("Hi, how can I help?"), "the first reply left the history: {history:#?}");
    assert!(said("Sure."), "the second reply is in the history: {history:#?}");
}
