//! Every tool call the history records carries JSON arguments: a call whose arguments the model
//! wrote malformed (a chat-template token leaked into them, the JSON cut short) is recorded as it
//! ran, repaired. The next request sends the history back to the model, and a provider refuses
//! invalid JSON there.
use lk_agents_abi::ChatItem;
use lk_agents_voice::{events::SessionEvent, trace::Step};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::Recording,
};

pub struct ToolCallArgumentsAreJson;

impl Rule for ToolCallArgumentsAreJson {
    const CONTRACT: Contract = Contract {
        id: "tool_call_arguments_are_json",
        guarantee: "Every tool call in the history carries JSON arguments, repaired when the model wrote them malformed.",
        examples: &[a_repaired_call],
        counterexamples: &[a_malformed_call],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        for (_, step) in r.journal() {
            let Step::Emitted(SessionEvent::ConversationItemAdded { item, .. }) = step else {
                continue;
            };
            let ChatItem::FunctionCall(call) = &**item else { continue };
            let valid = call.arguments.trim().is_empty()
                || serde_json::from_str::<serde_json::Value>(&call.arguments).is_ok();
            check.expect_with(Subject::Session, valid, || {
                format!(
                    "{} {} recorded with arguments {:?}",
                    call.name, call.call_id, call.arguments
                )
            });
        }
    }
}

fn call(arguments: &str) -> Recording {
    use lk_agents_voice::agent_activity::scheduling::SpeechId;
    let call = lk_agents_abi::FunctionCall {
        id: "item_c1".into(),
        call_id: "c1".into(),
        arguments: arguments.into(),
        name: "lookup".into(),
        created_at: std::time::SystemTime::UNIX_EPOCH,
        extra: Default::default(),
        group_id: None,
    };
    Authored::new()
        .journal(Step::Emitted(SessionEvent::ConversationItemAdded {
            item: Box::new(ChatItem::FunctionCall(call)),
            agent: "agent".into(),
            source: lk_agents_voice::agent::Source::ToolCall { speech: SpeechId(1), step: 1 },
            visibility: lk_agents_voice::agent::Visibility::Both,
            run: None,
        }))
        .done()
}

fn a_repaired_call() -> Recording {
    call(r#"{"label":"human"}"#)
}

fn a_malformed_call() -> Recording {
    call(r#"{"label":"human<|im_end|>"#)
}
