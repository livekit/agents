use std::collections::{BTreeMap, BTreeSet};

use lk_agents_abi::Role;
use lk_agents_voice::{events::SessionEvent, trace::Step};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Recording, RemoteStep},
};

// What the peer received is the black-box record of forwarding. A business message is never
// resent: a lost acknowledgement leaves its outcome to reconciliation. A human reply is the
// words of one committed user turn; it answers one waiting question, so the number of answers
// carrying a person's words never exceeds the turns in which they said them. Words the model
// chose itself (a new request, a clarification) are not a human reply.

pub struct RemoteReplyForwardedOnce;

impl Rule for RemoteReplyForwardedOnce {
    const CONTRACT: Contract = Contract {
        id: "remote_reply_forwarded_once",
        guarantee: "Each message reaches the remote peer once, and each human reply answers at \
                    most one waiting remote question, on exactly one task.",
        examples: &[answered_once, same_words_in_two_turns],
        counterexamples: &[message_resent, one_turn_answers_two_tasks],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let mut said: BTreeMap<String, u32> = BTreeMap::new();
        for (_, step) in r.journal() {
            if let Step::Emitted(SessionEvent::ConversationItemAdded { item, .. }) = step
                && let Some(message) = item.as_message()
                && message.role == Role::User
                && let Some(text) = message.raw_text_content()
            {
                *said.entry(text).or_default() += 1;
            }
        }
        let mut seen = BTreeSet::new();
        let mut answers: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for (_, step) in r.remote() {
            let RemoteStep::Received { task, message, answering, text } = step else { continue };
            check.expect_with(Subject::Session, seen.insert(message.as_str()), || {
                format!("message {message} reached the peer again, on {task}")
            });
            if answering.is_none() || !said.contains_key(text) {
                continue;
            }
            let tasks = answers.entry(text).or_default();
            tasks.push(task);
            let turns = said[text];
            check.expect_with(Subject::Session, tasks.len() as u32 <= turns, || {
                format!(
                    "the person said {text:?} in {turns} turn(s), and it answered questions on \
                     {tasks:?}"
                )
            });
        }
    }
}

fn user(text: &str) -> SessionEvent {
    let message = lk_agents_abi::ChatMessage::text("u", Role::User, text, std::time::UNIX_EPOCH);
    SessionEvent::ConversationItemAdded {
        item: Box::new(lk_agents_abi::ChatItem::Message(message)),
        agent: "agent".into(),
        source: lk_agents_voice::agent::Source::UserTyped,
        visibility: lk_agents_voice::agent::Visibility::Both,
    }
}

fn received(task: &str, message: &str, answering: Option<&str>, text: &str) -> RemoteStep {
    RemoteStep::Received {
        task: task.into(),
        message: message.into(),
        answering: answering.map(Into::into),
        text: text.into(),
    }
}

fn answered_once() -> Recording {
    Authored::new()
        .remote(received("task-1", "m1", None, "Book a table"))
        .journal(Step::Emitted(user("Friday")))
        .remote(received("task-1", "m2", Some("task-1-q1"), "Friday"))
        .done()
}

fn same_words_in_two_turns() -> Recording {
    Authored::new()
        .journal(Step::Emitted(user("Friday")))
        .remote(received("task-1", "m2", Some("task-1-q1"), "Friday"))
        .journal(Step::Emitted(user("Friday")))
        .remote(received("task-2", "m3", Some("task-2-q1"), "Friday"))
        .done()
}

fn message_resent() -> Recording {
    Authored::new()
        .remote(received("task-1", "m1", None, "Book a table"))
        .remote(received("task-1", "m1", None, "Book a table"))
        .done()
}

fn one_turn_answers_two_tasks() -> Recording {
    Authored::new()
        .journal(Step::Emitted(user("Friday")))
        .remote(received("task-1", "m2", Some("task-1-q1"), "Friday"))
        .remote(received("task-2", "m3", Some("task-2-q1"), "Friday"))
        .done()
}
