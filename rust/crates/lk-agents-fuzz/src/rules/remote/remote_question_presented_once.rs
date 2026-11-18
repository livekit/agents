use std::collections::BTreeMap;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Recording, RemoteStep},
};

// The remote driver decides to present a question once per occurrence: replays of the same
// task state (a reconciliation, a resubscription, a lost acknowledgement) observe it again but
// never hand the person the same question twice. Each peer question is unique in a run.

pub struct RemoteQuestionPresentedOnce;

impl Rule for RemoteQuestionPresentedOnce {
    const CONTRACT: Contract = Contract {
        id: "remote_question_presented_once",
        guarantee: "Each question a remote task asks is handed to the session at most once, and \
                    only after the peer asked it.",
        examples: &[asked_and_presented],
        counterexamples: &[presented_twice, presented_unasked],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let mut asked = Vec::new();
        let mut presented: BTreeMap<&str, u32> = BTreeMap::new();
        for (_, step) in r.remote() {
            match step {
                RemoteStep::Asked { text, .. } => asked.push(text.as_str()),
                RemoteStep::Presented { call, text } => {
                    let count = presented.entry(text).or_default();
                    *count += 1;
                    check.expect_with(Subject::Session, *count == 1, || {
                        format!("the remote question {text:?} was presented again by {call}")
                    });
                    check.expect_with(Subject::Session, asked.contains(&text.as_str()), || {
                        format!("{call} presented {text:?}, which the peer never asked")
                    });
                }
                _ => {}
            }
        }
    }
}

fn asked(text: &str) -> RemoteStep {
    RemoteStep::Asked { task: "task-1".into(), question: "task-1-q1".into(), text: text.into() }
}

fn presented(call: &str, text: &str) -> RemoteStep {
    RemoteStep::Presented { call: call.into(), text: text.into() }
}

fn asked_and_presented() -> Recording {
    Authored::new().remote(asked("Which day?")).remote(presented("item_a", "Which day?")).done()
}

fn presented_twice() -> Recording {
    Authored::new()
        .remote(asked("Which day?"))
        .remote(presented("item_a", "Which day?"))
        .remote(presented("item_a", "Which day?"))
        .done()
}

fn presented_unasked() -> Recording {
    Authored::new().remote(presented("item_a", "Which day?")).done()
}
