use crate::check::{Check, Subject};
use crate::text::{MessageRecord, TextContract, TextRecording, TextRule};

pub struct TextAppliedOnce;

impl TextRule for TextAppliedOnce {
    const CONTRACT: TextContract = TextContract {
        id: "text_applied_once",
        guarantee: "A message sent once or resent after a lost answer creates at most one task and lands in the history at most once.",
        examples: &[once],
        counterexamples: &[two_tasks, twice_in_history],
    };

    fn on_complete(r: &TextRecording, check: &mut Check) {
        for m in &r.messages {
            check.expect_with(Subject::Session, m.tasks <= 1 && m.items <= 1, || {
                format!("message {}: {} tasks, {} history items", m.message_id, m.tasks, m.items)
            });
        }
    }
}

fn message(tasks: u32, items: u32) -> TextRecording {
    let m = MessageRecord { message_id: "msg_1".into(), items, tasks };
    TextRecording { messages: vec![m], ..TextRecording::default() }
}

fn once() -> TextRecording {
    message(1, 1)
}

fn two_tasks() -> TextRecording {
    message(2, 1)
}

fn twice_in_history() -> TextRecording {
    message(1, 2)
}
