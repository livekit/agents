//! Recording, replacement, ordering and final output projection of a voice run.

use lk_agents_abi::voice::events::RunQuestion;
use lk_agents_abi::voice::run_view::{VoiceRun as Contract, VoiceRunOutput};
use lk_agents_abi::{ChatItem, HostRef, Role};
use lk_agents_core::tts::markup::ChatMessageExt;

pub struct VoiceRun {
    items: Vec<ChatItem>,
    settled: bool,
    status: String,
    output: Option<HostRef>,
    question: Option<RunQuestion>,
}

impl Contract for VoiceRun {
    fn new() -> Self {
        Self {
            items: Vec::new(),
            settled: false,
            status: "completed".into(),
            output: None,
            question: None,
        }
    }
    fn record(&mut self, item: ChatItem) {
        if self.settled
            || !matches!(
                item,
                ChatItem::Message(_) | ChatItem::FunctionCall(_) | ChatItem::FunctionCallOutput(_)
            )
        {
            return;
        }
        if let Some(known) = self.items.iter_mut().find(|known| {
            known.id() == item.id()
                && std::mem::discriminant(*known) == std::mem::discriminant(&item)
        }) {
            *known = item;
            return;
        }
        let index = self
            .items
            .iter()
            .rposition(|known| known.created_at() <= item.created_at())
            .map_or(0, |index| index + 1);
        self.items.insert(index, item);
    }
    fn settle(&mut self, status: String, output: Option<HostRef>, question: Option<RunQuestion>) {
        if self.settled {
            return;
        }
        self.settled = true;
        self.status = status;
        self.output = output;
        self.question = question;
    }
    fn status(&self) -> String {
        self.status.clone()
    }
    fn output(&self) -> Option<VoiceRunOutput> {
        if self.status != "completed" {
            return None;
        }
        if let Some(value) = &self.output {
            return Some(VoiceRunOutput::Value(value.clone()));
        }
        self.items
            .iter()
            .rev()
            .find_map(|item| match item {
                ChatItem::Message(message) if message.role == Role::Assistant => Some(message),
                _ => None,
            })
            .and_then(|message| message.text_content().map(VoiceRunOutput::Text))
    }
    fn question(&self) -> Option<String> {
        self.question.as_ref().map(|q| q.question.clone())
    }
    fn verbatim(&self) -> Option<String> {
        self.question.as_ref().filter(|q| q.verbatim).map(|q| q.question.clone())
    }
    fn items(&self) -> Vec<ChatItem> {
        self.items.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::ChatMessage;
    use std::time::{Duration, SystemTime};

    #[test]
    fn recording_replaces_identity_orders_equal_stamps_and_freezes_on_settlement() {
        let mut run = VoiceRun::new();
        let message = |id, text, seconds| {
            ChatMessage::text(
                id,
                Role::Assistant,
                text,
                SystemTime::UNIX_EPOCH + Duration::from_secs(seconds),
            )
            .into()
        };
        run.record(message("later", "old", 2));
        run.record(message("earlier", "first", 1));
        run.record(message("later", "updated", 2));
        run.record(message("equal", "last", 2));
        run.settle("completed".into(), None, None);
        run.record(message("late", "ignored", 3));
        assert_eq!(
            run.items().iter().map(ChatItem::id).collect::<Vec<_>>(),
            vec!["earlier", "later", "equal"]
        );
        assert!(matches!(run.output(), Some(VoiceRunOutput::Text(text)) if text == "last"));
    }

    #[test]
    fn a_question_cannot_fall_back_to_preceding_assistant_text() {
        let mut run = VoiceRun::new();
        run.record(
            ChatMessage::text("message", Role::Assistant, "earlier reply", SystemTime::UNIX_EPOCH)
                .into(),
        );
        run.settle(
            "input_required".into(),
            None,
            Some(RunQuestion { question: "Confirm?".into(), verbatim: true, answer_tool: None }),
        );
        assert!(run.output().is_none());
        assert_eq!(run.verbatim().as_deref(), Some("Confirm?"));
    }

    #[test]
    fn a_last_assistant_message_without_text_cannot_reuse_an_older_reply() {
        let mut run = VoiceRun::new();
        run.record(
            ChatMessage::text("old", Role::Assistant, "older reply", SystemTime::UNIX_EPOCH).into(),
        );
        let mut latest = ChatMessage::text("new", Role::Assistant, "", SystemTime::UNIX_EPOCH);
        latest.content.clear();
        run.record(latest.into());
        run.settle("completed".into(), None, None);
        assert!(run.output().is_none());
    }
}
