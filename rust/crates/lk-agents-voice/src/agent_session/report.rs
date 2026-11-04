//! The session's part of its `SessionReport` (docs/internals/telemetry.md "Session report"), read
//! from the core's own state at one instant. The job adds its identity, tags, events, options
//! and recording.

use crate::agent_session::machine::VoiceDriver;
use crate::telemetry;

impl VoiceDriver {
    /// The session's history, the turn span of each of its items, and its usage as of `now`.
    pub fn report(&self, now: std::time::SystemTime) -> telemetry::SessionReport {
        let turns = self.session.item_turns.items().iter();
        let turns = turns.map(|(item, span)| (item.clone(), (*span).into()));
        telemetry::SessionReport {
            chat_history: self.session.chat_ctx.items().cloned().collect(),
            turns: turns.collect(),
            usage: self.session.usage.snapshot(),
            timestamp: lk_agents_abi::unix_secs(now),
            ..telemetry::SessionReport::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use lk_agents_abi::{ChatItem, ChatMessage, Role};
    use std::time::{Duration, UNIX_EPOCH};

    // Catches: a report reading anything but the session's own committed history.
    #[test]
    fn the_report_snapshots_the_session_history() {
        let mut driver = crate::agent_session::tests::driver();
        let said = ChatMessage::text("item_1", Role::User, "hello", UNIX_EPOCH);
        driver.session.chat_ctx.update_agent_chat_ctx(vec![ChatItem::Message(said)], "agent");
        let now = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        let report = driver.report(now);
        let history: Vec<ChatItem> = driver.session.chat_ctx.items().cloned().collect();
        assert_eq!(report.chat_history, history);
        assert!(report.chat_history.iter().any(|i| i.id() == "item_1"));
        assert_eq!(report.timestamp, 1_700_000_000.0);
        assert!(report.usage.models.is_empty());
    }
}
