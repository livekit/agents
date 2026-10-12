//! Shared test fixtures (consumers use `lk_agents_inference::testkit`).

use std::sync::Arc;
use std::time::SystemTime;

use lk_agents_abi::{ChatItem, ChatMessage, Role};

use crate::realtime::dialect::Dialect;
use crate::realtime::machine::{RealtimeMachine, SessionOptions};
use crate::realtime::wire::SeqIds;

/// A fixed `created_at`, so item conversion is comparable.
pub const TIME: SystemTime = SystemTime::UNIX_EPOCH;

pub fn msg(id: &str, text: &str) -> ChatItem {
    ChatItem::Message(ChatMessage::text(id, Role::User, text, TIME))
}

/// A machine on a dialect, with deterministic ids and no image support.
pub fn machine(dialect: Dialect) -> RealtimeMachine {
    let opts = SessionOptions::from_dialect(&dialect, false);
    let caps = dialect.capabilities(opts.turn_detection.as_ref(), false, true, true);
    RealtimeMachine::new(
        dialect,
        opts,
        caps,
        Arc::new(SeqIds::default()),
        Arc::new(lk_agents_inference::testkit::NoImages),
    )
}
