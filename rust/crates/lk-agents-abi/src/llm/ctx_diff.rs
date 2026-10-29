//! `compute_chat_ctx_diff` (`llm/utils.py`) as the host calls it: the minimal edit script between
//! two contexts, for a realtime server that can only be told "insert this after that" or "delete
//! this". Core computes it.

use crate::ChatItem;

/// `(previous_item_id, id)`; `None` is first in the conversation.
pub type Anchored = (Option<String>, String);

#[crate::host(functions)]
pub trait ChatContextDiffs {
    /// The operations turning ``old`` into ``new``, as ``(to_remove, to_create, to_update)``; an
    /// update patches the item in place, keeping its server-side position.
    #[host(function)]
    fn compute_chat_ctx_diff(
        old: Vec<ChatItem>,
        new: Vec<ChatItem>,
    ) -> (Vec<String>, Vec<Anchored>, Vec<Anchored>);
}
