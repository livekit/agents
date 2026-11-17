//! Shared continuation rules; storage only loads and checkpoints their data.

use std::collections::BTreeSet;

use lk_agents_abi::{ChatItem, errors::Invalid};

/// An unknown tool outcome cannot be made safe by changing the conversation's delivery mode.
/// Reconcile the outcome before running a model against that saved context.
pub fn validate_resume(history: &[ChatItem]) -> Result<(), Invalid> {
    let answered: BTreeSet<_> = history
        .iter()
        .filter_map(|item| match item {
            ChatItem::FunctionCallOutput(output) => Some(output.call_id.as_str()),
            _ => None,
        })
        .collect();
    if history.iter().any(|item| matches!(item,ChatItem::FunctionCall(call) if !answered.contains(call.call_id.as_str()))) {
        return Err(Invalid("a stored tool has no recorded result; reconcile its outcome before resuming".into()));
    }
    Ok(())
}
