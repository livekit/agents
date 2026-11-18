//! Shared continuation rules; storage only loads and checkpoints their data.

use std::collections::BTreeSet;

use lk_agents_abi::session::snapshot::SessionSnapshot;
use lk_agents_abi::{ChatItem, errors::Invalid};

use crate::llm::history::{ItemFacts, tool_pairs};

pub(super) fn validate_voice_graph(state: &SessionSnapshot) -> Result<(), Invalid> {
    if state.active_frame_id != state.root_frame_id
        || !state.ancestors.is_empty()
        || state.frames.len() != 1
        || state.frames.values().any(|agent| !agent.pending_calls.is_empty())
    {
        return Err(Invalid(
            "saved local agent work must continue through TextSession or bound Agent.run".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "resume_tests.rs"]
mod tests;

/// Voice resumes the durable root, never a saved text child or a transient speaking role.
pub fn validate_voice(state: &SessionSnapshot, agent: &str) -> Result<(), Invalid> {
    validate_voice_graph(state)?;
    let root = state
        .frames
        .get(&state.root_frame_id)
        .ok_or_else(|| Invalid("the conversation has no root agent".into()))?;
    if root.agent_name != agent {
        return Err(Invalid(format!(
            "saved voice agent {:?} does not match configured agent {agent:?}",
            root.agent_name,
        )));
    }
    Ok(())
}

/// An unknown tool outcome cannot be made safe by changing the conversation's delivery mode.
/// Reconcile the outcome before running a model against that saved context.
pub fn validate_resume(history: &[ChatItem]) -> Result<(), Invalid> {
    if history.iter().any(|item| {
        matches!(item,
        ChatItem::FunctionCall(call) if crate::tool_calls::provisional(call))
    }) {
        return Err(Invalid(
            "a live local tool has no final outcome; its progress cannot be resumed".into(),
        ));
    }
    let facts: Vec<_> = history.iter().map(ItemFacts::of).collect();
    let answered: BTreeSet<_> = tool_pairs(&facts).into_iter().map(|(call, _)| call).collect();
    if facts.iter().enumerate().any(|(index, item)| {
        item.kind == lk_agents_abi::ItemKind::FunctionCall && !answered.contains(&index)
    }) {
        return Err(Invalid(
            "a stored tool has no recorded result; reconcile its outcome before resuming".into(),
        ));
    }
    Ok(())
}
