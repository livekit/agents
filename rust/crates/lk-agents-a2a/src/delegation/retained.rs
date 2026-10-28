//! Retained interaction selection and correlation on the caller's own function-call records.

use std::collections::BTreeMap;

use lk_agents_abi::{ChatItem, FunctionCall};
use serde_json::{Map, Value};

use super::{
    DelegationError, InteractionId, InteractionPhase, InteractionState, RemoteInteraction,
};
use crate::rpc::TaskRef;

pub const STATE_KEY: &str = "lk.delegation.state";

/// A new invocation can continue one pending request of this tool in its supplied conversation.
/// Terminal receipts supersede older pending snapshots; ambiguous destinations need explicit routing.
pub fn for_call(
    id: InteractionId,
    history: &[ChatItem],
) -> Result<RemoteInteraction, DelegationError> {
    let mut latest = BTreeMap::<String, (usize, InteractionState)>::new();
    for (index, item) in history.iter().enumerate() {
        let ChatItem::FunctionCall(call) = item else { continue };
        let Some(value) = call.extra.get(STATE_KEY) else { continue };
        let state: InteractionState =
            serde_json::from_value(value.clone()).map_err(|_| DelegationError::InvalidState)?;
        if state.id.endpoint != id.endpoint || state.id.tool_id != id.tool_id {
            continue;
        }
        if call.name != state.id.tool_id {
            return Err(DelegationError::InvalidState);
        }
        RemoteInteraction::restore(state.clone(), &state.id)?;
        latest.insert(state.id.delegation_id.clone(), (index, state));
    }
    let mut pending = latest
        .values()
        .map(|(_, state)| state)
        .filter(|state| state.phase == InteractionPhase::InputRequired);
    match (pending.next(), pending.next()) {
        (Some(state), None) => RemoteInteraction::restore(state.clone(), &state.id),
        (None, None) => {
            let last_context = latest
                .values()
                .filter(|(_, state)| state.phase == InteractionPhase::Settled)
                .max_by_key(|(index, _)| index)
                .and_then(|(_, state)| state.context_id.clone());
            let mut state = RemoteInteraction::new(id.clone())?.snapshot();
            state.context_id = last_context;
            RemoteInteraction::restore(state, &id)
        }
        _ => Err(DelegationError::InvalidState),
    }
}

pub fn call_extra(
    interaction: &RemoteInteraction,
    call: &FunctionCall,
    livekit: bool,
) -> Result<Map<String, Value>, DelegationError> {
    let state = interaction.snapshot();
    let mut call = call.clone();
    if let (Some(id), Some(context_id)) = (&state.pending_task_id, &state.context_id) {
        super::record_delegation(
            &mut call,
            &state.id,
            &TaskRef { id: id.clone(), context_id: context_id.clone() },
            livekit,
        );
    }
    call.extra.insert(
        STATE_KEY.into(),
        serde_json::to_value(state).map_err(|_| DelegationError::InvalidState)?,
    );
    Ok(call.extra)
}

/// The host record carries the reply decoder's assembled text. Correlation and merged-task
/// semantics are checked again before a retained interaction consumes it.
pub fn update_from_host(
    update: lk_agents_abi::a2a::client::TaskUpdate,
) -> Result<crate::client::TaskUpdate, DelegationError> {
    use lk_agents_abi::storage::TaskState;
    let state = TaskState::parse(&update.state).ok_or(DelegationError::InvalidState)?;
    let follow_task = match update.metadata.get("livekit/mergedInto") {
        None => None,
        Some(Value::String(id))
            if !id.is_empty() && id != &update.task_id && state == TaskState::Completed =>
        {
            Some(TaskRef { id: id.clone(), context_id: update.context_id.clone() })
        }
        Some(_) => return Err(DelegationError::InvalidState),
    };
    Ok(crate::client::TaskUpdate {
        state,
        text: update.text,
        task_id: update.task_id,
        context_id: update.context_id,
        metadata: update.metadata,
        artifacts: Vec::new(),
        follow_task,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::TaskUpdate;
    use lk_agents_abi::storage::TaskState;
    use std::time::UNIX_EPOCH;

    fn identity(conversation: &str) -> InteractionId {
        InteractionId {
            delegation_id: conversation.into(),
            tool_id: "verify_coverage".into(),
            parent_call_id: format!("call-{conversation}"),
            endpoint: "https://expert.test".into(),
        }
    }

    fn call(id: &str) -> FunctionCall {
        FunctionCall {
            id: id.into(),
            call_id: id.into(),
            name: "verify_coverage".into(),
            arguments: "{}".into(),
            created_at: UNIX_EPOCH,
            extra: Map::new(),
            group_id: None,
        }
    }

    fn question(conversation: &str) -> ChatItem {
        let mut interaction = RemoteInteraction::new(identity(conversation)).unwrap();
        interaction.message("help".into(), "message".into(), None, "", None, false, None).unwrap();
        interaction
            .update(&TaskUpdate {
                state: TaskState::InputRequired,
                text: "Which payer?".into(),
                task_id: format!("task-{conversation}"),
                context_id: format!("context-{conversation}"),
                metadata: Map::new(),
                artifacts: vec![],
                follow_task: None,
            })
            .unwrap();
        let mut call = call(&format!("call-{conversation}"));
        call.extra = call_extra(&interaction, &call, false).unwrap();
        ChatItem::FunctionCall(call)
    }

    #[test]
    fn reconstructed_conversations_resume_their_original_task_and_refuse_ambiguity() {
        let first = question("first");
        let second = question("second");
        for (conversation, item) in [("first", first.clone()), ("second", second.clone())] {
            let mut restored = for_call(identity("next"), &[item]).unwrap();
            let message = restored
                .message("Acme".into(), "answer".into(), None, "", None, false, None)
                .unwrap();
            assert_eq!(message.task_id, Some(format!("task-{conversation}")));
            assert_eq!(message.context_id, Some(format!("context-{conversation}")));
            assert_eq!(restored.snapshot().id.parent_call_id, format!("call-{conversation}"));
        }
        assert!(matches!(
            for_call(identity("next"), &[first, second]),
            Err(DelegationError::InvalidState)
        ));
        let unrelated = for_call(identity("new-conversation"), &[]).unwrap();
        assert_eq!(unrelated.snapshot().pending_task_id, None);
    }
    #[test]
    fn a_completed_receipt_supersedes_the_question_and_continues_only_its_context() {
        let question = question("first");
        let mut interaction = for_call(identity("answer"), &[question.clone()]).unwrap();
        interaction.message("Acme".into(), "answer".into(), None, "", None, false, None).unwrap();
        interaction
            .update(&TaskUpdate {
                state: TaskState::Completed,
                text: "Covered".into(),
                task_id: "task-first".into(),
                context_id: "context-first".into(),
                metadata: Map::new(),
                artifacts: vec![],
                follow_task: None,
            })
            .unwrap();
        let mut receipt = call("answer-call");
        receipt.extra = call_extra(&interaction, &receipt, false).unwrap();
        let mut next =
            for_call(identity("new-request"), &[question, ChatItem::FunctionCall(receipt)])
                .unwrap();
        let message = next
            .message("Another question".into(), "next-message".into(), None, "", None, false, None)
            .unwrap();
        assert_eq!(message.context_id.as_deref(), Some("context-first"));
        assert_eq!(message.task_id, None);
        assert_eq!(next.snapshot().id.parent_call_id, "call-new-request");
    }
}
