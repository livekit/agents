//! Conclusive accepted task observations, reconstructed without replaying mutable assembly.

use super::{RemoteError, invalid};
use lk_agents_abi::a2a_protocol as a2a;
use lk_agents_abi::delegation::remote::{
    A2AJournalPayload, A2AResponse, RemoteAgentState, RemoteTaskRef,
};
use std::collections::BTreeSet;

/// A stale transport observation cannot reopen work the owner already accepted as terminal.
/// Only its exact accepted terminal status supplies presentation flags. Result artifacts must
/// be complete within that status's own request prefix or explicit authoritative Task value.
pub fn observed(
    state: &RemoteAgentState,
    target: &RemoteTaskRef,
) -> Result<Option<a2a::SendMessageResponse>, RemoteError> {
    let saved = state.tasks.get(&target.task_id).ok_or_else(|| invalid("unknown observed task"))?;
    if target.task_id.is_empty()
        || target.context_id.is_empty()
        || saved.task.id != target.task_id
        || saved.task.context_id != target.context_id
    {
        return Err(invalid("the observed task identity or context changed"));
    }
    let Some(sequence) = saved.terminal_event else { return Ok(None) };
    let event = state
        .events
        .get(sequence as usize)
        .filter(|event| event.sequence == sequence)
        .ok_or_else(|| invalid("the accepted terminal event is missing"))?;
    let A2AJournalPayload::Response(response) = &event.payload else {
        return Err(invalid("accepted terminal position does not contain a response"));
    };
    let mut task = status(response, target)
        .filter(|task| {
            task.status.state.is_terminal() && task.status.state == saved.task.status.state
        })
        .ok_or_else(|| invalid("the accepted terminal event does not match this task"))?;
    task.artifacts = Some(artifacts(state, &event.request_id, sequence, target));
    // History is already retained for inspection; it is not a second tool-result transcript.
    task.history = None;
    Ok(Some(a2a::SendMessageResponse::Task(task)))
}

fn matches(task: &a2a::Task, target: &RemoteTaskRef) -> bool {
    task.id == target.task_id && task.context_id == target.context_id
}

fn status(response: &A2AResponse, target: &RemoteTaskRef) -> Option<a2a::Task> {
    match response {
        A2AResponse::Task(task)
        | A2AResponse::Send(a2a::SendMessageResponse::Task(task))
        | A2AResponse::Stream(a2a::StreamResponse::Task(task))
            if matches(task, target) =>
        {
            Some(task.clone())
        }
        A2AResponse::List(list) => list
            .tasks
            .iter()
            .find(|task| matches(task, target) && task.status.state.is_terminal())
            .cloned(),
        A2AResponse::Stream(a2a::StreamResponse::StatusUpdate(update))
            if update.task_id == target.task_id && update.context_id == target.context_id =>
        {
            Some(a2a::Task {
                id: update.task_id.clone(),
                context_id: update.context_id.clone(),
                status: update.status.clone(),
                metadata: update.metadata.clone(),
                artifacts: None,
                history: None,
            })
        }
        _ => None,
    }
}

fn artifacts(
    state: &RemoteAgentState,
    request: &str,
    terminal: u64,
    target: &RemoteTaskRef,
) -> Vec<a2a::Artifact> {
    let mut result: Vec<a2a::Artifact> = Vec::new();
    let mut incomplete = BTreeSet::new();
    for event in
        state.events.iter().take(terminal as usize + 1).filter(|event| event.request_id == request)
    {
        let A2AJournalPayload::Response(response) = &event.payload else { continue };
        if let Some(task) = status(response, target)
            && let Some(artifacts) = task.artifacts
        {
            result = artifacts;
            incomplete.clear();
        }
        let A2AResponse::Stream(a2a::StreamResponse::ArtifactUpdate(update)) = response.as_ref()
        else {
            continue;
        };
        if update.task_id != target.task_id || update.context_id != target.context_id {
            continue;
        }
        let position =
            result.iter().position(|artifact| artifact.artifact_id == update.artifact.artifact_id);
        match (position, update.append) {
            (Some(position), Some(true)) => {
                crate::a2a::client::append_artifact(&mut result[position], update.artifact.clone());
            }
            (Some(position), _) => {
                incomplete.remove(&update.artifact.artifact_id);
                result[position] = update.artifact.clone();
            }
            (None, append) => {
                if append == Some(true) {
                    incomplete.insert(update.artifact.artifact_id.clone());
                }
                result.push(update.artifact.clone());
            }
        }
    }
    // A terminal response after a stream gap must not present cached partial chunks as a
    // complete result. Its SDK state remains inspectable, and completion never triggers resend.
    if incomplete.is_empty() { result } else { Vec::new() }
}

#[cfg(test)]
#[path = "terminal_tests.rs"]
mod tests;
