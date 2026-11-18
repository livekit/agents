//! Remote communication decisions over the shared conversation snapshot. The execution owner
//! admits command lifetime and checkpoints the resulting state before acknowledging I/O.

use super::SessionSnapshot;
use lk_agents_abi::delegation::remote::{
    A2AJournalPayload, A2ARequest, RemoteAgentState, RemoteCommand, RemoteInput, RemoteRecovery,
    RemoteTaskRef,
};
use lk_agents_abi::session::{InputRequestRef, SessionRecovery, ToolCallRef};
use lk_agents_abi::storage::StorageError;
use std::collections::BTreeSet;

/// Ending a frontend stops new transport operations, while receipts already in flight settle.
pub fn admit(
    input: &lk_agents_abi::delegation::remote::RemoteInput,
    closing: bool,
    ended: bool,
) -> Result<(), StorageError> {
    use lk_agents_abi::delegation::remote::RemoteInput;
    if ended {
        return Err(StorageError::Invalid("the conversation owner has ended".into()));
    }
    if closing && matches!(input, RemoteInput::Prepare { .. } | RemoteInput::DispatchStarted { .. })
    {
        return Err(StorageError::Invalid(
            "the conversation owner is closing; no new remote dispatch is admitted".into(),
        ));
    }
    Ok(())
}

pub fn apply(
    snapshot: &mut SessionSnapshot,
    command: RemoteCommand,
) -> Result<Option<InputRequestRef>, StorageError> {
    if command.remote_id.is_empty() || command.endpoint.is_empty() {
        return Err(StorageError::Invalid("remote identity and endpoint are required".into()));
    }
    if let RemoteInput::Prepare { request_id, request: A2ARequest::Send(send), .. } = &command.input
        && let Some(task_id) = &send.message.task_id
    {
        let remote = state(snapshot, &command.remote_id, &command.endpoint)?
            .ok_or_else(|| StorageError::Invalid("this remote has no known tasks".into()))?;
        // Recovery reuses an immutable prepared request; its peer may already have completed.
        // A new request cannot continue a terminal task or guess an unobserved destination.
        if !remote.requests.contains_key(request_id) {
            let target = task(remote, task_id)?;
            if send.message.context_id.as_ref() != Some(&target.context_id) {
                return Err(StorageError::Invalid("the task belongs to another context".into()));
            }
        }
    }
    let cancel = match &command.input {
        RemoteInput::Prepare { request_id, request: A2ARequest::Cancel(cancel), .. } => {
            Some((request_id, Some(cancel.id.as_str())))
        }
        RemoteInput::DispatchStarted { request_id } => Some((request_id, None)),
        _ => None,
    };
    if let Some((request_id, preparing)) = cancel {
        let remote = state(snapshot, &command.remote_id, &command.endpoint)?;
        match (preparing, remote.and_then(|remote| remote.requests.get(request_id))) {
            (Some(task_id), None) => {
                let remote = remote.ok_or_else(|| {
                    StorageError::Invalid("this remote has no known tasks".into())
                })?;
                task(remote, task_id)?;
            }
            (None, Some(request)) if !request.dispatch_started => {
                let remote = remote.expect("the request belongs to this saved remote");
                let addressed =
                    match remote.event(request.request_event).map(|event| &event.payload) {
                        Some(A2AJournalPayload::Request(A2ARequest::Cancel(cancel))) => {
                            Some(cancel.id.as_str())
                        }
                        Some(A2AJournalPayload::Request(A2ARequest::Send(send))) => {
                            send.message.task_id.as_deref()
                        }
                        _ => None,
                    };
                if let Some(task_id) = addressed {
                    let target = task(remote, task_id)?;
                    if request.task.as_ref() != Some(&target) {
                        return Err(StorageError::Invalid(
                            "the addressed task context changed".into(),
                        ));
                    }
                }
            }
            _ => {}
        }
    }
    super::input::validate_dispatch(snapshot, &command)?;
    // Retention can only advance when an operation settles.
    let settles = matches!(command.input, RemoteInput::Ended { .. } | RemoteInput::Error { .. });
    let unresolved = if settles { unresolved_calls(&snapshot.frames) } else { BTreeSet::new() };
    let remote_id = command.remote_id;
    let remote = snapshot
        .remote_agents
        .entry(remote_id.clone())
        .or_insert_with(|| RemoteAgentState::new(command.endpoint.clone()));
    if remote.endpoint != command.endpoint {
        return Err(StorageError::Invalid("the saved remote endpoint changed".into()));
    }
    let input_request =
        crate::delegation::remote::journal::apply(&remote_id, remote, command.input)
            .map_err(|error| StorageError::Invalid(error.to_string()))?;
    if settles {
        crate::delegation::remote::journal::compact(
            remote,
            crate::delegation::remote::journal::RETAINED_SETTLED_EVENTS,
            |origin| unresolved.contains(&(origin.agent_id.as_str(), origin.call_id.as_str())),
        );
    }
    Ok(input_request)
}

/// Calls whose remote outcome a recovery may still need: no output yet, or a report that is
/// not final. Their originating requests stay in the journal.
fn unresolved_calls(
    frames: &std::collections::BTreeMap<
        lk_agents_abi::session::snapshot::FrameId,
        lk_agents_abi::session::snapshot::SessionFrame,
    >,
) -> BTreeSet<(&str, &str)> {
    use lk_agents_abi::ChatItem;
    use lk_agents_abi::llm::tool_executor::RemoteReport;
    let mut calls = BTreeSet::new();
    for frame in frames.values() {
        let outputs: BTreeSet<&str> = frame
            .history
            .iter()
            .filter_map(|item| match item {
                ChatItem::FunctionCallOutput(output) => Some(output.id.as_str()),
                _ => None,
            })
            .collect();
        for item in &frame.history {
            let ChatItem::FunctionCall(call) = item else { continue };
            let report = crate::tool_calls::remote_report(call);
            let settled = matches!(
                report,
                Some(RemoteReport::Completed | RemoteReport::Failed | RemoteReport::Cancelled)
            );
            if !settled || !outputs.contains(call.output_id().as_str()) {
                calls.insert((frame.id.value.as_str(), call.id.as_str()));
            }
        }
    }
    calls
}

/// A tool can address only work observed on its configured remote in this conversation.
/// Selecting a task grants no permission to consume an input occurrence or a human response.
pub fn task(remote: &RemoteAgentState, task_id: &str) -> Result<RemoteTaskRef, StorageError> {
    let saved = remote.tasks.get(task_id).ok_or_else(|| {
        StorageError::Invalid("the task is unknown on this remote in this conversation".into())
    })?;
    if task_id.is_empty() || saved.task.id != task_id || saved.task.context_id.is_empty() {
        return Err(StorageError::Invalid("the saved task identity is invalid".into()));
    }
    if saved.task.status.state.is_terminal() {
        return Err(StorageError::Invalid(
            "the task has ended; start new work without task_id".into(),
        ));
    }
    Ok(RemoteTaskRef { task_id: task_id.into(), context_id: saved.task.context_id.clone() })
}

/// An active owner can acknowledge an observed cancellation without another transport call.
/// Saved addresses still have to identify this exact task and context.
pub fn cancel_required(
    remote: &RemoteAgentState,
    target: &RemoteTaskRef,
) -> Result<bool, StorageError> {
    let saved = remote.tasks.get(&target.task_id).ok_or_else(|| {
        StorageError::Invalid("the task is unknown on this remote in this conversation".into())
    })?;
    if target.task_id.is_empty()
        || target.context_id.is_empty()
        || saved.task.id != target.task_id
        || saved.task.context_id != target.context_id
    {
        return Err(StorageError::Invalid("the saved task identity or context changed".into()));
    }
    if saved.task.status.state == lk_agents_abi::a2a_protocol::TaskState::Canceled {
        return Ok(false);
    }
    if saved.task.status.state.is_terminal() {
        return Err(StorageError::Invalid(
            "the remote task ended before cancellation was requested".into(),
        ));
    }
    Ok(true)
}

/// Saved addresses are pinned to their configured destination; they carry no credentials.
pub fn state<'a>(
    snapshot: &'a SessionSnapshot,
    remote_id: &str,
    endpoint: &str,
) -> Result<Option<&'a RemoteAgentState>, StorageError> {
    if remote_id.is_empty() || endpoint.is_empty() {
        return Err(StorageError::Invalid("remote identity and endpoint are required".into()));
    }
    let Some(remote) = snapshot.remote_agents.get(remote_id) else { return Ok(None) };
    if remote.endpoint != endpoint {
        return Err(StorageError::Invalid("the saved remote endpoint changed".into()));
    }
    Ok(Some(remote))
}

/// Only the initiating local call can recover its previous send. A later GetTask observes
/// that send; it never replaces the business operation whose outcome is being recovered.
pub fn recovery(
    remote: &RemoteAgentState,
    origin: &ToolCallRef,
) -> Result<Option<RemoteRecovery>, StorageError> {
    let request = remote
        .requests
        .iter()
        .filter(|(_, request)| {
            request.origin.as_ref() == Some(origin)
                && remote.event(request.request_event).is_some_and(|event| {
                    matches!(event.payload, A2AJournalPayload::Request(A2ARequest::Send(_)))
                })
        })
        .max_by_key(|(_, request)| request.request_event);
    let Some((id, _)) = request else { return Ok(None) };
    crate::delegation::remote::journal::recovery(remote, id)
        .map(Some)
        .map_err(|error| StorageError::Invalid(error.to_string()))
}

/// Recover one initiating operation or an explicit cancellation. A cancellation that may have
/// reached its peer is observed by task identity; it is never replaced by another control send.
pub fn recover(
    remote: &RemoteAgentState,
    target: &SessionRecovery,
) -> Result<Option<RemoteRecovery>, StorageError> {
    let (target, retry_after) = match target {
        SessionRecovery::Call(origin) => return recovery(remote, origin),
        SessionRecovery::Observation(task) => {
            return crate::delegation::remote::journal::observed(remote, task)
                .map(|response| {
                    response.map(|response| RemoteRecovery::Observed(Box::new(response)))
                })
                .map_err(|error| StorageError::Invalid(error.to_string()));
        }
        SessionRecovery::Cancellation { task, retry_after } => (task, retry_after.as_deref()),
    };
    let saved = remote.tasks.get(&target.task_id).ok_or_else(|| {
        StorageError::Invalid("the task is unknown on this remote in this conversation".into())
    })?;
    if target.task_id.is_empty()
        || target.context_id.is_empty()
        || saved.task.id != target.task_id
        || saved.task.context_id != target.context_id
    {
        return Err(StorageError::Invalid("the saved task identity or context changed".into()));
    }
    if saved.task.status.state == lk_agents_abi::a2a_protocol::TaskState::Canceled {
        return Ok(Some(RemoteRecovery::GetTask(target.clone())));
    }
    let previous = remote.requests.iter().filter(|(_, request)| {
        request.task.as_ref() == Some(target)
            && remote.event(request.request_event).is_some_and(|event| {
                matches!(&event.payload,
                    A2AJournalPayload::Request(A2ARequest::Cancel(cancel)) if cancel.id == target.task_id)
            })
    }).max_by_key(|(_, request)| request.request_event);
    if let Some((id, _)) = previous {
        if retry_after == Some(id.as_str())
            && crate::delegation::remote::cancel::retry_after(remote, target).as_deref()
                == Some(id.as_str())
            && !saved.task.status.state.is_terminal()
        {
            // Only an intent admitted after this exact refusal can start another request.
            // A concurrent intent frozen before the refusal still recovers the old one.
            return Ok(None);
        }
        if saved.task.status.state.is_terminal() {
            return Ok(Some(RemoteRecovery::GetTask(target.clone())));
        }
        return crate::delegation::remote::journal::recovery(remote, id)
            .map(Some)
            .map_err(|error| StorageError::Invalid(error.to_string()));
    }
    if saved.task.status.state.is_terminal() {
        return Err(StorageError::Invalid(
            "the remote task ended before cancellation was requested".into(),
        ));
    }
    Ok(None)
}

#[cfg(test)]
#[path = "remote_tests.rs"]
mod tests;
