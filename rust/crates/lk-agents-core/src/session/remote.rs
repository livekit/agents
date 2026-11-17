//! Remote communication decisions over the shared conversation snapshot. The execution owner
//! admits command lifetime and checkpoints the resulting state before acknowledging I/O.

use super::SessionSnapshot;
use lk_agents_abi::delegation::remote::{
    A2AJournalPayload, A2ARequest, RemoteAgentState, RemoteCommand, RemoteRecovery,
};
use lk_agents_abi::session::{InputRequestRef, ToolCallRef};
use lk_agents_abi::storage::StorageError;

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
    super::input::validate_dispatch(snapshot, &command)?;
    let remote_id = command.remote_id;
    let remote = snapshot
        .remote_agents
        .entry(remote_id.clone())
        .or_insert_with(|| RemoteAgentState::new(command.endpoint.clone()));
    if remote.endpoint != command.endpoint {
        return Err(StorageError::Invalid("the saved remote endpoint changed".into()));
    }
    crate::delegation::remote::journal::apply(&remote_id, remote, command.input)
        .map_err(|error| StorageError::Invalid(error.to_string()))
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
                && remote.events.get(request.request_event as usize).is_some_and(|event| {
                    matches!(event.payload, A2AJournalPayload::Request(A2ARequest::Send(_)))
                })
        })
        .max_by_key(|(_, request)| request.request_event);
    let Some((id, _)) = request else { return Ok(None) };
    crate::delegation::remote::journal::recovery(remote, id)
        .map(Some)
        .map_err(|error| StorageError::Invalid(error.to_string()))
}
