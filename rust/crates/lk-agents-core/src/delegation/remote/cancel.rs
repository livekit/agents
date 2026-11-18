//! Cancellation decisions over the original task owner's observations and journal recovery.

use super::InteractionState;
use crate::a2a::client::ClientError;
use crate::a2a::rpc::TaskRef;
use lk_agents_abi::a2a_protocol as a2a;
use lk_agents_abi::delegation::remote::{
    A2ARequest, RemoteAgentState, RemoteCancellationEvidence, RemoteControlScope, RemoteFailure,
    RemoteRecovery, RemoteTaskRef,
};

/// One approved native operation. This transient decision record owns no client or credential.
#[derive(Debug)]
pub enum Control {
    Dispatch { request_id: String, task: TaskRef, request: a2a::CancelTaskRequest },
    Observe { task: TaskRef },
    Reconcile { task: TaskRef, error: ClientError },
}

pub enum CancellationStep {
    Operation(Control),
    Finished(Box<CancellationResult>),
}

/// The observation belongs to the original call; confirmation belongs to the explicit control.
/// A refused or uncertain cancellation cannot discard a valid observation of continuing work.
pub struct CancellationResult {
    pub observation: Result<a2a::Task, ClientError>,
    pub confirmation: Result<(), ClientError>,
    /// Settle every already-admitted cancellation of this exact task before its driver ends.
    pub confirmed_task: Option<RemoteTaskRef>,
}

pub fn prepare(
    state: &InteractionState,
    target: &RemoteTaskRef,
    recovery: Option<RemoteRecovery>,
    fresh_request_id: String,
) -> Result<Control, ClientError> {
    if state.pending_task_id.as_ref() != Some(&target.task_id)
        || state.context_id.as_ref() != Some(&target.context_id)
    {
        return Err(ClientError::InvalidInput(
            "the live driver no longer owns this remote task".into(),
        ));
    }
    operation(target, recovery, fresh_request_id)
}

/// A retained task is controlled only after its caller has validated the current scoped
/// authority and read its exact recovery decision from the serialized journal owner.
pub fn operation(
    target: &RemoteTaskRef,
    recovery: Option<RemoteRecovery>,
    fresh_request_id: String,
) -> Result<Control, ClientError> {
    if target.task_id.is_empty() || target.context_id.is_empty() {
        return Err(ClientError::InvalidInput("task control needs an exact saved identity".into()));
    }
    let task = TaskRef { id: target.task_id.clone(), context_id: target.context_id.clone() };
    match recovery {
        None => Ok(Control::Dispatch {
            request_id: fresh_request_id,
            request: a2a::CancelTaskRequest { id: task.id.clone(), metadata: None, tenant: None },
            task,
        }),
        Some(RemoteRecovery::Dispatch { request_id, request, streaming: false, .. }) => {
            let A2ARequest::Cancel(request) = *request else {
                return Err(ClientError::InvalidInput(
                    "cancellation recovery needs its original control request".into(),
                ));
            };
            if request.id != task.id {
                return Err(ClientError::InvalidInput(
                    "the saved cancellation belongs to another task".into(),
                ));
            }
            Ok(Control::Dispatch { request_id, task, request })
        }
        Some(RemoteRecovery::GetTask(reference)) if reference == *target => {
            Ok(Control::Observe { task })
        }
        Some(_) => Err(ClientError::InvalidInput(
            "the cancellation has no safe recoverable operation".into(),
        )),
    }
}

impl Control {
    /// A write can commit before its acknowledgement is lost. Only a task observation may
    /// confirm cancellation; failure never replaces the recorded control with another send.
    pub fn observed(self, observation: Result<a2a::Task, ClientError>) -> CancellationStep {
        match (self, observation) {
            (Self::Dispatch { task, .. }, Err(error)) => {
                CancellationStep::Operation(Self::Reconcile { task, error })
            }
            (Self::Reconcile { error, .. }, Err(observation)) => {
                finish(Err(observation), Err(error))
            }
            (Self::Observe { .. }, Err(error)) => finish(Err(error.clone()), Err(error)),
            (control, Ok(task)) => {
                let confirmation = if task.status.state == a2a::TaskState::Canceled {
                    Ok(())
                } else if let Self::Reconcile { error, .. } = control {
                    Err(error)
                } else {
                    Err(ClientError::Protocol(
                        "cancellation is unconfirmed; inspect the retained request and task state"
                            .into(),
                    ))
                };
                finish(Ok(task), confirmation)
            }
        }
    }
}

fn finish(
    observation: Result<a2a::Task, ClientError>,
    confirmation: Result<(), ClientError>,
) -> CancellationStep {
    let confirmed_task = if confirmation.is_ok() {
        observation.as_ref().ok().map(|task| RemoteTaskRef {
            task_id: task.id.clone(),
            context_id: task.context_id.clone(),
        })
    } else {
        None
    };
    CancellationStep::Finished(Box::new(CancellationResult {
        observation,
        confirmation,
        confirmed_task,
    }))
}

/// A terminal cancellation settles matching queued controls, never another task's command.
pub fn queued_cancel(confirmed: &RemoteTaskRef, queued: &RemoteTaskRef) -> Result<(), ClientError> {
    if confirmed == queued {
        Ok(())
    } else {
        Err(ClientError::InvalidInput("the queued cancellation names another remote task".into()))
    }
}

/// Resource choice is explicit; an absent configured driver never widens saved authority.
pub enum CancellationRoute {
    Confirmed,
    Configured,
    Retained { recovery: Option<RemoteRecovery> },
}

pub fn route(
    remote: &RemoteAgentState,
    target: &RemoteTaskRef,
    configured: bool,
    current: Option<&RemoteControlScope>,
    retry_after: Option<&str>,
) -> Result<CancellationRoute, lk_agents_abi::storage::StorageError> {
    use lk_agents_abi::session::SessionRecovery;
    use lk_agents_abi::storage::StorageError;
    if !crate::session::remote::cancel_required(remote, target)? {
        return Ok(CancellationRoute::Confirmed);
    }
    if configured {
        return Ok(CancellationRoute::Configured);
    }
    let current = current.ok_or_else(|| {
        StorageError::Invalid(
            "this task has no configured client or verified current bound delegation scope".into(),
        )
    })?;
    authorize(remote, target, current).map_err(|error| StorageError::Invalid(error.to_string()))?;
    let recovery = crate::session::remote::recover(
        remote,
        &SessionRecovery::Cancellation {
            task: target.clone(),
            retry_after: retry_after.map(str::to_owned),
        },
    )?;
    Ok(CancellationRoute::Retained { recovery })
}

/// Refusals defined by the SDK prove that this control was not performed. Transport errors,
/// internal errors, invalid peer responses and unknown codes leave the write uncertain.
pub fn definitely_refused(code: i32) -> bool {
    use a2a::error_code;
    matches!(
        code,
        error_code::TASK_NOT_CANCELABLE
            | error_code::TASK_NOT_FOUND
            | error_code::UNSUPPORTED_OPERATION
            | error_code::METHOD_NOT_FOUND
            | error_code::INVALID_REQUEST
            | error_code::INVALID_PARAMS
    )
}

/// Freeze the latest definitely refused operation at explicit intent admission. A future
/// intent can retry it, while callers overlapping the original attempt share its result.
pub fn retry_after(remote: &RemoteAgentState, target: &RemoteTaskRef) -> Option<String> {
    retry_after_evidence(remote.cancellation_evidence(target).as_ref())
}

/// Only a refusal already observed by this caller can authorize a fresh explicit intent.
/// Waiting for another owner/fence never upgrades earlier intent into permission to resend.
pub fn retry_after_evidence(evidence: Option<&RemoteCancellationEvidence>) -> Option<String> {
    let evidence = evidence?;
    if !evidence.ended || !evidence.dispatch_started {
        return None;
    }
    match &evidence.failure {
        Some(RemoteFailure::Rpc { error }) if definitely_refused(error.code) => {
            Some(evidence.request_id.clone())
        }
        _ => None,
    }
}

/// A saved address is not authority. Only a send originally covered by a locally verified
/// grant for this same project, principal and bound thread can use current worker credentials.
pub fn authorize(
    remote: &RemoteAgentState,
    target: &RemoteTaskRef,
    current: &RemoteControlScope,
) -> Result<(), ClientError> {
    let observed = remote.tasks.get(&target.task_id).ok_or_else(|| {
        ClientError::InvalidInput("this bound conversation does not retain the task".into())
    })?;
    if observed.task.context_id != target.context_id || current.thread_id != target.context_id {
        return Err(ClientError::InvalidInput(
            "the current bound thread does not authorize this remote context".into(),
        ));
    }
    if !remote.requests.values().any(|request| {
        request.task.as_ref() == Some(target) && request.control_scope.as_ref() == Some(current)
    }) {
        return Err(ClientError::InvalidInput(
            "this task has no verified delegation scope for the current bound project and thread"
                .into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "cancel_tests.rs"]
mod tests;
