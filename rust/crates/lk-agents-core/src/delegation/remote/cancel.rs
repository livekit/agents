//! Cancellation decisions over the original task owner's observations and journal recovery.

use super::InteractionState;
use crate::a2a::client::ClientError;
use crate::a2a::rpc::TaskRef;
use lk_agents_abi::a2a_protocol as a2a;
use lk_agents_abi::delegation::remote::{A2ARequest, RemoteRecovery, RemoteTaskRef};

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

#[cfg(test)]
#[path = "cancel_tests.rs"]
mod tests;
