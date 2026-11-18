//! Execute cancellation decisions through the original task's client, journal and receipts.

use super::*;
use lk_agents_abi::delegation::remote::RemoteTaskRef;
use lk_agents_abi::session::SessionRecovery;
use lk_agents_core::delegation::remote::cancel::{self, CancellationStep, Control};

impl Delegation {
    pub(super) async fn cancel_command(
        &mut self,
        task: RemoteTaskRef,
        done: driver::Completion,
        reply: &mut Option<Owned<Reply>>,
    ) -> Result<Option<RemoteEvent>, AgentClientError> {
        let control = self.cancellation(&task).await;
        let mut control = match control {
            Ok(control) => control,
            Err(error) => {
                done.resolve(Err(error));
                return Ok(None);
            }
        };
        if let Some(previous) = reply.take()
            && let Err(error) = close_observation(previous).await
        {
            let failure = protocol(error.to_string());
            done.resolve(Err(error));
            return Err(failure);
        }
        loop {
            let observed = match &control {
                Control::Dispatch { request_id, task, request } => {
                    let observer =
                        self.observer_for_control(request_id.clone()).ok_or_else(|| {
                            protocol("task cancellation needs its active native owner")
                        })?;
                    self.client
                        .cancel_observed(task.clone(), request.clone(), &self.request.cx, observer)
                        .await
                }
                Control::Observe { task } | Control::Reconcile { task, .. } => {
                    let observer = self.observer();
                    self.client.get_task(task.clone(), &self.request.cx, observer).await
                }
            };
            match control.observed(observed) {
                CancellationStep::Operation(next) => control = next,
                CancellationStep::Finished(result) => {
                    if let Some(task) = &result.confirmed_task {
                        // Closing prevents new admission while recv drains every queued
                        // command, including a sender that already reserved its queue slot.
                        self.commands.close();
                        while let Some(command) = self.commands.recv().await {
                            let outcome = match command.message {
                                driver::Message::Cancel(queued) => {
                                    cancel::queued_cancel(task, &queued)
                                        .map_err(AgentClientError::from)
                                }
                                _ => Err(AgentClientError::InvalidInput(
                                    "the remote task was canceled before this command ran".into(),
                                )),
                            };
                            command.done.resolve(outcome);
                        }
                    }
                    done.resolve(result.confirmation.map_err(AgentClientError::from));
                    let task = result.observation.map_err(AgentClientError::from)?;
                    return self.interaction.reconciled(a2a::StreamResponse::Task(task)).map(Some);
                }
            }
        }
    }

    /// Read serialized recovery when this queued control starts, after earlier controls have
    /// checkpointed. The core selects its operation; this adapter only lends current resources.
    async fn cancellation(&mut self, task: &RemoteTaskRef) -> Result<Control, AgentClientError> {
        let state = self.interaction.0.lock().policy.snapshot();
        let owner = self
            .request
            .owner
            .as_ref()
            .ok_or_else(|| protocol("task cancellation needs an active conversation owner"))?;
        let endpoint = state.id.endpoint.clone();
        let recovery = owner
            .recover(&endpoint, &endpoint, &SessionRecovery::Cancellation(task.clone()))
            .await
            .map_err(input_error)?;
        cancel::prepare(&state, task, recovery, lk_agents_net::strings::shortuuid("request_"))
            .map_err(AgentClientError::from)
    }
}
