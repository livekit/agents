//! Execute cancellation decisions through the original task's client, journal and receipts.

use super::*;
use lk_agents_abi::delegation::remote::RemoteTaskRef;
use lk_agents_abi::session::SessionRecovery;
use lk_agents_core::delegation::remote::cancel::{self, CancellationStep, Control};

impl Delegation {
    pub(super) async fn cancel_command(
        &mut self,
        task: RemoteTaskRef,
        retry_after: Option<String>,
        done: driver::Completion,
        reply: &mut Option<Owned<Reply>>,
    ) -> Result<Option<RemoteEvent>, AgentClientError> {
        let control = self.cancellation(&task, retry_after).await;
        let control = match control {
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
        let client = Arc::clone(&self.client);
        let cx = self.request.cx.clone();
        let result = execute(&client, control, &cx, |step| {
            match step {
                Control::Dispatch { request_id, .. } => {
                    self.observer_for_control(request_id.clone())
                }
                Control::Observe { .. } | Control::Reconcile { .. } => self.observer(),
            }
            .expect("cancellation admission retained its native journal owner")
        })
        .await;
        if let Some(task) = &result.confirmed_task {
            self.commands.close();
            while let Some(command) = self.commands.recv().await {
                let outcome = match command.message {
                    driver::Message::Cancel { task: queued, .. } => {
                        cancel::queued_cancel(task, &queued).map_err(AgentClientError::from)
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
        self.interaction.reconciled(a2a::StreamResponse::Task(task)).map(Some)
    }

    /// Read serialized recovery when this queued control starts, after earlier controls have
    /// checkpointed. The core selects its operation; this adapter only lends current resources.
    async fn cancellation(
        &mut self,
        task: &RemoteTaskRef,
        retry_after: Option<String>,
    ) -> Result<Control, AgentClientError> {
        let state = self.interaction.0.lock().policy.snapshot();
        let owner = self
            .request
            .owner
            .as_ref()
            .ok_or_else(|| protocol("task cancellation needs an active conversation owner"))?;
        let recovery = owner
            .recover(
                &state.id.remote_id,
                &state.id.endpoint,
                &SessionRecovery::Cancellation { task: task.clone(), retry_after },
            )
            .await
            .map_err(input_error)?;
        cancel::prepare(&state, task, recovery, lk_agents_net::strings::shortuuid("request_"))
            .map_err(AgentClientError::from)
    }
}

/// One transport loop serves original live drivers, retained active-owner controls and idle
/// fenced controls. Only the core selects dispatch, observation, reconciliation and success.
pub(super) async fn execute(
    client: &AgentClient,
    mut control: Control,
    cx: &telemetry::Cx,
    mut observer: impl FnMut(&Control) -> Arc<dyn lk_agents_net::a2a_client::observer::RequestObserver>,
) -> Box<lk_agents_core::delegation::remote::cancel::CancellationResult> {
    loop {
        let journal = observer(&control);
        let observed = match &control {
            Control::Dispatch { task, request, .. } => {
                client.cancel_observed(task.clone(), request.clone(), cx, journal).await
            }
            Control::Observe { task } | Control::Reconcile { task, .. } => {
                client.get_task(task.clone(), cx, Some(journal)).await
            }
        };
        match control.observed(observed) {
            CancellationStep::Operation(next) => control = next,
            CancellationStep::Finished(result) => return result,
        }
    }
}
