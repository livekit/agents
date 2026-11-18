//! The native conversation command boundary shared by execution drivers.
//! Decisions remain in the core; the receiving driver checkpoints before acknowledging writes.

mod drivers;
mod live;

use std::sync::{Arc, Weak};

use lk_agents_abi::a2a::remote::{RemoteCallAdmission, RemoteTaskDriver};
use lk_agents_abi::delegation::routing::{CallRoute, CallRoutingRequest};
use lk_agents_abi::session::InputCandidate;
use lk_agents_abi::storage::StorageError;
use lk_agents_net::signal;
use tokio::sync::mpsc;

pub(crate) type Receipt<T> =
    signal::Producer<Option<Result<T, lk_agents_abi::storage::StorageError>>>;

pub(crate) enum SessionCommand {
    LiveInput(Box<crate::text_session::live::LiveInput>),
    RoutingContext {
        remote_id: String,
        endpoint: String,
        proposed: serde_json::Value,
        selected_task: Option<String>,
        offered: Vec<InputCandidate>,
        done: Receipt<CallRoutingRequest>,
    },
    AdmitCall {
        request: Box<CallRoutingRequest>,
        verdict: Option<Result<CallRoute, lk_agents_abi::errors::ProviderError>>,
        usage: Box<lk_agents_abi::usage::Usage>,
        candidate: Weak<dyn RemoteTaskDriver>,
        drivers: Arc<drivers::Drivers>,
        done: Receipt<RemoteCallAdmission>,
    },
    RemoteDriver {
        remote_id: String,
        endpoint: String,
        task: lk_agents_abi::delegation::remote::RemoteTaskRef,
        drivers: Arc<drivers::Drivers>,
        done: Receipt<Option<Arc<dyn RemoteTaskDriver>>>,
    },
    CancelRemote {
        target: lk_agents_abi::session::control::RemoteControlRef,
        scope: Option<lk_agents_abi::delegation::remote::RemoteControlScope>,
        candidate: Arc<dyn lk_agents_abi::session::control::RemoteCancellation>,
        retry_after: Option<String>,
        drivers: Arc<drivers::Drivers>,
        done: Receipt<lk_agents_abi::session::control::RemoteCancellationAdmission>,
    },
    CancelTask {
        remote_id: String,
        endpoint: String,
        task_id: String,
        scope: Option<lk_agents_abi::delegation::remote::RemoteControlScope>,
        candidate: Arc<dyn lk_agents_abi::session::control::RemoteCancellation>,
        drivers: Arc<drivers::Drivers>,
        done: Receipt<(
            lk_agents_abi::session::control::RemoteControlRef,
            lk_agents_abi::session::control::RemoteCancellationAdmission,
        )>,
    },
    CancelOutcome {
        target: lk_agents_abi::session::control::RemoteControlRef,
        done: Receipt<lk_agents_abi::delegation::remote::RemoteCancelOutcome>,
    },
    AdmitInput {
        request: Box<lk_agents_abi::session::InputRoutingRequest>,
        verdict: Option<
            Result<lk_agents_abi::session::InputRoute, lk_agents_abi::errors::ProviderError>,
        >,
        done: signal::Producer<
            Option<
                Result<
                    lk_agents_abi::session::InputAdmission,
                    lk_agents_abi::storage::StorageError,
                >,
            >,
        >,
    },
    Flush {
        done: signal::Producer<
            Option<Result<serde_json::Value, lk_agents_abi::storage::StorageError>>,
        >,
    },
    Input {
        command: lk_agents_abi::session::SessionInputCommand,
        done: signal::Producer<Option<Result<(), lk_agents_abi::storage::StorageError>>>,
    },
    Apply {
        command: Box<lk_agents_abi::delegation::remote::RemoteCommand>,
        done: signal::Producer<
            Option<
                Result<
                    Option<lk_agents_abi::delegation::remote::InputRequestRef>,
                    lk_agents_abi::storage::StorageError,
                >,
            >,
        >,
    },
    DefaultContext {
        remote_id: String,
        endpoint: String,
        done:
            signal::Producer<Option<Result<Option<String>, lk_agents_abi::storage::StorageError>>>,
    },
    Recover {
        remote_id: String,
        endpoint: String,
        target: lk_agents_abi::session::SessionRecovery,
        done: signal::Producer<
            Option<
                Result<
                    Option<lk_agents_abi::delegation::remote::RemoteRecovery>,
                    lk_agents_abi::storage::StorageError,
                >,
            >,
        >,
    },
}

impl SessionCommand {
    pub(crate) fn fail(self, error: lk_agents_abi::storage::StorageError) {
        match self {
            Self::LiveInput(input) => {
                // Dropping the producer closes this per-turn observer. It cannot retain a
                // success receipt after the sole conversation owner has died.
                drop(input);
            }
            Self::RoutingContext { done, .. } => done.resolve(Err(error)),
            Self::AdmitCall { done, .. } => done.resolve(Err(error)),
            Self::RemoteDriver { done, .. } => done.resolve(Err(error)),
            Self::CancelRemote { done, .. } => done.resolve(Err(error)),
            Self::CancelTask { done, .. } => done.resolve(Err(error)),
            Self::CancelOutcome { done, .. } => done.resolve(Err(error)),
            Self::AdmitInput { done, .. } => done.resolve(Err(error)),
            Self::Flush { done } => done.resolve(Err(error)),
            Self::Input { done, .. } => done.resolve(Err(error)),
            Self::Apply { done, .. } => done.resolve(Err(error)),
            Self::DefaultContext { done, .. } => done.resolve(Err(error)),
            Self::Recover { done, .. } => done.resolve(Err(error)),
        }
    }
}

/// The sender retains no owner or saved authority. The receiving execution owns each receipt;
/// ending it drops queued producers and settles every caller, even if another sender survives.
#[derive(Clone)]
pub(crate) struct CommandPort {
    pub(crate) live: Option<Arc<crate::text_session::live::LiveControl>>,
    sender: mpsc::Sender<SessionCommand>,
    drivers: Arc<drivers::Drivers>,
    routing_model: Option<Arc<dyn lk_agents_abi::Llm>>,
}

impl CommandPort {
    pub(crate) fn channel(
        capacity: usize,
        routing_model: Option<Arc<dyn lk_agents_abi::Llm>>,
    ) -> (Self, mpsc::Receiver<SessionCommand>) {
        let (sender, receiver) = mpsc::channel(capacity);
        (
            Self {
                sender,
                drivers: Arc::new(drivers::Drivers::default()),
                routing_model,
                live: None,
            },
            receiver,
        )
    }

    pub(crate) fn enqueue_live(
        &self,
        input: crate::text_session::live::LiveInput,
    ) -> Result<(), StorageError> {
        self.sender.try_send(SessionCommand::LiveInput(Box::new(input))).map_err(
            |error| match error {
                mpsc::error::TrySendError::Full(_) => {
                    StorageError::Unavailable("the live conversation inbox is full".into())
                }
                mpsc::error::TrySendError::Closed(_) => {
                    StorageError::Invalid("the live conversation owner ended".into())
                }
            },
        )
    }
}

impl lk_agents_abi::session::SessionCommands for CommandPort {
    fn ended(&self) -> lk_agents_abi::BoxFuture<'_, ()> {
        Box::pin(self.sender.closed())
    }

    fn routing_context<'a>(
        &'a self,
        remote_id: &'a str,
        endpoint: &'a str,
        proposed: serde_json::Value,
        selected_task: Option<&'a str>,
        offered: &'a [InputCandidate],
    ) -> lk_agents_abi::BoxFuture<'a, Result<CallRoutingRequest, StorageError>> {
        Box::pin(async move {
            let (done, result) = signal::once();
            self.sender
                .send(SessionCommand::RoutingContext {
                    remote_id: remote_id.into(),
                    endpoint: endpoint.into(),
                    proposed,
                    selected_task: selected_task.map(str::to_owned),
                    offered: offered.to_vec(),
                    done,
                })
                .await
                .map_err(|_| StorageError::Invalid("the conversation owner ended".into()))?;
            result.wait().await.map_err(|_| {
                StorageError::Invalid(
                    "the conversation owner ended before reading call routing context".into(),
                )
            })?
        })
    }

    fn admit_call<'a>(
        &'a self,
        request: &'a CallRoutingRequest,
        verdict: Option<Result<CallRoute, lk_agents_abi::errors::ProviderError>>,
        usage: lk_agents_abi::usage::Usage,
        candidate: Weak<dyn RemoteTaskDriver>,
    ) -> lk_agents_abi::BoxFuture<'a, Result<RemoteCallAdmission, StorageError>> {
        Box::pin(async move {
            let (done, result) = signal::once();
            self.sender
                .send(SessionCommand::AdmitCall {
                    request: Box::new(request.clone()),
                    verdict,
                    usage: Box::new(usage),
                    candidate,
                    drivers: self.drivers.clone(),
                    done,
                })
                .await
                .map_err(|_| StorageError::Invalid("the conversation owner ended".into()))?;
            result.wait().await.map_err(|_| {
                StorageError::Invalid(
                    "the conversation owner ended before admitting the remote call".into(),
                )
            })?
        })
    }

    fn routing_model(&self) -> Option<Arc<dyn lk_agents_abi::Llm>> {
        self.routing_model.clone()
    }

    fn register_remote_driver(
        &self,
        remote_id: &str,
        endpoint: &str,
        request_id: &str,
        driver: Weak<dyn RemoteTaskDriver>,
    ) {
        self.drivers.register(remote_id, endpoint, request_id, driver);
    }

    fn remote_driver<'a>(
        &'a self,
        remote_id: &'a str,
        endpoint: &'a str,
        task: &'a lk_agents_abi::delegation::remote::RemoteTaskRef,
    ) -> lk_agents_abi::BoxFuture<'a, Result<Option<Arc<dyn RemoteTaskDriver>>, StorageError>> {
        Box::pin(async move {
            let (done, result) = signal::once();
            self.sender
                .send(SessionCommand::RemoteDriver {
                    remote_id: remote_id.into(),
                    endpoint: endpoint.into(),
                    task: task.clone(),
                    drivers: self.drivers.clone(),
                    done,
                })
                .await
                .map_err(|_| StorageError::Invalid("the conversation owner ended".into()))?;
            result.wait().await.map_err(|_| {
                StorageError::Invalid(
                    "the conversation owner ended before resolving the live task driver".into(),
                )
            })?
        })
    }

    fn cancel_remote<'a>(
        &'a self,
        target: &'a lk_agents_abi::session::control::RemoteControlRef,
        scope: Option<&'a lk_agents_abi::delegation::remote::RemoteControlScope>,
        candidate: Arc<dyn lk_agents_abi::session::control::RemoteCancellation>,
        retry_after: Option<&'a str>,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            lk_agents_abi::session::control::RemoteCancellationAdmission,
            lk_agents_abi::a2a::client::AgentClientError,
        >,
    > {
        Box::pin(async move {
            use lk_agents_abi::a2a::client::AgentClientError;
            let (done, result) = signal::once();
            self.sender
                .send(SessionCommand::CancelRemote {
                    target: target.clone(),
                    scope: scope.cloned(),
                    candidate,
                    retry_after: retry_after.map(str::to_owned),
                    drivers: self.drivers.clone(),
                    done,
                })
                .await
                .map_err(|_| {
                    AgentClientError::InvalidInput("the conversation owner ended".into())
                })?;
            result
                .wait()
                .await
                .map_err(|_| {
                    AgentClientError::InvalidInput(
                        "the conversation owner ended before admitting task cancellation".into(),
                    )
                })?
                .map_err(|error| match error {
                    StorageError::Invalid(message) => AgentClientError::InvalidInput(message),
                    error => AgentClientError::Protocol(error.to_string()),
                })
        })
    }

    fn cancel_task<'a>(
        &'a self,
        remote_id: &'a str,
        endpoint: &'a str,
        task_id: &'a str,
        scope: Option<&'a lk_agents_abi::delegation::remote::RemoteControlScope>,
        candidate: Arc<dyn lk_agents_abi::session::control::RemoteCancellation>,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            (
                lk_agents_abi::session::control::RemoteControlRef,
                lk_agents_abi::session::control::RemoteCancellationAdmission,
            ),
            lk_agents_abi::a2a::client::AgentClientError,
        >,
    > {
        Box::pin(async move {
            use lk_agents_abi::a2a::client::AgentClientError;
            let (done, result) = signal::once();
            self.sender
                .send(SessionCommand::CancelTask {
                    remote_id: remote_id.into(),
                    endpoint: endpoint.into(),
                    task_id: task_id.into(),
                    scope: scope.cloned(),
                    candidate,
                    drivers: self.drivers.clone(),
                    done,
                })
                .await
                .map_err(|_| {
                    AgentClientError::InvalidInput("the conversation owner ended".into())
                })?;
            result
                .wait()
                .await
                .map_err(|_| {
                    AgentClientError::InvalidInput(
                        "the conversation owner ended before admitting task cancellation".into(),
                    )
                })?
                .map_err(|error| match error {
                    StorageError::Invalid(message) => AgentClientError::InvalidInput(message),
                    error => AgentClientError::Protocol(error.to_string()),
                })
        })
    }

    fn cancel_outcome<'a>(
        &'a self,
        target: &'a lk_agents_abi::session::control::RemoteControlRef,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<lk_agents_abi::delegation::remote::RemoteCancelOutcome, StorageError>,
    > {
        Box::pin(async move {
            let (done, result) = signal::once();
            self.sender
                .send(SessionCommand::CancelOutcome { target: target.clone(), done })
                .await
                .map_err(|_| StorageError::Invalid("the conversation owner ended".into()))?;
            result.wait().await.map_err(|_| {
                StorageError::Invalid(
                    "the conversation owner ended before reading the cancellation".into(),
                )
            })?
        })
    }

    fn admit_input(
        &self,
        request: lk_agents_abi::session::InputRoutingRequest,
        verdict: Option<
            Result<lk_agents_abi::session::InputRoute, lk_agents_abi::errors::ProviderError>,
        >,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<lk_agents_abi::session::InputAdmission, lk_agents_abi::storage::StorageError>,
    > {
        Box::pin(async move {
            use lk_agents_abi::storage::StorageError;
            let (done, result) = signal::once();
            self.sender
                .send(SessionCommand::AdmitInput { request: Box::new(request), verdict, done })
                .await
                .map_err(|_| StorageError::Invalid("the conversation owner ended".into()))?;
            result.wait().await.map_err(|_| {
                StorageError::Invalid(
                    "the conversation owner ended before admitting the response".into(),
                )
            })?
        })
    }

    fn checkpoint(
        &self,
    ) -> lk_agents_abi::BoxFuture<'_, Result<serde_json::Value, lk_agents_abi::storage::StorageError>>
    {
        Box::pin(async move {
            use lk_agents_abi::storage::StorageError;
            let (done, result) = signal::once();
            self.sender
                .send(SessionCommand::Flush { done })
                .await
                .map_err(|_| StorageError::Invalid("the conversation owner ended".into()))?;
            result.wait().await.map_err(|_| {
                StorageError::Invalid("the conversation owner ended before its checkpoint".into())
            })?
        })
    }

    fn input<'a>(
        &'a self,
        command: lk_agents_abi::session::SessionInputCommand,
    ) -> lk_agents_abi::BoxFuture<'a, Result<(), lk_agents_abi::storage::StorageError>> {
        Box::pin(async move {
            use lk_agents_abi::storage::StorageError;
            let sender = &self.sender;
            let (done, settled) = signal::once();
            sender
                .send(SessionCommand::Input { command, done })
                .await
                .map_err(|_| StorageError::Invalid("the conversation owner ended".into()))?;
            settled.wait().await.map_err(|_| {
                StorageError::Invalid(
                    "the conversation owner ended before checkpointing input".into(),
                )
            })?
        })
    }

    fn recover<'a>(
        &'a self,
        remote_id: &'a str,
        endpoint: &'a str,
        target: &'a lk_agents_abi::session::SessionRecovery,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            Option<lk_agents_abi::delegation::remote::RemoteRecovery>,
            lk_agents_abi::storage::StorageError,
        >,
    > {
        Box::pin(async move {
            use lk_agents_abi::storage::StorageError;
            let sender = &self.sender;
            let (done, settled) = signal::once();
            sender
                .send(SessionCommand::Recover {
                    remote_id: remote_id.into(),
                    endpoint: endpoint.into(),
                    target: target.clone(),
                    done,
                })
                .await
                .map_err(|_| StorageError::Invalid("the conversation owner ended".into()))?;
            settled.wait().await.map_err(|_| {
                StorageError::Invalid("the conversation owner ended before reading recovery".into())
            })?
        })
    }

    fn default_context<'a>(
        &'a self,
        remote_id: &'a str,
        endpoint: &'a str,
    ) -> lk_agents_abi::BoxFuture<'a, Result<Option<String>, lk_agents_abi::storage::StorageError>>
    {
        Box::pin(async move {
            use lk_agents_abi::storage::StorageError;
            let sender = &self.sender;
            let (done, settled) = signal::once();
            sender
                .send(SessionCommand::DefaultContext {
                    remote_id: remote_id.into(),
                    endpoint: endpoint.into(),
                    done,
                })
                .await
                .map_err(|_| StorageError::Invalid("the conversation owner ended".into()))?;
            settled.wait().await.map_err(|_| {
                StorageError::Invalid("the conversation owner ended before reading context".into())
            })?
        })
    }

    fn remote<'a>(
        &'a self,
        command: lk_agents_abi::delegation::remote::RemoteCommand,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            Option<lk_agents_abi::delegation::remote::InputRequestRef>,
            lk_agents_abi::storage::StorageError,
        >,
    > {
        Box::pin(async move {
            use lk_agents_abi::storage::StorageError;
            let sender = &self.sender;
            let (done, settled) = signal::once();
            sender
                .send(SessionCommand::Apply { command: Box::new(command), done })
                .await
                .map_err(|_| StorageError::Invalid("the conversation owner ended".into()))?;
            settled.wait().await.map_err(|_| {
                StorageError::Invalid("the conversation owner ended before checkpointing".into())
            })?
        })
    }
}

pub(crate) fn admit_cancellation(
    state: &lk_agents_abi::session::snapshot::SessionSnapshot,
    drivers: &drivers::Drivers,
    target: lk_agents_abi::session::control::RemoteControlRef,
    scope: Option<&lk_agents_abi::delegation::remote::RemoteControlScope>,
    candidate: Arc<dyn lk_agents_abi::session::control::RemoteCancellation>,
    retry_after: Option<&str>,
) -> Result<lk_agents_abi::session::control::RemoteCancellationAdmission, StorageError> {
    use lk_agents_abi::session::control::RemoteCancellationAdmission;
    use lk_agents_core::delegation::remote::cancel::{self, CancellationRoute};
    let remote =
        lk_agents_core::session::remote::state(state, &target.remote_id, &target.endpoint)?
            .ok_or_else(|| StorageError::Invalid("this remote has no known tasks".into()))?;
    let driver = drivers.get(state, &target.remote_id, &target.endpoint, &target.task)?;
    let route = cancel::route(remote, &target.task, driver.is_some(), scope, retry_after)?;
    if matches!(route, CancellationRoute::Confirmed) {
        return Ok(RemoteCancellationAdmission::Confirmed);
    }
    let tracked = drivers.claim_cancellation(&target, candidate.clone())?;
    if !Arc::ptr_eq(&tracked, &candidate) {
        return Ok(RemoteCancellationAdmission::Pending(tracked));
    }
    match route {
        CancellationRoute::Confirmed => unreachable!("confirmed controls need no resource claim"),
        CancellationRoute::Configured => Ok(RemoteCancellationAdmission::Pending(
            driver
                .expect("the core selected the supplied configured resource")
                .cancellation(target.task, retry_after.map(str::to_owned)),
        )),
        CancellationRoute::Retained { recovery } => {
            Ok(RemoteCancellationAdmission::Retained { recovery })
        }
    }
}

/// The model's cancel names a task by ID; the core resolves it and freezes this intent's
/// retry evidence against the owner's current state before the shared admission.
pub(crate) fn admit_task_cancellation(
    state: &lk_agents_abi::session::snapshot::SessionSnapshot,
    drivers: &drivers::Drivers,
    remote_id: String,
    endpoint: String,
    task_id: &str,
    scope: Option<&lk_agents_abi::delegation::remote::RemoteControlScope>,
    candidate: Arc<dyn lk_agents_abi::session::control::RemoteCancellation>,
) -> Result<
    (
        lk_agents_abi::session::control::RemoteControlRef,
        lk_agents_abi::session::control::RemoteCancellationAdmission,
    ),
    StorageError,
> {
    use lk_agents_core::delegation::remote::cancel;
    let remote = lk_agents_core::session::remote::state(state, &remote_id, &endpoint)?;
    let (task, retry_after) = cancel::model_target(remote, task_id)?;
    let target = lk_agents_abi::session::control::RemoteControlRef { remote_id, endpoint, task };
    let admission = admit_cancellation(
        state,
        drivers,
        target.clone(),
        scope,
        candidate,
        retry_after.as_deref(),
    )?;
    Ok((target, admission))
}

pub(crate) fn cancel_outcome(
    state: &lk_agents_abi::session::snapshot::SessionSnapshot,
    target: &lk_agents_abi::session::control::RemoteControlRef,
) -> Result<lk_agents_abi::delegation::remote::RemoteCancelOutcome, StorageError> {
    let remote =
        lk_agents_core::session::remote::state(state, &target.remote_id, &target.endpoint)?
            .ok_or_else(|| StorageError::Invalid("this remote has no known tasks".into()))?;
    Ok(lk_agents_core::delegation::remote::cancel::model_outcome(remote, &target.task))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::session::SessionCommands as _;

    #[tokio::test]
    async fn owner_loss_settles_received_queued_and_capacity_waiting_commands() {
        let (port, mut owner) = CommandPort::channel(1, None);
        let survivor = port.clone();
        let mut received = port.default_context("booking", "https://booking.test");
        std::future::poll_fn(|cx| {
            assert!(received.as_mut().poll(cx).is_pending());
            std::task::Poll::Ready(())
        })
        .await;
        let unacknowledged = owner.recv().await.unwrap();
        let mut queued = survivor.default_context("booking", "https://booking.test");
        let mut waiting = port.default_context("booking", "https://booking.test");
        std::future::poll_fn(|cx| {
            assert!(queued.as_mut().poll(cx).is_pending());
            assert!(waiting.as_mut().poll(cx).is_pending());
            std::task::Poll::Ready(())
        })
        .await;
        drop(owner);
        drop(unacknowledged);
        for receipt in [received, queued, waiting] {
            assert!(matches!(receipt.await, Err(lk_agents_abi::storage::StorageError::Invalid(_))));
        }
        assert!(matches!(
            survivor.default_context("booking", "https://booking.test").await,
            Err(lk_agents_abi::storage::StorageError::Invalid(_))
        ));
    }
}
