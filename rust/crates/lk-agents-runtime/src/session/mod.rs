//! The native conversation command boundary shared by execution drivers.
//! Decisions remain in the core; the receiving driver checkpoints before acknowledging writes.

mod drivers;

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
        remote_id: String,
        endpoint: String,
        task: lk_agents_abi::delegation::remote::RemoteTaskRef,
        drivers: Arc<drivers::Drivers>,
        /// None is the core's already-observed cancellation, with no control I/O needed.
        done: Receipt<Option<Arc<dyn RemoteTaskDriver>>>,
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
            Self::RoutingContext { done, .. } => done.resolve(Err(error)),
            Self::AdmitCall { done, .. } => done.resolve(Err(error)),
            Self::RemoteDriver { done, .. } => done.resolve(Err(error)),
            Self::CancelRemote { done, .. } => done.resolve(Err(error)),
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
        (Self { sender, drivers: Arc::new(drivers::Drivers::default()), routing_model }, receiver)
    }
}

impl lk_agents_abi::session::SessionCommands for CommandPort {
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
        remote_id: &'a str,
        endpoint: &'a str,
        task: &'a lk_agents_abi::delegation::remote::RemoteTaskRef,
    ) -> lk_agents_abi::BoxFuture<'a, Result<(), lk_agents_abi::a2a::client::AgentClientError>>
    {
        Box::pin(async move {
            use lk_agents_abi::a2a::client::AgentClientError;
            let (done, result) = signal::once();
            self.sender
                .send(SessionCommand::CancelRemote {
                    remote_id: remote_id.into(),
                    endpoint: endpoint.into(),
                    task: task.clone(),
                    drivers: self.drivers.clone(),
                    done,
                })
                .await
                .map_err(|_| {
                    AgentClientError::InvalidInput("the conversation owner ended".into())
                })?;
            let driver = result
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
                })?;
            let Some(driver) = driver else { return Ok(()) };
            // The owner loop has released this read receipt before the driver can journal
            // its cancellation. Awaiting control I/O inside that loop would deadlock it.
            // Dropping this waiter leaves any enqueued control with its original driver.
            let cancel = driver.cancel(task.clone());
            tokio::pin!(cancel);
            tokio::select! {
                biased;
                result = &mut cancel => result,
                () = self.sender.closed() => Err(AgentClientError::InvalidInput(
                    "the conversation owner ended before remote cancellation was confirmed".into(),
                )),
            }
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
