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
    HeldInput {
        remote_id: String,
        endpoint: String,
        task_id: Option<String>,
        done: signal::Producer<
            Option<
                Result<
                    Option<lk_agents_abi::session::InputRequestRef>,
                    lk_agents_abi::storage::StorageError,
                >,
            >,
        >,
    },
    Task {
        remote_id: String,
        endpoint: String,
        task_id: String,
        done: Receipt<lk_agents_abi::delegation::remote::RemoteTaskRef>,
    },
    Response {
        reference: lk_agents_abi::session::InputRequestRef,
        done: signal::Producer<
            Option<Result<lk_agents_abi::ChatMessage, lk_agents_abi::storage::StorageError>>,
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
        origin: lk_agents_abi::session::ToolCallRef,
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
            Self::AdmitInput { done, .. } => done.resolve(Err(error)),
            Self::Flush { done } => done.resolve(Err(error)),
            Self::HeldInput { done, .. } => done.resolve(Err(error)),
            Self::Task { done, .. } => done.resolve(Err(error)),
            Self::Response { done, .. } => done.resolve(Err(error)),
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

    fn held_input<'a>(
        &'a self,
        remote_id: &'a str,
        endpoint: &'a str,
        task_id: Option<&'a str>,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            Option<lk_agents_abi::session::InputRequestRef>,
            lk_agents_abi::storage::StorageError,
        >,
    > {
        Box::pin(async move {
            use lk_agents_abi::storage::StorageError;
            let sender = &self.sender;
            let (done, settled) = signal::once();
            sender
                .send(SessionCommand::HeldInput {
                    remote_id: remote_id.into(),
                    endpoint: endpoint.into(),
                    task_id: task_id.map(str::to_owned),
                    done,
                })
                .await
                .map_err(|_| StorageError::Invalid("the conversation owner ended".into()))?;
            settled.wait().await.map_err(|_| {
                StorageError::Invalid(
                    "the conversation owner ended before selecting the response".into(),
                )
            })?
        })
    }

    fn task<'a>(
        &'a self,
        remote_id: &'a str,
        endpoint: &'a str,
        task_id: &'a str,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            lk_agents_abi::delegation::remote::RemoteTaskRef,
            lk_agents_abi::storage::StorageError,
        >,
    > {
        Box::pin(async move {
            use lk_agents_abi::storage::StorageError;
            let (done, settled) = signal::once();
            self.sender
                .send(SessionCommand::Task {
                    remote_id: remote_id.into(),
                    endpoint: endpoint.into(),
                    task_id: task_id.into(),
                    done,
                })
                .await
                .map_err(|_| StorageError::Invalid("the conversation owner ended".into()))?;
            settled.wait().await.map_err(|_| {
                StorageError::Invalid(
                    "the conversation owner ended before resolving the task".into(),
                )
            })?
        })
    }

    fn response<'a>(
        &'a self,
        reference: &'a lk_agents_abi::session::InputRequestRef,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<lk_agents_abi::ChatMessage, lk_agents_abi::storage::StorageError>,
    > {
        Box::pin(async move {
            use lk_agents_abi::storage::StorageError;
            let sender = &self.sender;
            let (done, settled) = signal::once();
            sender
                .send(SessionCommand::Response { reference: reference.clone(), done })
                .await
                .map_err(|_| StorageError::Invalid("the conversation owner ended".into()))?;
            settled.wait().await.map_err(|_| {
                StorageError::Invalid(
                    "the conversation owner ended before reading the response".into(),
                )
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
        origin: &'a lk_agents_abi::session::ToolCallRef,
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
                    origin: origin.clone(),
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
