//! The native conversation command boundary shared by execution drivers.
//! Decisions remain in the core; the receiving driver checkpoints before acknowledging writes.

use lk_agents_net::signal;
use tokio::sync::mpsc;

pub(crate) enum SessionCommand {
    HeldInput {
        remote_id: String,
        endpoint: String,
        done: signal::Producer<
            Option<
                Result<
                    Option<lk_agents_abi::session::InputRequestRef>,
                    lk_agents_abi::storage::StorageError,
                >,
            >,
        >,
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

/// The sender retains no owner or saved authority. The receiving execution owns each receipt;
/// ending it drops queued producers and settles every caller, even if another sender survives.
#[derive(Clone)]
pub(crate) struct CommandPort(mpsc::Sender<SessionCommand>);

impl CommandPort {
    pub(crate) fn channel(capacity: usize) -> (Self, mpsc::Receiver<SessionCommand>) {
        let (sender, receiver) = mpsc::channel(capacity);
        (Self(sender), receiver)
    }
}

impl lk_agents_abi::session::SessionCommands for CommandPort {
    fn held_input<'a>(
        &'a self,
        remote_id: &'a str,
        endpoint: &'a str,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            Option<lk_agents_abi::session::InputRequestRef>,
            lk_agents_abi::storage::StorageError,
        >,
    > {
        Box::pin(async move {
            use lk_agents_abi::storage::StorageError;
            let sender = &self.0;
            let (done, settled) = signal::once();
            sender
                .send(SessionCommand::HeldInput {
                    remote_id: remote_id.into(),
                    endpoint: endpoint.into(),
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

    fn response<'a>(
        &'a self,
        reference: &'a lk_agents_abi::session::InputRequestRef,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<lk_agents_abi::ChatMessage, lk_agents_abi::storage::StorageError>,
    > {
        Box::pin(async move {
            use lk_agents_abi::storage::StorageError;
            let sender = &self.0;
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
            let sender = &self.0;
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
            let sender = &self.0;
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
            let sender = &self.0;
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
            let sender = &self.0;
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
        let (port, mut owner) = CommandPort::channel(1);
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
